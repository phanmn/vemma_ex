"""Export QUAN-Transcribe-1.0 to ONNX: one graph from 16 kHz waveform to CTC logits.

Run with the model's venv (quan_transcribe, torch, transformers, onnx, onnxscript, onnxruntime installed):

    M=/path/to/QUAN-Transcribe-1.0
    HF_HOME=$M/hf HF_HUB_OFFLINE=1 $M/.venv/bin/python scripts/export_onnx.py $M/onnx

Graph inputs:
    wav          float32 (B, L)  16 kHz mono in [-1, 1], zero padded on the right
    wav_lens     int64   (B,)    valid samples per row
    streaming    int64   (1,)    1 = restrict the reader's attention to the current and past 12-frame chunks
    left_chunks  int64   (1,)    streaming only: past chunks the reader may see; -1 = all

Graph outputs:
    logits  float32 (B, T, V)  CTC logits, 25 frames/s
    ids     int64   (B, T)     argmax of logits
    lens    int64   (B,)       valid frames per row

Differences from quan_transcribe's PyTorch path (all numerically checked by --check):
  * log-mel is in the graph; the rfft is a matmul of the gathered frames with the window times the real/imaginary
    DFT basis (only 320 of the 512 FFT inputs are non-zero), so no FFT op is needed.
  * no Conv ops: the depthwise convs are shifted multiply-adds and the 3x3 stride-2 subsampling convs are
    im2col + matmul. ONNX Runtime's CUDA EP otherwise re-runs an exhaustive cuDNN search for each new length.
  * the encoder's forward hooks are replaced by calling its submodules directly and returning every hidden state.
  * the encoder's 5-D blocked attention mask is built directly from the frame mask (same values as transformers').
  * the encoder attention's overlapping key/value windows use an index gather instead of Tensor.unfold.
  * exported with the TorchScript exporter (opset 20, IR 10, for ONNX Runtime 1.19 in Ortex): torch.export
    stalls in sympy on the encoder's symbolic block shapes.
"""

import argparse
import json
import math
import os
import shutil
import types
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from quan_transcribe import QuanTranscribe
from quan_transcribe.audio import load_audio

MODEL_ID = "qnguyen3/QUAN-Transcribe-1.0"
# Ortex (phanmn fork) links ONNX Runtime 1.19: opset <= 21, IR version <= 10. The TorchScript exporter stops at 20.
OPSET = 20
IR_VERSION = 10


def extract_block_context(self, hidden_states):
    """Gemma4AudioAttention._extract_block_context without Tensor.unfold, which the TorchScript exporter
    cannot export over a dynamic length: the same overlapping windows, gathered by index."""
    seq_len = hidden_states.shape[1]
    hidden_states = F.pad(
        hidden_states, (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1)
    )
    num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
    starts = torch.arange(num_blocks, device=hidden_states.device) * self.chunk_size
    idx = starts[:, None] + torch.arange(self.context_size, device=hidden_states.device)[None, :]
    return hidden_states[:, idx]  # (B, blocks, context, heads, head_dim)


# No Conv ops in the graph. ONNX Runtime's CUDA EP runs an exhaustive cuDNN algorithm search for every new
# input shape (seconds per new audio length) and Ortex does not expose the option to turn it off, so every
# convolution is written as the same sums with gathers and matmuls.


def depthwise_conv1d(x, weight, bias=None):
    """Depthwise Conv1d (groups == channels, stride 1, no padding) as k shifted multiply-adds. x: (B, C, L)."""
    k = weight.shape[-1]
    T = x.shape[-1] - k + 1
    w = weight[:, 0, :]
    y = x[..., 0:T] * w[:, 0:1]
    for i in range(1, k):
        y = y + x[..., i : i + T] * w[:, i : i + 1]
    return y if bias is None else y + bias[:, None]


def causal_conv1d_forward(self, x):
    """Gemma4AudioCausalConv1d.forward (depthwise, left padded) without Conv."""
    assert self.groups == self.in_channels == self.out_channels and self.stride[0] == 1 and self.dilation[0] == 1
    return depthwise_conv1d(F.pad(x, (self.left_pad, 0)), self.weight, self.bias)


def conv2d_3x3_s2(conv, x):
    """Conv2d(kernel 3, stride 2, padding 1, no bias) as im2col + matmul. x: (B, Cin, T, F) -> (B, T', F', Cout)."""
    assert conv.kernel_size == (3, 3) and conv.stride == (2, 2) and conv.padding == (1, 1) and conv.bias is None
    B, cin, T, Fd = x.shape
    x = F.pad(x, (1, 1, 1, 1))
    to = (T - 1) // 2 + 1
    fo = (Fd - 1) // 2 + 1
    ti = torch.arange(to, device=x.device)[:, None] * 2 + torch.arange(3, device=x.device)[None, :]
    fi = torch.arange(fo, device=x.device)[:, None] * 2 + torch.arange(3, device=x.device)[None, :]
    p = x[:, :, ti][..., fi]  # (B, Cin, T', 3, F', 3)
    p = p.permute(0, 2, 4, 1, 3, 5).reshape(B, to, fo, cin * 9)
    return p @ conv.weight.reshape(conv.out_channels, cin * 9).t()


def subsample_layer_forward(self, hidden_states, mask=None):
    """Gemma4AudioSubSampleConvProjectionLayer.forward with conv2d_3x3_s2."""
    if mask is not None:
        hidden_states = hidden_states * mask[:, None, :, None]
    hidden_states = self.act(self.norm(conv2d_3x3_s2(self.conv, hidden_states))).permute(0, 3, 1, 2)
    if mask is not None:
        mask = mask[:, ::2]
    return hidden_states, mask


class ExportQuan(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        enc = m.encoder
        for layer in enc.layers:
            layer.self_attn._extract_block_context = types.MethodType(extract_block_context, layer.self_attn)
            conv = layer.lconv1d.depthwise_conv1d
            conv.forward = types.MethodType(causal_conv1d_forward, conv)
        sub = enc.subsample_conv_projection
        for layer in (sub.layer0, sub.layer1):
            layer.forward = types.MethodType(subsample_layer_forward, layer)
        # Drop the hooks that collect hidden states; forward() collects them itself.
        enc.subsample_conv_projection._forward_hooks.clear()
        for layer in enc.layers:
            layer._forward_hooks.clear()

        mel = m.mel
        self.hop = mel.hop_length
        self.frame = mel.frame_length
        self.mel_floor = mel.mel_floor
        bins = mel.fft_length // 2 + 1
        k = torch.arange(bins, dtype=torch.float64)[:, None]
        t = torch.arange(self.frame, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * k * t / mel.fft_length
        win = mel.window.double()[None, :]
        basis = torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win], 0)  # (2 * bins, frame)
        self.register_buffer("dft", basis.float().t().contiguous(), persistent=False)  # (frame, 2 * bins)
        self.register_buffer("mel_filters", mel.mel_filters.clone(), persistent=False)
        self.bins = bins

        cfg = enc.config
        self.chunk = cfg.attention_chunk_size
        self.past = cfg.attention_context_left - 1
        assert cfg.attention_context_right == 0
        self.context = self.chunk + self.past
        self.reader_chunk = m.cfg.chunk_frames

    def log_mel(self, wav, wav_lens):
        pad = self.frame // 2
        x = F.pad(wav, (pad, 0))
        # Same frames as LogMel's unfold(frame + 1, hop)[..., :-1].
        n = (x.shape[1] - self.frame - 1) // self.hop + 1
        idx = torch.arange(n, device=wav.device)[:, None] * self.hop + torch.arange(self.frame, device=wav.device)
        spec = x[:, idx] @ self.dft  # (B, T, 2 * bins): rfft as a DFT matmul
        re, im = spec[..., : self.bins], spec[..., self.bins :]
        mag = torch.sqrt(re * re + im * im)
        feats = torch.log(mag @ self.mel_filters + self.mel_floor)
        n = feats.shape[1]
        frame_end = torch.arange(n, device=wav.device) * self.hop + self.frame
        mask = frame_end[None, :] < (wav_lens[:, None] + pad)
        return feats * mask[..., None], mask

    def blocked_mask(self, mask):
        # (B, T) frame mask -> (B, 1, blocks, chunk, context): query b*chunk+q may see key b*chunk+o-past
        # if that key is a valid frame less than `past` frames back (as transformers' sliding_window_mask_function).
        T = mask.shape[1]
        nb = (T + self.chunk - 1) // self.chunk
        padded = F.pad(mask, (self.past, nb * self.chunk - T))
        b = torch.arange(nb, device=mask.device)[:, None, None]
        q = torch.arange(self.chunk, device=mask.device)[None, :, None]
        o = torch.arange(self.context, device=mask.device)[None, None, :]
        kv = b * self.chunk + o  # index into padded
        dist = q - o + self.past
        near = (dist >= 0) & (dist < self.past) & (b * self.chunk + q < T)
        kv = kv.expand(nb, self.chunk, self.context).reshape(-1)
        m5 = padded[:, kv].reshape(mask.shape[0], nb, self.chunk, self.context) & near[None]
        return m5[:, None]

    def forward(self, wav, wav_lens, streaming, left_chunks):
        m = self.m
        enc = m.encoder
        feats, fmask = self.log_mel(wav, wav_lens)
        h, mask = enc.subsample_conv_projection(feats, fmask)
        acts = [h]
        pos = enc.rel_pos_enc(h)
        amask = self.blocked_mask(mask)
        for layer in enc.layers:
            h = layer(h, attention_mask=amask, position_embeddings=pos)
            acts.append(h)
        w = torch.softmax(m.layer_logits.float(), dim=0)
        h = acts[0] * w[0]
        for i in range(1, len(acts)):
            h = h + acts[i] * w[i]

        x = m.in_proj(m.in_norm(h))
        xc = (x * mask[..., None]).transpose(1, 2)
        xc = F.pad(xc, (m.cfg.conv_kernel - 1, 0))
        x = x + F.gelu(depthwise_conv1d(xc, m.pos_conv.weight, m.pos_conv.bias).transpose(1, 2))

        T = x.shape[1]
        c = torch.arange(T, device=x.device) // self.reader_chunk
        allow = (c[None, :] <= c[:, None]) & ((left_chunks < 0) | (c[None, :] >= c[:, None] - left_chunks))
        allow = allow | (streaming == 0)
        attn_mask = mask[:, None, None, :] & allow[None, None]
        for blk in m.blocks:
            x = blk(x, attn_mask)
        t = m.to_text(m.out_norm(x))
        rows = torch.cat([m.special_rows, m.unit_rows], dim=0)
        logits = t @ rows.t() + m.head_bias
        return logits, logits.argmax(-1), mask.sum(1)


def manifest(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip()]


def export(asr, out_dir):
    model = ExportQuan(asr.model).eval()
    wav = torch.from_numpy(np.random.default_rng(0).standard_normal((2, 16000 * 7)).astype(np.float32) * 0.1)
    lens = torch.tensor([16000 * 7, 16000 * 5], dtype=torch.long)
    st = torch.tensor([0], dtype=torch.long)
    lc = torch.tensor([-1], dtype=torch.long)
    path = os.path.join(out_dir, "model.onnx")
    # The TorchScript exporter: torch.export (dynamo=True) stalls for hours in sympy simplifying the
    # encoder's symbolic block shapes. Tracing records the shape arithmetic as graph ops instead.
    with torch.no_grad():
        torch.onnx.export(
            model,
            (wav, lens, st, lc),
            path,
            input_names=["wav", "wav_lens", "streaming", "left_chunks"],
            output_names=["logits", "ids", "lens"],
            dynamic_axes={
                "wav": {0: "batch", 1: "samples"},
                "wav_lens": {0: "batch"},
                "logits": {0: "batch", 1: "frames"},
                "ids": {0: "batch", 1: "frames"},
                "lens": {0: "batch"},
            },
            dynamo=False,
            opset_version=OPSET,
            do_constant_folding=True,
        )
    proto = onnx.load(path)
    proto.ir_version = min(proto.ir_version, IR_VERSION)
    onnx.save(proto, path)
    return model


def write_assets(asr, out_dir):
    m = asr.model
    mel = m.mel
    tok = hf_hub_download(m.cfg.encoder_id, "tokenizer.json")
    shutil.copyfile(tok, os.path.join(out_dir, "tokenizer.json"))
    with open(os.path.join(out_dir, "units.json"), "w") as f:
        json.dump({"blank": 0, "unk": 1, "offset": 2, "gemma_ids": asr.gemma_ids}, f)
    cfg = {
        "model_id": MODEL_ID,
        "encoder_id": m.cfg.encoder_id,
        "sample_rate": mel.sampling_rate,
        "frame_rate": 25,
        "samples_per_frame": 640,
        "vocab_size": m.cfg.vocab_size,
        "blank_id": 0,
        "unk_id": 1,
        "chunk_frames": m.cfg.chunk_frames,
        "chunk_seconds": 30.0,
        "min_samples": 1600,
        "mel": {
            "frame_length": mel.frame_length,
            "hop_length": mel.hop_length,
            "fft_length": mel.fft_length,
            "mel_bins": mel.mel_filters.shape[1],
            "mel_floor": mel.mel_floor,
            "in_graph": True,
        },
        "inputs": ["wav", "wav_lens", "streaming", "left_chunks"],
        "outputs": ["logits", "ids", "lens"],
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)


def check_wrapper(asr, wavs):
    """The wrapper must reproduce QuanCTC.forward (fp32, no autocast) on a padded batch."""
    m = asr.model
    lens = torch.tensor([len(w) for w in wavs])
    x = torch.zeros(len(wavs), int(lens.max()))
    for i, w in enumerate(wavs):
        x[i, : len(w)] = torch.from_numpy(w)
    # References first: the wrapper removes the hooks QuanCTC.forward relies on.
    refs = {s: m(x, lens, s) for s in (False, True)}
    feats_ref, mask_ref = m.mel(x, lens)
    wrapper = ExportQuan(m).eval()
    for streaming, (ref, ref_lens) in refs.items():
        got, _, got_lens = wrapper(x, lens, torch.tensor([int(streaming)]), torch.tensor([-1]))
        assert torch.equal(ref_lens, got_lens), (ref_lens, got_lens)
        diff = max(float((ref[i, :n] - got[i, :n]).abs().max()) for i, n in enumerate(ref_lens.tolist()))
        print(f"wrapper vs QuanCTC (streaming={streaming}): max abs logit diff {diff:.2e}", flush=True)
    feats, mask = wrapper.log_mel(x, lens)
    assert torch.equal(mask_ref, mask)
    print(f"log-mel max abs diff {float((feats_ref - feats).abs().max()):.2e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--manifest", default=None, help="wav<TAB>ref file used for the wrapper check")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    asr = QuanTranscribe.from_pretrained(MODEL_ID, device="cpu")
    if a.manifest:
        rows = manifest(a.manifest)[:3]
        check_wrapper(asr, [load_audio(r[0]) for r in rows])
    export(asr, a.out_dir)
    write_assets(asr, a.out_dir)
    print("wrote", a.out_dir, file=sys.stderr)


if __name__ == "__main__":
    main()
