"""Parity of the exported ONNX graph (onnxruntime) against quan_transcribe (PyTorch), greedy decoding.

    M=/path/to/QUAN-Transcribe-1.0
    HF_HOME=$M/hf HF_HUB_OFFLINE=1 $M/.venv/bin/python scripts/parity.py $M/onnx manifest.tsv out_dir

For every wav in the manifest (wav<TAB>reference) and both modes (normal, streaming) it compares:
  * ORT text vs PyTorch fp32 text (CPU, one file at a time) - must be equal; also the max abs logit diff
  * ORT text vs PyTorch's default path (CUDA bf16 autocast, batched) - reported, not required to match
and writes out_dir/{ort,torch_fp32,torch_default}_{normal,streaming}.tsv as ref<TAB>hyp (for wer.py) plus
out_dir/ort_{normal,streaming}.texts (wav<TAB>hyp, the reference for the Elixir tests).
"""

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

from quan_transcribe import QuanTranscribe
from quan_transcribe.audio import load_audio, split_quiet
from quan_transcribe.model import BLANK

MODEL_ID = "qnguyen3/QUAN-Transcribe-1.0"


def collapse(ids):
    """Greedy CTC on per-frame argmax ids: collapse repeats, drop blanks (as quan_transcribe.model.greedy_decode)."""
    out = [u for i, u in enumerate(ids) if i == 0 or u != ids[i - 1]]
    return [u for u in out if u != BLANK]


def ort_text(sess, asr, wav, streaming):
    parts, logits = [], []
    for piece in split_quiet(wav, 30.0):
        out = sess.run(
            None,
            {
                "wav": piece[None, :].astype(np.float32),
                "wav_lens": np.array([len(piece)], dtype=np.int64),
                "streaming": np.array([int(streaming)], dtype=np.int64),
                "left_chunks": np.array([-1], dtype=np.int64),
            },
        )
        lg, ids, lens = out
        n = int(lens[0])
        logits.append(lg[0, :n])
        units = collapse(ids[0, :n].tolist())
        parts.append(asr.decode_units(units))
    return " ".join(t for t in parts if t), logits


def torch_logits(asr, wav, streaming):
    out = []
    for piece in split_quiet(wav, 30.0):
        lg, lens = asr._forward([piece], streaming, None)
        out.append(lg[0, : int(lens[0])].float().cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_dir")
    ap.add_argument("manifest")
    ap.add_argument("out_dir")
    ap.add_argument("--providers", default="CPUExecutionProvider")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    rows = [l.rstrip("\n").split("\t") for l in open(a.manifest, encoding="utf-8") if l.strip()]

    so = ort.SessionOptions()
    sess = ort.InferenceSession(os.path.join(a.onnx_dir, "model.onnx"), so, providers=a.providers.split(","))
    cpu = QuanTranscribe.from_pretrained(MODEL_ID, device="cpu")
    gpu = QuanTranscribe.from_pretrained(MODEL_ID) if torch.cuda.is_available() else None

    ok = True
    for streaming in (False, True):
        mode = "streaming" if streaming else "normal"
        res = {"ort": [], "torch_fp32": [], "torch_default": []}
        wavs = [load_audio(r[0]) for r in rows]
        if gpu is not None:
            res["torch_default"] = gpu.transcribe(wavs, use_lm=False, streaming=streaming)
        max_diff, mismatches, diff_default = 0.0, 0, 0
        for (path, ref), wav in zip(rows, wavs):
            text, lg = ort_text(sess, cpu, wav, streaming)
            ref_text = cpu.transcribe(wav, use_lm=False, streaming=streaming)
            tl = torch_logits(cpu, wav, streaming)
            max_diff = max([max_diff] + [float(np.abs(x - y).max()) for x, y in zip(lg, tl)])
            res["ort"].append(text)
            res["torch_fp32"].append(ref_text)
            if text != ref_text:
                mismatches += 1
                print(f"MISMATCH {mode} {path}\n  ort:   {text}\n  torch: {ref_text}", flush=True)
        if gpu is not None:
            diff_default = sum(x != y for x, y in zip(res["ort"], res["torch_default"]))
        print(
            f"{mode}: ORT vs PyTorch fp32 text mismatches {mismatches}/{len(rows)}, max abs logit diff {max_diff:.2e}; "
            f"ORT vs PyTorch default (CUDA bf16) differing files {diff_default}/{len(rows)}",
            flush=True,
        )
        ok &= mismatches == 0
        for name, hyps in res.items():
            if hyps:
                with open(os.path.join(a.out_dir, f"{name}_{mode}.tsv"), "w", encoding="utf-8") as f:
                    for (_, ref), h in zip(rows, hyps):
                        f.write(f"{ref}\t{h}\n")
        with open(os.path.join(a.out_dir, f"ort_{mode}.texts"), "w", encoding="utf-8") as f:
            for (path, _), h in zip(rows, res["ort"]):
                f.write(f"{path}\t{h}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
