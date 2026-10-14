# VemmaEx

Native BEAM inference for [QUAN-Transcribe-1.0](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0),
a Vietnamese CTC speech recognizer built on EmbeddingGemma 2's audio encoder (356M params).

There is no Python at runtime. The whole network, log-mel included, is a single ONNX graph that
runs through [Ortex](https://github.com/phanmn/ortex) (ONNX Runtime). Greedy CTC decoding and
Gemma detokenization ([`tokenizers`](https://hex.pm/packages/tokenizers)) run in Elixir. EXLA is
not needed.

Output is lowercase Vietnamese with no punctuation, and numbers are spelled out as words.

The interface follows [smart_turn_ex](https://github.com/phanmn/smart_turn_ex): a registry of named
models, auto-download with `mix vemma.download`, and an opt-in `Nx.Serving`.

## Install

```elixir
def deps do
  [
    {:vemma_ex, github: "phanmn/vemma_ex"}
  ]
end
```

Ortex comes from a fork with precompiled NIFs (CPU, ONNX Runtime 1.19), so no Rust toolchain is
needed. See [CUDA](#cuda) to build it with the CUDA provider.

The model (1.4 GB) is downloaded on first load, so there is no separate setup step. To fetch it
ahead of time (baking it into a release image, or avoiding a stall on first boot):

```
mix vemma.download
```

## Use

Name your models and load them at startup, before your supervision children:

```elixir
defmodule MyApp.Application do
  use Application

  @impl true
  def start(_type, _args) do
    VemmaEx.Model.Registry.load!(:asr)

    Supervisor.start_link(children(), strategy: :one_for_one, name: MyApp.Supervisor)
  end
end
```

```elixir
{:ok, text} = VemmaEx.transcribe(:asr, samples)                 # {n} f32 tensor, f32-le binary or list, 16 kHz mono
{:ok, text} = VemmaEx.transcribe_file(:asr, "speech.wav")       # 16 kHz WAV (PCM 16/24/32 or float)
{:ok, text} = VemmaEx.transcribe(:asr, {:file, "speech.wav"}, streaming: true)

# s16 PCM (telephony/WebRTC):
samples = VemmaEx.Audio.from_pcm_s16(pcm)
```

Loading *before* `Supervisor.start_link/2` gives boot-time fail-fast: missing or corrupt files
raise before any child accepts traffic.

A single unnamed model is `VemmaEx.Model.Registry.load!()` (registered as `:default`), reached as
`VemmaEx.transcribe(samples)`. Or skip the registry and hold the struct yourself, with no global
state:

```elixir
model = VemmaEx.load("/models/QUAN-Transcribe-1.0/onnx")
{:ok, text} = VemmaEx.transcribe(model, samples)
```

Every entry point (`transcribe/3`, `transcribe_file/3`, `transcribe_piece/3`,
`transcribe_batch/3`, `VemmaEx.Live.new/2`) takes a `%VemmaEx.Model{}`, a registered name, or no
model (the `:default` name).

Each loaded name is its own ONNX Runtime session, about 1.9 GB resident. Per-call `streaming:` and
`left_chunks:` let one model serve both modes, so you rarely need two.

## Configuration

Models are configured at the call site, not in application config:

```elixir
VemmaEx.Model.Registry.load!(:asr)
VemmaEx.Model.Registry.load!(:pinned, path: "/var/models/QUAN-Transcribe-1.0/onnx")
VemmaEx.Model.Registry.load!(:strict, download: [auto: false])
VemmaEx.Model.Registry.load!(:gpu, providers: [:cuda, :cpu])
```

| option | meaning | default |
| ------ | ------- | ------- |
| `:path` | the model directory; wins over everything else | `priv/models/<version>` |
| `:version` | release tag of the model files | `"model-v1"` |
| `:download` | how to fetch missing files, see below | `[auto: true]` |
| `:providers` | ONNX Runtime execution providers, e.g. `[:cuda, :cpu]` | `[:cpu]` |
| `:optimization_level` | graph optimization 1-3 | `3` |
| `:file` | graph file inside the directory | `"model.onnx"` |
| `:streaming` / `:left_chunks` | defaults for `transcribe/3` | `false` / `nil` (all) |

A model directory holds `model.onnx`, `tokenizer.json`, `units.json` and `config.json`. With no
`:path`, it is `priv/models/<version>` inside `:vemma_ex`'s priv dir, which is where
`mix vemma.download` puts the files.

### Auto-download

`VemmaEx.Model.load/2` fetches missing files over `:httpc`, by default from this repository's
GitHub release assets: `https://github.com/phanmn/vemma_ex/releases/download/<version>/<file>`.
Each file streams to a temporary file next to its destination (the 1.4 GB graph never sits in
memory), its SHA-256 is checked, and only then is it renamed into place. A mismatch discards the
file and raises.

The expected SHA-256 of each version's files ships with the library in
`priv/checksums/<version>.sha256` (`sha256sum` format) and is compiled in, so the checksums come
from the code you depend on, not from the server you download from. Nothing is downloaded without
a checksum: for a version the library does not know, pass `checksums:` yourself.

| sub-option | meaning | default |
| ---------- | ------- | ------- |
| `:auto` | fetch missing files instead of raising | `true` |
| `:repo` | base URL to fetch from (the filename is appended): a mirror, proxy or bucket | the GitHub release of `:version` |
| `:headers` | request headers as `{name, value}` string pairs; values are never logged | none |
| `:checksums` | `%{"model.onnx" => sha256, ...}` for all four files | bundled for the version |
| `:timeout` | per-file request timeout, ms | 1 hour |

```elixir
VemmaEx.Model.Registry.load!(:asr,
  download: [
    repo: "https://artifacts.internal/vemma/model-v1",
    headers: [{"authorization", "Bearer " <> System.fetch_env!("ARTIFACT_TOKEN")}]
  ]
)
```

```
mix vemma.download                                   # model-v1 into priv/models/model-v1
mix vemma.download --output /var/models/asr          # somewhere else
mix vemma.download --repo https://artifacts.internal/vemma/model-v1
mix vemma.download --version model-v1 --force
```

`load!/2` blocks for the length of the download. In production, bake the files into your image or
pass `download: [auto: false]` to fail fast instead.

A custom `:file` that is missing is not downloaded (releases ship `model.onnx`), and raises.

## Concurrency and `Nx.Serving`

Calling `VemmaEx.transcribe/3` runs the forward pass in the calling process. Ortex runs on a dirty
CPU scheduler, so concurrent callers run in parallel and never block a normal scheduler.

`VemmaEx.Nx.Serving` runs it in one serving process that batches pieces from many callers into one
graph run. Each recording is cut at quiet points into pieces of at most `:chunk_seconds` (30 s),
exactly as `transcribe/3` does, each piece is a batch entry, and the texts are joined back per
recording. The graph masks padding, so batched and unbatched text are identical: on the 20 FLEURS
clips, per-frame ids match exactly in both normal and streaming mode (`test/vemma_ex/nx/serving_test.exs`).

```elixir
children = [
  {VemmaEx.Nx.Serving, name: MyApp.ASR, model: :asr, batch_size: 4}
]

{:ok, text} = VemmaEx.Nx.Serving.transcribe(samples, MyApp.ASR)
[{:ok, a}, {:error, :enoent}] = VemmaEx.Nx.Serving.transcribe([samples, {:file, "missing.wav"}], MyApp.ASR)
```

`:streaming` and `:left_chunks` are per serving (one batch is one graph run). `:model` fronts a
registered model instead of loading a second copy.

On CPU, batching is **slower**: the 20 FLEURS clips took 8.7 s one at a time and 20.4 s in batches
of 5 (Ryzen 7 7800X3D). ONNX Runtime already uses every core for one clip, and padding adds work.
Use the serving on a GPU, or when you want its queue as backpressure. `VemmaEx.transcribe_batch/3`
is the same batching without a process.

## Streaming

`streaming: true` makes the reader attend only to the current 0.48 s chunk and earlier ones, so
text for finished chunks never changes; `left_chunks: n` limits how far back it looks.

`VemmaEx.Live` is a port of `quan_transcribe.live.LiveSession` for a live 16 kHz stream:

```elixir
session = VemmaEx.Live.new(:asr)
{finals, session} = VemmaEx.Live.accept(session, samples)   # feed ~0.48 s (7680 samples) at a time
session.partial                                             # text so far of the utterance in progress
{text, session} = VemmaEx.Live.flush(session)
```

`accept/2` re-decodes the current utterance in streaming mode. An utterance ends after 0.8 s with no
new tokens; it is then re-decoded with full context and returned as final.

## Long audio

Audio longer than `:chunk_seconds` (default 30) is cut at the quietest 50 ms window in the last
5 s, and the pieces are joined with a space. This is a port of `split_quiet`. On a 96 s test file
its cut points and the final text match Python's exactly. `VemmaEx.transcribe_piece/3` runs one
forward pass with no splitting.

## Limitations

- Input must be 16 kHz mono. There is no resampler, and WAVs at other rates return
  `{:error, {:sample_rate, rate}}`.
- Only greedy decoding is supported. The upstream KenLM beam search (7.7% WER on FLEURS) is not
  ported. The `logits` output is there if you want to add it.
- Streaming is "re-run the growing buffer", the same as upstream's live mode. Each call costs a
  full forward pass over the utterance, which is bounded by `:max_utterance_seconds` (30 s).
  There is no incremental encoder state.
- Pieces of 0.1 s or less are not run and give `""`. The reader's attention is quadratic in
  piece length, so call `transcribe_piece/3` on a few minutes at most.
- Only fp32 is exported. There is no fp16 or int8 variant yet.
- PyTorch's default CUDA path uses bf16 autocast, and the graph is fp32. On FLEURS, 2 of 20
  texts differ from it, at the same WER. The graph matches PyTorch fp32 exactly (see below).

## Exporting the model

`scripts/export_onnx.py` runs in the model's Python venv, which has `quan_transcribe`, torch,
transformers, `onnx`, `onnxscript` and `onnxruntime` installed:

```sh
M=/run/media/phanmn/01DBCEF7F80235E0/codings/llama-cpp/models/QUAN-Transcribe-1.0
HF_HOME=$M/hf HF_HUB_OFFLINE=1 $M/.venv/bin/python scripts/export_onnx.py $M/onnx \
  --manifest /path/to/fleurs/vi_vn/manifest.tsv   # optional: checks the wrapper against QuanCTC first
```

The script writes these files to `$M/onnx/`. The ONNX file is not kept in this repo.

| file | content |
|---|---|
| `model.onnx` (1.4 GB, fp32) | inputs `wav f32 {B, L}`, `wav_lens s64 {B}`, `streaming s64 {1}`, `left_chunks s64 {1}` (-1 = all); outputs `logits f32 {B, T, 14919}`, `ids s64 {B, T}` (argmax), `lens s64 {B}` |
| `tokenizer.json` | Gemma tokenizer (from google/embeddinggemma-2) |
| `units.json` | `gemma_ids`: CTC unit `u >= 2` -> Gemma token id `gemma_ids[u - 2]`; 0 = blank, 1 = unk |
| `config.json` | sample rate, 25 frames/s, blank/unk ids, `chunk_frames` 12, mel params |

How the export works, and where it differs from `quan_transcribe`'s PyTorch path:

- **Log-mel is in the graph.** The rfft becomes a matmul of the gathered 320-sample frames with
  `window x DFT basis`, so no FFT op is needed. It differs from `torch.fft` by at most 2.3e-4 in
  log-mel.
- **Forward hooks.** `quan_transcribe` collects the 13 encoder hidden states with hooks. The
  export wrapper calls the encoder's submodules itself and mixes the states explicitly.
- **Masks.** The encoder's 5-D chunked attention mask is built directly from the frame mask. It
  is bit-identical to the one transformers builds. The reader gets the key-padding mask, AND-ed
  with the streaming chunk mask when `streaming == 1`, so one graph serves both modes.
- **No `Tensor.unfold` and no Conv ops.** The legacy exporter cannot export `unfold` over a
  dynamic length, so overlapping windows are index gathers instead. Convolutions are written as
  the same sums: depthwise convs as shifted multiply-adds, and the 3x3 stride-2 subsampling as
  im2col + matmul. With Conv ops, ONNX Runtime's CUDA EP re-ran cuDNN's exhaustive algorithm
  search for every new audio length: 101 s instead of 1.4 s on the test set. Ortex has no option
  to turn that search off.
- **Exporter and format.** The TorchScript exporter is used, at opset 20 and IR version 10, which
  the ONNX Runtime 1.19 linked into Ortex can load. `torch.export` (`dynamo=True`) stalled for
  hours in sympy, simplifying the encoder's symbolic block shapes.

`scripts/parity.py` checks the graph against PyTorch. It writes the texts that
`test/fixtures/` holds for the Elixir tests:

```sh
HF_HOME=$M/hf HF_HUB_OFFLINE=1 $M/.venv/bin/python scripts/parity.py $M/onnx manifest.tsv /tmp/parity
```

## Results (FLEURS vi_vn, 20 files, 239 s of audio)

| | normal | streaming |
|---|---|---|
| Python ORT vs PyTorch fp32: text | 20/20 identical | 20/20 identical |
| Python ORT vs PyTorch fp32: max abs logit diff | 7.2e-4 | 1.1e-3 |
| Elixir vs Python ORT: text (CPU and CUDA) | 20/20 identical | 20/20 identical |
| WER, Elixir (`wer.py`) | **9.6%** | **11.3%** |
| WER, PyTorch default (CUDA bf16) | 9.6% | 11.3% |

Speed was measured with `bench/fleurs.exs` on a Ryzen 7 7800X3D (8 cores) and an RTX 4090.
Pieces run one at a time with batch 1, after one warm-up file:

| provider | normal | streaming | peak RSS | GPU memory |
|---|---|---|---|---|
| CPU (precompiled Ortex NIF) | 8.9 s (RTF 0.037) | 6.9 s (RTF 0.029) | 1.9 GB | - |
| CUDA (Ortex built with `cuda`) | 1.4 s (RTF 0.006) | 1.1 s (RTF 0.005) | 1.9 GB | ~2.5 GB |

Model load takes about 3 s.

### CUDA

The precompiled Ortex NIF is CPU only. If you ask for `[:cuda, :cpu]`, it silently falls back to
CPU. To use CUDA:

1. In the host app, add `{:rustler, "~> 0.29.0", runtime: false}` and
   `config :ortex, Ortex.Native, features: ["cuda"]`, then build with `ORTEX_BUILD=1`. ort
   downloads ONNX Runtime 1.19 with the CUDA provider into `_build/.../ortex/priv/native/`.
2. That provider needs CUDA 12 and cuDNN 9 at runtime. This machine only has CUDA 13, so the
   pip wheels were installed into the model dir:
   `uv pip install --target $M/cuda12/site nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10" nvidia-cufft-cu12 nvidia-cuda-runtime-cu12 nvidia-curand-cu12`.
   Then run with
   `LD_LIBRARY_PATH=$(ls -d $M/cuda12/site/nvidia/*/lib | tr '\n' ':')`.

## Tests

```sh
mix test                                  # unit tests, downloader against a local HTTP server
VEMMA_MODEL_DIR=$M/onnx mix test           # + FLEURS parity, registry, Nx.Serving batch == single, Live
VEMMA_MODEL_DIR=$M/onnx mix test --include download   # + download all 1.4 GB from a local server
# VEMMA_FLEURS_DIR overrides the wav dir (default: the asr-testset path on this machine)
# VEMMA_PROVIDERS=cuda,cpu to run the FLEURS tests on a CUDA-enabled Ortex build

VEMMA_MODEL_DIR=$M/onnx VEMMA_PROVIDERS=cpu mix run bench/fleurs.exs manifest.tsv /tmp/out
python3 wer.py /tmp/out/normal.tsv
```

Run Elixir with `LANG=C.UTF-8` (or `ELIXIR_ERL_OPTIONS="+fnu"`).

## Publishing the model

`scripts/release_assets.sh [onnx_dir] [tag]` checks the files against
`priv/checksums/<tag>.sha256` and prints the `gh release create` command that uploads them, with
`NOTICE` and the checksum file. A new export is a new tag: add its `priv/checksums/<tag>.sha256`
(`sha256sum config.json units.json tokenizer.json model.onnx`) and bump the default version in
`VemmaEx.Model`.

## License

- The code in this repository is our own and MIT licensed, see [LICENSE](LICENSE).
- The model files published as release assets (and downloaded by this library) are distributed
  under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). `model.onnx` is a modified
  (ONNX-converted) derivative of qnguyen3/QUAN-Transcribe-1.0, itself a derivative of
  google/embeddinggemma-2, and its use must follow the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). See
  [NOTICE](NOTICE). The MIT license does not cover them.
- The upstream code (github.com/qnguyen3/QUAN-Transcribe) has no license stated. Nothing from it
  is copied here. The export wrapper imports the installed package and re-expresses its forward
  pass.
