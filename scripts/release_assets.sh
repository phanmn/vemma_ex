#!/usr/bin/env bash
# Prints the `gh release create` command that publishes the model files as
# release assets of phanmn/vemma_ex. It runs nothing against GitHub.
#
#     scripts/release_assets.sh [onnx_dir] [tag]
#
# First checks the files in onnx_dir against priv/checksums/<tag>.sha256, the
# checksums VemmaEx.Downloader verifies downloads against. NOTICE goes up with
# them (it is informational: the downloader neither fetches nor checks it), and
# so does the checksum file, for `sha256sum -c` by hand.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dir="${1:-/run/media/phanmn/01DBCEF7F80235E0/codings/llama-cpp/models/QUAN-Transcribe-1.0/onnx}"
tag="${2:-model-v1}"
sums="$root/priv/checksums/$tag.sha256"

[ -f "$sums" ] || { echo "no checksums for $tag at $sums" >&2; exit 1; }
(cd "$dir" && sha256sum --quiet -c "$sums") >&2 || { echo "checksum mismatch in $dir" >&2; exit 1; }

notes=$(cat <<NOTES
QUAN-Transcribe-1.0 exported to ONNX for vemma_ex (fp32, one graph from 16 kHz waveform to CTC ids).

Downloaded and verified automatically by VemmaEx.Model.Registry.load!/2 or \`mix vemma.download\`.
SHA-256: see $tag.sha256 (\`sha256sum -c $tag.sha256\`).

Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms

These files are a modified (ONNX-converted) derivative of qnguyen3/QUAN-Transcribe-1.0, itself a
derivative of google/embeddinggemma-2. Use must follow the Gemma Prohibited Use Policy
(ai.google.dev/gemma/prohibited_use_policy). See NOTICE. The vemma_ex code is MIT; that license
does not cover these files.
NOTES
)

q() { printf '%q' "$1"; }

echo "# Assets:"
for f in config.json units.json tokenizer.json model.onnx; do
  printf '#   %-15s %12s bytes  %s\n' "$f" "$(stat -c %s "$dir/$f")" "$(grep "  $f\$" "$sums" | cut -d' ' -f1)"
done
echo "#   NOTICE"
echo "#   $tag.sha256"
echo
echo "gh release create $(q "$tag") --repo phanmn/vemma_ex \\"
echo "  --title $(q "Model $tag: QUAN-Transcribe-1.0 ONNX") \\"
echo "  --notes $(q "$notes") \\"
for f in config.json units.json tokenizer.json model.onnx; do
  echo "  $(q "$dir/$f") \\"
done
echo "  $(q "$root/NOTICE") \\"
echo "  $(q "$sums")"
