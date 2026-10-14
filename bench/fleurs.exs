# Transcribes the FLEURS vi_vn manifest and reports wall time; writes ref<TAB>hyp files for wer.py.
#
#     VEMMA_MODEL_DIR=.../onnx VEMMA_PROVIDERS=cpu mix run bench/fleurs.exs manifest.tsv out_dir
#     python3 wer.py out_dir/normal.tsv
[manifest, out_dir] = System.argv()

providers =
  "VEMMA_PROVIDERS"
  |> System.get_env("cpu")
  |> String.split(",")
  |> Enum.map(&String.to_atom/1)

{load_us, model} =
  :timer.tc(fn -> VemmaEx.load(System.fetch_env!("VEMMA_MODEL_DIR"), providers: providers) end)

rows =
  manifest
  |> File.read!()
  |> String.split("\n", trim: true)
  |> Enum.map(&(&1 |> String.split("\t") |> List.to_tuple()))

audio = Enum.map(rows, fn {wav, ref} -> {elem(VemmaEx.Audio.read_wav(wav), 1), ref} end)
seconds = audio |> Enum.map(fn {s, _} -> Nx.axis_size(s, 0) / 16_000 end) |> Enum.sum()
File.mkdir_p!(out_dir)
IO.puts("providers #{inspect(providers)}, load #{Float.round(load_us / 1.0e6, 2)} s, audio #{Float.round(seconds, 1)} s")

# Warm-up: ONNX Runtime allocates and (on CUDA) tunes kernels on the first runs.
{:ok, _} = VemmaEx.transcribe(model, elem(hd(audio), 0))

for mode <- [:normal, :streaming] do
  {us, lines} =
    :timer.tc(fn ->
      Enum.map(audio, fn {samples, ref} ->
        {:ok, hyp} = VemmaEx.transcribe(model, samples, streaming: mode == :streaming)
        "#{ref}\t#{hyp}\n"
      end)
    end)

  File.write!(Path.join(out_dir, "#{mode}.tsv"), lines)
  IO.puts("#{mode}: #{Float.round(us / 1.0e6, 2)} s (RTF #{Float.round(us / 1.0e6 / seconds, 4)})")
end

IO.puts("BEAM total memory #{div(:erlang.memory(:total), 1_048_576)} MB")
