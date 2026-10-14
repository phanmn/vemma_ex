defmodule VemmaEx.FleursTest do
  # Parity with the Python onnxruntime run of the same graph (scripts/parity.py), whose texts
  # are in test/fixtures. Needs VEMMA_MODEL_DIR and the FLEURS vi_vn wavs in VEMMA_FLEURS_DIR.
  # The model goes in the context as :asr; :model is the tag.
  use ExUnit.Case, async: false

  @moduletag :model
  @moduletag timeout: 600_000

  @fleurs_dir System.get_env(
                "VEMMA_FLEURS_DIR",
                "/run/media/phanmn/01DBCEF7F80235E0/codings/asr-testset/fleurs/vi_vn"
              )

  setup_all do
    providers =
      "VEMMA_PROVIDERS"
      |> System.get_env("cpu")
      |> String.split(",")
      |> Enum.map(&String.to_existing_atom/1)

    %{asr: VemmaEx.load(System.fetch_env!("VEMMA_MODEL_DIR"), providers: providers)}
  end

  defp expected(mode) do
    "test/fixtures/fleurs_vi_ort_#{mode}.tsv"
    |> File.read!()
    |> String.split("\n", trim: true)
    |> Enum.map(&(&1 |> String.split("\t") |> List.to_tuple()))
  end

  for mode <- [:normal, :streaming] do
    test "#{mode} text equals Python onnxruntime on all 20 files", %{asr: model} do
      mismatches =
        unquote(mode)
        |> expected()
        |> Enum.map(fn {wav, text} ->
          {:ok, got} =
            VemmaEx.transcribe_file(model, Path.join(@fleurs_dir, wav),
              streaming: unquote(mode) == :streaming
            )

          {wav, text, got}
        end)
        |> Enum.reject(fn {_, text, got} -> text == got end)

      assert mismatches == []
    end
  end

  test "binary f32 input gives the same text as a tensor", %{asr: model} do
    {:ok, samples} = VemmaEx.Audio.read_wav(Path.join(@fleurs_dir, "0000.wav"))

    assert VemmaEx.transcribe(model, samples) ==
             VemmaEx.transcribe(model, Nx.to_binary(samples))
  end

  test "live session finalizes the utterance", %{asr: model} do
    {:ok, samples} = VemmaEx.Audio.read_wav(Path.join(@fleurs_dir, "0000.wav"))
    {:ok, full} = VemmaEx.transcribe(model, samples)
    chunk = 7680
    n = Nx.axis_size(samples, 0)

    {finals, session} =
      0
      |> Stream.iterate(&(&1 + chunk))
      |> Enum.take_while(&(&1 < n))
      |> Enum.reduce({[], VemmaEx.Live.new(model)}, fn start, {acc, session} ->
        piece = Nx.slice(samples, [start], [min(chunk, n - start)])
        {finals, session} = VemmaEx.Live.accept(session, piece)
        {acc ++ finals, session}
      end)

    {last, _session} = VemmaEx.Live.flush(session)
    text = (finals ++ [last]) |> Enum.reject(&(&1 == "")) |> Enum.join(" ")
    assert String.jaro_distance(text, full) > 0.9
  end
end
