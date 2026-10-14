defmodule VemmaEx.Nx.ServingTest do
  # Batched runs must give exactly the text of one-at-a-time runs: padding is
  # masked out inside the graph. Checked on the 20 FLEURS vi_vn clips.
  use ExUnit.Case, async: false

  @moduletag :model
  @moduletag timeout: 900_000

  @fleurs_dir System.get_env(
                "VEMMA_FLEURS_DIR",
                "/run/media/phanmn/01DBCEF7F80235E0/codings/asr-testset/fleurs/vi_vn"
              )

  setup_all do
    VemmaEx.Model.Registry.load!(:served,
      path: System.fetch_env!("VEMMA_MODEL_DIR"),
      download: [auto: false]
    )

    on_exit(fn -> VemmaEx.Model.Registry.unload(:served) end)

    clips =
      for i <- 0..19 do
        {:ok, samples} =
          VemmaEx.Audio.read_wav(
            Path.join(@fleurs_dir, String.pad_leading("#{i}", 4, "0") <> ".wav")
          )

        samples
      end

    {:ok, clips: clips}
  end

  defp start(name, opts) do
    start_supervised!(
      {VemmaEx.Nx.Serving, [name: name, model: :served, batch_timeout: 50] ++ opts},
      id: name
    )
  end

  for mode <- [:normal, :streaming] do
    test "#{mode}: 20 clips batched give the same text as one at a time", %{clips: clips} do
      streaming = unquote(mode) == :streaming
      start(__MODULE__.Parity, batch_size: 8, streaming: streaming)

      single = Enum.map(clips, &VemmaEx.transcribe(:served, &1, streaming: streaming))
      batched = VemmaEx.Nx.Serving.transcribe(clips, __MODULE__.Parity)

      assert batched == single

      assert VemmaEx.transcribe_batch(:served, clips, streaming: streaming, batch_size: 5) ==
               single

      # And both equal the Python onnxruntime texts in test/fixtures.
      assert Enum.map(single, &elem(&1, 1)) == Enum.map(expected(unquote(mode)), &elem(&1, 1))
    end
  end

  test "recordings over 30 s are split at quiet points like transcribe/3", %{clips: clips} do
    start(__MODULE__.Long, batch_size: 4)
    # ~65 s (three pieces) and ~45 s, plus a short clip in the same request.
    long = Nx.concatenate(Enum.take(clips, 6))
    other = Nx.concatenate(Enum.slice(clips, 6, 3))
    assert Nx.axis_size(long, 0) > 2 * 30 * 16_000

    expected = Enum.map([long, other, Enum.at(clips, 9)], &VemmaEx.transcribe(:served, &1))

    assert VemmaEx.Nx.Serving.transcribe([long, other, Enum.at(clips, 9)], __MODULE__.Long) ==
             expected

    assert VemmaEx.Nx.Serving.transcribe(long, __MODULE__.Long) == hd(expected)
  end

  test "single inputs, files, errors and silence", %{clips: clips} do
    start(__MODULE__.Single, [])
    wav = Path.join(@fleurs_dir, "0003.wav")
    {:ok, text} = VemmaEx.transcribe(:served, {:file, wav})

    assert VemmaEx.Nx.Serving.transcribe({:file, wav}, __MODULE__.Single) == {:ok, text}

    assert VemmaEx.Nx.Serving.transcribe(Nx.to_binary(Enum.at(clips, 3)), __MODULE__.Single) ==
             {:ok, text}

    assert VemmaEx.Nx.Serving.transcribe({:file, "/nonexistent.wav"}, __MODULE__.Single) ==
             {:error, :enoent}

    assert VemmaEx.Nx.Serving.transcribe(<<>>, __MODULE__.Single) == {:ok, ""}

    assert VemmaEx.Nx.Serving.transcribe([{:file, "/nonexistent.wav"}], __MODULE__.Single) == [
             {:error, :enoent}
           ]
  end

  test "concurrent callers get their own text", %{clips: clips} do
    start(__MODULE__.Concurrent, batch_size: 4)
    picks = Enum.take(clips, 6)
    expected = Enum.map(picks, &VemmaEx.transcribe(:served, &1))

    results =
      picks
      |> Enum.zip(expected)
      |> List.duplicate(2)
      |> List.flatten()
      |> Task.async_stream(
        fn {clip, want} -> {VemmaEx.Nx.Serving.transcribe(clip, __MODULE__.Concurrent), want} end,
        max_concurrency: 12,
        timeout: 300_000
      )
      |> Enum.map(fn {:ok, result} -> result end)

    assert length(results) == 12
    for {got, want} <- results, do: assert(got == want)
  end

  defp expected(mode) do
    "test/fixtures/fleurs_vi_ort_#{mode}.tsv"
    |> File.read!()
    |> String.split("\n", trim: true)
    |> Enum.map(&(&1 |> String.split("\t") |> List.to_tuple()))
  end
end
