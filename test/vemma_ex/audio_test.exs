defmodule VemmaEx.AudioTest do
  use ExUnit.Case, async: true

  @rate 16_000

  defp tone(seconds), do: Nx.broadcast(Nx.tensor(0.5, type: :f32), {trunc(seconds * @rate)})
  defp silence(seconds), do: Nx.broadcast(Nx.tensor(0.0, type: :f32), {trunc(seconds * @rate)})

  test "short audio is one piece" do
    assert [piece] = VemmaEx.Audio.split_quiet(tone(10), 30.0)
    assert Nx.axis_size(piece, 0) == 10 * @rate
  end

  test "long audio is cut in the middle of the first quiet window near the end" do
    audio = Nx.concatenate([tone(27), silence(0.2), tone(20)])
    assert [a, b] = VemmaEx.Audio.split_quiet(audio, 30.0)
    # The first all-zero 50 ms window starts at 27 s; the cut is its centre.
    assert Nx.axis_size(a, 0) == 27 * @rate + 400
    assert Nx.axis_size(a, 0) + Nx.axis_size(b, 0) == Nx.axis_size(audio, 0)
  end

  test "pieces of 0.1 s or less are dropped" do
    assert [] = VemmaEx.Audio.split_quiet(tone(0.1), 30.0)
  end

  test "reads 16-bit PCM WAV, averaging channels" do
    path = Path.join(System.tmp_dir!(), "vemma_ex_test.wav")
    # Two frames, stereo: (1000, 3000) and (-2000, 0).
    data = <<1000::little-signed-16, 3000::little-signed-16, -2000::little-signed-16, 0::16>>

    fmt =
      <<1::little-16, 2::little-16, @rate::little-32, @rate * 4::little-32, 4::little-16,
        16::little-16>>

    wav =
      <<"RIFF", 36 + byte_size(data)::little-32, "WAVE", "fmt ", 16::little-32, fmt::binary,
        "data", byte_size(data)::little-32, data::binary>>

    File.write!(path, wav)
    assert {:ok, samples} = VemmaEx.Audio.read_wav(path)
    assert Nx.to_flat_list(samples) == [2000 / 32768, -1000 / 32768]
    File.rm!(path)
  end

  test "rejects other sample rates" do
    path = Path.join(System.tmp_dir!(), "vemma_ex_test_8k.wav")

    fmt =
      <<1::little-16, 1::little-16, 8000::little-32, 16000::little-32, 2::little-16,
        16::little-16>>

    File.write!(
      path,
      <<"RIFF", 38::little-32, "WAVE", "fmt ", 16::little-32, fmt::binary, "data", 2::little-32,
        0::16>>
    )

    assert {:error, {:sample_rate, 8000}} = VemmaEx.Audio.read_wav(path)
    File.rm!(path)
  end
end
