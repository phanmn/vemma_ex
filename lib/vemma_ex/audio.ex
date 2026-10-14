defmodule VemmaEx.Audio do
  @moduledoc """
  Audio input helpers: PCM conversion, WAV reading and splitting long
  recordings at quiet points (`quan_transcribe.audio.split_quiet`).

  Everything here is 16 kHz mono. There is no resampler: a WAV at another
  rate is rejected.
  """

  @sample_rate 16_000
  # Energy window used to find a quiet cut point: 50 ms.
  @window div(@sample_rate, 20)

  def sample_rate, do: @sample_rate

  @doc """
  Converts input audio to a `{n}` f32 tensor.

  Accepts an `Nx` tensor, a binary of little-endian f32 samples, or a list of numbers.
  """
  @spec to_tensor(Nx.Tensor.t() | binary() | [number()]) :: Nx.Tensor.t()
  def to_tensor(%Nx.Tensor{} = tensor), do: tensor |> Nx.as_type(:f32) |> Nx.flatten()
  def to_tensor(audio) when is_binary(audio), do: Nx.from_binary(audio, :f32)
  def to_tensor(audio) when is_list(audio), do: Nx.tensor(audio, type: :f32)

  @doc "Decodes little-endian signed 16-bit PCM into a `[-1, 1)` f32 tensor."
  @spec from_pcm_s16(binary()) :: Nx.Tensor.t()
  def from_pcm_s16(binary) when is_binary(binary) do
    binary
    |> Nx.from_binary(:s16)
    |> Nx.as_type(:f32)
    |> Nx.divide(32768.0)
  end

  @doc """
  Reads a 16 kHz WAV file (PCM 16/24/32-bit or float32, any channel count) into
  a mono f32 tensor. Channels are averaged, as `quan_transcribe.audio.load_audio` does.
  """
  @spec read_wav(Path.t()) :: {:ok, Nx.Tensor.t()} | {:error, term()}
  def read_wav(path) do
    path
    |> File.read()
    |> case do
      {:ok, <<"RIFF", _size::little-32, "WAVE", chunks::binary>>} -> parse_chunks(chunks, nil)
      {:ok, _} -> {:error, :not_a_wav}
      error -> error
    end
  end

  defp parse_chunks(<<"fmt ", size::little-32, rest::binary>>, _fmt) do
    <<body::binary-size(^size), rest::binary>> = rest

    <<tag::little-16, channels::little-16, rate::little-32, _::little-32, _::little-16,
      bits::little-16, ext::binary>> = body

    parse_chunks(skip_pad(rest, size), {format(tag, ext), channels, rate, bits})
  end

  defp parse_chunks(<<"data", size::little-32, rest::binary>>, {format, channels, rate, bits}) do
    size = min(size, byte_size(rest))
    <<data::binary-size(^size), _::binary>> = rest

    rate
    |> case do
      @sample_rate -> decode(data, {format, bits}) |> to_mono(channels)
      other -> {:error, {:sample_rate, other}}
    end
  end

  defp parse_chunks(<<"data", _::binary>>, nil), do: {:error, :missing_fmt_chunk}

  defp parse_chunks(<<_id::binary-4, size::little-32, rest::binary>>, fmt)
       when byte_size(rest) >= size,
       do:
         rest |> binary_part(size, byte_size(rest) - size) |> skip_pad(size) |> parse_chunks(fmt)

  defp parse_chunks(_, _), do: {:error, :missing_data_chunk}

  # Chunks are word aligned.
  defp skip_pad(<<_, rest::binary>>, size) when rem(size, 2) == 1, do: rest
  defp skip_pad(rest, _size), do: rest

  # WAVE_FORMAT_EXTENSIBLE carries the real format in the first two bytes of the subformat GUID.
  defp format(0xFFFE, <<_::binary-8, sub::little-16, _::binary>>), do: format(sub, <<>>)
  defp format(1, _), do: :pcm
  defp format(3, _), do: :float
  defp format(tag, _), do: {:unsupported, tag}

  defp decode(data, {:pcm, 16}), do: {:ok, from_pcm_s16(trim(data, 2))}

  defp decode(data, {:pcm, 32}),
    do: {:ok, data |> trim(4) |> Nx.from_binary(:s32) |> Nx.as_type(:f32) |> Nx.divide(2.0 ** 31)}

  defp decode(data, {:pcm, 24}) do
    samples =
      for <<s::little-signed-24 <- trim(data, 3)>>, into: <<>>, do: <<s::little-signed-32>>

    {:ok, samples |> Nx.from_binary(:s32) |> Nx.as_type(:f32) |> Nx.divide(2.0 ** 23)}
  end

  defp decode(data, {:float, 32}), do: {:ok, Nx.from_binary(trim(data, 4), :f32)}
  defp decode(_data, other), do: {:error, {:unsupported_format, other}}

  defp trim(data, width), do: binary_part(data, 0, byte_size(data) - rem(byte_size(data), width))

  defp to_mono({:ok, samples}, 1), do: {:ok, samples}

  defp to_mono({:ok, samples}, channels) do
    frames = div(Nx.axis_size(samples, 0), channels)

    {:ok,
     samples
     |> Nx.slice([0], [frames * channels])
     |> Nx.reshape({frames, channels})
     |> Nx.mean(axes: [1])}
  end

  defp to_mono(error, _channels), do: error

  @doc """
  Splits a waveform into pieces of at most `chunk_seconds`, cutting at the
  lowest-energy 50 ms window in the last `:search_seconds` (default 5) of each
  piece. Pieces of 0.1 s or less are dropped.

  Mirrors `quan_transcribe.audio.split_quiet`.
  """
  @spec split_quiet(Nx.Tensor.t(), number(), keyword()) :: [Nx.Tensor.t()]
  def split_quiet(samples, chunk_seconds, opts \\ []) do
    n = trunc(chunk_seconds * @sample_rate)
    search = min(trunc(Keyword.get(opts, :search_seconds, 5.0) * @sample_rate), n - @window)

    samples
    |> cut_points(%{n: n, search: search, start: 0})
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(fn [from, to] -> Nx.slice(samples, [from], [to - from]) end)
    |> Enum.filter(&(Nx.axis_size(&1, 0) > div(@sample_rate, 10)))
  end

  defp cut_points(samples, %{n: n, search: search, start: start} = state) do
    total = Nx.axis_size(samples, 0)

    (total - start > n)
    |> case do
      true ->
        lo = start + n - search
        cut = lo + quietest(Nx.slice(samples, [lo], [search])) + div(@window, 2)
        [start | cut_points(samples, %{state | start: cut})]

      false ->
        [start, total]
    end
  end

  # Start of the lowest-energy window (first on ties), like np.argmin over
  # np.convolve(x ** 2, ones(win), "valid"): squares in f32, sums in f64.
  defp quietest(segment) do
    sums =
      segment
      |> Nx.multiply(segment)
      |> Nx.as_type(:f64)
      |> Nx.to_flat_list()
      |> Enum.scan(&(&1 + &2))

    # Window i covers samples i..i+win-1: sums[i+win-1] - sums[i-1].
    [0.0 | sums]
    |> Enum.zip_with(Enum.drop(sums, @window - 1), fn before, last -> last - before end)
    |> Enum.with_index()
    |> Enum.min_by(&elem(&1, 0))
    |> elem(1)
  end
end
