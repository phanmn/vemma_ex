defmodule VemmaEx do
  @moduledoc """
  Native BEAM inference for [QUAN-Transcribe-1.0](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0),
  a Vietnamese CTC speech recognizer on EmbeddingGemma 2's audio encoder.

  No Python at runtime: the whole network, log-mel included, is one ONNX graph
  run through `Ortex`; greedy CTC decoding and detokenization (`Tokenizers`)
  happen here.

      VemmaEx.Model.Registry.load!(:asr)
      {:ok, text} = VemmaEx.transcribe(:asr, samples)
      {:ok, text} = VemmaEx.transcribe(:asr, {:file, "speech.wav"}, streaming: true)

  Or hold the struct yourself, with no global state:

      model = VemmaEx.load("/models/QUAN-Transcribe-1.0/onnx")
      {:ok, text} = VemmaEx.transcribe(model, samples)

  Files missing from the model directory are downloaded on first load
  (`VemmaEx.Downloader`, `mix vemma.download`).

  Output is lowercase Vietnamese without punctuation, numbers as words.

  ## Input format

  16 kHz mono f32: an `Nx` tensor, a binary of little-endian f32 samples, a
  list of floats, or `{:file, path}` for a 16 kHz WAV. For s16 PCM use
  `VemmaEx.Audio.from_pcm_s16/1` first.

  The forward pass runs in the calling process (Ortex runs on a dirty
  scheduler), so concurrent callers run in parallel. `VemmaEx.Nx.Serving` is
  the batching alternative.
  """

  @type audio :: Nx.Tensor.t() | binary() | [number()] | {:file, Path.t()}
  @type model :: VemmaEx.Model.t() | atom()

  @doc """
  Loads a model directory. See `VemmaEx.Model.load/2` for options
  (`:providers`, `:optimization_level`, `:file`, `:streaming`, `:left_chunks`,
  `:download`). To load by name instead, see `VemmaEx.Model.Registry`.
  """
  @spec load(Path.t() | nil, keyword()) :: VemmaEx.Model.t()
  defdelegate load(dir \\ nil, opts \\ []), to: VemmaEx.Model

  @doc """
  Transcribes audio to text.

  The model can be given three ways. An atom names a model registered with
  `VemmaEx.Model.Registry`; a `%VemmaEx.Model{}` is used directly; anything
  else is treated as audio and resolved against the `:default` registered
  model.

      VemmaEx.transcribe(audio)
      VemmaEx.transcribe(:asr, audio, streaming: true)
      VemmaEx.transcribe(model, {:file, "speech.wav"})

  Recordings longer than `:chunk_seconds` (default 30) are cut at quiet
  points and the pieces transcribed one by one, joined with a space; this is
  what `quan_transcribe` does.

  ## Options

    * `:streaming` - decode as the model would in streaming mode: the reader
      only attends to the current 0.48 s chunk and the ones before it, so text
      for finished chunks never changes. Defaults to the model's setting.
    * `:left_chunks` - in streaming mode, how many past chunks the reader may
      attend to (`nil` = all).
    * `:chunk_seconds` - maximum piece length, default `30.0`.
  """
  @spec transcribe(model() | audio(), audio() | keyword(), keyword()) ::
          {:ok, String.t()} | {:error, term()}
  def transcribe(model_or_audio, audio_or_opts \\ [], opts \\ [])

  def transcribe(%VemmaEx.Model{} = model, audio, opts), do: one(model, audio, opts)

  def transcribe(name, audio, opts) when is_atom(name),
    do: one(VemmaEx.Model.Registry.fetch!(name), audio, opts)

  def transcribe(audio, opts, []), do: one(VemmaEx.Model.Registry.fetch!(), audio, opts)

  defp one(model, audio, opts) do
    audio
    |> pieces(opts)
    |> case do
      {:ok, pieces} -> {:ok, pieces |> Enum.map(&transcribe_piece(model, &1, opts)) |> join()}
      error -> error
    end
  end

  @doc """
  Transcribes several recordings, running their pieces through the graph
  `:batch_size` (default 4) at a time, zero padded to the longest.

  Takes the same three model forms and options as `transcribe/3`, and returns
  one result per input, in order. The text is identical to `transcribe/3`'s
  (padding is masked out). On CPU this is *slower* than one call per
  recording: see `VemmaEx.Nx.Serving` for when batching pays.
  """
  @spec transcribe_batch(model() | [audio()], [audio()] | keyword(), keyword()) ::
          [{:ok, String.t()} | {:error, term()}]
  def transcribe_batch(model_or_audios, audios_or_opts \\ [], opts \\ [])

  def transcribe_batch(%VemmaEx.Model{} = model, audios, opts) when is_list(audios),
    do: many(model, audios, opts)

  def transcribe_batch(name, audios, opts) when is_atom(name) and is_list(audios),
    do: many(VemmaEx.Model.Registry.fetch!(name), audios, opts)

  def transcribe_batch(audios, opts, []) when is_list(audios),
    do: many(VemmaEx.Model.Registry.fetch!(), audios, opts)

  defp many(model, audios, opts) do
    split = Enum.map(audios, &pieces(&1, opts))

    texts =
      split
      |> Enum.flat_map(fn
        {:ok, pieces} -> pieces
        _error -> []
      end)
      |> Enum.chunk_every(Keyword.get(opts, :batch_size, 4))
      |> Enum.flat_map(&VemmaEx.Model.run_batch(model, &1, opts))
      |> Enum.map(&VemmaEx.CTC.decode(model, &1))

    assemble(split, texts)
  end

  @doc """
  Transcribes a 16 kHz WAV file. Same model forms and options as `transcribe/3`.
  """
  @spec transcribe_file(model() | Path.t(), Path.t() | keyword(), keyword()) ::
          {:ok, String.t()} | {:error, term()}
  def transcribe_file(model_or_path, path_or_opts \\ [], opts \\ [])

  def transcribe_file(path, opts, []) when is_binary(path),
    do: transcribe({:file, path}, opts, [])

  def transcribe_file(model, path, opts), do: transcribe(model, {:file, path}, opts)

  @doc """
  Transcribes one waveform in a single forward pass, without splitting.
  Same model forms as `transcribe/3`.

  The reader's attention is quadratic in length, so keep pieces to a few
  minutes at most.
  """
  @spec transcribe_piece(model() | Nx.Tensor.t(), Nx.Tensor.t() | keyword(), keyword()) ::
          String.t()
  def transcribe_piece(model_or_samples, samples_or_opts \\ [], opts \\ [])

  def transcribe_piece(%Nx.Tensor{} = samples, opts, []),
    do: transcribe_piece(VemmaEx.Model.Registry.fetch!(), samples, opts)

  def transcribe_piece(model, samples, opts) do
    model = VemmaEx.Model.Registry.resolve!(model)

    model
    |> VemmaEx.Model.run(samples, opts)
    |> then(&VemmaEx.CTC.decode(model, &1))
  end

  @doc false
  # Audio -> {:ok, pieces of at most :chunk_seconds} | {:error, reason}.
  def pieces({:file, path}, opts) do
    path
    |> VemmaEx.Audio.read_wav()
    |> case do
      {:ok, samples} -> pieces(samples, opts)
      error -> error
    end
  end

  # Nx has no empty tensors.
  def pieces(audio, _opts) when audio in [<<>>, []], do: {:ok, []}

  def pieces(audio, opts) do
    {:ok,
     audio
     |> VemmaEx.Audio.to_tensor()
     |> VemmaEx.Audio.split_quiet(Keyword.get(opts, :chunk_seconds, 30.0))}
  end

  @doc false
  # Joins piece texts back per input: `split` is each input's pieces/2 result,
  # `texts` the texts of all their pieces in order.
  def assemble(split, texts) do
    split
    |> Enum.map_reduce(texts, fn
      {:ok, pieces}, texts ->
        {mine, rest} = Enum.split(texts, length(pieces))
        {{:ok, join(mine)}, rest}

      error, texts ->
        {error, texts}
    end)
    |> elem(0)
  end

  defp join(texts) do
    texts
    |> Enum.reject(&(&1 == ""))
    |> Enum.join(" ")
  end
end
