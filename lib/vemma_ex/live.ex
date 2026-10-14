defmodule VemmaEx.Live do
  @moduledoc """
  Incremental decoding of a 16 kHz mono stream, after `quan_transcribe.live.LiveSession`.

  The current utterance is re-decoded in streaming mode on every `accept/2`.
  Streaming mode never looks more than one 0.48 s chunk ahead, so text from
  finished chunks stays put and only the newest words can still change. An
  utterance ends after `:endpoint_seconds` with no new tokens; it is then
  decoded once more with full context (more accurate), returned as final, and
  the buffer starts over.

      session = VemmaEx.Live.new(:asr)
      {finals, session} = VemmaEx.Live.accept(session, samples)
      session.partial
      {text, session} = VemmaEx.Live.flush(session)

  Each call re-runs the whole utterance, so the cost per call grows with
  utterance length (bounded by `:max_utterance_seconds`). Feed it about one
  chunk (0.48 s, 7680 samples) at a time.
  """

  # One encoder frame (40 ms).
  @frame_samples 640
  @sample_rate 16_000

  defstruct [
    :model,
    :left_chunks,
    :endpoint_frames,
    :max_utterance_seconds,
    # f32 little-endian samples of the utterance in progress
    buffer: <<>>,
    partial: ""
  ]

  @type t :: %__MODULE__{}

  @doc """
  Starts a session on a `%VemmaEx.Model{}` or the name of one registered with
  `VemmaEx.Model.Registry` (default `:default`). A name is looked up once, here.

  ## Options

    * `:left_chunks` - past chunks the reader may attend to while streaming (`nil` = all).
    * `:endpoint_seconds` - silence (no new tokens) that ends an utterance, default `0.8`.
    * `:max_utterance_seconds` - longer stretches without a pause are cut at
      the quietest point, default `30.0`.
  """
  @spec new(VemmaEx.Model.t() | atom(), keyword()) :: t()
  def new(model \\ VemmaEx.Model.Registry.default_name(), opts \\ []) do
    %__MODULE__{
      model: VemmaEx.Model.Registry.resolve!(model),
      left_chunks: Keyword.get(opts, :left_chunks),
      endpoint_frames:
        round(Keyword.get(opts, :endpoint_seconds, 0.8) * @sample_rate / @frame_samples),
      max_utterance_seconds: Keyword.get(opts, :max_utterance_seconds, 30.0)
    }
  end

  @doc """
  Adds audio and decodes. Returns the utterances that finished (non-empty
  texts) and the new session; `session.partial` holds the text so far of the
  utterance in progress.
  """
  @spec accept(t(), Nx.Tensor.t() | binary() | [number()]) :: {[String.t()], t()}
  def accept(%__MODULE__{} = session, samples) do
    buffer = session.buffer <> to_binary(samples)

    {finals, buffer} = cut_long(session, buffer)
    {more, session} = decode(%{session | buffer: buffer})

    {Enum.reject(finals ++ more, &(&1 == "")), session}
  end

  @doc "Finishes the current utterance (e.g. when the input stops)."
  @spec flush(t()) :: {String.t(), t()}
  def flush(%__MODULE__{} = session) do
    text =
      session.buffer
      |> byte_size()
      |> case do
        bytes when bytes > div(@sample_rate, 10) * 4 -> final(session, session.buffer)
        _ -> ""
      end

    {text, %{session | buffer: <<>>, partial: ""}}
  end

  defp to_binary(samples) when is_binary(samples), do: samples
  defp to_binary([]), do: <<>>
  defp to_binary(samples), do: samples |> VemmaEx.Audio.to_tensor() |> Nx.to_binary()

  # A long stretch without a pause: finish it at the quietest point near the end.
  defp cut_long(session, buffer) do
    (byte_size(buffer) > session.max_utterance_seconds * @sample_rate * 4)
    |> case do
      true ->
        [head | _] =
          buffer
          |> Nx.from_binary(:f32)
          |> VemmaEx.Audio.split_quiet(session.max_utterance_seconds)

        head = Nx.to_binary(head)
        rest = binary_part(buffer, byte_size(head), byte_size(buffer) - byte_size(head))
        {[final(session, head)], rest}

      false ->
        {[], buffer}
    end
  end

  defp decode(%{buffer: <<>>} = session), do: {[], session}

  defp decode(session) do
    ids =
      VemmaEx.Model.run(session.model, Nx.from_binary(session.buffer, :f32),
        streaming: true,
        left_chunks: session.left_chunks
      )

    partial = VemmaEx.CTC.decode(session.model, ids)
    session = %{session | partial: partial}
    bytes = byte_size(session.buffer)

    {partial, VemmaEx.CTC.last_emitted(ids)}
    |> case do
      # Nothing said yet: drop old silence, keeping enough for a word that has started.
      {"", _} when bytes > 3 * @sample_rate * 4 ->
        {[],
         %{
           session
           | buffer: binary_part(session.buffer, bytes - @sample_rate * 4, @sample_rate * 4)
         }}

      {"", _} ->
        {[], session}

      {_, last} when length(ids) - 1 - last >= session.endpoint_frames ->
        {[final(session, session.buffer)], %{session | buffer: <<>>, partial: ""}}

      _ ->
        {[], session}
    end
  end

  # Full-context (non-streaming) decode of a finished utterance.
  defp final(session, buffer),
    do: VemmaEx.transcribe_piece(session.model, Nx.from_binary(buffer, :f32))
end
