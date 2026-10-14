defmodule VemmaEx.Nx.Serving do
  @moduledoc """
  `Nx.Serving` wrapper around the model: the batching alternative to calling
  `VemmaEx.transcribe/3` directly.

  The two differ in **which process executes the forward pass**. Calling
  `VemmaEx.transcribe/3` runs it in the caller; this runs it in a single
  serving process that collects requests into a padded batch.

  Each recording is cut at quiet points into pieces of at most
  `:chunk_seconds`, exactly as `VemmaEx.transcribe/3` does, and every piece is
  one batch entry; the texts are joined back per recording. Padding is masked
  out inside the graph, so the text is identical to unbatched calls (checked
  on FLEURS vi_vn, normal and streaming).

  ## This is slower on CPU

  On a Ryzen 7 7800X3D with the CPU ONNX Runtime, the 20 FLEURS clips took
  8.7 s one at a time and 20.4 s in batches of 5: ONNX Runtime already uses
  every core for one clip, and padding adds work. Batching pays off on a GPU,
  or when you want the serving's queue as backpressure.

  ## Use

      children = [{VemmaEx.Nx.Serving, name: MyApp.ASR, model: :asr, batch_size: 4}]

      {:ok, text} = VemmaEx.Nx.Serving.transcribe(samples, MyApp.ASR)
      [{:ok, a}, {:ok, b}] = VemmaEx.Nx.Serving.transcribe([s1, {:file, "b.wav"}], MyApp.ASR)

  ## Options

    * `:name` - process name. Defaults to `VemmaEx.Nx.Serving`.
    * `:model` - name of a model registered with `VemmaEx.Model.Registry` (or a
      `%VemmaEx.Model{}`) to front. When given, the serving reuses that model
      instead of loading its own.
    * `:path` / `:version` / `:providers` / `:optimization_level` / `:file` /
      `:download` - passed to `VemmaEx.Model.load/2` when `:model` is not given.
    * `:streaming` / `:left_chunks` - decoding mode for every request; one
      batch is one graph run, so these are per serving, not per call.
      Default to the model's.
    * `:chunk_seconds` - maximum piece length, default `30.0`. Every batch
      entry is padded to it on the client and trimmed to the longest piece
      before the graph runs.
    * `:batch_size` - maximum pieces per forward pass. Defaults to `4`.
    * `:batch_timeout` - milliseconds to wait for a batch to fill. Defaults to `10`.
  """

  @default_name __MODULE__

  @doc """
  Builds the `Nx.Serving` struct. See the module docs for options.
  """
  def new(opts \\ []) do
    model =
      opts
      |> Keyword.fetch(:model)
      |> case do
        {:ok, name} when is_atom(name) -> VemmaEx.Model.Registry.fetch!(name)
        {:ok, %VemmaEx.Model{} = model} -> model
        :error -> VemmaEx.Model.load(nil, opts)
      end

    run_opts = Keyword.take(opts, [:streaming, :left_chunks, :chunk_seconds])
    max_samples = trunc(Keyword.get(opts, :chunk_seconds, 30.0) * VemmaEx.Audio.sample_rate())
    frame = Map.get(model.config, "samples_per_frame", 640)
    shape = %{max_frames: div(max_samples + frame - 1, frame) + 1, opts: run_opts}

    Nx.Serving.new(fn _defn_opts -> &run(model, &1, shape) end)
    |> Nx.Serving.client_preprocessing(fn input ->
      {audios, batched?} = unwrap(input)
      split = Enum.map(audios, &VemmaEx.pieces(&1, run_opts))

      entries =
        split
        |> Enum.flat_map(fn
          {:ok, pieces} -> pieces
          _error -> []
        end)
        |> case do
          # A batch cannot be empty: run a little silence, assemble/2 ignores its text.
          [] -> [Nx.broadcast(Nx.tensor(0.0, type: :f32), {VemmaEx.Model.min_samples() + 1})]
          pieces -> pieces
        end
        |> Enum.map(&entry(&1, max_samples))

      {Nx.Batch.stack(entries), {split, batched?}}
    end)
    |> Nx.Serving.client_postprocessing(fn {output, _metadata}, {split, batched?} ->
      texts =
        output
        |> VemmaEx.Model.to_lists()
        |> Enum.map(&VemmaEx.CTC.decode(model, &1))

      results = VemmaEx.assemble(split, texts)
      if batched?, do: results, else: hd(results)
    end)
  end

  # Fixed-size entries, so requests can share a batch: the piece zero padded to
  # max_samples, and its length.
  defp entry(piece, max_samples) do
    n = Nx.axis_size(piece, 0)

    {piece |> Nx.as_type(:f32) |> Nx.pad(0.0, [{0, max_samples - n, 0}]),
     Nx.tensor(n, type: :s64)}
  end

  # Trims the padding down to the longest piece, runs the graph, and pads ids to
  # a fixed frame count: Nx.Serving concatenates outputs of split runs.
  defp run(model, batch, %{max_frames: max_frames, opts: opts}) do
    {wav, lens} = Nx.Defn.jit_apply(&Function.identity/1, [batch])
    longest = lens |> Nx.reduce_max() |> Nx.to_number()

    {ids, lens} =
      VemmaEx.Model.forward(model, {Nx.slice_along_axis(wav, 0, longest, axis: 1), lens}, opts)

    {Nx.pad(ids, 0, [{0, 0, 0}, {0, max_frames - Nx.axis_size(ids, 1), 0}]), lens}
  end

  # A list of numbers is one waveform, not a batch of scalars. Anything else
  # that is a list is a batch; anything else is a single recording.
  defp unwrap([head | _] = audio) when is_number(head), do: {[audio], false}
  defp unwrap(audios) when is_list(audios), do: {audios, true}
  defp unwrap(audio), do: {[audio], false}

  @doc false
  def child_spec(opts) do
    {name, opts} = Keyword.pop(opts, :name, @default_name)
    {batch_size, opts} = Keyword.pop(opts, :batch_size, 4)
    {batch_timeout, opts} = Keyword.pop(opts, :batch_timeout, 10)

    Nx.Serving.child_spec(
      serving: new(opts),
      name: name,
      batch_size: batch_size,
      batch_timeout: batch_timeout
    )
  end

  @doc """
  Transcribes through a supervised serving.

  Pass one recording (tensor, f32 binary, list of floats or `{:file, path}`)
  for `{:ok, text} | {:error, reason}`, or a list of them for a list of results.
  """
  def transcribe(audio, name \\ @default_name) do
    Nx.Serving.batched_run(name, audio)
  end
end
