defmodule VemmaEx.Model do
  @moduledoc """
  Loads the exported QUAN-Transcribe ONNX graph via Ortex and runs a forward pass.

  A model is a directory of four files, the output of `scripts/export_onnx.py`:

    * `model.onnx` - waveform to CTC logits, log-mel included (1.4 GB, fp32)
    * `tokenizer.json` - the Gemma tokenizer
    * `units.json` - CTC unit index to Gemma token id (`gemma_ids[u - 2]`)
    * `config.json` - sample rate, frame rate, blank id, chunk size, mel params

  The graph takes `wav {B, L}` f32 (zero padded on the right), `wav_lens {B}`,
  `streaming {1}` and `left_chunks {1}` (s64), and returns `logits`, `ids`
  (per-frame argmax) and `lens`. Only `ids` and `lens` are copied out of ONNX
  Runtime.
  """

  @default_version "model-v1"
  @files ["config.json", "units.json", "tokenizer.json", "model.onnx"]

  # Shorter input is not run (quan_transcribe drops pieces of 0.1 s or less).
  @min_samples 1600

  defstruct [:ortex, :tokenizer, :units, :config, :path, streaming: false, left_chunks: nil]

  @type t :: %__MODULE__{}

  @doc """
  Resolves which model directory to load, given per-call options.

  There is no application configuration for model selection, so two names can
  describe two different models:

      resolve_path([])
      resolve_path(version: "model-v2")
      resolve_path(path: "/var/models/QUAN-Transcribe-1.0/onnx")

  `:path` wins outright. Otherwise `:version` (a release tag, see
  `VemmaEx.Downloader`) selects a directory in `priv/models`, defaulting to
  `"#{@default_version}"`.
  """
  @spec resolve_path(keyword()) :: Path.t()
  def resolve_path(opts \\ []) do
    opts
    |> Keyword.get(:path)
    |> case do
      nil ->
        Path.join([:code.priv_dir(:vemma_ex), "models", opts[:version] || @default_version])

      path ->
        path
    end
  end

  @doc """
  Directory of the default model, `priv/models/#{@default_version}`.

  Equivalent to `resolve_path/1` with no options. This is what a bare
  `VemmaEx.Model.Registry.load!/0` and `mix vemma.download` use.
  """
  def default_path, do: resolve_path()

  @doc "The default model version (release tag), `#{@default_version}`."
  def default_version, do: @default_version

  @doc "The files that make up a model directory, small ones first."
  def files, do: @files

  @doc """
  Loads a model directory.

  Returns a `%VemmaEx.Model{}` you can pass to `VemmaEx.transcribe/3`. To load
  once at startup and reach it by name instead, see `VemmaEx.Model.Registry`.

  ## Options

    * `:path` - the model directory, same as the positional argument.
    * `:version` - release tag selecting a directory in `priv/models`.
      Defaults to `"#{@default_version}"`.
    * `:providers` - execution providers in priority order, e.g. `[:cuda, :cpu]`.
      Defaults to `[:cpu]`. Ortex silently falls back to the next provider when
      one is unavailable in its build.
    * `:optimization_level` - ONNX Runtime graph optimization level, 1..3. Defaults to `3`.
    * `:file` - graph file inside the directory. Defaults to `"model.onnx"`.
    * `:streaming` - default for `VemmaEx.transcribe/3`. Defaults to `false`.
    * `:left_chunks` - default past-chunk limit in streaming mode; `nil` = all.
    * `:download` - how to fetch files that are not on disk. See
      `VemmaEx.Downloader` for `:auto`, `:repo`, `:headers` and `:checksums`.
      Pass `download: [auto: false]` in production if you bake the model into
      your image and would rather fail fast than block on a 1.4 GB download.
  """
  @spec load(Path.t() | nil, keyword()) :: t()
  def load(dir \\ nil, opts \\ []) do
    dir = dir || resolve_path(opts)

    dir
    |> missing(opts)
    |> case do
      [] ->
        :ok

      missing ->
        opts
        |> VemmaEx.Downloader.auto?()
        |> case do
          true -> VemmaEx.Downloader.ensure!(dir, opts)
          false -> raise missing_message(dir, missing)
        end
    end

    open(dir, opts)
  end

  defp open(dir, opts) do
    config = read_json(dir, "config.json")

    units =
      dir
      |> read_json("units.json")
      |> Map.fetch!("gemma_ids")
      |> List.to_tuple()

    {:ok, tokenizer} = dir |> Path.join("tokenizer.json") |> Tokenizers.Tokenizer.from_file()
    path = Path.join(dir, graph_file(opts))

    ortex =
      Ortex.load(
        path,
        Keyword.get(opts, :providers, [:cpu]),
        Keyword.get(opts, :optimization_level, 3)
      )

    %__MODULE__{
      ortex: ortex,
      tokenizer: tokenizer,
      units: units,
      config: config,
      path: path,
      streaming: Keyword.get(opts, :streaming, false),
      left_chunks: Keyword.get(opts, :left_chunks)
    }
  end

  @doc false
  def graph_file(opts), do: Keyword.get(opts, :file, "model.onnx")

  @doc false
  # Files of the model directory not on disk, with `:file` in place of model.onnx.
  def missing(dir, opts) do
    @files
    |> List.replace_at(-1, graph_file(opts))
    |> Enum.reject(&File.exists?(Path.join(dir, &1)))
  end

  defp missing_message(dir, missing) do
    """
    QUAN-Transcribe model files missing in #{dir}: #{Enum.join(missing, ", ")}

    Run `mix vemma.download` to fetch them, or let them download themselves:

        VemmaEx.Model.Registry.load!(name, download: [auto: true])
    """
  end

  @doc "Inputs of this many samples (0.1 s) or fewer are not run."
  def min_samples, do: @min_samples

  @doc """
  Runs one waveform (`{n}` f32 tensor, 16 kHz) and returns the per-frame best
  CTC unit ids (25 per second) of its valid frames. Input of 0.1 s or less
  returns `[]` without running the graph.

  Options: `:streaming`, `:left_chunks`, defaulting to the model's.
  """
  @spec run(t(), Nx.Tensor.t(), keyword()) :: [non_neg_integer()]
  def run(%__MODULE__{} = model, samples, opts \\ []) do
    [ids] = run_batch(model, [samples], opts)
    ids
  end

  @doc """
  Runs several waveforms in one forward pass, zero padded to the longest, and
  returns each one's ids as `run/3` would. Inputs of 0.1 s or less give `[]`
  and are left out of the batch.
  """
  @spec run_batch(t(), [Nx.Tensor.t()], keyword()) :: [[non_neg_integer()]]
  def run_batch(%__MODULE__{} = model, samples, opts \\ []) do
    lens = Enum.map(samples, &Nx.axis_size(&1, 0))

    samples
    |> Enum.zip(lens)
    |> Enum.filter(fn {_, n} -> n > @min_samples end)
    |> case do
      [] ->
        Enum.map(samples, fn _ -> [] end)

      runnable ->
        max = runnable |> Enum.map(&elem(&1, 1)) |> Enum.max()

        wav =
          runnable
          |> Enum.map(fn {s, n} -> s |> Nx.as_type(:f32) |> Nx.pad(0.0, [{0, max - n, 0}]) end)
          |> Nx.stack()

        wav_lens = runnable |> Enum.map(&elem(&1, 1)) |> Nx.tensor(type: :s64)

        model
        |> forward({wav, wav_lens}, opts)
        |> to_lists()
        |> refill(lens)
    end
  end

  # Puts [] back for the inputs too short to run.
  defp refill(results, lens) do
    lens
    |> Enum.map_reduce(results, fn
      n, rest when n <= @min_samples -> {[], rest}
      _n, [ids | rest] -> {ids, rest}
    end)
    |> elem(0)
  end

  @doc """
  Runs a padded batch: `wav {B, L}` f32 and `wav_lens {B}` s64.

  Returns `{ids, lens}`: per-frame best ids `{B, T}` and valid frames `{B}`,
  both on `Nx.BinaryBackend`. Options: `:streaming`, `:left_chunks`.
  """
  @spec forward(t(), {Nx.Tensor.t(), Nx.Tensor.t()}, keyword()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def forward(%__MODULE__{} = model, {wav, wav_lens}, opts \\ []) do
    streaming = if Keyword.get(opts, :streaming, model.streaming), do: 1, else: 0
    left_chunks = Keyword.get(opts, :left_chunks, model.left_chunks) || -1

    # backend_copy, not backend_transfer: transfer would free a caller-owned tensor.
    inputs =
      {wav |> Nx.as_type(:f32) |> Nx.backend_copy(Nx.BinaryBackend),
       wav_lens |> Nx.as_type(:s64) |> Nx.backend_copy(Nx.BinaryBackend),
       Nx.tensor([streaming], type: :s64), Nx.tensor([left_chunks], type: :s64)}

    {_logits, ids, lens} = Ortex.run(model.ortex, inputs)

    {Nx.backend_transfer(ids, Nx.BinaryBackend), Nx.backend_transfer(lens, Nx.BinaryBackend)}
  end

  @doc false
  # {ids {B, T}, lens {B}} -> the valid ids of each row.
  def to_lists({ids, lens}) do
    ids
    |> Nx.to_list()
    |> Enum.zip_with(Nx.to_flat_list(lens), &Enum.take/2)
  end

  defp read_json(dir, file) do
    dir
    |> Path.join(file)
    |> File.read!()
    |> JSON.decode!()
  end
end
