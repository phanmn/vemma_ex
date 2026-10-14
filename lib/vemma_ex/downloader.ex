defmodule VemmaEx.Downloader do
  @moduledoc """
  Fetches the model files at runtime, by default from this library's GitHub
  release assets.

  Uses `:httpc` rather than shelling out to `curl`, so this works inside a slim
  release image. `VemmaEx.Model.load/2` calls `ensure!/2` automatically when
  files are missing.

  Each file streams to a temporary file next to its destination (the 1.4 GB
  `model.onnx` never sits in memory), its SHA-256 is checked, and only then is
  it renamed into place. An interrupted or tampered download can never leave a
  truncated or wrong file behind a model name.

  ## Versions and checksums

  A model version is a release tag, `#{inspect(VemmaEx.Model.default_version())}` by default,
  passed as the top-level `:version` option. The expected SHA-256 of every file of
  each known version ships with this library in `priv/checksums/<version>.sha256`
  (`sha256sum` format, so `sha256sum -c` checks a directory by hand) and is
  compiled in. Nothing is downloaded without a checksum to verify it against:
  for a version this library does not know, pass `:checksums` yourself.

  ## The `:download` option

  Everything about fetching lives under one key, passed with the rest of a
  model's description:

      VemmaEx.Model.Registry.load!(:asr,
        download: [
          auto: true,
          repo: "https://artifacts.internal/vemma/model-v1",
          headers: [{"authorization", "Bearer " <> token}]
        ]
      )

    * `:auto` - fetch missing files instead of raising. Defaults to `true`.
    * `:repo` - base URL to fetch from. The filename is appended, so the example
      above fetches `.../vemma/model-v1/model.onnx`. Defaults to
      `default_repo/1` of the version: `#{"https://github.com/phanmn/vemma_ex/releases/download/<version>"}`.
    * `:headers` - request headers as `{name, value}` string pairs. Use these to
      reach a mirror behind auth. None are sent unless you pass them.
    * `:checksums` - `%{"model.onnx" => "<sha256 hex>", ...}` for every file, for
      a version without bundled checksums. Overrides the bundled ones.
    * `:timeout` - per-file request timeout in ms. Defaults to one hour.

  Nothing here logs or reports header values, which is what makes an
  `authorization` header safe to pass.
  """

  require Logger

  @github "https://github.com/phanmn/vemma_ex/releases/download"
  @timeout :timer.hours(1)

  @checksums_dir Path.expand("../../priv/checksums", __DIR__)
  @checksum_files Path.wildcard(Path.join(@checksums_dir, "*.sha256"))

  for file <- @checksum_files, do: @external_resource(file)

  @checksums Map.new(@checksum_files, fn file ->
               sums =
                 file
                 |> File.read!()
                 |> String.split("\n", trim: true)
                 |> Map.new(fn line ->
                   [sha, name] = String.split(line, ~r/\s+\*?/, parts: 2)
                   {name, String.downcase(sha)}
                 end)

               {Path.basename(file, ".sha256"), sums}
             end)

  # A checksum file added for a new version recompiles this module.
  @doc false
  def __mix_recompile__?,
    do: Path.wildcard(Path.join(@checksums_dir, "*.sha256")) != @checksum_files

  @doc "Base URL for a version when `:download`'s `:repo` is not given."
  def default_repo(version \\ VemmaEx.Model.default_version()), do: "#{@github}/#{version}"

  @doc "The bundled SHA-256 checksums of a version, `%{file => hex}`, or `nil`."
  @spec checksums(String.t()) :: %{String.t() => String.t()} | nil
  def checksums(version), do: Map.get(@checksums, version)

  @doc "Versions with bundled checksums."
  def versions, do: @checksums |> Map.keys() |> Enum.sort()

  @doc """
  Reads the `:download` keyword out of a model's options.
  """
  @spec download_opts(keyword()) :: keyword()
  def download_opts(opts), do: Keyword.get(opts, :download, [])

  @doc """
  Whether missing files should be fetched rather than raise.
  """
  @spec auto?(keyword()) :: boolean()
  def auto?(opts), do: opts |> download_opts() |> Keyword.get(:auto, true)

  @doc """
  Ensures every model file exists in `dir`, downloading the missing ones.

  Takes a model's full options: `:version` for which files belong in `dir`,
  `:file` for the graph's name, and `:download` for how to fetch them.

  A custom `:file` that is missing raises rather than download `model.onnx`
  under another name: a graph file named something else was put there by hand.
  """
  @spec ensure!(Path.t(), keyword()) :: Path.t()
  def ensure!(dir, opts \\ []) do
    graph = VemmaEx.Model.graph_file(opts)

    dir
    |> VemmaEx.Model.missing(opts)
    |> case do
      [] ->
        dir

      missing ->
        (graph != "model.onnx" and graph in missing)
        |> case do
          true -> raise ArgumentError, custom_file_message(dir, graph)
          false -> :ok
        end

        download = Keyword.put(download_opts(opts), :version, version(opts))
        Enum.each(missing, &download!(&1, Path.join(dir, &1), download))
        dir
    end
  end

  @doc """
  Downloads model file `file` (e.g. `"model.onnx"`) to `dest`, verifying its SHA-256.

  Takes the `:download` options directly (`:repo`, `:headers`, `:checksums`,
  `:timeout`) plus `:version`. `:auto` is not consulted here; calling this *is*
  the decision to download.
  """
  @spec download!(String.t(), Path.t(), keyword()) :: Path.t()
  def download!(file, dest, opts \\ []) do
    expected = expected_sha!(file, opts)
    url = "#{repo(opts)}/#{file}"

    {:ok, _} = Application.ensure_all_started(:inets)
    {:ok, _} = Application.ensure_all_started(:ssl)

    File.mkdir_p!(Path.dirname(dest))
    tmp = "#{dest}.download.#{System.unique_integer([:positive])}"

    Logger.info("vemma_ex: downloading #{file} -> #{dest}")

    url
    |> request(%{
      tmp: tmp,
      headers: headers(opts),
      timeout: Keyword.get(opts, :timeout, @timeout)
    })
    |> verify(%{tmp: tmp, url: url, expected: expected})
    |> case do
      {:ok, size} ->
        File.rename!(tmp, dest)
        Logger.info("vemma_ex: downloaded #{file} (#{mb(size)} MB)")
        dest

      {:error, message} ->
        File.rm(tmp)
        raise message
    end
  end

  defp request(url, %{tmp: tmp, headers: headers, timeout: timeout}) do
    http_opts = [
      timeout: timeout,
      connect_timeout: :timer.seconds(30),
      ssl: [
        verify: :verify_peer,
        cacerts: :public_key.cacerts_get(),
        depth: 3,
        customize_hostname_check: [
          match_fun: :public_key.pkix_verify_hostname_match_fun(:https)
        ]
      ]
    ]

    # stream: writes the body to tmp as it arrives instead of buffering it.
    :get
    |> :httpc.request({String.to_charlist(url), headers}, http_opts,
      stream: String.to_charlist(tmp)
    )
    |> case do
      {:ok, :saved_to_file} ->
        :ok

      {:ok, {{_, status, _}, _headers, _body}} ->
        {:error, "download of #{url} failed with HTTP #{status}"}

      {:error, reason} ->
        {:error, "download of #{url} failed: #{inspect(reason)}"}
    end
  end

  defp verify(:ok, %{tmp: tmp, url: url, expected: expected}) do
    tmp
    |> sha256()
    |> case do
      ^expected ->
        {:ok, File.stat!(tmp).size}

      actual ->
        {:error,
         "checksum mismatch for #{url}: expected sha256 #{expected}, got #{actual}. " <>
           "The file was discarded."}
    end
  end

  defp verify(error, _), do: error

  @doc "SHA-256 of a file as lowercase hex, read in 1 MB blocks."
  @spec sha256(Path.t()) :: String.t()
  def sha256(path) do
    path
    |> File.stream!(1_048_576)
    |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
    |> :crypto.hash_final()
    |> Base.encode16(case: :lower)
  end

  defp version(opts), do: Keyword.get(opts, :version) || VemmaEx.Model.default_version()

  defp expected_sha!(file, opts) do
    version = version(opts)

    opts
    |> Keyword.get(:checksums)
    |> case do
      nil -> checksums(version)
      sums -> sums
    end
    |> case do
      %{^file => sha} when is_binary(sha) -> String.downcase(sha)
      _ -> raise ArgumentError, unknown_checksum_message(file, version)
    end
  end

  # Trailing slash is trimmed so both ".../model-v1" and ".../model-v1/" work, and the
  # base is checked here rather than letting :httpc fail on a mangled charlist.
  defp repo(opts) do
    opts
    |> Keyword.get(:repo, default_repo(version(opts)))
    |> validate_repo()
  end

  defp validate_repo(repo) when is_binary(repo) do
    repo
    |> URI.parse()
    |> case do
      %URI{scheme: scheme, host: host} when is_binary(scheme) and is_binary(host) ->
        String.trim_trailing(repo, "/")

      _ ->
        raise ArgumentError, ":repo must be an absolute base URL, got: #{inspect(repo)}"
    end
  end

  defp validate_repo(other) do
    raise ArgumentError, ":repo must be a string, got: #{inspect(other)}"
  end

  defp headers(opts) do
    opts
    |> Keyword.get(:headers, [])
    |> Enum.map(fn {name, value} ->
      {String.to_charlist(name), String.to_charlist(value)}
    end)
  end

  defp unknown_checksum_message(file, version) do
    """
    no SHA-256 known for #{file} of model version #{inspect(version)}

    Bundled versions: #{Enum.map_join(versions(), ", ", &inspect/1)}. For another
    version, pass the checksum of every file:

        download: [checksums: %{"model.onnx" => "<sha256>", "tokenizer.json" => "<sha256>", ...}]
    """
  end

  defp custom_file_message(dir, graph) do
    """
    #{Path.join(dir, graph)} is missing, and a custom :file is not downloaded

    Releases ship the graph as model.onnx. Either drop `file: #{inspect(graph)}` to
    download it, or put the file there yourself and turn the download off:

        VemmaEx.Model.Registry.load!(:name, path: #{inspect(dir)}, file: #{inspect(graph)}, download: [auto: false])
    """
  end

  defp mb(bytes), do: Float.round(bytes / 1_048_576, 1)
end
