defmodule VemmaEx.Model.Registry do
  @moduledoc """
  Loads models once at startup and holds them by name.

  This is a plain module, not a process. Memoizing a struct does not need a
  GenServer, and inference never goes through here anyway: `VemmaEx.transcribe/3`
  reads the registered model and runs the forward pass in the *calling* process.

  ## Declaring your models

  Name each model and pass its configuration where you use it, from your
  `Application.start/2`:

      defmodule MyApp.Application do
        use Application

        @impl true
        def start(_type, _args) do
          VemmaEx.Model.Registry.load!(:asr)
          VemmaEx.Model.Registry.load!(:live, streaming: true, left_chunks: 8)
          VemmaEx.Model.Registry.load!(:pinned, path: "/var/models/QUAN-Transcribe-1.0/onnx")

          Supervisor.start_link(children(), strategy: :one_for_one, name: MyApp.Supervisor)
        end
      end

  Then:

      VemmaEx.transcribe(:asr, audio)

  Loading *before* `Supervisor.start_link/2` is what gives you boot-time
  fail-fast: missing or corrupt files raise before any child accepts traffic.

  Each name holds its own ONNX Runtime session (~1.9 GB resident for fp32), so
  two names on the same files cost twice the memory. Per-call `:streaming` and
  `:left_chunks` options let one loaded model serve both modes.

  A single unnamed model is spelled `VemmaEx.Model.Registry.load!()`, which
  registers under `#{inspect(:default)}`.
  """

  @default :default

  @doc "The name used when none is given."
  def default_name, do: @default

  @doc """
  Loads a model and registers it under `name`.

  Accepts every option `VemmaEx.Model.load/2` does: `:path`, `:version`,
  `:providers`, `:optimization_level`, `:file`, `:streaming`, `:left_chunks`,
  `:download`.

  Reloading a name replaces it, but this is not cheap: `:persistent_term.put/2`
  triggers a global scan proportional to the number of processes on the node.
  Call this at boot, never per request.

  Returns the loaded `%VemmaEx.Model{}`.
  """
  @spec load!(atom(), keyword()) :: VemmaEx.Model.t()
  def load!(name \\ @default, opts \\ []) when is_atom(name) do
    model = VemmaEx.Model.load(nil, opts)
    :persistent_term.put(key(name), model)
    model
  end

  @doc """
  Returns the model registered under `name`, raising if there is none.
  """
  @spec fetch!(atom()) :: VemmaEx.Model.t()
  def fetch!(name \\ @default) when is_atom(name) do
    name
    |> fetch()
    |> case do
      {:ok, model} ->
        model

      :error ->
        raise """
        no QUAN-Transcribe model registered under #{inspect(name)}#{registered_hint()}

        Load it at startup, before your supervision children:

            VemmaEx.Model.Registry.load!(#{inspect(name)})
        """
    end
  end

  @doc """
  Returns `{:ok, model}` if `name` is registered, `:error` otherwise.
  """
  @spec fetch(atom()) :: {:ok, VemmaEx.Model.t()} | :error
  def fetch(name \\ @default) when is_atom(name) do
    name
    |> key()
    |> :persistent_term.get(:undefined)
    |> case do
      :undefined -> :error
      model -> {:ok, model}
    end
  end

  @doc """
  Lists the names currently registered.
  """
  @spec loaded() :: [atom()]
  def loaded do
    for {{__MODULE__, name}, _model} <- :persistent_term.get(), do: name
  end

  @doc """
  Removes the model registered under `name`.

  Returns `true` if something was removed. Like `load!/2`, this is a global
  operation, for tests and hot-swaps, not steady state.
  """
  @spec unload(atom()) :: boolean()
  def unload(name \\ @default) when is_atom(name) do
    :persistent_term.erase(key(name))
  end

  @doc false
  # A model struct as is, a name through fetch!/1.
  def resolve!(%VemmaEx.Model{} = model), do: model
  def resolve!(name) when is_atom(name), do: fetch!(name)

  defp key(name), do: {__MODULE__, name}

  defp registered_hint do
    loaded()
    |> case do
      [] -> ""
      names -> "\n\nRegistered: #{Enum.map_join(Enum.sort(names), ", ", &inspect/1)}"
    end
  end
end
