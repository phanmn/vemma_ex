defmodule VemmaEx.Model.RegistryTest do
  use ExUnit.Case, async: false

  setup do
    on_exit(fn ->
      Enum.each(VemmaEx.Model.Registry.loaded(), &VemmaEx.Model.Registry.unload/1)
    end)

    :ok
  end

  describe "missing models" do
    test "fetch/1 returns :error rather than raising" do
      assert VemmaEx.Model.Registry.fetch(:nope) == :error
      assert VemmaEx.Model.Registry.fetch() == :error
    end

    test "fetch!/1 explains how to load the model" do
      error = assert_raise RuntimeError, fn -> VemmaEx.Model.Registry.fetch!(:asr) end

      assert Exception.message(error) =~ "no QUAN-Transcribe model registered under :asr"
      assert Exception.message(error) =~ "VemmaEx.Model.Registry.load!(:asr)"
    end

    test "every entry point raises through an unregistered name" do
      samples = Nx.broadcast(Nx.tensor(0.0, type: :f32), {16_000})

      for call <- [
            fn -> VemmaEx.transcribe(samples) end,
            fn -> VemmaEx.transcribe(:asr, samples) end,
            fn -> VemmaEx.transcribe_file("x.wav") end,
            fn -> VemmaEx.transcribe_piece(samples) end,
            fn -> VemmaEx.transcribe_batch([samples]) end,
            fn -> VemmaEx.Live.new() end
          ] do
        assert_raise RuntimeError, ~r/no QUAN-Transcribe model registered/, call
      end
    end

    test "load!/2 with nothing on disk and downloads off raises before registering" do
      dir = Path.join(System.tmp_dir!(), "vemma_none_#{System.unique_integer([:positive])}")

      assert_raise RuntimeError, ~r/mix vemma.download/, fn ->
        VemmaEx.Model.Registry.load!(:asr, path: dir, download: [auto: false])
      end

      assert VemmaEx.Model.Registry.loaded() == []
    end
  end
end

defmodule VemmaEx.Model.RegistryModelTest do
  use ExUnit.Case, async: false

  @moduletag :model
  @moduletag timeout: 300_000

  setup_all do
    dir = System.fetch_env!("VEMMA_MODEL_DIR")
    VemmaEx.Model.Registry.load!(:default, path: dir, download: [auto: false])

    on_exit(fn ->
      Enum.each(VemmaEx.Model.Registry.loaded(), &VemmaEx.Model.Registry.unload/1)
    end)

    {:ok, dir: dir}
  end

  test "load!/1 registers under :default and the bare calls use it", %{dir: dir} do
    assert :default in VemmaEx.Model.Registry.loaded()
    assert VemmaEx.Model.Registry.fetch!().path == Path.join(dir, "model.onnx")

    {:ok, samples} = VemmaEx.Audio.read_wav(fleurs_wav())
    {:ok, text} = VemmaEx.transcribe(VemmaEx.Model.Registry.fetch!(), samples)

    assert text != ""
    assert VemmaEx.transcribe(samples) == {:ok, text}
    assert VemmaEx.transcribe(:default, samples) == {:ok, text}
    assert VemmaEx.transcribe_file(fleurs_wav()) == {:ok, text}
    assert VemmaEx.transcribe_file(:default, fleurs_wav()) == {:ok, text}
    assert VemmaEx.transcribe_piece(samples) == text
    assert VemmaEx.transcribe_piece(:default, samples) == text

    assert VemmaEx.transcribe_batch([samples, {:file, "/nonexistent.wav"}, []]) ==
             [{:ok, text}, {:error, :enoent}, {:ok, ""}]
  end

  test "a second name is a second model, and unload/1 removes only it", %{dir: dir} do
    VemmaEx.Model.Registry.load!(:live, path: dir, streaming: true, download: [auto: false])

    assert Enum.sort(VemmaEx.Model.Registry.loaded()) == [:default, :live]
    assert VemmaEx.Model.Registry.fetch!(:live).streaming
    refute VemmaEx.Model.Registry.fetch!(:default).streaming

    assert VemmaEx.Model.Registry.unload(:live)
    refute VemmaEx.Model.Registry.unload(:live)
    assert VemmaEx.Model.Registry.loaded() == [:default]
  end

  test "fetch!/1 names what is registered" do
    error = assert_raise RuntimeError, fn -> VemmaEx.Model.Registry.fetch!(:defualt) end
    assert Exception.message(error) =~ "Registered: :default"
  end

  test "Live sessions start from a name" do
    assert %VemmaEx.Live{model: %VemmaEx.Model{}} = VemmaEx.Live.new()
    assert %VemmaEx.Live{left_chunks: 4} = VemmaEx.Live.new(:default, left_chunks: 4)
  end

  defp fleurs_wav do
    "VEMMA_FLEURS_DIR"
    |> System.get_env("/run/media/phanmn/01DBCEF7F80235E0/codings/asr-testset/fleurs/vi_vn")
    |> Path.join("0000.wav")
  end
end
