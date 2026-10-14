defmodule VemmaEx.DownloaderTest do
  use ExUnit.Case, async: false

  # Small stand-ins for the four model files, served over local HTTP.
  @fake %{
    "config.json" => ~s({"sample_rate": 16000}),
    "units.json" => ~s({"gemma_ids": [5, 6, 7]}),
    "tokenizer.json" => "{}",
    "model.onnx" => String.duplicate("not really a graph ", 50_000)
  }

  # 127.0.0.1:1 refuses immediately, so these assert on the URL that was built
  # without reaching the network. The failure message carries the URL.
  @dead "http://127.0.0.1:1/mirror"

  setup do
    tmp = Path.join(System.tmp_dir!(), "vemma_dl_#{System.unique_integer([:positive])}")
    served = Path.join(tmp, "served")
    dest = Path.join(tmp, "model")
    File.mkdir_p!(served)
    Enum.each(@fake, fn {file, body} -> File.write!(Path.join(served, file), body) end)

    {pid, base} = VemmaEx.TestServer.start!(served)

    on_exit(fn ->
      VemmaEx.TestServer.stop(pid)
      File.rm_rf(tmp)
    end)

    sums = Map.new(@fake, fn {file, body} -> {file, sha(body)} end)
    {:ok, served: served, dest: dest, base: base, sums: sums}
  end

  defp sha(body), do: :sha256 |> :crypto.hash(body) |> Base.encode16(case: :lower)

  describe "ensure!/2" do
    test "downloads every missing file and verifies it", %{dest: dest, base: base, sums: sums} do
      assert VemmaEx.Downloader.ensure!(dest, download: [repo: base, checksums: sums]) == dest

      for {file, body} <- @fake, do: assert(File.read!(Path.join(dest, file)) == body)
      # Nothing but the files themselves: temp files were renamed, not copied.
      assert Enum.sort(File.ls!(dest)) == Enum.sort(Map.keys(@fake))
    end

    test "fetches only what is missing", %{dest: dest, base: base, sums: sums} do
      File.mkdir_p!(dest)
      File.write!(Path.join(dest, "config.json"), "mine")

      VemmaEx.Downloader.ensure!(dest, download: [repo: base, checksums: sums])

      assert File.read!(Path.join(dest, "config.json")) == "mine"
      assert File.read!(Path.join(dest, "model.onnx")) == @fake["model.onnx"]
    end

    test "is a no-op when nothing is missing", %{dest: dest} do
      File.mkdir_p!(dest)
      Enum.each(@fake, fn {file, _} -> File.write!(Path.join(dest, file), "x") end)

      assert VemmaEx.Downloader.ensure!(dest, download: [repo: @dead]) == dest
    end

    test "a checksum mismatch discards the file", %{dest: dest, base: base, sums: sums} do
      sums = Map.put(sums, "model.onnx", String.duplicate("0", 64))

      error =
        assert_raise RuntimeError, fn ->
          VemmaEx.Downloader.ensure!(dest, download: [repo: base, checksums: sums])
        end

      assert Exception.message(error) =~ "checksum mismatch for #{base}/model.onnx"
      assert Exception.message(error) =~ sums["model.onnx"]
      refute File.exists?(Path.join(dest, "model.onnx"))
      # The files before it verified and stay; no temp file is left behind.
      assert Enum.sort(File.ls!(dest)) == ["config.json", "tokenizer.json", "units.json"]
    end

    test "the bundled checksums reject files that are not the release", %{dest: dest, base: base} do
      assert_raise RuntimeError, ~r/checksum mismatch for .*config.json/, fn ->
        VemmaEx.Downloader.ensure!(dest, download: [repo: base])
      end

      assert File.ls(dest) == {:ok, []}
    end

    test "an HTTP error leaves no partial file", %{served: served, dest: dest, base: base} do
      File.rm!(Path.join(served, "units.json"))
      sums = Map.new(@fake, fn {file, body} -> {file, sha(body)} end)

      assert_raise RuntimeError, ~r/#{base}\/units.json failed with HTTP 404/, fn ->
        VemmaEx.Downloader.ensure!(dest, download: [repo: base, checksums: sums])
      end

      assert File.ls!(dest) == ["config.json"]
    end

    test "a version without bundled checksums raises before downloading", %{dest: dest} do
      error =
        assert_raise ArgumentError, fn ->
          VemmaEx.Downloader.ensure!(dest, version: "model-v0", download: [repo: @dead])
        end

      assert Exception.message(error) =~
               ~s(no SHA-256 known for config.json of model version "model-v0")

      assert Exception.message(error) =~ "checksums:"
      refute File.exists?(dest)
    end

    test "a missing custom :file is not downloaded under another name", %{dest: dest} do
      assert_raise ArgumentError, ~r/a custom :file is not downloaded/, fn ->
        VemmaEx.Downloader.ensure!(dest, file: "int8.onnx", download: [repo: @dead])
      end
    end
  end

  describe "download!/3" do
    test "streams a file to its destination", %{dest: dest, base: base, sums: sums} do
      path = Path.join(dest, "graph.onnx")

      assert VemmaEx.Downloader.download!("model.onnx", path, repo: base, checksums: sums) == path
      assert VemmaEx.Downloader.sha256(path) == sums["model.onnx"]
    end
  end

  describe ":repo" do
    test "defaults to this library's GitHub release of the version" do
      assert VemmaEx.Downloader.default_repo() ==
               "https://github.com/phanmn/vemma_ex/releases/download/model-v1"

      assert VemmaEx.Downloader.default_repo("model-v2") =~ "/releases/download/model-v2"
    end

    test "a trailing slash does not produce a doubled separator", %{dest: dest} do
      error =
        assert_raise RuntimeError, fn ->
          VemmaEx.Downloader.ensure!(dest, download: [repo: @dead <> "/"])
        end

      assert Exception.message(error) =~ "#{@dead}/config.json"
      refute Exception.message(error) =~ "//config.json"
    end

    test "reaches the downloader through Model.load/2 and the registry", %{dest: dest} do
      for call <- [
            fn -> VemmaEx.Model.load(dest, download: [repo: @dead]) end,
            fn -> VemmaEx.Model.Registry.load!(:mirrored, path: dest, download: [repo: @dead]) end
          ] do
        error = assert_raise RuntimeError, call
        assert Exception.message(error) =~ "#{@dead}/config.json"
      end
    end

    test "headers are accepted and their values never surface", %{dest: dest} do
      error =
        assert_raise RuntimeError, fn ->
          VemmaEx.Downloader.ensure!(dest,
            download: [repo: @dead, headers: [{"authorization", "Bearer sekrit"}]]
          )
        end

      assert Exception.message(error) =~ "#{@dead}/config.json"
      refute Exception.message(error) =~ "sekrit"
    end

    test "rejects a base that is not an absolute URL", %{dest: dest} do
      for bad <- ["/var/models", "artifacts.internal", ""] do
        assert_raise ArgumentError, ~r/absolute base URL/, fn ->
          VemmaEx.Downloader.ensure!(dest, download: [repo: bad])
        end
      end

      assert_raise ArgumentError, ~r/must be a string/, fn ->
        VemmaEx.Downloader.ensure!(dest, download: [repo: :internal])
      end
    end
  end

  describe "load/2 with download: [auto: false]" do
    test "raises, naming the missing files and the mix task", %{dest: dest} do
      error =
        assert_raise RuntimeError, fn ->
          VemmaEx.Model.load(dest, download: [auto: false])
        end

      assert Exception.message(error) =~ "mix vemma.download"
      assert Exception.message(error) =~ "config.json, units.json, tokenizer.json, model.onnx"
      refute File.exists?(dest)
    end
  end

  describe "bundled checksums" do
    test "cover the four files of model-v1" do
      sums = VemmaEx.Downloader.checksums("model-v1")

      assert Enum.sort(Map.keys(sums)) == Enum.sort(VemmaEx.Model.files())
      assert Enum.all?(Map.values(sums), &(&1 =~ ~r/^[0-9a-f]{64}$/))
      assert "model-v1" in VemmaEx.Downloader.versions()
    end

    test "sha256/1 hashes a file in blocks", %{served: served} do
      path = Path.join(served, "model.onnx")
      assert VemmaEx.Downloader.sha256(path) == sha(File.read!(path))
    end
  end

  # Serves VEMMA_MODEL_DIR and downloads all 1.4 GB of it against the bundled
  # model-v1 checksums, then transcribes with what landed.
  @tag :download
  @tag :model
  @tag timeout: 600_000
  test "downloads, verifies and loads the real model" do
    {pid, base} = VemmaEx.TestServer.start!(System.fetch_env!("VEMMA_MODEL_DIR"))
    dest = Path.expand("../../tmp/download_test", __DIR__)
    File.rm_rf!(dest)

    on_exit(fn ->
      VemmaEx.TestServer.stop(pid)
      File.rm_rf(dest)
    end)

    before = :erlang.memory(:binary)
    VemmaEx.Downloader.ensure!(dest, download: [repo: base])
    # Streamed to disk: the 1.4 GB body never sat in memory.
    assert :erlang.memory(:binary) - before < 100_000_000

    assert Enum.sort(File.ls!(dest)) == Enum.sort(VemmaEx.Model.files())

    model = VemmaEx.Model.Registry.load!(:downloaded, path: dest, download: [auto: false])
    on_exit(fn -> VemmaEx.Model.Registry.unload(:downloaded) end)

    assert {:ok, text} = VemmaEx.transcribe(:downloaded, {:file, fleurs_wav()})
    assert text != ""
    assert model.path == Path.join(dest, "model.onnx")
  end

  defp fleurs_wav do
    "VEMMA_FLEURS_DIR"
    |> System.get_env("/run/media/phanmn/01DBCEF7F80235E0/codings/asr-testset/fleurs/vi_vn")
    |> Path.join("0000.wav")
  end
end
