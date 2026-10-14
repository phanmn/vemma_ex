defmodule VemmaEx.ModelTest do
  use ExUnit.Case, async: true

  describe "resolve_path/1" do
    # Which model to load is described entirely by options; there is no
    # application configuration for it, so two named models can differ.

    test "defaults to the model-v1 directory in priv/models" do
      assert VemmaEx.Model.resolve_path() =~ ~r{priv/models/model-v1$}
      assert VemmaEx.Model.resolve_path() == VemmaEx.Model.default_path()
    end

    test ":version selects the release" do
      assert VemmaEx.Model.resolve_path(version: "model-v2") =~ ~r{priv/models/model-v2$}
    end

    test ":path wins outright" do
      assert VemmaEx.Model.resolve_path(path: "/var/models/asr", version: "model-v2") ==
               "/var/models/asr"
    end
  end

  test "missing/2 lists absent files, with :file in place of model.onnx" do
    dir = Path.join(System.tmp_dir!(), "vemma_missing_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf(dir) end)
    File.write!(Path.join(dir, "config.json"), "{}")

    assert VemmaEx.Model.missing(dir, []) == ["units.json", "tokenizer.json", "model.onnx"]

    assert VemmaEx.Model.missing(dir, file: "x.onnx") == [
             "units.json",
             "tokenizer.json",
             "x.onnx"
           ]
  end
end
