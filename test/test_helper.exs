defmodule VemmaEx.TestServer do
  @moduledoc false
  # Serves a directory over HTTP on 127.0.0.1 with :inets' httpd, for downloader tests.

  def start!(dir) do
    {:ok, _} = Application.ensure_all_started(:inets)

    {:ok, pid} =
      :inets.start(:httpd,
        port: 0,
        bind_address: {127, 0, 0, 1},
        server_name: ~c"vemma_test",
        server_root: String.to_charlist(dir),
        document_root: String.to_charlist(dir)
      )

    port = pid |> :httpd.info() |> Keyword.fetch!(:port)
    {pid, "http://127.0.0.1:#{port}"}
  end

  def stop(pid), do: :inets.stop(:httpd, pid)
end

# Tests tagged :model run the real ONNX graph. They need the exported model directory:
#
#     VEMMA_MODEL_DIR=/path/to/QUAN-Transcribe-1.0/onnx mix test
#
# The FLEURS tests also read the 20 wavs from VEMMA_FLEURS_DIR (see fleurs_test.exs).
# Tests tagged :download serve VEMMA_MODEL_DIR over local HTTP and download all of it
# (1.4 GB) through VemmaEx.Downloader; run them with `--include download`.
exclude = if System.get_env("VEMMA_MODEL_DIR"), do: [:download], else: [:model, :download]
ExUnit.start(exclude: exclude, capture_log: true)
