defmodule VemmaEx.MixProject do
  use Mix.Project

  def project do
    [
      app: :vemma_ex,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    # No `mod:`: this library starts nothing. Hosts declare their models with
    # VemmaEx.Model.Registry.load!/2 from their own Application.start/2.
    # :inets, :ssl and :crypto are for VemmaEx.Downloader.
    [extra_applications: [:logger, :inets, :ssl, :crypto]]
  end

  defp deps do
    [
      {:nx, "~> 0.13"},
      # Fork of :ortex with precompiled NIFs (ONNX Runtime 1.19 linked in), as smart_turn_ex uses.
      # Set ORTEX_BUILD=1 to build from source instead.
      {:ortex, github: "phanmn/ortex", branch: "main"},
      # Hugging Face tokenizers, to turn Gemma token ids into text.
      {:tokenizers, "~> 0.5"}
    ]
  end
end
