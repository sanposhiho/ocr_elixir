defmodule Aimodoki.MixProject do
  use Mix.Project

  def project do
    [
      app: :aimodoki,
      version: "0.1.0",
      elixir: "~> 1.8",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      escript: escript()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:matrix, "~> 0.3.0"},
    ]
  end

  defp escript do    #add
    [main_module: Aimodoki]
  end
end
