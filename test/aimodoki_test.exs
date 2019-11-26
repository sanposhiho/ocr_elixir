defmodule AimodokiTest do
  use ExUnit.Case
  doctest Aimodoki

  test "greets the world" do
    assert Aimodoki.hello() == :world
  end
end
