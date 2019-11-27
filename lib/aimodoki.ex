defmodule Aimodoki do

  #m*n行列の作成
  def new(m, n, default \\ 1, matrix \\[]) do
    case m do
      0 -> case n do
            0 -> matrix
            _ -> matrix = Enum.map(matrix, &(&1 ++ [default]))
                 new(m, n-1, default, matrix)
           end
      _ -> matrix = matrix ++ [[]]
           new(m-1, n, default, matrix)
    end
  end

  #m*n行列xの表示
  def print(_m, n, x) do
    List.flatten(x)
    |> Enum.with_index(1)
    |> Enum.each(fn {xx, index} -> if rem(index, n) == 0, do: IO.write(inspect(xx)<>"\n"), else: IO.write(inspect(xx)<>" ")  end)
  end

  #入力xに対し、y=A*x+bを計算する関数
  def fc(_m, _n, x, a, b) do
    x = List.flatten(x)
    b
    |> List.flatten()
    |> Enum.with_index()
    |> Enum.map(fn {bx, index} ->
        result =
          Enum.at(a, index)
          |>Enum.with_index()
          |>Enum.map(fn{ax, a_index} ->
              ax * Enum.at(x, a_index)
            end)
    |> Enum.sum()
        result + bx
      end)
    |> Enum.chunk_every(1)
  end

  #入力xに対し、0以下の要素を0にし それ以外はｘをｙに代入する関数
  def relu(_n, x) do
    x
    |> List.flatten()
    |> Enum.map(fn xx -> if xx < 0, do: 0, else: xx end)
    |> Enum.chunk_every(1)
  end

  #Softmax演算
  #入力xに対し、yk =exp(xk −xmax)/∑ i exp(xi −xmax)を求め、それをyに代入する関数
  #(引数　x:n行列ベクトル　戻り値　y:n行列ベクトル)
  def softmax(n, x) do
    x_max = x |> List.flatten() |> Enum.max()
    denominator = x |> List.flatten() |> Enum.map(fn xx -> :math.exp(xx - x_max) end) |> Enum.sum()
    x
    |> List.flatten()
    |> Enum.map(fn xx -> :math.exp(xx - x_max)/denominator end)
    |> Enum.chunk_every(1)
  end

  

  def main(args \\ []) do
    a = [[1,2,3],[4,5,6]]
    m = 2
    n=3
    b = [[3],[4]]
    x=[[1],[4],[7]]
    matrix = softmax(n, x)
    print(n,1,matrix)
  end
end
