defmodule Aimodoki do
  use Task, restart: :transient

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
  def fc(m, n, x, a, b) do
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
  def relu(x, n) when is_integer(x) == false, do: relu(n, x)
  def relu(_n, x) do
    x
    |> List.flatten()
    |> Enum.map(fn xx -> if xx < 0, do: 0, else: xx end)
    |> Enum.chunk_every(1)
  end

  #Softmax演算
  #入力xに対し、yk =exp(xk −xmax)/∑ i exp(xi −xmax)を求め、それをyに代入する関数
  #(引数　x:n行列ベクトル　戻り値　y:n行列ベクトル)
  def softmax(x, n) when is_integer(x) == false, do: softmax(n, x)
  def softmax(n, x) do
    x_max = x |> List.flatten() |> Enum.max()
    denominator = x |> List.flatten() |> Enum.map(fn xx -> :math.exp(xx - x_max) end) |> Enum.sum()
    x
    |> List.flatten()
    |> Enum.map(fn xx -> :math.exp(xx - x_max)/denominator end)
    |> Enum.chunk_every(1)
  end

  def softmaxwithloss_bwd(m, y, t) do
    y = List.flatten(y)

    [0,0,0,0,0,0,0,0,0,0]
      |> Enum.with_index(0)
      |> Enum.map(fn {ans, index} -> if index == t, do: {index-1, 1}, else: {index-1, 0} end)
      |> Enum.map(fn {ans, index} ->
            Enum.at(y, index) - ans
          end)
      |> Enum.chunk_every(1)
  end

  def relu_bwd(m, x, dEdy) do
    x = List.flatten(x)

    dEdy |> List.flatten()
         |> Enum.with_index()
         |> Enum.map(fn {dedy, index} -> if Enum.at(x, index) > 0, do: dedy, else: 0 end)
         |> Enum.chunk_every(1)
  end

  #memo x, dEdy, dEdbは列ベクトル dEdxは行ベクトル(?) a, dEdaは行列
  def fc_bwd(m, n, x, dEdy, a, dEda, dEdb, dEdx) do
    dEdb = dEdy
    a = a |> List.flatten()
    dEdy = dEdy |> List.flatten()
    x = x |> List.flatten()
    dEda =
      Task.async(fn ->
                  dEda
                  |> Enum.with_index()
                  |> Enum.map(fn {deda, index} ->
                              Task.async(fn ->
                                Enum.with_index(deda)
                                |> Enum.map(fn {dedaa, indexx} ->
                                              Enum.at(dEdy, index) * Enum.at(x, indexx)
                                            end)
                                end)
                              end)
                  |> Enum.map(fn task -> Task.await(task, 100000) end)
                end)

    dEdx =
      Task.async(fn ->
                    dEdx
                    |> Enum.with_index()
                    |> Enum.map(fn {dedx, index} ->
                        Task.async(fn ->
                          a
                          |> Enum.with_index()
                          |> Enum.take_while(fn {aa, indexx} -> rem(indexx, n) == index end)
                          |> Enum.map(fn {aa, indexx} -> aa * Enum.at(dEdy, index)  end)
                          |> Enum.sum()
                        end)
                      end)
                    |> Enum.map(fn task -> Task.await(task, 100000) end)
                  end)
    dEda = Task.await(dEda, 1000000)
    dEdx = Task.await(dEdx, 1000000)
    {dEda, dEdb, dEdx}
  end

  def shuffle(matrix) do
    matrix
    |> Enum.shuffle()
  end

  def cross_entropy_error(y, t) do
    target = List.flatten(y) |> Enum.at(t)
    -1*:math.log(target + 1.0e-7)
  end


  #乱数で初期化　列ベクトル
  def rand_init(n, matrix \\[]) do
    case n do
      0 -> matrix
      _ ->
        matrix = matrix ++ [[:rand.uniform()]]
        rand_init(n-1, matrix)
    end
  end


  # maxのindexを返す
  def inference6(a1, a2, a3, b1, b2, b3, x, b, y) do
    y1 = fc(50, 784, x, a1, b1)  |> relu(50)
    y2 = fc(100, 50, y1, a2, b2) |> relu(100)
    y =  fc(10, 100, y2, a3, b3) |> softmax(10)

    y
    |> List.flatten()
    |> Enum.with_index()
    |> Enum.sort_by(&(elem(&1, 0)), &>=/2)
    |> Enum.at(0)
  end

  #何を返すかは謎 yを減らした
  def backward6(a1, a2, a3, b1, b2, b3, x, t,  deda1, deda2, deda3, dedb1, dedb2, dedb3) do
    copy_fc1   = x
    copy_relu1 = fc(50, 784, x, a1, b1)
    copy_fc2   = relu(50, copy_relu1)
    copy_relu2 = fc(100, 50, copy_fc2, a2, b2)
    copy_fc3   = relu(100, copy_relu2)
    y          = fc(10, 100, copy_fc3, a3, b3) |> softmax(10)
    y0         = Matrix.new(784, 1)
    y1         = Matrix.new(50, 1)
    y2         = Matrix.new(100, 1)

    y3 = softmaxwithloss_bwd(10, y, t)
    {deda3, dedb3, y2} = fc_bwd(10, 100, copy_fc3, y3, a3, deda3, dedb3, y2)
    y2 = relu_bwd(100, copy_relu2, y2)
    {deda2, dedb2, y1} = fc_bwd(100, 50, copy_fc2, y2, a2, deda2, dedb2, y1)
    IO.puts "もうちょい！"
    y1 = relu_bwd(50, copy_relu1, y1)
    {deda1, dedb1, y0} = fc_bwd(50, 784, copy_fc1, y1, a1, deda1, dedb1, y0)

    IO.puts "おわた！"
    {deda1, deda2, deda3, dedb1, dedb2, dedb3}
  end

  def learn_6layers(train_count, train_x, train_y) do
    epoch = 10
    batch = 100
    h = 0.01

    deda1 = Matrix.new(50, 784)
    deda2 = Matrix.new(100, 50)
    deda3 = Matrix.new(10, 100)
    dedb1 = Matrix.new(50, 1)
    dedb2 = Matrix.new(100, 1)
    dedb3 = Matrix.new(10, 1)

    a1 = Matrix.rand(50, 784)
    a2 = Matrix.rand(100, 50)
    a3 = Matrix.rand(10, 100)
    b1 = Matrix.rand(50, 1)
    b2 = Matrix.rand(100, 1)
    b3 = Matrix.rand(10, 1)

    index = 0..train_count

    {a1, a2, a3, b1, b2, b3} = learn_single(a1, a2, a3, b1, b2, b3, deda1, deda2, deda3, dedb1, dedb2, dedb3,  train_x, train_y, index, batch, epoch, train_count)
    #save
  end

  def learn_single(a1, a2, a3, b1, b2, b3, deda1, deda2, deda3, dedb1, dedb2, dedb3,  train_x, train_y, index, batch, epoch, train_count, i \\ 1) do
    case i do
      ^epoch ->
        IO.puts("processing...   Epoch:#{epoch}/#{epoch}")

        index = shuffle(index)
        #train_count/batchは割り切れると仮定
        learn_batch(a1, a2, a3, b1, b2, b3, deda1, deda2, deda3, dedb1, dedb2, dedb3,  train_x, train_y, index, batch, train_count/batch)

      ii ->
        IO.puts("processing...   Epoch: #{ii}/#{epoch}")

        index = shuffle(index)
        #train_count/batchは割り切れると仮定
        {a1, a2, a3, b1, b2, b3} = learn_batch(a1, a2, a3, b1, b2, b3, deda1, deda2, deda3, dedb1, dedb2, dedb3,  train_x, train_y, index, batch, round(train_count/batch)-1)
        learn_single(a1, a2, a3, b1, b2, b3, deda1, deda2, deda3, dedb1, dedb2, dedb3,  train_x, train_y, index, batch, epoch, train_count, ii+1)
    end
  end

  def learn_batch(a1, a2, a3, b1, b2, b3, deda1, deda2, deda3, dedb1, dedb2, dedb3,  train_x, train_y, index, batch, j) do
    case j do
      0 -> {a1, a2, a3, b1, b2, b3}
      jj ->
        #平均勾配の初期化
        deda1_av = new(784, 50, 0)
        deda2_av = new(50, 100, 0)
        deda3_av = new(100, 10, 0)
        dedb1_av = new(50, 1, 0)
        dedb2_av = new(100, 1, 0)
        dedb3_av = new(10, 1, 0)
        {a1, a2, a3, b1, b2, b3} = cal_gradient(a1, a2, a3, b1, b2, b3, deda1, deda2, deda3, dedb1, dedb2, dedb3, deda1_av, deda2_av, deda3_av, dedb1_av, dedb2_av, dedb3_av, train_x, train_y, index, batch, j, batch-1)
        learn_batch(a1, a2, a3, b1, b2, b3, deda1, deda2, deda3, dedb1, dedb2, dedb3,  train_x, train_y, index, batch, jj-1)
    end
  end

  def cal_gradient(a1, a2, a3, b1, b2, b3, deda1, deda2, deda3, dedb1, dedb2, dedb3, deda1_av, deda2_av, deda3_av, dedb1_av, dedb2_av, dedb3_av, train_x, train_y, index, batch, j, time, tasks_list \\ []) do
    case time do
      0 -> IO.puts("\n")
           IO.puts("wait task finishing..")
          {deda1_av, deda2_av, deda3_av, dedb1_av, dedb2_av, dedb3_av} =
             tasks_list
             |> Enum.map(fn task ->
               IO.write "."
               Task.await(task, 1000000)
             end)
             |> Enum.reduce(fn {deda1, deda2, deda3, dedb1, dedb2, dedb3}, {deda1_av, deda2_av, deda3_av, dedb1_av, dedb2_av, dedb3_av} ->
                             deda1_av = Matrix.add(deda1, deda1_av)
                             deda2_av = Matrix.add(deda2, deda2_av)
                             deda3_av = Matrix.add(deda3, deda3_av)
                             dedb1_av = Matrix.add(dedb1, dedb1_av)
                             dedb2_av = Matrix.add(dedb2, dedb2_av)
                             dedb3_av = Matrix.add(dedb3, dedb3_av)
                            {deda1_av, deda2_av, deda3_av, dedb1_av, dedb2_av, dedb3_av}
                           end)
           #ミニバッチで割って平均を求める
           h = 0.01
           deda1_av = Matrix.scale(deda1_av, -h/batch)
           deda2_av = Matrix.scale(deda2_av, -h/batch)
           deda3_av = Matrix.scale(deda3_av, -h/batch)
           dedb1_av = Matrix.scale(dedb1_av, -h/batch)
           dedb2_av = Matrix.scale(dedb2_av, -h/batch)
           dedb3_av = Matrix.scale(dedb3_av, -h/batch)

           #係数a, bを更新
           a1 = Matrix.add(deda1_av, a1)
           a2 = Matrix.add(deda2_av, a2)
           a3 = Matrix.add(deda3_av, a3)
           b1 = Matrix.add(dedb1_av, b1)
           b2 = Matrix.add(dedb2_av, b2)
           b3 = Matrix.add(dedb3_av, b3)

           IO.puts("\n")
           IO.puts("finished")
           {a1, a2, a3, b1, b2, b3}
      n ->task = Task.async(Aimodoki, :backward6, [a1, a2, a3, b1, b2, b3, Enum.at(train_x, Enum.at(index, j*batch+n)) ,Enum.at(train_y, Enum.at(index, j*batch+n)),  deda1, deda2, deda3, dedb1, dedb2, dedb3])
          tasks_list = tasks_list ++ [task]
           IO.write "."
          cal_gradient(a1, a2, a3, b1, b2, b3, deda1, deda2, deda3, dedb1, dedb2, dedb3, deda1_av, deda2_av, deda3_av, dedb1_av, dedb2_av, dedb3_av, train_x, train_y, index, batch, j, n-1, tasks_list)
    end
  end

  def is_learn_mode?(args) do
    {opts, target, _} =
      args
      |> OptionParser.parse(switches: [learn: :boolean])

    opts[:learn]
  end

  def main(args \\ []) do
    if is_learn_mode?(args) do
      IO.puts "------ learn mode ------"
      train_x = MNIST.train_image()
      train_y = MNIST.train_label()
      train_count = 60000
      learn_6layers(train_count, train_x, train_y)
      IO.puts "== end =="
    else
      IO.puts "------ inference mode ------"
      #inference_number = inference6(a1, a2, a3, b1, b2, b3, x, b, y)
      #IO.puts "The number is #{inference_number}, I think!"
      IO.puts "== end =="
    end
  end
end
