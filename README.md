# Aimodoki

 Elixirを使った手書き数字認識を行うプログラムです。
 以下のQiitaに概要を説明しています。
 
Elixirで作るニューラルネットワークを用いた手書き数字認識① 〜関数の準備編〜
https://qiita.com/sanpo_shiho/items/457c02a0007406dc381d 

Elixirで作るニューラルネットワークを用いた手書き数字認識② 〜完成編〜
https://qiita.com/sanpo_shiho/items/18d56e44fb37a3969900

# 実行

ビルド後に実行してください。
また、学習モードを一度も実行したことのない状態で推論モードは実行できません。

```
mix escript.build
./aimodoki --learn true   #学習モード
./aimodoki                #推論モード
```

# 権利関係
良識の範囲内でコピペしてもらって全然構いませんが、一応Twitterまで連絡をくださると嬉しいです。
https://twitter.com/sanpo_shiho
