# [AHC036](https://atcoder.jp/contests/ahc036)

## 何をするか

- T個の都市を順に訪問する
- 訪問するために信号を制御し、信号の制御回数をなるべく少なくする

- N(=600)個の都市があり、最初は都市0におり、T(=600)個の都市を順に訪問したい(訪問する都市は重複あり。ただし連続して同じ都市に訪問はしない)
- 都市間の道路はM(N-1 <= M <= 3N-6)個あり、連結である(必ずすべての都市はつながっている)
- 都市に移動するには、移動先の信号が青の時のみ移動可能でる
- 予定した訪問先に順に訪問する際、信号の制御が可能で、信号の制御の回数をなるべく少なくしたい
- 信号の制御はLA(N <= LA <= 2N)の長さの配列AとLB(4 <= LB <= 24)の長さの配列Bを使用する
- 配列Aと配列Bから同じ長さの区間を選択して、配列Aから選択した区間を、配列Bの指定した区間で書き換える
- 配列Bで指定されている都市の信号が青となる
- 配列Aは初回に決めることができる(その後は変更不可)
- 移動と信号の制御の行動を合わせて10^5以下にする

## 理想的な解

- 一度に信号を青にできるのはLB個
- 都市間の移動は、0から開始しT個の都市を順に訪問する
- 都市間の移動は、BFSなどで最短経路を求める事ができる。最短経路の和が移動回数の最小で、LTと置く
- 理想的な制御ができたとすると、LT/LBが、信号の制御回数の最小値

## 理想的な解に近づけるために行うこと

- 都市間の移動は最短経路である方がよい。ただし信号の制御回数を抑えることができるなら、遠回りしてもよい
- 一度に信号を青にできる個数はLB個と限られており、一度に信号を青にできる対象は配列Aに依存する
- 配列Aの並びは、都市間の道路で隣接しているor近い都市を並べた方が良い

- 初期: 配列Aの並びを最適化するにはどうすればよいか
  - 移動中の方法を考える中で初期の配列Aも最適化する必要がありそう
- 移動中: => この中で初期の配列Aを最適化する？
  - 移動の経路を決定する
  - 信号を青にする対象を決定する

- 配列Aが決まったときに、最適な行動はとれるか？
- 配列Aを部分問題として最適化できるか？
  - 各都市の隣接したものを順に入れていく(Done)
  - 経路を計算して、一番利用する経路から入れていく => ある都市に対して一番利用する経路を順に入れていっても一致率は減少 => 経路に沿いつつ、出現数が多い都市を優先して入れていくとよいのではないか？

- 一致率(最短経路に対して、配列Bがどの程度一致しているか)
  - switchごとの一致率を見てもよいかも(switchしないタイミング多少ずれるのは当たり前)

### 配列A

- must
  - 経路にあるすべての都市を含むこと
- want
  - 経路に沿って訪れる都市が連続して含むこと
  - 重複がないこと(連続する都市やLB範囲)
- match_rateの範囲
  - 最大: ある都市からの経路がLB範囲ですべて同じ: 100%
  - 二分木(2経路に半分ずつ分かれていく経路): 最初の都市が8回経路に含みLB=4の場合は、(8+4+4+2)/(8*4)=18/32=56% > 50%
  - 最小: すべての経路が別々: (N+LB-1)/(N*LB) => 1/LB (N => ∞)
- ある経路の配列AのLB範囲においては、次の経路の中で最大の出現数をとるのが最適
  - 次の経路の出現数の合計は、一つ前の経路の出現数と一致する
  - 以下のデータ構造を持つといいのではないか
    - P: 最適化する最初の都市(Vec)
      - l: 経路の何番目か(Vec)
        - HashMap(key)
          - pre: 一つ前の都市(l=0の場合はp)
          - Pl: 最適化する最初の都市からl番目の都市(P0=P)
        - HashMap(value)
          - cnt: 出現数

### 経路

- 出現都市を少なく
- 重複都市を多く

## 実験

- 理想的な制御回数LT/LBを算出 => 200サンプルで平均768
- BFSなどで最短経路を算出(Done)
- 信号を青にする対象を探索 => 実装したが、配列Aが適切でないとほぼ1つずつしか青にできない => v3で対処(Done)

## その他気になること

- x, yはビジュアライザー用の経路がクロスしないような配置。ある程度の制約がある並びなので、最適化のヒントになるのではないか
  - 入力作成時に、x, yの距離が一定以上離れている点を採用している
  - 辺の追加時には、一定以下の距離のものを優先して採用している。一定超のものも最大値を設定して、後から採用する(数は少ないと思われる)
  - 上記の追加できる辺をすべて追加したのちに、確率で辺を除外する
  - 上記の結果でMが決まると思われるが、入力生成にMの記述はない。理論的に、N-1 <= M <= 3N-6 になるのか？

## ToDo

- 一括実行処理(Done)
- 実行結果の検証(Done)
- 配列Aをすべて使う(Done)
- 複数の経路から最適な経路を選択する(Done)
- 配列Aを最適化する
  - 配列Aの後半はランダム
  - swichしたタイミングの一致率を見て、シミュレーションで配列Aを最適化する★
  - 統計的な考えから配列Aを最適化できないか
  - 連続が必要なため直近の出現数を参照する
- 次の都市の移動経路を考慮する(つなぎの部分は考慮していない)
  - すべての経路をつなげてから移動する？ => その場合現状の都市間の移動ごとの複数経路を試す処理が行えなくなる => 複数経路を試す処理は削除(Done)
  - 移動経路をつなげるときに最適化する？(Done)
- 最短経路を取得するBFSが間違っている
- 通らない経路があるため除外する(Done)
- 配列Aのランダムに出現数が多い都市から最適解を入れていく(Done)

## History

- v1: サンプルコードを改良し、幅優先探索で最短経路を選択(スコア: 356,812, 予測: 367,451)
- v2: 信号の制御でbを最大限使用する(スコア: 334,740, 予測: 349,674)
- v3: 配列Aを都市間の道路の隣接順に設定する(スコア: 181,773, 予測: 188,880)
- v4: 配列Aをすべて使用し、一番連続して青にできる箇所を使用(スコア: 171,054, 予測: 176,741)
- v5: 都市間の経路を最大10パターンまで取得して、最適な制御回数を選択する(スコア: 166,570, 予測: 172,577)
- v6: 配列Aの選択で出現数を考慮(スコア: 136,733, 予測: 140,769)
- v7: 連続することが大切なため、出現数を直近のみを参照(スコア: 136,646, 予測: 140,162)
- v8: 都市間のつなぎ部分を連続して最適化(スコア: 130,657, 予測: 134,272)
- v9: 配列Aで連続する出現数が一番多い都市を選択していく(スコア: 109,486, 予測: 121,472)
- v10: 配列Aの残りをランダムから最適化(スコア: 92,418, 予測: 102,626)
- v11: pathに出てくる都市を少ないように最適化(各都市間2パターン)(スコア: 90,535, 予測: 100,732)
- v12: いくつか(4つ)のpathの最適スコアを採用(スコア: 89,082, 予測: 99,493)
- v13: すでに利用した経路を優先して使用する(スコア: 88,961, 予測: 98,782)

## 実行コマンド

- 1テストケース実行

```
cargo build
cat .\in\0000.txt | .\target\debug\ahc036.exe > .\out\0000.txt
.\tools\vis.exe .\in\0000.txt .\out\0000.txt
```

- 一括実行

```
cargo build
python .\simulator.py
```

