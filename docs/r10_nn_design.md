# R10: NN第二モデル 設計書（実装前・設計のみ）

> 対応: `docs/strategy_20260710.md` §3 P5。前提: `docs/playbooks/00_common.md`（リーク鉄則・pooled RMSE・API）、
> `docs/design.md`（全体設計・EDA実測値）、`analysis/experiment_ledger.md`（実験の正）、
> `cdeotte/nn-starter-cv-15-5 (Kaggle)`（下敷き参照実装）、`scripts/run_stack_v23_cv.py`（現行ベストの特徴・CV設計）。
> **本書は設計のみ。実装は着手しない。**

## 0. 位置づけと結論の先出し

現行ベストは stack v2.3（LGBM、66特徴、5-seed平均、pooled OOF **8.8573**）。金圏（LB ~6.30）にはあと
−2.5ft必要だが、GBDT系の特徴注入路線（R7〜R9）は「定数offset推定系の天井」（design.md §0: stack v2.3 ≈
const-oracle 9.07 近傍）に漸近しつつあり、単独では届かない。3位Tuckerが早期に「NN CV 8.5」（当時のGBDT系
~11水準を上回る）を得ていたという証言は、**アーキテクチャを変えることで現行GBDT系の天井を突破できるという
存在証明**である。R10はこの唯一の跳躍候補に着手する設計。

**結論（詳細は各節）**:
- ターゲット空間は絶対TVTでもanchor直下の生ドリフトでもなく、**現行stackが既に検証済みの「PF blend残差」**
  （`target = TVT − pf_blend`）を踏襲する。理由は§1.2。
- 本命アーキは **dilated 1D-CNN（TCN, cdeotte参照実装の直系）**、保険は **BiGRUエンコーダ+非自己回帰デコーダ**。
  理由と比較は§2。
- 特徴チャネルは stack v2.3 の66特徴（重要度上位: `spatial_d`, `dz_from_ps` 等）をほぼ全量、系列/静的チャネルに
  翻訳する（§3）。
- CVは stack と**同一 well-GroupKFold(5, seed=42) を `outputs/stack_v2_oof.npz` から復元して再利用**し、OOFを
  直接比較可能にする（§4）。
- ゲート: 単体 pooled OOF **≤9.5** でアンサンブル原資合格、**≤8.8** で単体競争力、**1週間で9.0を切らなければ撤退**
  （§5、strategy_20260710.md P5のキル基準と同一）。

---

## 1. 問題の定式化

### 1.1 系列の単位とスコープ

1 well = 1 系列。行順序は `MD`（全773/773 wellで厳密1ft刻み、design.md §3.3）。境界は `data.eval_mask(h)`
（`TVT_input` がNaNの末尾連続ブロック）。実測レンジ（design.md §3.3〜3.4、train 773well）:

| 量 | min | median | mean | max |
|---|---|---|---|---|
| well全行数 | 2,058 | 6,576 | — | 12,141 |
| eval（予測対象）行数 | 407 | 4,840 | 4,895 | 10,052 |

cdeotte参照実装（`nn-starter-cv-15-5.ipynb`）に倣い、**known区間+evalゾーンを1本の系列として丸ごとモデルに渡す**
（既知区間はエンコーダ文脈、evalゾーンは損失を計算する対象行、`target_mask` で区別）。この方式は実装が簡潔で、
コンテキスト境界をモデル自身が `known_tvt_mask` チャネルから学習できる（cdeotte参照実装で既に検証済みのI/O
形状）。

### 1.2 目的変数: PF blend残差を踏襲する

**重要な既存知見（`run_stack_v23_cv.py` L356-357、`analysis/experiment_ledger.md` 2026-07-06行）**: 現行stackは
`target = TVT − pf_blend`（`pf_blend = anchor + 0.7*(pf_tvt − anchor)`、PF blend w0.7）を予測しており、これは
2026-07-06の実測で確定した設計選択である — 同一特徴での比較で **(a) `U = TVT − anchor` 直接回帰 = 11.077 ≫
(b) PF残差 = 10.651**。以降 stack v2〜v2.3 は全てこの (b) の系譜（floorをPF blendに固定し、その上の残差だけを
学習）。

NNもこれを踏襲する。理由:
1. **既に自データで検証済みの設計**（GBDTとNNで比較実験の重複を避け、既知の勝ち筋を継承）。
2. **必要な受容野を構造的に小さくする**。生の anchor相対ドリフト `U = TVT − anchor` を直接回帰させると、
   モデルは「anchorからevalゾーン末尾までの低周波ドリフト全体」（|last−anchor| median 11ft・p99 73ft・max
   104ft、eval行数最大10,052）を系列の受容野の中で積分し直す必要がある。PF blend は既にこの低周波ドリフトを
   物理トラッカー（rate状態を持つ粒子フィルタ）として追跡済みなので、残差ターゲットは**局所的な誤登録の補正**
   （オーダーは数〜数十ft）に縮退する。これは本命アーキ（§2.1）の受容野設計（±500〜1000行）と整合する。
3. R3-v3の教訓（絶対TVT空間で木モデルが外挿崩壊、anchor相対化で正常化）はNNにも構造的に当てはまりうる —
   PF残差ターゲットは anchor相対よりさらに「原点非依存」の性質が強く、同種の崩壊に対してより頑健。

**アブレーション用のフォールバック**として、`U = TVT − anchor` を予測する設定も学習コード側にフラグで残す
（config切替のみ、実装コストほぼゼロ）。PF残差が想定通り機能しない場合の切り分けに使う。

### 1.3 損失関数: pooled RMSEと整合する行重み

**設計上の落とし穴（本書で明示的に固定する）**: バッチ内 well の eval行数は最大 10,052/最小 407 と25倍以上の
開きがある。cdeotte参照実装の `masked_mse`（`torch.mean(diff[mask] ** 2)`、バッチ×時間軸をフラットにして
mask位置だけ平均）は、**バッチ内の全 unmasked 行を等重みで平均する**ため、pooled RMSE（`ΣSE/N`、well平均で
はなく行数で加重）の定義と一致する。これを踏襲する。

**やってはいけない実装**: well ごとにRMSE/MSEを計算してからwell間で平均する損失（per-well平均）。CLAUDE.md
の鉄則「指標 = pooled RMSE、well ごとに平均しない」はローカル評価だけでなく学習時の損失関数にも及ぶ —
per-well平均損失は短いevalゾーンのwellを不当に重く学習してしまい、pooled RMSE最適化からズレる。

具体的な実装契約:
- padding位置は必ず `target_mask=False` とし、`pred[mask]` のみを損失に使う（cdeotte参照実装のcollate方式を
  踏襲）。
- ミニバッチは「well数固定」ではなく「バッチ内総行数の上限」で組む（§4.2）。総行数ベースのバッチ化により、
  1バッチ内の行重みが均一に近づく。
- ターゲットの標準化（mean/std）は**学習foldのみから計算**（cdeotte `compute_target_scaler` と同じ fold-safe
  パターン）。

---

## 2. アーキテクチャ案（本命 + 保険）

両案ともパラメータ数 **<2M**（要求上限5Mに対し十分な余裕）とし、CPU推論（提出時9h予算、200 well）を最初から
制約に入れる。

### 2.1 本命: Dilated 1D-CNN（TCN）— cdeotte参照実装の直系改造

cdeotte参照実装（`TCNResidualModel`: `input_proj` Conv1d→`ResidualBlock`×N（kernel=5, dilation=2^i, BN+SiLU+
Dropout, residual add）→`head` Conv1d）をほぼそのまま踏襲し、**チャネル設計とターゲットのみ§1.2/§3に差し替える**。

| ハイパラ | 値 | 備考 |
|---|---|---|
| hidden channels C | 128 | cdeotte参照は96、対称パディングなのでeval行だけでなく既知区間からも文脈を得る |
| block数 B | 7〜8 | 7で受容野±508行、8で±1020行（下記） |
| kernel | 5 | cdeotte参照と同じ |
| dropout | 0.10 | 同上 |

**受容野**: 対称パディング(non-causal)のResidualBlockをB個、kernel=5・dilation=2^iで積むと、
受容野 = `1 + 8*(2^B − 1)` 行。B=7で **1,017行**（片側±508行 ≒ 508ft）、B=8で **2,041行**（±1,020ft）。
median eval長4,840行・max 10,052行に対しては全域をカバーしない — が、§1.2の設計によりターゲットは
「PF blend残差」（局所誤差の補正）なので、全域受容野は不要という仮説に立つ。この仮説が誤っていれば§3.2の
well-level静的ブロードキャストチャネル（既知区間全体の統計量）で補い、なお不足するならBを9〜10まで安価に
増やせる（1ブロック追加で約+165Kパラメータのみ）。

**パラメータ数見積り**（入力チャネル数≈60、C=128、B=7）: input_proj ≈6.5K + 7×164.6K(block) ≈1,152K +
head ≈8.3K → **合計 ≈1.17M**。B=8で ≈1.33M。

**採用理由**:
1. 系列長方向に完全並列（Conv1dはRNNと違い時間軸に逐次依存がない）→ GPU学習・CPU推論とも高速。200 well×
   最大12,141行のCPU推論が9h予算に収まる確度が最も高い構成（§4.4のFLOPs見積り参照）。
2. **既に動く参照実装を持つ**（`cdeotte/nn-starter-cv-15-5 (Kaggle)`）ため、Dataset/collate/masked
   loss/GroupKFold/fold平均の足回りを流用でき、2〜3日の予算内でチャネル設計とターゲット差し替えに工数を
   集中できる（エンジニアリングリスクが最も低い）。
3. 局所受容野が§1.2のPF残差ターゲット設計と構造的に整合する（「全域積分」ではなく「局所補正」を学習すれば
   よいタスクに、局所受容野モデルを当てるのは過剰な複雑化を避ける選択 — Tuckerの「simple」示唆とも一致）。

**cdeotte参照実装からの重要な変更点**（§3で詳述）: 参照実装は `MD, X, Y, Z` を**絶対値**のままチャネルに
含めている（cell 8 `feature_dict["MD"]=md` 等）。これはR3-v3の教訓（絶対座標系での木モデルの外挿崩壊）と
同種のリスクをNNにも持ち込む — well間で座標系が全く異なるため、絶対座標を見た時点で「このwellはこの座標
レンジ」という記憶に頼った過学習が起きうる。**R10では絶対座標チャネルを採用せず、anchor相対の相対量
（`*_from_ps`）のみを使う**（stack v2.3の既存設計と同じ境界）。

**cdeotte参照のCV参考値**: この参照notebookは通常設定（EPOCHS=6, hidden=96, blocks=5, batch_size=6）で
CV≈15.5（notebook名に明記）— carry_last floor 15.9099よりわずかに良い程度で、当方のcarry_last/PF/stack
基準からは大きく劣る。ただしこれは「汎用特徴（絶対座標・生GR・生TVT_input）+ `TVT−anchor`直接ターゲット」
での結果であり、**アーキテクチャ自体の限界ではなく、当方の物理トラッカー特徴（PF/beam/spatial）とPF残差
ターゲットを持たないことの限界**と解釈する（§3のチャネル設計が本設計の主要な賭け）。

### 2.2 保険: BiGRUエンコーダ + 非自己回帰デコーダ

「既知区間をエンコーダ、evalゾーンをデコーダで一括予測（自己回帰なし）」という、より明示的なencoder-decoder
分割。

- **エンコーダ**: 既知区間の直近window（末尾最大3,000〜4,000行、features.pyの `_KNOWN_TAIL_LONG=200`/
  host hint「予測開始直前の50点が最重要」を一般化した設計）を2層BiGRU（hidden=64/方向、双方向で128次元）に
  通し、(i) 最終隠れ状態の結合（=既知区間全体の要約コンテキストベクトル）と (ii) 逐次隠れ状態列を得る。
- **デコーダ**: evalゾーンの各行について `[コンテキストベクトル, その行の位置/幾何/トラッカー系チャネル
  （§3.1）]` を結合し、1〜2層の単方向GRU（hidden=96、非自己回帰 — 前ステップの**予測TVT**は入力に戻さない、
  流れるのはエンジニアリングされた入力チャネルのみ）+ 小型MLPヘッドへ通し、行ごとの残差を一括出力する。

**パラメータ数見積り**: エンコーダ ≈118K + デコーダGRU ≈126K + ヘッド ≈6K → **合計 ≈250K**。TCN案よりさらに
小さく、hidden次元を2〜4倍に拡張しても5M上限に対して余裕がある。

**採用理由（保険としての位置づけ）**:
1. タスク仕様の例示（(a) GRU/LSTM encoder-decoder）に最も忠実な構成であり、**受容野に上限がない**
   （GRUの隠れ状態は既知区間window全体を逐次要約するため、TCN案の固定受容野±500〜1,000行という仮定が
   §1.2で外れた場合の直接的な代替になる）。
2. TCN案とアーキテクチャ的に十分異質（畳み込み vs 再帰、局所 vs 大域要約）なため、**もし両方が採用ラインに
   乗った場合、R9のLGBM seed-ensemble（−0.1232）と同型の「多様性のある弱学習器の平均」に使える**（誤差が
   相関しにくい2本目のNNとしての価値）。
3. 逆に**主命にしない理由**: GRUは時間軸方向に逐次計算（cuDNN実装でも時間ステップ間の依存は残る）のため、
   TCN案と比べてGPU学習・CPU推論とも遅い。evalゾーン最大10,052行×200 wellのCPU推論時間は実測が必要で、
   9h予算超過（design.mdが最重要リスクとして明記する「Submission Scoring Error」＝静かな不採点）のリスクが
   TCN案より高い。

### 2.3 比較表

| 項目 | (a) 本命: TCN | (b) 保険: BiGRU enc-dec |
|---|---|---|
| パラメータ数見積り | ≈1.2〜1.3M | ≈0.25M |
| 受容野 | 固定（±500〜1,000行、B調整可） | 実質無制限（隠れ状態が既知区間全体を要約） |
| 学習/推論の並列性 | 系列長方向に完全並列 | 時間軸に逐次依存 |
| CPU推論9h予算への確度 | 高（§4.4のFLOPs見積り） | 中（実測未了、長well懸念） |
| 実装リスク（2-3日予算内） | 低（動く参照実装を直接改造） | 中（encoder-decoder配線を新規実装） |
| 採用順位 | 1本目（着手） | 2本目（時間が余れば/TCNが受容野不足で伸び悩んだ場合） |

---

## 3. 特徴設計（チャネル）

方針: stack v2.3の66特徴（`src/rogii/features.py`＋`src/rogii/stack_features.py`）を、可能な限りそのまま
系列/静的チャネルへ翻訳する。新規の特徴発明はしない（既存の重要度上位=`spatial_d`, `dz_from_ps`等の資産を
再利用することが目的で、NN側での特徴探索はスコープ外）。

### 3.1 per-row系列チャネル（時間軸に沿う、長さT=well全行数）

| 系統 | チャネル | 出典 |
|---|---|---|
| 位置（**相対のみ、絶対X/Y/Z不使用**） | `md_from_ps, dx_from_ps, dy_from_ps, dz_from_ps, dist3d_from_ps, row_frac` | `features.py` L171-192 |
| 軌跡微分（傾き=入射角の近似） | `dz_dmd_roll51, dx_dmd_roll51, dy_dmd_roll51` | `features.py` L120-137 |
| GR（弱信号、補助扱い — §7） | `gr, gr_missing, gr_diff_1, gr_diff_10, gr_roll_mean/std@{11,51,151}` | `features.py` L152-160 |
| typewellとの照合 | `gr_minus_twgr_anchor_calibrated`, **新規**: `tw_gr_at_carry`（下記） | `features.py` L162-183, `typewell.py::tw_gr_lookup` |
| 物理トラッカー（per-row） | `pf_d, pf_std, beam_d, beam_margin, pf_beam_abs_diff, pf_d_damp, pf_std_is_inf, pf_beam_sign_agree` | `stack_features.py::build_tracker_features`（FEATURE_COLUMNS 8本） |
| 空間事前分布（per-row） | `spatial_d, spatial_prefix_rmse, spatial_nn_dist_median, spatial_gated_d` | `stack_features.py::build_spatial_features` |
| R7/R8候補デルタ（per-row） | `pfbma_v2_d, beam_grid_d, pfbma_v2_minus_pf, pfbma_v2_s5/8/12_d, pfbma_v2_scale_range, pfbma_v2_s3_std` | `run_stack_v23_cv.py` config (c) |
| 系列境界マーカー | `known_tvt_mask`（0/1）, `tvt_input_delta_last`（既知区間はffill値−anchor、evalは0埋め） | cdeotte参照 cell8 踏襲 |

**`tw_gr_at_carry`（新規、既存物理トラッカーの発想を素直に翻訳したチャネル）**: 各行について
`anchor_tvt + slope_md_last200 * (md − ps_md)` という「既知区間末尾の局所傾きで単純外挿したTVT」（=PFの
motion model初期化と同じ発想、`particle.py`の`_calibrate_rate`が使う`_RATE_CAL_ROWS=200`と同一窓）を
`carry_tvt` として計算し、`TW.tw_gr_lookup(tw)(carry_tvt)` でtypewell GRを引く。`gr − tw_gr_at_carry` を
併置することで、PFが内部で計算している「観測GR vs 期待GR」の尤度入力そのものをNNにも直接渡す。

### 3.2 well-level静的ブロードキャストチャネル（全T行に同一値を複製）

TCN案の固定受容野（§2.1、±500〜1,000行）がmedian eval長4,840行に届かないことを補うための設計要件。
stack v2.3のwell集約8特徴＋既知区間統計をそのまま複製する。

`anchor_tvt, slope_md_all, slope_md_last200, slope_md_last50, slope_z_all, known_tvt_{min,max,range,mean,
std}, n_known_rows, pf_d_final, pf_d_mean, pf_std_well_{mean,max}, beam_margin_well_mean,
pf_beam_abs_diff_well_mean, gr_nan_frac, eval_len`（`features.py`の既知区間統計＋
`stack_features.py::build_well_aggregate_features`のWELL_FEATURE_COLUMNS全量）。

合計チャネル数は概算 **55〜65**（per-row系列 約43 + 静的ブロードキャスト 約19、実装フェーズでの重複整理・
GR系サブセットの取捨で前後する）— stack v2.3の66特徴とほぼ同スケール。

### 3.3 リーク境界

全チャネルは既存のリーク監査済みキャッシュ（`data/processed/tracker_cache/`, `data/processed/spatial_cache/`,
`outputs/path_bank_pfbma_v2.npz`, `outputs/path_bank_beam.npz`）と `features.py`/`stack_features.py` の
既存関数から構築し、**トラッカー計算自体をNNパイプライン内で再実装しない**（車輪の再発明とリーク再監査コスト
を避ける）。`h["TVT"]` はターゲット構築（`TVT − pf_blend`）以外で一切読まない。`ANCC/ASTNU/ASTNL/EGFDU/
EGFDL/BUDA`・typewell `Geology` は不使用（既存鉄則を継承）。

---

## 4. CV / 学習プロトコル

### 4.1 fold再利用（stack_v2_oofとの直接比較可能性）

`outputs/stack_v2_oof.npz` は `used_wells`（773 well id）・`well_idx`（行→well番号）・`row_fold`（行→fold番号、
`cv.well_folds(wells, n_splits=5, seed=42)` 由来）を保持している。NN側は独立に `cv.well_folds` を呼び直すの
ではなく、**このnpzから `well_fold[used_wells[i]] = row_fold[well_idx==i][0]` を復元し、assertで773 well全件
一致を確認した上でwell→fold対応として使う**。これによりNNのOOFとLGBM stack（v2/v2.1/v2.2/v2.3）のOOFが
**完全に同一の分割**の上で計算され、pooled RMSEの差がフォールド運の違いではなくモデル自体の差であると
言い切れる。

ES（early stopping）分割も同一パターン（`_split_fit_es`, `ES_SPLIT_SEED=42`, `es_frac=0.15`）を踏襲し、学習
foldの中から検証専用wellを切り出す（GBDTと同じくscore foldには一切触れさせない3分割: fit/ES/score）。

### 4.2 バッチ化・可変長処理

well全行数のレンジが2,058〜12,141（5.9倍）と広いため、固定well数バッチはメモリ使用量が最大12倍振れる。
**総行数ベースの動的バッチ化**（NLPのtoken-budgetバッチングと同型: 1バッチの合計行数が目標値
（例: 8,000〜16,000行、GPUメモリから逆算）を超えない範囲でwellを詰める）を採用し、padding+`target_mask`は
cdeotte参照実装の`collate_wells`パターンを踏襲する。

### 4.3 augmentation

| 種別 | 内容 | 優先度 |
|---|---|---|
| 既知区間window打ち切り | 既知区間の**古い方**から末尾最大3,000〜4,000行までに切り詰め（host hint「直前50点が最重要」と整合、既知区間の先頭は情報価値が低い）。メモリ上限を安全に抑える主目的で、副次的にaugmentationとしても働く | 必須（§2.1/2.2双方の実装に組み込む） |
| トラッカーチャネルのランダムdropout | 学習時に一定確率でPF/beam/spatialチャネルを0埋め+"欠落"フラグに置換し、トラッカー不信頼時の頑健性を学習させる | 推奨（低コスト） |
| 連続チャネルへのGaussian jitter | 位置・GR系チャネルに小さいノイズを加える正則化 | 推奨（低コスト） |
| prefix-cut（合成anchor） | 既知の教師ありwellで、anchorをより手前に人工的に動かし、本来の既知区間の一部を疑似evalゾーンとして学習に使う。anchorから終端までの距離分布を薄く広くする効果が理論上大きい | **保留（stretch）**。トラッカー特徴（PF/beam/spatial）は各wellの**実際のanchor**でのみ事前計算済みで、合成anchorでの再計算は同じコストのバッチジョブが再度必要（R2-a/b相当の重い計算のKaggle再オフロード）。実装コストが2-3日予算に見合わないため、まずトラッカーchannelを疑似evalゾーンで0埋め+フラグ扱いにする簡易版で代替できるか検証してから可否判断 |

### 4.4 Kaggle GPU学習の1実験サイクル時間見積り（未実測・事前見積り）

本命TCN（C=128, B=7, 入力チャネル≈60）について、well 1本あたりのforward FLOPsを概算する: 1ブロックあたり
2畳み込み×(C×C×kernel×2) ≈ 2×(128×128×5×2) ≈ 327K FLOPs/行、7ブロックで ≈2.3M FLOPs/行。median well長
6,576行なら ≈15B FLOPs/well（forward）、backward込みで概ね3倍として ≈45B FLOPs/well。学習用well ≈618本/fold
（773×4/5）なら1エポック ≈28T FLOPs。Kaggle T4クラスGPU（FP32数TFLOPS〜、mixed precisionでさらに高速）を
仮定すると**演算自体は1エポックあたり数十秒〜数分のオーダー**だが、実運用はデータロード・Python側オーバー
ヘッド・可変長パディングの無駄計算が支配的になりやすく、**1エポック実測は数分レンジになる可能性が高い**
（cdeotte参照のデフォルトはEPOCHS=6・N_FOLDS=5）。

**見積り（保守的、未実測）**: 5fold × 10〜15epoch（early stopping込み）× 数分/epoch ≈ **単一GPUセッションで
3〜6時間/実験構成**。Kaggleの無料GPUクォータは週30時間程度が実務的な目安であり、フルスケール実験を2〜3構成
（本命1本+ハイパラ振り1〜2本）流せる範囲。**この見積りは実装ステップ3〜4（§6）の最初のKaggle実行で必ず
実測に置き換える** — 数値の楽観的先取りを避けるため、実装フェーズの報告では実測値のみをledgerに記載する。
まず40〜80 well程度のsmoke実行でオーダーを確認してからフル773 well学習に進む2段階運用とする。

---

## 5. 成功基準とキル基準

| 基準 | pooled OOF | 判定 |
|---|---|---|
| アンサンブル原資合格 | **≤9.5** | 単体ではstack v2.3（8.8573）に劣っても、誤差が非相関ならアンサンブル/特徴注入で正味の改善に使える |
| 単体競争力 | **≤8.8** | 現行ベスト8.8573と同水準以上。R7〜R9の特徴注入路線の天井を単独で突破したことを意味する |
| 1週間キル基準 | **9.0を切れない** | 撤退（strategy_20260710.md P5と同一基準）。着手日から7日以内に何らかの構成でOOF<9.0を達成できなければ、R10全体を打ち切りbank/spatial等の既存資産強化に戻る |

**追加の統合判断（≤9.5未達でも即ゼロにしない救済チェック）**: pooled OOFがゲート未達でも、per-well符号付き
残差の stack v2.3 OOFとの相関が低ければ（目安 Pearson r<0.5）、`run_fixed_blend_grid.py` のGram行列評価
（数分で完了）で出力blend/特徴注入の価値を機械的に確認する。これはstrategy_20260710.mdの「すべてを特徴量
としてstackに注入する」路線と整合する — **NN予測を `nn_pred_d = nn_pred_tvt − pf_blend` という単一の
delta特徴としてstack v2.4に追加し、R7/R8/R9と同じ手順（前ベスト−0.10ゲート）で採否判定する**方が、NN単体の
標準では9.5/8.8ラインに届かなくても価値を回収できる可能性が高い（出力blendは較正点#4/probe実測で転移しない
ことが確定済み、特徴注入だけが実証済みの転移経路）。単体性能の基準（9.5/8.8/9.0）は本節の通り厳守しつつ、
「単体基準未達＝即ゼロ」ではなく「特徴注入経路での再評価」を必ず一段挟む。

---

## 6. 実装ステップ分解（1ユニット=1ワーカータスク）

1. **データセットビルダー**（`scripts/build_nn_sequence_dataset.py`、Kaggle実行想定）: 773 well分の§3チャネル
   （per-row+静的ブロードキャスト）とターゲット（`TVT − pf_blend`）を、可変長を扱えるフォーマット
   （well毎npz、またはoffsets+flat配列のCSR風単一npz）でビルドし、Kaggle datasetとしてアップロード
   （773well合計 ≈4.9M行×約50ch×4byte ≈1GB、ローカル3.8GB箱では厳しいためKaggleでビルド）。既存の
   `block_slices`/`reindex_to_master`/y_true突合パターンを再利用してリーク監査を継承。単体テスト: 数well分の
   チャネル次元・NaN方針・境界（known_tvt_mask）の正しさ。
2. **PyTorch Dataset/collate + fold-safeスケーラ**: cdeotte参照実装の`WellSequenceDataset`/`collate_wells`を
   土台に、総行数ベースの動的バッチング（§4.2）と学習fold限定のfeature/targetスケーラ（§1.3）を実装。単体
   テスト: pad位置がmaskで損失から除外されること、スケーラがvalidation/scoreフォールドの値を一切参照しない
   こと。
3. **モデル定義（TCN本命）+ 学習ループ**: §2.1のTCN、masked pooled-weighted MSE損失、AdamW+cosine LR、
   ES-holdoutでのearly stopping。ローカルは40〜80 well規模のCPU smokeのみ（動作確認、収束は問わない）。
4. **Kaggle GPU学習カーネル化**（`enable_gpu: true`, `dataset_sources: [ステップ1のdataset]`,
   `competition_sources`）: §4.1のfold復元で5-fold OOFを算出し、pooled RMSE実測・時間実測（§4.4の見積りを
   置き換える）・fold毎重みをKaggle Model（`create_model`/`update_model`）として保存。
5. **OOF評価+ゲート判定**: 既存 `run_stack_v2x_cv.py` 群と同じ報告フォーマット（pooled RMSE、per-well
   median/p90/max、helped/hurt）でledgerに記録。stack v2.3 OOFとのper-well相関を算出し§5の救済チェックに使う
   データを揃える。
6. **（ゲート通過 or 相関低の場合）特徴注入ルート**: `nn_pred_d = nn_pred_tvt − pf_blend` をstack v2.4の67本目
   特徴として追加し、R7〜R9と同一のCVゲート（前ベスト−0.10）で採否判定。
7. **（十分な単体競争力 or 特徴注入が大きく効いた場合のみ）提出カーネル統合**: 学習済み重み（5-fold平均、
   または複数seed平均）をKaggle Model経由で提出カーネルに`dataset_sources`添付、CPU推論（torch CPU、200
   well想定）。playbook 03の既存規律（well単位try/except→carry_last縮退、Kaggleはローカルの~1.5倍遅い、
   進行print、9h予算の実測ゲート）をそのまま継承する。

保険アーキ（§2.2 BiGRU）は上記1本目（TCN）の結果を見てから着手判断する独立ユニットとして扱い、本ステップ
分解には含めない（時間が余る/TCNが受容野不足で伸び悩んだ場合に3〜5相当を差し替えて再実行）。

---

## 7. リスク

| リスク | 内容 | 備考/緩和 |
|---|---|---|
| GR弱信号への過信 | R4-b実測: GRはshuffle検定を通らない（true TVTでもcorr~0.7、shuffled GRとtop10 coverage同等）。GBDT importanceでもGR系は下位 | §3のチャネル設計はGRを主軸に置かず、位置/トラッカー/typewell照合チャネルを優先。GR系チャネルの重要度をOOF後に確認し、極端に低ければ削減を検討 |
| 系列長のばらつき（407〜10,052 eval行、2,058〜12,141 全行） | padding/mask管理のバグは pooled RMSE を不当に楽観化する典型的な落とし穴（padding行が損失やOOF集計に混入） | §1.3/§4.2の契約をテストで固定。OOF集計時も`target_mask`を貫通させ、pad行がy_trueに混ざらないことをassertで守る |
| fold間の分布差 | field-CV v2監査（2026-07-09実測）でspatial特徴がfield隔離時に大幅悪化（9.23→10.69）した前例 = 近傍train well依存のリスクは既に実証済み | NNもtracker/spatialチャネルに強く依存する設計のため同種のリスクを継承しうる。低優先度だが時間があればfield-isolated監査を追加 |
| Kaggle GPUクォータ（週30時間程度、無料枠） | 2アーキ×5fold×複数epoch×ハイパラ探索をフルスケールで多数回すと枯渇しうる | §4.4の2段階運用（40〜80 well smoke→フル773 well）を徹底。実験ごとに使用時間をledgerに記録 |
| CPU提出推論の未実測 | GPU学習後のCPU推論速度は未実測。TCNは並列化が効くため比較的安全という設計上の期待だが、実測するまで確証はない | 実装ステップ7で per-well時間×200+マージンのゲート実測を必須化（design.md §11のR5と同じ規律） |
| 学習済み重みのリーク混入 | オフライン学習→重み固定でdataset添付する設計のため、学習時にリークが混入すると提出後まで気づきにくい | 既存のリーク監査パターン（`block_slices`/`reindex_to_master`/y_true突合、`h["TVT"]`はターゲット構築以外不読）を完全再利用し、reviewerによる査読を必須にする |
| 1週間キル基準の運用が後手に回る | 「2-3日は様子を見る」という運用だと手戻りが大きい | 着手初日からデータセットビルダー（ステップ1）を優先し、遅くとも3日目までにTCNの最初のフルスケールOOFを得る。日次で150 well程度のsmoke OOFを計測し、9.0への接近トレンドが見えなければ早期に撤退判断する |

---

## 付録: 用語の対応表（本書 ↔ 既存コード）

| 本書の用語 | 既存コードでの対応 |
|---|---|
| PF blend floor | `run_stack_v23_cv.py`: `pf_blend_arr = anchor + PF_BLEND_W * (pf_tvt − anchor)`, `PF_BLEND_W=0.7` |
| anchor | `data.last_known_tvt(h)` |
| well-level静的ブロードキャストチャネル | `stack_features.py::build_well_aggregate_features`（WELL_FEATURE_COLUMNS 8本）+ `features.py`の既知区間統計ブロック |
| per-row物理トラッカーチャネル | `stack_features.py::build_tracker_features`（FEATURE_COLUMNS 8本）+ `build_spatial_features`（SPATIAL_FEATURE_COLUMNS 4本） |
| fold復元元 | `outputs/stack_v2_oof.npz`（`used_wells`, `well_idx`, `row_fold`） |
