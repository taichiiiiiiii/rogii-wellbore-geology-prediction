# ROGII - Wellbore Geology Prediction — 設計書 (Design Document)

> 調査日 2026-07-06。出典 = Kaggle コンペページ（Overview/Data/Evaluation/Rules/Code Requirements）、
> 公開 Leaderboard、Discussion、公開 Notebook 5本（Kaggle 上で参照）、およびローカル train 全 773 well の EDA。
> **揮発する数値（スコア・採否）は memory `project_rogii_kaggle.md` と `analysis/experiment_ledger.md`（新設予定）が最新。本書は設計の骨格。**

---

## 0. ゴール: 金メダル（設計ターゲット・**2026-07-09 全面改訂**）

> 改訂根拠 = 2026-07-09 のLB全量分析・discussion 再調査（712037/716699/718670/721549/722041/723647 等）・自前実測4本（slope診断・field-CV v2監査・test/train重複チェック・hard-well分解）。旧版の記述は §14 に移設・注記。

- **メダル数（4,522 teams・2026-07-09 LB実測）**: Gold = top 10+0.2% = **19位・LB 6.328**（安全圏18位 6.137。18↔19位間 Δ0.191 はトップ200内最大の断絶＝金圏境界そのもの）／Silver ≈ 226位（7.176）／Bronze ≈ 452位（7.21）。
- **公開LB（2026-07-09）**: 首位 5.262（7/6から不動）。コミュニティ最終予想（720701）: 金 ~5.5・最上位 ~4.5–4.8。**設計ターゲット = private-robust RMSE ≤5.5**（締切8/5までの圧縮込み）。
- **現在地**: stack v2 OOF **9.2309**（well-GroupKFold）＝LB換算 ~2049位相当。金まで **−2.9〜−3.7ft**。
- **🔑 ギャップの構造（実測で確定）**:
  - oracle階段: per-well **const** 9.07 / **line** 6.59 / **smooth** 3.00 / **候補パス選択**（likpf/pf/beam/formation の per-well 最良、721549 Ochir）**4.5–6** / 我々の4-way手法ルータ（stack/carry/PF/spatial）7.77
  - **stack v2 9.23 ≈ const oracle 9.07 ＝「定数offset推定」系の天井に到達済み**。同系統の改良（特徴追加・チューニング・平滑化）では金に届かない（P4平滑化 −0.03 却下が実証）
  - **slope（dip）の特徴量回帰は不可能と確定**（2026-07-09 実測: 16特徴×Ridge OOF R²=−0.066・nested適用で+0.05悪化。forum独立3系統と一致: direction R²=−0.16／James-Stein shrink=0／「evalゾーンのdipはGRに符号化されていない積分定数」）
  - ∴ **金への実証済み経路は唯一つ: 「候補パスの多様化 × per-well 選択/hedge」**。パス選択の難しさも実測済み（単一信号の識別 recall 0.30）だが、**detection AUC 0.69 ＝ trust-gate は成立**（ずれの検知はできる・方向補正はできない → 高確信時のみ切替、曖昧 well は posterior-mean hedge）
- **正攻法で5ft台に到達できる証拠**: LB#3 Tucker Arrants 本人証言（712037/723647）＝ **per-well 情報のみ（空間・external tops 不使用）で CV 5.0–5.4**。公開 7.1–7.3 クラスタ（917チーム密集、ravaghi artifacts 系統 ρ≈0.89）はリークでなく手法の質で抜ける。
- **spatial 特徴の降格（field-CV v2 監査 2026-07-09）**: stack v2 の −1.33 利得は**近傍 train well 依存**（field隔離で 10.69 に逆戻り・field fold の best_iter 1–51 に崩壊）。hidden test の空間性（train と混在か隔離か）は **stack v2 提出＝較正点#3 が直接判定**。**設計は spatial 利得に依存しない per-well 正攻法を主軸**とし、spatial は「混在なら上乗せ」の保険に格下げ。
- **リーク疑惑は撤回で決着（712037 提唱者本人撤回・722041 上位勢確認）**: visible test 3 well（train 完全複製、ローカル実証済み）は**スコアリング時に隠し ~200 well へ置換**される。leak override は設計から削除（開発時のダミー well 挙動確認にのみ意味）。
- **降格済みレバー（実測棄却、再挑戦禁止）**: slope特徴量回帰／平滑化系後処理／NCC単独／beam単独／within-well GBDT／空間クラスタ・IDWのslope転写（隣接wellのdrift slope相関≈0、offset相関0.98のみ）／CNN inversion／合成事前学習／leak override。

---

## 1. コンペ概要（事実）

| 項目 | 値 |
|---|---|
| 正式名 | ROGII - Wellbore Geology Prediction（id 132265, Featured） |
| 主催 | ROGII（Houston, Texas。データは Texas 地質） |
| 目的 | 水平坑井に沿った地質＝**真垂直層厚 `TVT`（ft）** を予測し、ジオステアリングを自動化 |
| 指標 | **RMSE（ft）**（Evaluation ページの式が正）。※metadata の "Mean Squared Error" は単調同値でランキング一致。公開LBの数値(例5.26)は ft=RMSE |
| 賞金 | $50,000（1st $25k / 2nd $13k / 3rd $7k / 4th $5k） |
| Working Note Award | 2×$2,500（締切 2026-07-06 UTC・medal zone 必須。**現状 medal-zone 未達で本サイクルは対象外**。審査基準は §11 の開発台帳指針として活用） |
| 参加規模 | **4,330 teams**。1日最大 **5 提出**、最終選択 **2 件**、チーム最大 5 |
| 形式 | **Code competition**（Notebook のみ、CPU/GPU ≤9h、**インターネット遮断**、公開外部データ/事前学習モデル可、`submission.csv` 出力） |
| 期日 | Start 2026-05-05 ／ Entry・Team Merge **2026-07-29** ／ **最終提出 2026-08-05** |
| データ利用 | Competition use only（**再配布禁止**、リポジトリにコミットしない） |

**スコア地形（現状の目安）**

| 水準 | RMSE (ft) | 出典 |
|---|---|---|
| carry_last ベースライン | **15.91** | ローカル実測（pooled, 3.78M 行） |
| 素朴 tabular（XGB/NN starter） | ~15.0–15.5 CV | cdeotte starter Notebook |
| 公開 registration+stack（ridge-sp 系） | LB **7.78–7.88** / CV ~9.2 | lightningv08 / pixiux Notebook |
| **公開LB 先頭（2026-07-05）** | **5.262**、上位クラスタ 5.2–6.3 | Leaderboard 実測 |

→ 「素朴 tabular で 15、GR レジストレーションを入れて 9→7.8、先頭は 5 台」。**改善の主軸は GR 信号の位置合わせ（registration）**。

---

## 2. 問題定式化（物理）

水平坑井は測定深度 `MD`（1ft 刻み）に沿って進み、各点で **Gamma-Ray `GR`(API)** を記録する。垂直リファレンス
**タイプウェル**は `GR` を層序深度 `TVT` の関数として与える（`typewell: TVT→GR`）。地層はほぼ層状なので、
**水平坑井の GR(MD) は、タイプウェルの GR(TVT) を局所的にシフト/ワープしたコピー**。よって

> **各横方向点 MD に対し、水平 GR がタイプウェル GR と最も一致する TVT を求める = 信号/画像レジストレーション**。

補助構造:
- **`TVT + Z` フレーム**: `Z`(TVD) は既知。地質位置は概ね `TVT ≈ −Z + const + 局所ドリフト`。目的変数を
  `U = TVT − anchor`（または `TVT + Z − anchor`）にすると、坑井の幾何を差し引いた**残差ドリフトの学習**になり過学習を抑える（公開上位が全採用）。
- **入射角 / incidence angle**: 坑井が地層を斜めに横切るため、GR(MD)→TVT の局所変換率は坑井傾斜と地層 dip に依存。
  軌跡微分（`dZ/dMD, dX/dMD, dY/dMD`）で近似（Evaluation の審査軸に明記された "estimation of incidence angle"）。
- **spurious correlation（偽の一致）**: 似た GR シグネチャが層序上に反復 → GR だけの照合は**誤ブランチに固着**しうる。
  多仮説（particle/beam）＋物理事前分布（Z, formation plane）＋ロバスト平滑で抑える（審査軸に明記）。

---

## 3. データ（EDA 実測・ローカル train 773 well）

### 3.1 ファイル構成
- `train/{well}__horizontal_well.csv` / `{well}__typewell.csv` / `{well}.png`（773 well）
- `test/{well}__horizontal_well.csv` / `{well}__typewell.csv`（**ローカルは 3 well の例のみ。本番 hidden test ≈200 well**）
- `sample_submission.csv`（`id={well}_{row_index}`, `tvt`）
- 全体 2,327 ファイル（csv 1,553 / png 773 / pptx 1）

### 3.2 スキーマ（列の存在＝リーク境界）

| ファイル | 列 |
|---|---|
| horizontal (train) | `MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, TVT, GR, TVT_input` |
| horizontal (**test**) | `MD, X, Y, Z, GR, TVT_input` |
| typewell (train) | `TVT, GR, Geology` |
| typewell (**test**) | `TVT, GR` |

- **train 限定（予測時に使用不可）**: 6 formation 深度 `ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA` ＋ 目的 `TVT` ＋ typewell `Geology`。
- formation 列は forum 708167 いわく **typewell から導出**（独立 3D サーフェスではない）。ただし **test typewell に `Geology` 列が無い**（実測確認）ため、新規 well の contact 深度は復元できない。`tvt_from_contacts` の再構成が成立するのは **train∩test の重複 well 限定の leak**（§8）であって、**一般特徴量ではない**（private では効かない）。

### 3.3 主要インバリアント

| 指標 | 値 |
|---|---|
| MD ステップ | **全 773/773 well で厳密に 1.0 ft**（rows = 固定ステップ系列, MD index ≈ depth ft） |
| horizontal 行数 min/median/max | 2,058 / 6,576 / 12,141 |
| **評価ゾーン = `TVT_input` NaN**：常に末尾連続1ブロック | **違反 0/773**、anchor 無し 0、eval ゾーン無し 0 |
| eval 長 min/median/mean/max | 407 / 4,840 / 4,895 / 10,052 |
| **ローカル採点 N（全 eval 行合計）** | **3,783,989** |
| eval / 全行（median 比） | 0.74 |
| typewell 行数 min/median/max | 636 / 1,874 / 10,043 |
| typewell TVT が horizontal TVT を包含 | **760/773 = 98.3%**（13 well は外挿注意） |

### 3.4 目的と carry_last の天井

| 指標 | 値 (ft) |
|---|---|
| eval TVT min/max/mean/std | 10,039 / 12,894 / 11,546 / 626 |
| drift（first-eval − anchor） | ≈0（median 0.000, max 0.53） |
| **\|last-eval − anchor\|** median/p90/p95/p99/max | 10.97 / 31.96 / 41.29 / 73.26 / 103.78 |
| carry_last pooled RMSE | **15.9099** |
| carry_last per-well RMSE mean/median/max | 12.81 / 10.67 / 70.64 |

→ **eval ゾーン先頭では anchor がほぼ正解**（drift 0）。末尾に向かって TVT がドリフトし誤差が蓄積。分布は**強い右歪**で、
少数の "drift well"（p99=73ft, max=104ft）が pooled RMSE を支配。pooled(15.9) ≫ per-well median(10.7) はこの重み付けの帰結。
**改善は末尾ドリフトの回復＝レジストレーション**が本丸。

### 3.5 GR の欠損（重要リスク）

| 系列 | min/max/mean/std | NaN |
|---|---|---|
| horizontal GR | 13.9 / 487 / 87.8 / 23.8 | **~1,507,972（≈30%）**、しばしば **eval ゾーンで欠損** |
| typewell GR | 18.5 / 434 / 89.1 / 33.7 | 0 |

→ トラッカーは GR 必須だが eval ゾーンで GR が抜けやすい。**GR 欠損時は幾何/事前分布へフォールバック**する設計が必須。
GR のスケールは水平/垂直で近い（mean 88 vs 89）が **affine 校正**（gain/offset）で一致度が上がる（公開上位採用）。

### 3.6 Geology ラベル
43 種。上位6＝`ANCC/EGFDL/ASTNL/BUDA/ASTNU/EGFDU`（formation 列と一致）、tail は `OLMOS/MNSS/UPSN/...`（希少 facies）。train 限定。

---

## 4. 評価と CV 設計

- **本番指標 = hidden test（≈200 well）の全 eval 行 pooled RMSE**。ローカルも **pooled**（well 平均でなく ΣSE→√(ΣSE/N)）で測る（`scripts/run_baseline_cv.py::pooled_rmse` に準拠）。
- **CV = `GroupKFold(n_splits=5)` by well**（同一 well を train/valid に跨がせない）。公開上位・starter が全員この設計。base model・meta・spatial imputer は **leave-self-out**（対象 well を除外して近傍を作る）。
- **CV↔LB の規律（epistemics）**:
  - public LB は test の代表サンプル、private が最終。**private は再スコア済み**（forum 707695「Private Test Update and Rescore」）＝過去 LB との単純比較は不可。
  - **pooled GroupKFold OOF を第一の羅針盤**にし、提出ごとに public LB と突き合わせ、乖離したら分割/リーク/ゾーン処理を疑う。
  - 後処理・ブレンド重みの grid search は **OOF を厳密に改善する時のみ採用**（公開上位の "accept only if beats OOF" 規律を踏襲）。
- 誤差の右歪を踏まえ、**pooled RMSE に加えて per-well RMSE 分布（median/p90/max）と drift-well 群の内訳**も併記して回帰を監視。
- **尺度確定 probe**: metadata は "Mean Squared Error" 表記だが Evaluation ページは RMSE。初回に **carry_last（ローカル RMSE=15.91）を提出**し、LB 表示値との一致で確定（forum 実測で titericz の last-value baseline = **LB 15.883** ≈ ローカル 15.91 → RMSE(ft) スケールでほぼ確定済み。probe はこの再確認を兼ねる）。
- **pooled per-row（非グループ）CV は禁止**（楽観・リーキー。rank3 の「pooled CV ~5」は field-grouped 再現で ~10 との報告あり、forum 712037/714514）。
- **field-grouped CV を併設**: host 発言「表層深度は近傍 10 well から R²>0.99 で補間可能」＝ well 単位 GroupKFold でも**空間リーク**が残り得る。X,Y の空間クラスタで fold を切る field-grouped 変種も測り、**どちらが LB と整合するかを probe 提出で較正**してから本命 CV を固定する。
- **LB ノイズ規律**: public は test の ~25%（~50 well）のみ採点、同一コードの reseed だけで ±0.09–0.38 変動（forum 718670 実測）→ **±0.3 未満の LB 差で採否を決めない**。CV–LB ギャップ >3ft は過学習/リーク/バグのサイン（rank3 Tucker）。
- **CV↔LB 較正の追加知見（2026-07-09）**: CV→LB オフセットは +0.28〜+0.34 で安定（yu4u 5fold×5seed 平均, 719389）。**CV<6 では public LB との相関が崩壊**（~50 well 抽出ノイズ支配）→ 終盤は CV を信じる（Tucker「trust your CV」, 723647）。長い well が MSE 分散を支配（Var∝Σn²/N², working note 716699）→ per-well RMSE 分布（median/p90/max）の併記を継続。
- **field-CV の運用（2026-07-09 v2 監査後）**: spatial 特徴を含む構成は **well-GroupKFold（楽観側）と field-GroupKFold + field-aware spatial 再計算（悲観側）の両方**を測る（v2 実測: 9.2308 ↔ 10.6886）。どちらが hidden test に近いかは LB 較正点（R1 提出）で判定し、以後その側を主 CV とする。

---

## 5. ソリューション・アーキテクチャ（層構造）

```
Layer0   baseline anchor           : carry_last = last_known TVT（目的は残差 U=TVT−anchor）
Layer1   GR registration trackers  : ①Particle Filter（rate状態・本命）②Beam DP ③NCC（弱特徴）→ 候補TVT曲線+不確実性
Layer1.5 candidate path bank ★NEW : PF変種（尤度スケール/particle-seed配分/GR denoise）× beam変種 × formation/spatial
                                     パス ＝ per-well 複数候補曲線の銀行（パスoracle 4.5–6ft が金の源泉）
Layer2   GBDT stack（(b)残差型）    : トラッカー=床、GBDT=残差補正。ES-holdout 正直プロトコル（v2実測 9.23）
Layer3   per-well router/hedge ★NEW: trust-gate＝高確信 well のみパス切替、曖昧 well は posterior-mean hedge。
                                     nested 検証必須（誤切替リスク実測済み: spatial一律切替は+6.4）
Layer4   postproc（縮小）           : warm-up damping / affine GR校正のみ（IRLS/SavGol平滑化系は実測棄却済み）
```
※ 旧 Layer4「leak override」は**削除**（2026-07-09: visible test はスコアリング時に置換と確定、forum 712037/722041）。

**設計思想**: 各層は前層を**恒久置換せず加算/補正**する。トラッカーは「候補と不確実性」を出し、GBDT が残差を補正し、
router が「どの well でどの候補パスをどれだけ信じるか」を判断する（＝物理と ML の分業）。
**金圏の伸びしろは Layer1.5×Layer3 に集中**（oracle: 手法レベル 7.77 → パスレベル 4.5–6）。Layer2 までは実装済みの足場。

---

## 6. 特徴量設計（Layer2 GBDT）

目的変数 `U = TVT − last_known_TVT`（または `TVT + Z − anchor`）。**train 限定 6 列 + `TVT` 列は特徴にしない**。
**推論時は `TVT` 列を一切読まない**（test に存在しない）。既知区間の TVT・傾き等は **`TVT_input` の非NaNプレフィックス**から算出する（`data.last_known_tvt` と同源）。この出所を守らないと train では通って test でクラッシュ/劣化する train-only 特徴を作り込む罠に陥る。

| 群 | 特徴例 |
|---|---|
| トラッカー出力 | NCC/beam/PF の候補 `Û`（delta 表現）、複数スケールの候補、トラッカー間 disagreement |
| GR | rolling mean/std（窓 11/51/151）、diff(1,10)、`GR − typewell_GR@候補TVT` 残差、affine 校正後 GR、`gr_missing` フラグ |
| 幾何/入射角 | `dZ/dMD, dX/dMD, dY/dMD`、`_pf_z`（縦速度回帰）、prediction-start からの距離・`md_from_ps`・`row_frac` |
| 既知セグメント統計 | anchor 前 known 区間の TVT-vs-MD 傾き（全体/直近200）、TVT-vs-Z 傾き、min/max/range/mean/std（robust_slope ガード） |
| 空間事前 | leave-self-out `FormationPlaneKNN`、`DenseANCCImputer`（`TVT=−Z+formation_depth+offset`） |
| 不確実性 | PF std、beam コスト、KNN 分散、tracker 合意度（§11 の deliverable も兼ねる） |

モデル: LightGBM ×3 + CatBoost ×2 を **positive Ridge meta**（alpha≈1.5–2）で融合（公開上位構成）。GPU 可（Notebook ≤9h）。

---

## 7. Layer1 レジストレーション（コア）

**⚠️ 実測済みの前提（2026-07-06）**: 近 anchor の局所勾配は median **0.01–0.04 ft/行**＝水平 GR はタイプウェル GR の **~25–100 倍ストレッチ**。この伸縮を無視した素朴な窓相関（逐次 NCC 単独）は**全 well 実測で 47.08 と大敗**（carry_last 15.91 未満のブレンドなし）。→ **トラッカーは伸縮率（rate）を状態に持つことが必須**。NCC は GBDT の弱特徴に降格。

1. **Particle Filter（本命・sequential Monte Carlo）** — state=`pos(=TVT or TVT+Z)`,**`rate`**。`pos += rate·dMD`、期待 GR=`interp(pos, tw_TVT, tw_GR)`、尤度 `exp(−½((GR−eg)/gs)²)`、128 seeds × scales{3,5,8,12}、resampling。rate 事前 = 既知区間 `TVT_input` 傾き＋軌跡傾斜（`_pf_z`）。公開系で単独 LB ~8.8 実績（sunnywu27 物理モデル）。Notebook 予算（§10）の主対象。
2. **Beam / DTW（Viterbi DP）** — 水平 GR をタイプウェル TVT グリッドへ整列。コスト = mismatch + move penalty、±2 index 移動（グリッド刻みをストレッチに合わせ sub-ft に）。多仮説 DP＝PF と誤差非相関を狙う。
3. **NCC（multi-scale）** — 単独では負（上記実測）。**GBDT の1特徴として弱く投入するに留める**（禁止はしない: 公開スタックでも1特徴扱い）。

各トラッカーは **候補 TVT 曲線 + dispersion（不確実性）** を返し、Layer2 の特徴になる。**GR 欠損区間は幾何/priors にフォールバック**、各トラッカーは try/except で必ず last_known にフォールバック（クラッシュ＝提出 Error）。

---

## 8. リーク / 物理妥当性

- **train∩test 重複リークは決着済み（2026-07-09）**: ローカル test 3 well は train の完全複製（well-ID/XY/GR/MD 全一致、eval TVT 全行復元可能 — `check_test_train_overlap.py` 実証）。**しかし visible test はスコアリング時に隠し ~200 well へ置換される**（forum 712037 で leak 説の提唱者本人が撤回・独立再現一致・722041 で上位勢確認。hidden well は `MD,X,Y,Z,GR,TVT_input` のみ・markers 剥離済み）。→ **override 系は本番採点で無価値のため設計から削除**。公開 LB と我々の CV の差はリークではなく手法の質。
- **物理妥当性（Working Note 審査軸3）**: 「metric 最適化」ではなく「地質的に妥当な解」を志向。`TVT+Z` フレーム・formation plane・
  ロバスト IRLS で wrong-branch を down-weight。過剰なブレンド/平滑は OOF 改善が physical plausibility を犠牲にしていないか点検。

---

## 9. Layer3 後処理
- **warm-up damping**: eval 先頭は anchor がほぼ正解（§3.4）＝ `1−exp(−md_since/τ)`（τ≈85）で anchor から滑らかに立ち上げる。
- **robust IRLS degree-4 polyfit**: `U vs 正規化MD` を IRLS で当て、`0.25·raw + 0.75·fit` で外れ枝を抑制。
- **Savitzky-Golay**（窓17, order3）で高周波ノイズ除去。
- **affine GR 校正**: 水平 GR とタイプウェル GR の gain/offset 合わせ。

---

## 10. 提出 Notebook 設計（Code competition）
- **自己完結**: `/kaggle/input/...` を読み `submission.csv` 出力。インターネット遮断・外部依存追加なし。
- **⚠️ 実測済みマウント（2026-07-06 probe）**: Kaggle 実行環境のデータは定石パスでなく **`/kaggle/input/competitions/rogii-wellbore-geology-prediction`** にマウント。root 検出は `/kaggle/input` 配下の `rglob("sample_submission.csv")` フォールバック必須（`notebooks/submission/probe_carry_last.py` 実装済み。**TODO: `src/rogii/data.py::data_root` にも同フォールバックを追加**）。
- **⚠️ 提出 API（実測）**: `kaggle competitions submit -k <kernel> -v <ver>` は 400（CLI が file_name を送らない）。**`KaggleApi.competition_submit_code(file_name="submission.csv", kernel=..., kernel_version=...)` 直呼びが確実**（probe ref 54377455 で実証）。
- **提出実績**: carry_last probe = **LB 15.883**（local 15.9099, Δ=−0.027）＝ LB は RMSE(ft)・提出経路は rerun 完走を実証済み。
- **事前学習モデルは Kaggle Dataset として添付**（学習はオフライン、Notebook は推論のみ＝時間節約。事前学習/公開外部データは規約上可）。
- **≤9h 予算**: 支配項は PF（128 seeds × 4 scales × ~5,000 行 × ~200 well）。Numba JIT・ベクトル化・per-well selector で重い構成を必要 well のみに。GBDT 推論は安い。**時間実測はローカル test が 3 well のみのため不可** → **train well を代理**に 1-well あたり PF/beam コストを計測し ~200 well へ外挿、余裕率込みで 9h 内を確認（submitter が担当）。
- **id 整合**: 出力 `id` 集合を `sample_submission.csv` と完全一致（`data.submission_ids`）。`tvt` に NaN/inf 無し。

---

## 11. 開発ロードマップ（フェーズ & 目標）

**（2026-07-09 全面改訂。P0–P4 は完了・実測済み、以降は R 系列に再編）**

| Phase | 内容 | 目標 pooled CV (ft) | 状態 |
|---|---|---|---|
| P0–P3 | scaffold / carry_last 15.91 / トラッカー（PF blend 11.06=LB 10.559）/ stack v1 10.65 → **stack v2 9.23** | — | ✅ 完了 |
| P4 | 平滑化系 postproc → **実測棄却**（−0.03 <ゲート）。slope特徴量回帰 → **実測棄却**（+0.05悪化） | — | ✅ 決着（負） |
| **R1** | **stack v2 提出＝較正点#3**（学習系の初LB転移テスト・spatial利得の hidden test 転移判定: LB~8.7–9.2なら転移/~10.2–10.7ならwell-CVアーティファクト） | LB実測待ち | 🚀 kernel push済み（2026-07-09） |
| **R2** | **候補パス bank（Layer1.5）**: PF変種（尤度スケール格子・particle↑seed↓・GR-rotation denoise・複数 typewell 整合）× beam変種 × formation/spatial パス。**per-well K候補曲線＋品質診断**を吐く | パス oracle を 7.77→6 前後へ拡張 | 未 |
| **R3** | **per-well router/hedge（Layer3）**: 候補間の選択分類器＋trust-gate（高確信のみ切替）＋曖昧 well は posterior-mean/blend hedge。**nested 検証必須** | oracle の部分刈り取りで **−1〜−2** | 未 |
| **R4** | **独立第二パイプライン**: simple sequence NN（forum 722236: CV8.9/LB8.39 実績帯）or host直伝特徴系（PS前 lateral GR 自己相関・直前50点）。公開クラスタ非依存・stack との誤差非相関を狙いブレンド | ブレンドで −0.3〜−0.8 | 未 |
| **R5** | **最終化**: runtime 実測ゲート（⚠️ 723856: commit は3ダミーwell・採点は~200 well で9h超過＝不採点リスク。per-well 時間×200+マージン必須）／robust（正直CV最良）+ aggressive（router/hedge 全部入り）の2本立て最終選択 | **≤5.5（金圏）** | 未 |
| （撤去） | ~~leak override~~（visible test は採点時に置換と確定・無価値） | — | ❌ 削除 |

各 Phase は **Issue 起票 → TDD 実装(implementer) → リークレビュー(reviewer) → GroupKFold pooled RMSE(experimenter) → 前ベスト比で採否 → PR**（CLAUDE.md 準拠）。

**Working Note の審査軸（§1）は開発台帳の指針**として常時記録: ①異なる approach の広さ/深さ（負の結果も分析付きで等価）②well 差の洞察 ③物理妥当性 ④各アイデアの寄与を定量 ⑤不確実性推定。→ `analysis/experiment_ledger.md` に各実験の「idea/motivation・CV・結論」を残す。

---

## 12. リポジトリ / モジュール設計（`src/rogii/`）

| モジュール | 責務 | 状態 |
|---|---|---|
| `data.py` | I/O・`eval_mask`・anchor・`submission_ids`・`data_root`（local/Kaggle 自動解決） | ✅ 有 |
| `baseline.py` | carry_last（残差の基準） | ✅ 有 |
| `typewell.py` | `TVT→GR` 補間・affine GR 校正 | 新設 |
| `trajectory.py` | 幾何・入射角（`dZ/dMD` 等）・`_pf_z` | 新設 |
| `registration/ncc.py` / `beam.py` / `particle.py` | トラッカー（候補TVT+不確実性） | 新設 |
| `features.py` | Layer2 特徴量（train 限定列を排除） | 新設 |
| `contacts.py` | `tvt_from_contacts` 再構成・leak マッチ | 新設 |
| `cv.py` | GroupKFold by well・pooled RMSE | 新設 |
| `stack.py` | LGBM/CatBoost + positive Ridge meta | 新設 |
| `postproc.py` | damping / IRLS polyfit / Savitzky-Golay | 新設 |
| `selector.py` | per-well 分類・構成ルーティング | 新設 |
| `predict.py` | 全層を統合し submission 生成 | 新設 |

scripts: `run_baseline_cv.py`(有) → `run_registration_cv.py` / `run_stack_cv.py` / `build_submission_notebook.py`。
tests: `tests/` を新設（predictor 契約=`eval_mask` 整列、リーク不使用、pooled RMSE、contacts 再構成、id 整合）。

---

## 13. リスク & 対策

| リスク | 対策 |
|---|---|
| CV↔LB 乖離・private 再スコア | pooled GroupKFold OOF を主軸、後処理は OOF 改善時のみ採用、public は参考 |
| eval ゾーンの GR 欠損(~30%) | GR 欠損時は幾何/priors フォールバック・`gr_missing` フラグ・tracker try/except |
| Notebook ≤9h 超過（PF が重い） | Numba JIT・per-well selector で重処理を限定・~200 well 時間実測・学習はオフライン+Dataset 添付 |
| spurious/wrong-branch 固着 | 多仮説(PF/beam)・物理事前(Z/plane)・robust IRLS・`TVT+Z` フレーム |
| leak override の private 逆効果 | ガード付き（prefix RMSE<1ft, MD 補間照合）で blend 以上を保証＝実質 no-op に設計 |
| metric 過適合 vs 物理妥当性 | Working Note 審査軸で自己点検、ledger に物理解釈を記録 |
| データ再配布違反 | `data/` を git-ignore、`kaggle competitions download` で再取得、提出は Kaggle 内のみ |

---

## 14. 金ギャップ調査所見（discussion 15 スレ・2026-07-06）

### レバー証拠ランキング（期待インパクト × 実装コスト）

| # | レバー | 期待インパクト | コスト | 根拠 |
|---|---|---|---|---|
| 1 | flat anchor 堅守＋傾き系は特徴量に降格＋GR マッチをガード | **高**（実測 19.0→14.3 の例） | 低 | 718670, 717445 |
| 2 | 正直な CV（well/field-group）＋ reseed 分散把握 | 高（誤採用防止） | 低 | 712037, 714514, 718670 |
| 3 | host 直伝: PS 前 lateral GR 自己相関 > typewell／直前50点／近傍 well 補間 | 中〜高 | 低 | **698825（HOST）**, 701041 |
| 4 | 公開クラスタ非依存の独立第二パイプライン（ρ≈0.89 の相関天井回避） | 高（推定） | 高 | 718670, 712037 |
| 5 | hard-well 検出＋worst-decile 限定 hedge（二峰性 datum） | 低〜中（worst-decile 5.8→5.1） | 低 | 711878, 712037 |
| 6 | test-time fine-tuning（LGB/CatBoost で −0.15〜−0.37 実測） | 中 | 低〜中 | 698002（**運営未公認**） |
| 7 | MTP top-K 候補パス | 中→低（本人「正味 +0.03ft」） | 中〜高 | 699853 |
| 8 | CNN+SDF／合成 forward-sim 事前学習 | 低（合成→実 転移完全失敗） | 高 | 699853, 702474 |
| 9 | DTW | 不明（スコア報告ゼロ、「機能しなかった」1件） | 中 | 697431 |
| 10 | Foundation model（PatchTST 等） | 投機的（ROGII 実測ゼロ） | 高 | 701041 |

### 確定的な定量所見
- **oracle 階段**（全773 well）: per-well const 9.07 / line 6.59 / smooth 3.00。金圏 ≤5.5 は「datum＋dip をほぼ回復」した水準。
- **hard-well**: 49% の well が二峰性コスト分布、**~15% が datum バイアス ≥ 半バンドル（~8ft）で二乗誤差の ~65%**、~12% は oracle 較正でもコインフリップ。**NCC マージンは正誤を予測しない（r=+0.054, p=0.30）**＝「confidence が高い＝正しい」は成り立たない。不確実性設計は margin 以外（多仮説の分散・prefix 整合）で。
- **titericz last-value baseline = LB 15.883**（ローカル carry_last 15.91 と一致）＝ LB は RMSE(ft) スケール。
- naive XGBoost（X,Y,Z,MD,GR）は **LB ~9.7 で頭打ち**（host 発言）。

### 警告（epistemics）
- フォーラム数値の大半は**自己申告・第三者検証なし**（一部スレは AI 生成分析の疑いをコミュニティ自身が指摘）。**設計判断は必ず自前データで再現してから**。
- 公開 notebook 群（7.8–9 台）は「CV リーク＋public LB 過学習」の両方を抱えるとの rank3 警告 → 公開スコアで自分の期待値を較正しない。
- TTT はホスト明示回答なし（GM の慣行容認のみ）→ 採用時はフォーラムでホストに直接確認する。

### 追補（2026-07-09 再調査: LB全量 + discussion 新スレ + working note 群）

**証拠ランキング v2（旧ランキングの更新。①②は自前実測で決着済み）**

| # | レバー | 期待 | 根拠 | 状態 |
|---|---|---|---|---|
| 1 | **候補パス bank + per-well 選択/hedge** | 高（パス oracle 4.5–6） | 721549 Ochir・自前 4-way oracle 7.77 | **主軸（R2/R3）** |
| 2 | heel affine GR 校正（定位 80% ≈ oracle 82%）+ GR-rotation denoise（→84%） | 済/小 | 712037 v7 実測 | fit_affine_gr 済・denoise 未 |
| 3 | 不確実性 = **trust-gate**（検知 AUC 0.69）。方向補正には使えない（R²=−0.16） | 中 | 716699 working note 群 | R3 に統合 |
| 4 | 独立第二パイプライン（simple NN CV8.9/LB8.39 帯） | 中 | 722236・718670 ρ天井 | R4 |
| 5 | bimodal hedge（posterior mean, worst-decile 5.8→5.1） | 小〜中 | 712037。⚠️ PF ベースでは粒子分布が既に hedge＝冗長の反論あり（716699 Villa）→ 要実測 | R3 で実測 |
| ― | ~~slope 特徴量回帰~~ | 0 | 自前実測 R²=−0.066・適用+0.05 悪化 | ❌ 棄却 |
| ― | ~~空間クラスタ/IDW の slope 転写~~ | 0 | 711308: 隣接 well の drift slope 相関 ≈0（offset 相関 0.98 のみ）→ LB~12.8 止まり | ❌ 棄却 |
| ― | ~~leak override~~ | 0 | 712037 撤回・722041 | ❌ 削除 |

**working note 群（716699）の構造的知見**（金設計の物理的裏付け）:
- 「The Wiggle Is Free, the Trend Is the Wall」: TVT=surface−Z 分解により高周波成分は既知 trajectory そのもの（モデル不要）。**残る per-well trend が "broadband-irreducible"＝evalゾーンの dip は GR に符号化されていない積分定数**
- identifiability lemma: 0.1° の dip 誤差が 5kft で ~8.7ft に積分。James-Stein shrink 係数 =0.000（offset 推定の SNR ゼロ）＝点推定でなく事後分布で扱え
- coherence spectrum: 識別帯域(5–32ft)のコヒーレンス 0.03–0.11 ＝判別信号が計測ノイズ床以下（物理限界であってモデリングの失敗ではない）
- detection AUC 0.69 vs direction R² −0.16 ＝「ずれの検知」と「補正方向」の非対称 → **gate はできる、補正はできない**
- leak-free ceiling ~8.3ft の主張（Subbotin note）vs Tucker per-well CV 5.0 の実在 → ceiling 論は保守的すぎる可能性。**Tucker の存在が「per-well 情報だけで 5ft 台」の実在証明**

**公開 notebook 解剖（2026-07-09、LB上位4本のソース精読）**: 公開 7.09–7.30 帯の実体は単一「sp45/fleongg」ファミリー（ravaghi artifacts 系譜、LGBM×3+CatBoost×2+Ridge メタ・well-GroupKFold は正しい）で、**honest 核の内部 CV は 9.21–9.7 ＝ 我々の stack v2 (9.23) と同水準**。LB との乖離は (a) train/test well-ID 重複 lookup（visible 3 well でのみ確実に発火・hidden での発火は未検証）(b) LB 反復チューニング（由来不明の閾値・well ID 3個の決め打ちルール・LB を見ながらの外部ブレンド重み選別）。**LB 首位 5.262 に対応する公開コードは存在しない**。→ 公開 7.1 帯は「honest に超えるべき壁」ではない（既に並走中）。
移植候補（優先順）: ①**尤度加重 PF-BMA**（128seed 独立実行→累積対数尤度 softmax(ll/scale) 統合。我々は 8seed 平均）②**visible-prefix cut backtest による per-well 候補選択**（cut_fracs 3点・min_gain/consistency/alpha-cap の confidence gate ＝ R3 の実装様式）③**beam 多構成格子**（(bs,mc,es,平滑化) の 7–14 構成）④Z 速度カップリング PF ⑤自己参照 multi-scale NCC（既知 prefix→eval の同一 well 内マッチング。棄却済み typewell-NCC とは別物）。**移植禁止**: overlap lookup／LB スコア参照ブレンド（リーク・水増し、方針非両立）。

**LB 地形 v2（2026-07-09 全量分析）**: 4,522 チーム。金圏 19位=6.328（安全圏 18位=6.137）。**7.10–7.30 に 917 チーム密集**（公開 kernel 系譜。Bronze 圏 452位=7.21 がこのクラスタ内部を通過＝公開 kernel をそのまま出すと Bronze 当落線上）。15.883 ちょうどに 185 チーム（carry_last 系）。8.863×26・14.336×25・9.150×17 の中規模スパイク＝別系統公開 notebook。先頭集団（<6.0）は 12 チームのみ。首位 5.262 は 3 日間不動。

**提出運用の追加罠（723856）**: commit 実行は 3 ダミー well・採点実行は ~200 well。**9h 超過は「Submission Scoring Error」でなく静かな不採点になりうる** → per-well 実測×200+マージンのゲートを R5 の必須項目に（stack v2 kernel は文書化済み: 推定 ~3.5h/9h）。private 採点は例外=即不採点 → well 単位 try/except + carry_last 縮退の徹底（721343）。

## 付録: 一次情報リンク（Discussion）
- 「Diagram of the problem」(697418, 161票) — 問題の正準図
- 「Formation Columns Are Derived from Typewell…」(708167) — formation 列＝typewell 由来
- 「Duplicate type wells for different horizontal wells」(698449) — leak/重複 typewell
- 「Private Test Update and Rescore」(707695) — private 再スコア
- 「besides regression, also dwt (time warping)!」(697431) — DTW ヒント
- 「Problem Breakdown」(708367) / 「How Geologists Interpret Wells」(698825) / 「Geological Formations (Texas)」(697406)
- **2026-07-09 追補分**: 「Fork the ruler, not the model」(712037, leak撤回+heel較正80%+oracle階段) / Working Note集約スレ(716699) / 公開kernel分解(718670, ρ≈0.89+seed noise) / 候補パスoracle 4.5–6(721549) / 上位勢ヒント+hidden well仕様(722041) / Tucker CV5.0証言(723647) / runtime罠(723856) / カットオフ予想(720701) / 空間slope転写の限界(711308) / CV↔LB較正(719389)
