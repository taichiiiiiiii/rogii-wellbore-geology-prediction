# ROGII 提出パイプライン詳細アーキテクチャ（2026-07-27 時点）

我々の提出ライン（slot1 = GS1.45 系 6.411）の内部構造・系譜・knob 地図・実測特性の完全な記録。
一次データは experiment_ledger.md R47-R69。

> **⚠️ この文書は 2026-07-27 時点の記録であり、一部はその後の実験で反証されています。**
> 特に §4 の「複製宝くじ（ビット同一の再提出を繰り返して良いドローを選ぶ）が有利」および
> 「同一ランで public と private がシードを共有する」という記述は **R112 で明確に否定されました**
> （ρ_seed の点推定 0、宝くじの価値は上界でも 0.02 ft 未満）。さらに R137 で、可視出力が
> ビット同一の2本が隠し再実行で 6.470 / 6.500 に分かれることが実測され、public スコアは
> 単なるドローであることが確定しています。§3 と §6 の「現在の構成」も当時のもので、
> 最終提出は別系統（physics フォーク）に移りました。訂正後の立場はリポジトリの README を
> 参照してください。

## 1. 系譜（lineage）

```
blacklions/rogii-contact-gated-stratigraphic-alignment   ← 家系の源流（R54で判明）
  └─ johnjanson/hahaha-nondet-agi V1                     公開 6.594（自己採点）
       └─ iaztec/top-pf-config-branch-conservative      = janson V1 + GS×1.3 + 読み取り専用EDA
            └─ 我々の fork 系列:
                 rogii-janson-v1-fork-probe   GS1.0   {6.671, 6.700}
                 rogii-iaztec-gs13-fork-probe GS1.3   {6.551, 6.464}
                 rogii-gs145-probe            GS1.45  {6.411, 6.514} ★主力
                 rogii-gs16-probe             GS1.6   6.711（過剰・死蔵）
                 rogii-gsboth13-probe         site2   7.020（有害・死蔵）
```

通称「Contact-Gated Stratigraphic Alignment」家系。45 code cells / 単一ノートブック /
internet OFF / docker pin `python@sha256:37c64f7d...`。

## 2. 入力

- **競技データ**: train 773 wells（`<wid8>__horizontal_well.csv` = MD/Z/GR/TVT_input/TVT +
  formation列, `<wid8>__typewell.csv` = TVT/GR/Geology）。可視 test = 3 dummy（train複製）、
  採点は隠し ~200 wells の再実行（public=固定~52 / private=残り、同一runの部分集合）。
- **マウント dataset（7本）**:
  | dataset | 役割 |
  |---|---|
  | phongnguyn23021656/koolbox-offline | 学習基盤ライブラリ（Trainer/CV枠組み） |
  | ravaghi/wellbore-geology-prediction-artifacts | 事前計算特徴テーブル(7.4GB)+事前学習 LGBM×3/CatBoost×2 |
  | fleongg/rogii-claude-models-pub | learned branch 用事前学習 booster（lgb0/1/2.pkl + features.json） |
  | pilkwang/rogii-model-package | drift 補正モデル（gate 付き最終補正） |
  | nina2025/rogii-03, thbdh5765/…, needless090/… | **dead mount**（コードから参照ゼロ、Opusレビューで確認済） |

## 3. 処理ステージ（実行順）

### Stage 1: Ridge/PF アンカー
- **residual ensemble**: ravaghi の事前学習 LGBM×3 + CatBoost×2（195特徴・OOF保持）→ Ridge stack。
  事前学習が見つからない場合は from-scratch fallback（CVハーネスで実証済みの経路）。
- **粒子フィルタ（PF）**: 128-seed 尤度重み付き PF（numba JIT, cell46）。状態=(pos, rate)、
  観測尤度 = exp(−0.5·((GR−G_tw(T))/gs)²)。**この gs が我々の主 knob**。
- **SP45 selector**: well形状ビン（eval行数閾値 4840 / Z-span 閾値）ごとに PF 設定
  （scale 3/5/8/12 × beam/hold 変種）を選ぶ selector 経路。ここにも独立した gs 計算（cell20）
  があるが**増幅してはならない**（site2×1.3 = 7.020 で実証）。
- アンカー合成: T_A = 0.30·T_ridge + 0.70·T_selector。

### Stage 2: SP45 射影（層序レベル平滑化）
- U = T + Z（含意層序レベル）を正規化MD空間のロバスト多項式で平滑化:
  T_proj = (1−λ_p)·T_A + λ_p·(Û − Z), λ_p = 0.75。

### Stage 3: learned branch とのブレンド
- fleongg boosters（無ければ from-scratch train_stack）による学習軌道 T_learned。
- T_blend = w_s·T_proj + (1−w_s)·T_learned, **w_s = SP45_BLEND_WEIGHT = 0.60**
  （profile 由来。w0.58/0.55/0.62 の実機ラダーを 07-27 に測定中）。

### Stage 4: guarded same-well contact override
- test well ID が train に存在する場合のみ、formation contact（ANCC/EGFDU等）から
  TVT を再構成し、可視 prefix RMSE < 1.0ft の自己検証を通過した well だけ上書き。
- **隠し test では 0 発火**（隠し well は新規 ID。R39 の dup-overlay 3提出で実証済み）。
  可視3井では 3/3 発火（ログで確認済）＝可視スコアの見かけを作る層。

### Stage 5: visible-prefix 校正（VP、この家系の核心）
- 既知 prefix の末尾を隠して（cut_fracs 0.50/0.65/0.75）、候補生成器群を holdout RMSE で
  ランキング → profile 出力を最終軌道に採用（visible_prefix_final_selection='profile'）。
- 内部パラメータ: cal_seeds=24, final_seeds=48, particles=350, α multiplier 1.30
  （heel GR affine 較正 (α_w, β_w) を含む）。profile='balanced'。
- CVハーネス（劣化構成）では conservative が −0.28 良かったが、実機の swarm 合意は
  balanced（未検証仮説として貯蔵、R59 の歯止め参照）。

### Stage 6: model-package 補正
- pilkwang drift モデルの gated 補正。gate: max_weight=0.00425, scale=6.0,
  disagreement p95 > 25ft で自動無効。移動量は小さい（gate が絞る）。

### Stage 7: PF seed-branch midpoint hedge
- PF の branch 分岐が二峰の well に対し、posterior 中点へのシフト。
  strength=0.60, min mass=0.25, separation 4-40ft, **cap = _BH_CAP = 2.00ft**
  （cap2.5 は muelsyse111 監査で悪化 6.667、blacklions a23 の 6.593 主張はノイズと判定済）。

### 出力監査
- sample_submission と id/順序/有限性を突合、SHA-256 記録。id 形式 = `<wid8>_<row>`
  （row = 全行CSVの0-based index、TVT_input が NaN の行のみ）。

## 4. GS knob の機構（我々の主発見）

```
gs = clip( nanstd(GR_known − typewell_GR_at_known), 10, 60 ) × M
```
- gs は PF 観測尤度の温度。M↑ → 尤度平坦化 → branch 選択の過信を抑制。
- **用量反応曲線（ペア平均, 隠しセット実測）**:
  M=1.0: 6.685 → M=1.3: 6.508 → **M=1.45: 6.463（最適）** → M=1.6: 6.711（追跡崩壊）。
  非対称パラボラ（右側急峻）。
- **適用は cell46（PF workhorse）のみ**。cell20（selector側）の増幅は selector 判断を
  壊し +0.5 悪化（site2 probe 7.020）。
- **副作用: nondet 分散の拡大**。バイト同一 rerun の複製帯:
  M=1.0 で 0.029 → M=1.3 で 0.087 → M=1.45 で 0.103。
  → 複製宝くじ（byte-identical 再提出の best-of-N）が knob 探索と同等以上の EV。
  同一 rerun 内で public/private が seed を共有するため、良い public draw の選択は
  private でも +EV。

## 5. 実測特性まとめ

| 特性 | 値 | 出所 |
|---|---|---|
| 可視3井の決定論性 | 完全（RMS diff 0.0000, 源流とも一致） | R55/R57 |
| 可視出力の knob 不感性 | GS/w 変更でも可視出力は不変（Stage4/5 が支配） | R57/R61 |
| 隠し採点遅延 | ~21h | R55 |
| 偏差相関 | janson↔pilkwang 0.952（同族）/ checkpoint 0.115（唯一の脱相関） | R50/R53 |
| 家系の信頼できる実測 | 我々の probe 群 + muelsyse111 監査のみ（ref無し主張はハッシュ矛盾多発） | R53/R68 |

## 6. 現在の構成（提出中の主力）

- kernel: `taichiiiii/rogii-gs145-probe` v1 = iaztec fork + cell46 gs×1.45（単一行diff）
- profile: `vp_balanced_modelpkg_005`（VP balanced + modelpkg gate 0.00425/scale6, w_s=0.60, cap2.00）
- 最終2枠（暫定, 08-04 明示select）: slot1 = ref54990438 (6.411) /
  slot2 = janson run1 6.671 へ見直し中（final2_redesign.md 07-27 追記参照）
- 進行中ラダー: 複製宝くじ（5draw {6.411, 6.447, 6.460, 6.514, 6.545}, mean 6.475, σ0.05, run6採点待ち）
  + SP45 w ラダー: **w0.58 = 6.595 = 有害（R70）→ w下げ棄却・w0.55 死蔵**、次候補 w0.62（ビルド済）
