# R6-c: chunk-level DP traceback 候補帯 — 設計ノート

> 表記: **事実**＝コード/notebookに書いてある内容の引用。**推測**＝筆者（本ノート作成者）の解釈・仮説。
> 対象 notebook: `mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore (Kaggle)drift-targeting-ncc-tree-based-rogii-wellbore.ipynb`（全34セル、json parse で全文抽出済み）。

## 0. 最重要の先出し結論

**事実**: この notebook には「chunk-level DP traceback候補帯」という機構は存在しない。

- `grep -i "chunk"` の全ヒットは Savitzky-Golay 後処理（`for wid in ...: chunk = blend_oof[mask]`）のみで、DP/Viterbi とは無関係。
- `grep -i "band\|candidate band\|traceback"` はゼロヒット。
- ledger の主張数値「9.99→8.76 oracle / 9.47→9.24 smoke」は notebook 全文（cell出力のprint文字列含む）を文字列検索してもゼロヒット。
- notebook にある Viterbi 機構は「4種のシグマ設定を持つ単純な forward-beam Viterbi（1 well = 1グリッド = 1回のDP、chunk分割なし、単一最適パスのみ出力、候補帯/traceback帯という概念なし）」で、**その標準単体RMSEは4設定全てが null（15.91ft）より悪い**（18.97〜36.05ft）。

**推測**: `analysis/experiment_ledger.md` R4-b 行（2026-07-10）の原文を読むと、"chunk-level DP traceback候補帯" は **topic 699853「MTP deep CNN」（93メッセージ）+ topic 699289 のフォーラム調査**から得た発見であり、この notebook から得たものではない（ledger 原文: 「hengck23のnotebook 2本のコード実体まで突合」「副産物=非深層の高ROI発見3つ」という書き方から、フォーラムのテキスト議論内の主張であって、mitchgansemer notebook を指してはいない）。本タスクの前提（"機構の実体は mitchgansemer notebook にあると推定される"）は**「Viterbi」という単語が9回出現する notebook を grep で見つけた結果の誤帰属である可能性が高い**。実際の一次情報源（フォーラム topic 699853 の該当メッセージ）は本タスクでは未取得・未検証。

→ 以降 §1 は「mitchgansemer notebook に実在する Viterbi 機構」を忠実に記述する（これ自体は R6-c の一設計材料として無駄ではない）。§5 で「chunk-DP候補帯」の数値そのものの信頼性を別途評価する。

---

## §1. mitchgansemer notebook の Viterbi 機構（完全記述）

### 1.1 アクセス制約（事実）

- `MODEL_DIR`（`models/` または Kaggle 上 `rogii-wellbore-models` dataset）から `utils.py` を `import utils` している（cell 1: `sys.path.insert(0, str(MODEL_DIR) ...)`）。
- `viterbi_tvt` / `particle_filter_tvt` / `multi_scale_ncc` / `FormationPlaneKNN` / `rmse` 等、**実装の中身は全てこの外部 `utils.py` にあり、ダウンロードした .ipynb には含まれていない**（`kernel-metadata.json` の `dataset_sources: [""]` — 空文字列で、紐付くデータセットの参照すら失われている）。
- したがって以下は **「呼び出し側コード＋markdown説明文」から復元できる範囲**の記述であり、コスト関数の正確な数式・シグマ定数値・traceback実装の細部は「不明（コード非公開）」として扱う。

### 1.2 状態空間・呼び出しシグネチャ（事実、cell 28）

```python
TW_OFFSETS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], dtype=np.float32)  # 別機能(tw_diff特徴)用、Viterbiの状態空間ではない
BEAM_RADIUS = 80.0   # search_radius に渡す定数（cell 1 定義）

for name, emit_s, move_s in BEAM_VARIANTS:               # 4 variants: tight / med1 / med2 / loose（med2は標準RMSE表に出現しないが signal-divergence特徴で使用=cell845）
    path = viterbi_tvt(
        gr_beam, tw_tvt, tw_gr,
        last_tvt, tvt_step_per_row,
        emit_sigma=emit_s, move_sigma=move_s,
        search_radius=BEAM_RADIUS, grid_step=1.0,
    )
    beam_results[f'{name}_delta'] = (path - last_tvt).astype(np.float32)
```

- **状態空間（事実＋推測）**: `search_radius=80.0`, `grid_step=1.0` → 呼び出し規約から見て、`last_tvt`（アンカー = `data.last_known_tvt` 相当）を中心に **±80ft を 1.0ft 刻みでグリッド化（推測: 約160状態）**。当方 `beam.py` の「±120ft を 0.2ft 刻み（約1200状態）」より**グリッドが6倍粗く、探索半径も2/3狭い**。
- **観測列（事実）**: `gr_beam` は評価ゾーンの GR のみから作られる — `eval_z['GR']` を `interpolate(limit_direction='both') → fillna(typewell平均) → rolling(5,center=True).mean()` した系列（cell 786-791）。**タイプウェルGRと既知ゾーンの外側情報は使っていない、TVT/Geology は一切参照しない**（§4で再確認）。
- **1 well = 1回のグリッド構築 + 1回のDP呼び出し**（chunk分割の形跡なし）。`BEAM_VARIANTS` の4回ループは「同じグリッド構造で emit_sigma/move_sigma だけ変えた再実行」であり、chunkに分けての反復ではない。

### 1.3 コスト関数（推測、markdown記述からの復元）

cell 12（markdown）: 「an HMM over typewell GR. At each step the most likely TVT position is found by forward-beam Viterbi with varying emission and transition sigmas.」

- **推測**: 標準的な HMM Viterbi 風に読むなら、観測コスト（emission）は `-(gr_row - tw_gr[state])^2 / (2*emit_sigma^2)` 型のガウス対数尤度、遷移コスト（transition）は `-(move)^2 / (2*move_sigma^2)` 型（当方 `beam.py` は L1 型 `move_penalty*|d|` だが、こちらは "sigma" と命名されており **ガウス系＝L2型の可能性が高い**、確証はない）。
- **"forward-beam Viterbi" という表現の解釈（推測、不確実性明記）**: 名前は `viterbi_tvt`（Viterbiを名乗る）だが、markdown文中の "at each step the most likely TVT position is found" という言い回しは、**各行で argmax を単純に取っていく「forward filtering（各ステップのMAP点推定、backward tracebackなし）」を指している可能性**と、**通常のexact Viterbi（forward DP＋最終行からのbacktrace）**の両方に読める。コード非公開のためどちらか確定できない。
- **4 variants の意味（事実）**: `BEAM_VARIANTS = [(name, emit_sigma, move_sigma), ...]` の4組。具体的な数値（tight/med1/med2/looseの emit_sigma, move_sigma）は `utils.py` 内定数であり、**この notebook には数値そのものは出現しない**（変数名だけ import され、値は不明）。

### 1.4 単体性能（事実、cell 12markdown表）

| Estimator | 単体 TVT RMSE |
|---|---|
| Particle filter (ANCC) | 13.38 ft |
| Null (last_anchor_tvt) | **15.91 ft** |
| Viterbi beam (tight) | 18.97 ft |
| Particle filter (TVT rate) | 29.70 ft |
| Viterbi beam (med1) | 30.73 ft |
| Viterbi beam (loose) | 36.05 ft |

**事実**: **4 Viterbi variant のうち表に出た3つ全てが null（15.91）より悪い**。med2 は表に無いが cell 845 の signal-divergence特徴でのみ使用される（単体RMSE非開示）。

**推測**: `y_tvt`（cell 5: `gt['TVT'].values`、773 well 全体を concat した1次元配列）に対して `rmse()` を呼んでいる形跡から、**この RMSE は pooled RMSE（当方の指標と同じ設計）である可能性が高い** — ただし `rmse()` 自体の実装も `utils.py` 内にあり非公開なので断定はできない（§5でこの点を再評価）。

### 1.5 前処理（事実）

- GR補完: `impute_gr_with_typewell`（既知ゾーン）＋ eval zone は `tw_interp(tvt_extrap[null_mask])`（線形外挿位置のタイプウェルGRで欠損埋め）。
- Viterbi専用の追加平滑化: `rolling(5, center=True).mean()`（5行centered窓）。
- GRアフィン較正: このViterbi呼び出し自体には見当たらない（`gr_beam` は生GR由来、affine calibration は別の特徴群 `visible_gr_shift_fit` でのみ使用されている模様）。**当方 `beam.py` は逆に `typewell.fit_affine_gr` による較正済みGRをDPに渡している** — この差は§2で比較。
- typewell側: `tw.sort_values('TVT').dropna(subset=['GR'])` でソート済み・NaN除去済みの `(tw_tvt, tw_gr)` を渡す。

### 1.6 claimed metric の集計単位（推測、断定不可）

- 表の "OOF RMSE" 列（cell 0 の R2〜R11表）は fold-averaged pooled RMSE の可能性が高い（`GroupKFold(5)` セクションが明示的にあり、"OOF predictions use fold models trained on the other 4 folds" と明記=cell19markdown）。
- ただし cell 12 の Viterbi/PF単体表は**モデルOOF予測ではなく「素の推定器をそのまま予測として使った場合のRMSE」**（tree modelを介さない）。これが**pooled**か**per-well平均**かは、`rmse()` の実装が非公開のため確認不能。CLAUDE.mdの鉄則（「pooled RMSE以外は本番指標と乖離しうる」）に照らし、**この単体表の数値はそのまま信用しない（推測: おそらくpooledだが未確認）**。

---

## §2. 当方 beam.py との差分表

| 項目 | mitchgansemer `viterbi_tvt`（推測含む） | 当方 `src/rogii/registration/beam.py::track_beam` |
|---|---|---|
| DPの厳密性 | 不明（forward filteringかexact Viterbiか非公開コードにより確定不可） | **事実: exact full-grid forward Viterbi + 明示的backtrace**（`_forward_viterbi`、`backptr` 配列で厳密再構成、コード内 `# an exact forward Viterbi dynamic program, not an approximate/truncated beam search`） |
| 状態空間 | ±80ft @ 1.0ft刻み（推測、約160状態） | ±120ft @ 0.2ft刻み（事実、約1000〜2000状態） — 6倍細かく、探索範囲も1.5倍広い |
| 遷移コスト形 | 不明（"sigma"命名からガウス/L2の可能性、推測） | **事実: L1型** `move_penalty * |d|`（`max_move_per_row` でd を±3ステップに制限） |
| 観測コスト形 | 不明（emit_sigma=ガウス対数尤度の可能性） | **事実**: `(gr_eval[i] - grid_gr[s])**2 / mismatch_scale`（L2、有効パラメータは`move_penalty*mismatch_scale`の積のみ — 実測で確認済み） |
| GR較正 | Viterbi呼び出し自体には見当たらず（生GRベース、推測） | **事実**: `typewell.fit_affine_gr`で較正済みGRを使用（既知ゾーンのみでfit、leak-safe） |
| アンサンブル | 4 variant（emit/move sigma違い）を**別特徴として**GBDTに供給（`beam_med2_delta`等）、出力blendはしていない | **事実**: `track_beam_multi`で3設定（move_penalty 16/20/25 @ mismatch_scale=150）を**重み付き平均**して1本のBeamResultに集約（`beam_grid.py`はさらにGR較正on/off・平滑化半径を振った候補バンクを別途生成） |
| 候補帯/不確実性の出力 | **無し**（単一パスのみ返す、`path`のみ） | **有り**: `BeamResult.margin`（各行で「グリッド全体でのbest-vs-2nd-bestコスト差」を既に返す。`_forward_viterbi`内`part = np.partition(cost_i, 1)`） |
| chunk分割 | **無し**（1 well = 1回のDP） | **無し**（同じく1 well = 1回のDP、行単位で全評価ゾーンを一括処理） |
| 単体性能（同一指標なら比較可） | 15.91(null)超え失敗、best=18.97 | **事実**: `beam_raw` pooled RMSE **15.80**（nullを0.11ft下回る、2026-07-06実測） — mitchgansemerの最良variant(18.97)より優れる |
| 既にstack特徴として使われているか | 不明（`build_features`のこの後の工程で `beam_med2_delta`等が163特徴の一部として使われている＝事実） | **事実**: `beam_d`（アンカーからのdelta）と`beam_margin`が`stack_features.py::build_tracker_features`で既にGBDTに供給済み（stack v2以降、LB 9.022で転移確認済み） |

**結論（推測込み）**: mitchgansemerのViterbi機構は当方beam.pyの**劣位互換**に近い（粗いグリッド・狭い探索半径・非較正GR・単体性能で劣る）。この notebook から新たに輸入すべき「chunk-level DP」という独自機構は実在しない。当方が既に持つ`beam.py`/`beam_grid.py`/`stack_features.py`の枠組みの方が進んでいる。

---

## §3. 我々への実装案

前提の再確認: 「chunk-level DP候補帯」の一次情報源（フォーラムtopic 699853）は本タスクでは未読。以下は**その実体が確認できないまま**、ledgerに書かれた発想（"chunk単位でDPを解き、near-optimalな候補帯＝レジストレーション曖昧度を出す"）を、**当方の既存資産の上に自前で具体化する場合の設計案**。R4-bの「chunk-DPはGR依存＝同じ壁に注意」という警句を前提に、コスト・リスクの低い順に並べる。

### (b) 【優先・低リスク】per-row 候補帯特徴（near-optimal band width）— 先に着手すべき

**現状**: `beam_margin`（best-vs-2nd-bestのコスト差、スカラー1個/行）は**既に実装・既にstack特徴として本番投入済み**（`stack_features.py`の`beam_margin`列、LB 9.022の中に含まれる）。task説明の「best と 2nd の margin」は**再発明ではなく既存機能の確認**という結果になった。

**新規性がある部分**: marginは「コスト差」というスカラーであり、「TVTの物理的な帯の幅（ft）」ではない。真に新しいのは、**各行で `cost[i,s] <= min_cost[i] + threshold` を満たす状態集合のTVTレンジ（max−min、ft単位）＝ near-optimal band の物理幅**を特徴化すること。これは「コストがどれだけ急峻か」ではなく「その行でどれだけ広いTVT範囲が『ほぼ同じくらいもっともらしいか』」という、marginとは異なる情報を持つ（急峻な単峰でも2位との差が小さいケース＝margin小・band幅小、対して緩やかな多峰プラトーでmargin小だがband幅は大、を区別できる）。

**具体実装（案）**: `src/rogii/registration/beam.py::_forward_viterbi`は既に各行で`cost_i`（グリッド全状態のコスト配列）をローカル変数として保持している（`margin`計算に使っている箇所）。同じループ内で、閾値`τ`（例: `mismatch_scale`の何倍か、またはmargin自体の分布から较正）に対し

```python
band_mask = cost_i <= (cost_i.min() + tau)
band_width_ft = grid_tvt[band_mask].max() - grid_tvt[band_mask].min()  # 単峰なら小さい、多峰プラトーなら大きい
```

を追加コストほぼゼロ（既に計算済みの`cost_i`を再利用）で計算可能。`BeamResult`に`band_width`フィールドを追加 → `stack_features.py`に`beam_band_width`列として追加 → 既存の「G1/G2アブレーション」と同じ手順（R8参照: 新特徴群を既存59特徴に足してseed42+複数seedでゲート判定、単独ではなく複数seed必須のR15教訓を厳守）で採否判定。

**リスク評価**: この特徴は`beam.py`の出力（既にLB転移実証済み）から導出される**追加のスカラー1列**であり、新しい「候補パス」を生成するわけではない。R7（特徴注入で成功: pooled -0.25、LB 9.014）と同じ「特徴注入」形態であり、R6-b/R7-9で確立した「候補は集約すると常時有効、bank由来の新規候補ほどLB転移が壊れやすい」という教訓に照らしても、**既存の転移実証済みDP出力から追加スカラーを1個増やすだけ**なのでリスクは低い。

### (a) 【後回し・高リスク】chunk単位のDP経路候補（新規パス）

**内容（案）**: 評価ゾーンを固定長チャンク（例: 100〜300行）に分割し、各チャンクの終端で「そのチャンク内の最良コストのTVT」に再アンカーしてDPを打ち切り・再起動する（＝チャンク境界でのみtracebackする、チャンク内は当方既存の`_forward_viterbi`をそのまま再利用可能）。狙いは「長い評価ゾーンで単一グローバル最適パスが序盤の誤りを引きずる」問題を、チャンクごとの再アンカーで軽減すること。

**リスク（重要）**: これは**新しい候補パス**であり、bank（`path_bank_beam.npz`等）に新規追加してGram再解 → 固定重みblendまたは特徴注入 → CV改善確認、という既存フローに乗せる必要がある。しかし本プロジェクトの実測記録は「**新規bank候補は繰り返しLB転移に失敗している**」ことを強く示している:
  - probe#2（pf_bma単体）: train 12.18 → **LB 14.64**（+2.5ギャップ）
  - probe（pf_bma_v2単体）: train 11.80 → **LB 14.41**（+2.6ギャップ）
  - sub-v6（新候補を特徴注入）: CV 8.98 → **LB 10.01**（+1.03、stack v2の9.022より悪化）
  - sub-v7（5-seed平均+新候補）: CV 8.86 → **LB 9.66**（+0.80、依然悪化）
  - sub-v4（新候補を出力blend）: CV 9.11 → **LB 9.456**（+0.35、悪化）

  加えて、**mitchgansemer自身の実測でも4種のViterbi単体は全てnullより悪い**（§1.4）。これは「GRベースの新規経路は、当方の既存beam.py/particle.pyより強くなる根拠が薄い」ことの独立傍証。

**推奨手順**: (a)に着手するなら、**必ず二重ゲート**（① well-GroupKFold CVでの改善確認 → ② 小規模でも良いのでLB probeでの転移確認）を経てから本採用する。R6-a・probe#2・sub-v6/v7の教訓（「CVで勝ってもLBで負ける新規bank候補」パターン）を踏まえ、**(b)のゲート結果が出るまで(a)の着手は保留**が妥当。

### 実装順序の推奨

1. `beam.py::_forward_viterbi`に`band_width`計算を追加（既存コード凍結方針があるため、`beam_grid.py`同様の別モジュール拡張、または`BeamResult`への非破壊的フィールド追加で対応。既存関数群は「凍結」と明記されているため、変更方針はレビュー時に要確認）。
2. `stack_features.py`に`beam_band_width`列を追加、複数seedでゲート判定（閾値-0.10ft、R15/R20の教訓通り単一seedで判断しない）。
3. (2)が通れば、(a)のchunk-DP新候補は「フォーラムtopic 699853の原文を実際に読んでから」検討する（現状は伝聞の伝聞であり、実装コストに見合う根拠が確認できていない）。

---

## §4. リーク面の判定

**判定: notebookに可視のViterbi機構自体（cell 28の`viterbi_tvt`呼び出し）にはリーク無し（事実ベースで確認可能な範囲では）。**

- 入力: `gr_beam`（評価ゾーン自身のGRのみ、TVT/Geology不使用）／ `last_tvt`, `tvt_step_per_row`（既知ゾーンから計算、legal）／ `tw_tvt, tw_gr`（タイプウェルのTVT・GR、CLAUDE.mdで明示的に許可: 「タイプウェルのTVTは train/test 両方にあるので使ってよい」）。
- 評価ゾーンの`TVT`/`TVT_input`列、タイプウェルの`Geology`列（test側に存在しないと確定済み、R21分析）は、このViterbi呼び出しには一切登場しない。

**部分監査である点の明記**: `viterbi_tvt`の中身（`utils.py`）は非公開なので、「内部で何かのグローバル変数・キャッシュ経由でリークしていないか」までは**確認不能**（性善説での判定）。当方の`beam.py`は自前実装かつ全コード可視・テスト済みなので、この点で優位。

**同一notebook内の他機能で気になる点（Viterbi本体ではないが§4の趣旨に関連するため記録）**:
- `formation_plane_knn` / `row_knn` / `dense_imputer`は**train限定の formation深度列（ANCC等）を「他の学習用well」からKNNで借りてくる**方式（`exclude=well_id if loo else None`で自分自身を除外）。これは当方の`spatial.py`と同型の「他well由来の物理事前分布」であり、**自分の禁止列を直接読むリークではない**が、これらのクラス自体の実装（除外ロジックが本当に正しいか）は非公開のため完全検証はできない。
- notebook自身の「What didn't work」セクション（cell 26）が **自己申告で重要な警告**をしている: 「Coordinate-overlap post-processing」は「their visible test wellsがtraining wellsと100%座標重複しているために見かけ上効いているだけで、実際のhidden test（重複率ほぼ0%）には汎化しない」と明記。これは当方が2026-07-09に独立実測した「visible test 3/3 well が train の完全複製」という発見と完全に一致する（ledger該当行参照）。**Kaggle公開notebookの「効いた」報告は、visible test wellの特殊性（train重複）に依存した見かけ上の改善を含みうる**、という一般的な信頼性への警鐘として、chunk-DP候補帯の評価にも適用すべき教訓。

---

## §5. claimed score の信頼性評価

### 5.1 notebook内のViterbi単体表（18.97〜36.05ft）について

- **相対的に信頼できる**: 4 variant全てがnull（15.91）より悪いと自己申告しており、著者に「良く見せる」インセンティブがあるなら書かない類の数値。オラクル混入・チェリーピッキングの兆候は薄い。
- ただし集計方式（pooled vs well平均）は`rmse()`実装非公開のため**未確認**（§1.6）。

### 5.2 「chunk-level DP traceback候補帯（9.99→8.76 oracle / 9.47→9.24 smoke）」について

**この数値は本タスクで検証不能（一次資料未取得）。以下は状況証拠からの推測評価:**

- **oracle→smoke の落差が大きい（重要な赤旗パターン）**: oracle改善 `9.99→8.76`（**-1.23ft**）に対し、smoke（実運用に近い小規模実測とみられる）改善は `9.47→9.24`（**-0.23ft**）と、**oracleの5分の1以下に縮小**している。本プロジェクトで独立に確認済みの同型パターン:
  - dz-cumsumオフセット候補: oracle 7.7 → 合法実現は**carryの劣化コピー**（17.96、改善どころか悪化）。ledgerが明記: 「噂の7.7はオラクル」。
  - R2 router bank: 18候補のfull oracle **5.9588**（金圏6.303を下回る）に対し、学習型選択の実現値は**9.11〜9.38**（router 3連敗）。
  - これらは全て「oracle改善は本物だが、それを合法的に・GRのみから・学習ベースで再現しようとすると崩壊する」という、本コンペのGR情報量の壁（フォーラムでも独立に確認された「GRはshuffle検定を通らない」）に起因する構造的パターン。
  - "9.99→8.76"のoracleも同じ壁にぶつかっている可能性が高い（**推測**）。"9.47→9.24"のsmoke値は相対的にはマシだが、-0.23ftという幅は「well数の少ないsmokeテストのノイズ」（当プロジェクトの経験則: 単一seed/小標本での-0.03〜-0.09ft級の"改善"は複数seed/フルセットで消える例が頻発— R15, R17, R20）と区別がつかない小ささである。
- **出典不明**: 数値の well数・fold構成・pooled/per-well平均のどちらか・oracle計算の具体的な定義（何をoracleとして選んでいるのか＝chunk単位で正解に最も近い候補を後知恵で選ぶ、という意味なら定義上リークを含む比較であり「上限」の意味しか持たない）が一切わからない。ledger自身「claimed 7.7、ただしwell平均指標・ANCCリーク性要検証」という注記を並列候補①（ANCC区分線形）には付けているが、②（chunk-DP）には同種の注記がない＝**まだ検証されていない未検証claim**という位置づけがR4-b時点で既に示唆されている。

### 5.3 総合判定

- **"9.99→8.76"（oracle）: 信頼度 低**。本コンペで繰り返し確認された「oracle改善は合法再現不可能」パターンと整合的で、鵜呑みにすべきでない。
- **"9.47→9.24"（smoke）: 信頼度 中低・要再検証**。仮に本物でも改善幅が小さく、single-runノイズと区別できない。母数（何well、何fold）が不明な限り採否判断の根拠にできない。
- **最重要**: これらの数値の一次情報源（フォーラムtopic 699853）を本タスクでは読んでいない。R6-c着手前に、まずこの一次情報源を実際に取得・精読し、chunk-DPの具体的な機構定義（何をchunkと呼んでいるか、traceback帯の定義、母数）を確認することが必須の前提条件である。それなしに「9.47→9.24」を目標値・ゲート基準として使うべきではない。
