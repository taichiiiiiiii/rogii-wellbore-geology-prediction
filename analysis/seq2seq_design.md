# Full-well seq2seq (prefix-in-context) — 設計 (2026-07-20, 1-cycle disciplined attempt)

> **⚠️ 2026-07-20 事前Opusレビューで中核根拠を撤回・再フレーム（R48）**。当初の根拠「full-well seq2seqは
> R37の prefix自己較正信号(oracle 6.79の源泉)をネイティブに持つ」は**R37の完走結論と矛盾**していた:
> R37は忠実な prefix-holdout proxy を既に実施し純選択利得 **−0.07ft**（=prefixはeval尾へ転移しない、
> oracle 6.79はeval真値を要する選択バイアスの蜃気楼, R32 tail-oracle 10.84と同型）。∴この試みは
> **「oracle 6.79回収ベット」ではない**。正直な位置づけ＝**低prior の "アーキ・クラス再現ベット"**（Tucker rank8 /
> James Day rank6 が "1well=1系列" で CV4.5-5.8＝これだけが正の外部証拠。上位優位はR32/R35で「非公開の
> 根本的に優れた登録primitive」と結論済みで、小型modelがfold0で再発見する期待値は低い）。**staged killで安く反証する**のが目的。

**目的（再定義）**: 学習型 full-well 登録モデルに **prefix文脈(ch6/ch7)** を足すと R34 の val14 プラトーを破れるかを、
**最小差分の段階ablationで安く白黒つける**（破れなければ即kill）。下限は pilkwang 7.074 が常に確保＝downside=GPU 1サイクル。

## H-3 実測（系列長, 2026-07-20 / 773 wells）
`L_total p50=6576 / p90=8056 / max=12141`、`L_known p50=1703`、`L_eval p50=4840 / max=10052`、**eval=well全体の74%**。
→ 全well双方向self-attentionは O(L²) 43M(p50)〜147M(max)/well＝**T4非現実的**。**確定: TCN/1D-CNN(dilated, O(L)) backbone**
＋短いtypewell(数百行)への **cross-attn(L_well×L_tw, cheap)**。=R34と同一backbone（Stage Aで流用）。
※ eval=74%＝ch6(prefix drift)はeval行で全区間0が中央値4840行続く＝レビューHIGH-1の悲観を実測が補強。

## R34（失敗, val14）との決定的差分
| | R34 (kill) | 本設計 |
|---|---|---|
| query系列 | **eval領域のみ** | **full-well（known prefix + eval）** |
| known prefixのTVT | GR affine較正のスカラー要約のみ | **TVT_input ドリフト軌跡を毎行チャネル入力** |
| 自己較正信号(R37, 6.79 oracleの源泉) | **無し** | **構造的にネイティブ** |
| typewell | key/value cross-attn | 同左（維持） |

R37: 「per-well oracle 6.79 を回収する信号＝各候補の"自well既知prefix再構成誤差"」。full-well seq2seqは
prefix(既知TVT)とeval尾を1系列で見るため、この信号を明示的に学習できる。これが val14 と CV<5 の 9ft ギャップの仮説。

## サンプル / 入力チャネル（全て test に存在＝leak-safe）
- **サンプル = 1 well**、行index順（MD単調）。
- 毎行チャネル（horizontal_well 由来, 全て test にある列のみ）:
  1. `GR` （**global z-scored**; 正規化統計は train全体）
  2. `dGR` = GRの隣接差分（局所勾配）
  3. `MD_rel` = MD − MD[0]
  4. `Z_rel` = Z − Z[0]
  5. `X_rel`, `Y_rel` = X−X[0], Y−Y[0]（横位置ドリフト）
  6. **`TVT_prefix_rel`** = (TVT_input − anchor) where known, **else 0**  ← ★中核（prefix drift軌跡）
  7. **`is_known`** = 1 (TVT_input既知) / 0 (eval)  ← ★maskチャネル
  - anchor = `last_known_tvt`（eval直前の既知TVT_input）
- **typewell 参照**（cross-attention の key/value, ragged per well）: `tw_gr`(global z-scored), `tw_tvt_rel`(TVT−anchor)。
  typewell の Geology は **test に無い→使わない**。

### 🚫 leak 厳禁（reviewer 必須検査）
- eval領域の `TVT` / `TVT_input`(NaN) を**入力にしない**（ch6はeval行では0固定, ch7 mask=0）。
- formation列(`ANCC/…/BUDA`)・typewell `Geology`・未来情報を使わない。
- 正規化統計・fold割当は **train のみ**から。val well の真値は CV 測定時のみ使用。

## ターゲット / 損失（★C-3修正: 補助損失を in-model prefix-holdout に）
- target = `TVT − anchor`（heel anchorからのドリフト, R34と同一でstack比較可能）。
- **主損失** = eval行の MSE。
- **補助損失（修正版）** = 学習時に **known行の一部をランダムマスク**（そのブロックだけ ch6=0 / is_known=0）し、
  マスクした行の `(TVT_input−anchor)` を GR＋近傍＋typewell から**再構成**させる MSE（重み~0.3）。
  ⚠️旧案（known行のch6をそのまま補助target）は**ch6が答えを与える恒等コピー＝no-op**でボツ（C-3）。マスクして初めて
  「prefix自己較正」を学習させたと言える。マスク率~30%, ブロック連続マスク（尾側regime模擬）。
- 出力: 毎行 drift 予測 → anchor加算 → TVT。**採点は eval行のみ**の pooled RMSE。

## アーキテクチャ（小型, Kaggle T4）
- well系列 Transformer encoder（bidirectional; eval領域GRは合法なので双方向可）: d_model 192, layers 4–6, heads 4, FFN 4×。
- typewell encoder（同d_model, 2 layers）→ well tokens が **cross-attention**。
- head: per-row linear → drift。
- 位置: MD_rel/Z_rel を連続位置として入力（学習位置埋め込み併用可）。
- パラメータ ~1–3M。系列長 = well行数（数百〜千, padding+mask）。

## CV / kill 基準（★H-2/H-4修正: 尾RMSE一次指標 + staged ablation）
- **well-GroupKFold(5, seed=42)** を stack OOF から再利用（比較可能性）。**cycle1 = fold0 のみ**（高速）。
- val well は自身の TVT_input NaN 尾を eval mask として持つ（organizerのvisible-prefix/hidden-suffix maskを再現）。
- **skip集合を stack OOF と一致**（空eval / anchor無し well をskip, cv.py契約）。padding/known行を採点に混ぜない（M-1）。
- 指標（**尾RMSEを一次**に, H-2）:
  - **`tail_rmse` = eval後半50%行の pooled RMSE**（誤差集中領域, honest手法が負ける本丸。R34の敗因=tail23）。
  - `pooled_rmse`（eval全行）は副指標（anchor近傍の易行に薄まる）。
  - fold0 を**最低2 seed分割**で回し点推定でなく帯で判定（±0.4 churn対策）。
- **★段階ablation（H-4, 変数隔離）**:
  - **Stage A = R34-exact 再現**（eval-only query, 既存 scripts/train_registration.py 流用）→ **~14 が出れば pipeline sanity OK**。
  - **Stage B = +ch6/ch7 のみ追加**（typewellはR34とバイト同一固定 + in-model prefix-holdout aux）→ fold0。
- **kill 基準（規律・非対称を排除）**:
  - **即停止(仮説死)** = Stage B の tail_rmse が R34帯(~14) から**有意に改善せず**（例: >12）。typewellチューニング等は不要、そこで終了。
  - **続行(cycle2検討)** = tail_rmse が**明瞭に改善**（例: <10 かつ pooled が stack 9.14 を明確に下回る）。ただし
    **単独では"有望"止まり**（CV改善→LB反転の履歴多数, sub-v4/v6/v7）。cycle2で 5-fold + 提出でLB転移を必ず確認。
  - デッドゾーン(改善はあるが<10未満届かず)= 帯とtailの内訳を見て判断、安易に青信号にしない。
- 下限は pilkwang 7.074 が常に確保 → downside = GPU 1サイクル分のみ。

## 実行（学習全面Kaggle）
1. `scripts/build_seq2seq_dataset.py`: full-well leak-safe 系列を npz 化。R34 builderの leak-safe 正規化/anchor/typewell を流用、**ch6/7 追加・eval-only→full-well query 化**。
   - **★leakガード（C-1/C-2 必須, テストで強制）**: (a)正規化統計・fold割当は **fold-train well のみ**（val well を統計から除外, assert）。(b)`assert np.all(ch6[eval_mask]==0)` 全well。(c)builderが `h["TVT"]` に触れるのは **target構築の1箇所のみ**（grep/テストで保証）。(d)formation/Geology列を一切読まない。
   - ローカルは smoke(数well)のみ（3.8GB RAM, 実fit禁止）。
2. `src/rogii/seq2seq_model.py`: **TCN/1D-CNN(dilated) backbone + typewell cross-attn**（O(L), H-3対応）。padding mask を is_known と独立に持ち、loss mask=(eval_mask AND not padding)（M-2）。typewell 退化(<50行/低std)は learnable null-token フォールバック（M-3）。
3. Kaggle T4 kernel で dataset build→**Stage A→Stage B** fold0 学習→**tail_rmse + pooled** 報告。
4. reviewer で leak 検査（C-1/C-2/C-3, M-1/M-2）→ push。
5. 測定 → kill 判定。有望なら cycle2（5-fold + 提出でLB転移確認 + typewell強化）。


## 参考（外部一次情報, 07-19 sweep）
- Tucker Arrants(rank8): 単一modelでCV<5(近傍well無), 各サンプル=complete well(非tabular), 5fold best4.5/worst5.3, random group-by-well, 外れ25本は特別扱い無。
- James Day(rank6): 5.77 pooled 5-fold。
- #727149: sub-6は end-to-end学習 or engineered alignment(DTW/PF/HMM)をrefineか、が論点。per-well drift は合法特徴から**回帰では**学習不能(field-grouped OOF R²<0) → 系列文脈(prefix)が鍵。
- CV-LB: CV>6で相関堅い(LB=CV+0.3程度)、CV<5.8-6で崩れる(台帳13) → sub-5 CVはLB~6着地の可能性(それでも銀7.047/銅7.093射程)。
