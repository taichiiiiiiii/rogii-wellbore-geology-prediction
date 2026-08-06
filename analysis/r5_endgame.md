# R5: エンドゲーム準備（最終2枠の設計と運用チェックリスト）

作成 2026-07-11（R5 prep、実行は最終週 7/29〜8/5）。台帳キュー#9 の準備成果物。

## 締切（固定事実）
- **エントリー/チームマージ: 2026-07-29**
- **最終提出: 2026-08-05**（code comp、Notebook ≤9h CPU、インターネット遮断）
- 最終評価に使う提出は **2枠を明示選択**する（放置すると public ベスト自動選択 —
  public は50 well・再ランノイズ±0.09〜0.38 実測（topic 718670 teardown +
  我々の較正）なので **必ず CV 基準で手動選択**する）。

## 🔒 2026-07-17 最終2枠 構成で確定（goal「privateでも精度」の session内完了形）
**外部律速の明示**: privateスコアは競技仕様で08-05最終評価まで秘匿・公開LBもrerunキュー律速。∴private精度の*測定*はsession内で不可能。→ **privateスコアが無い時の正当な検証法＝(i)honesty forensics（リーク=dead/per-row bias無/固定重み軽微 を精読確認済 R35/R36）(ii)CV（773-well, 独立手法のみ可）(iii)脱相関（分散hedge）** で最終2枠を*構成で*確定する。
- **枠A（medal bet, private-robust化の主成果）= ayodeji-conservative**（ref 54761480, profile `contact_gated_anchor`＝prefix検証核 self_verified_anchor のみ・R32でコイン投げ実証のbimodal補正OFF・model-package補正OFF＝public過適合レバーを除去した最保守honest版）。フォーク一族で最も private-robust。
- **枠B（脱相関hedge）= pilkwang 7.074**（ref 54715010, honest確定＝physical+learned ensemble + prefix検証bounded補正・LB-probe hack無）。ayodejiの contact-gated stratigraphic alignment とは**別手法**＝private分散hedge。両者③依存は共有するが head/機構が脱相関。
- **却下: 独立③非依存kernel（R37完走評価）**＝天井~9.0でbronze 7.09に非到達、medal hedgeにならず（oracle6.79はwell内非定常で回収不能の蜃気楼）。∴最終2枠は両方フォーク一族から選ぶのが正（独立kernelを枠に入れると medal を捨てることになる）。sub-v8(9.014)は**枠外の緊急floor**として保持（両フォークが万一privateで壊滅した場合の最終保険、但しmedal圏外）。
- **採点着弾後の唯一の調整**: balanced(6.768)↔conservative の実LB差を見る。差が小さい（≲0.3）→conservative採用（private安全）が正当。差が大（balancedがLB-fit的に良い）→balancedのその上振れはpublic過適合疑いでconservative維持。**public LB数値そのもので枠を選ばない**（50well±0.4ノイズ＝bronze↔silver差0.046の10倍）。
- **本決定は採点を待たず構成で確定済み**＝goalの「改修」は成果物・枠選定とも完了、残るは外部採点の受動的到着のみ。

## 🟢 2026-07-16 R36 fork戦略確定＋private-robust改修（user goal「privateでも精度」）
**再現可能フォークのランキング（全依存公開・honest検証済）**: ayodeji **6.768(silver級)** > kersaoyagi 7.039 > iaztec 7.032(tail-guard) > kimdoong 7.061 > pilkwang 7.074(実行確認済)。municef1 6.985=❌`offset_krige_lookup.csv`欠落で再現不能。**上位ほど著者固有precomputedアーティファクト依存で再現不能になりがち**。
**リーク/固定値の実態（municef1/ayodeji精読）**: train-copy overlap+tvt_from_contacts+precomputed submission は全て`if wid in train_wells`/id一致でgate → **隠しtestで0発火(dead leak)**、`RUN_EXACT_MATCH_RECOVERY=False`。per-row bias offset無(reverse-bias型なし)。固定ブレンド重み(sp45 0.60等)はオフラインtuned=**軽微config-overfitリスク**(public 7.0→private ~7.1-7.4想定)。
**private-robust改修(goal)**: フォーク内部の再tuneはOOFの壁で不能(事前学習モデルがtrain全学習=クリーンOOF取れず)。honest改修=**(1)保守化profile: ayodeji `SUBMISSION_PROFILE=contact_gated_anchor`=prefix検証核のみ・model-package/bimodal補正OFF(R32でbimodal=コイン投げ実証)→ayodeji-conservative(ref 54761480)。balanced比 mean|diff| 0.27ft(可視)=補正影響小。(2)等重みアンサンブル=分散低減(重み調整なし=overfit無)。フォーク自身のprefix検証(self_verified_anchor)が各private wellの安全装置**。
**⚠️public LBチューニング厳禁**: public~50well・±0.4ノイズ、bronze↔silver差0.046=ノイズ1/10 → public最適化はprivate劣化。CV(773well)/prefix検証のみが合法private-proxy。
**最終2枠候補(2026-07-16提出、採点待ち)**: ayodeji-balanced(54760881,public最適)・**ayodeji-conservative(54761480,private-robust)**・AmgedAlfaqih(54758250)・iaztec-tailguard(54761607,脱相関)・pilkwang(54715010,7.074確定)。**方針=最終2枠は{最良honest fork or conservative, 脱相関fork}でprivate variance hedge。スコア着弾後にbalanced↔conservative実LB差で採否確定**。

## 🟢 2026-07-16 R35 訂正: bronzeは手中（pilkwang honest確定）＝メダル狙える
- **実LB(5008チーム)メダル閾値**: GOLD 6.137 / SILVER 7.047 / **BRONZE 7.093（rank500/top10%）**。
- **pilkwang 7.074 = rank~390(top7.8%) = public bronze圏**。精読で**honest確定**（physical+learned ensemble + prefix-verified bounded corrections + heel GR affine較正、LB-probe/frozen-tuned/per-well fix 無、①overlapは0発火で無影響）。∴**private ~7.0-7.3 = bronze圏**。**shakeupはpilkwangに有利**（LB-fittersが崩落→honest pilkwangのrank上昇）。
- **旧「pilkwang=②④ LB-fit崩落」は誤り撤回**（R35、user「多角的視点」指摘で自分のforensics矛盾に気づく）。
- **最終2枠 = {pilkwang 7.074（bronze bet, ref 54715010, 選択必須）, checkpoint-only 8.160（honest safety）or 第2 honest fork（private variance hedge）}**。
- **メダル不能・維持モードは撤回**。R34 neural registration killは有効（あれは5-6/尾用で不要）だが結論が誤りだった。

## 🔑 2026-07-16 更新（R32/R33）: robust floor が sub-v8 → checkpoint-only(8.160) に改善
- **R33 決定打**: `checkpoint-only`（koolbox base + ③ravaghi/fleongg 事前学習checkpoint、①overlap/②frozenCSV/④LB定数 全OFF）= **LB 8.160 = honest**（隠しpublic train非重複=正当held-out=privateでも保つ）。∴**③は本物のhonest転移改善＝我々base(9.014)より+0.85ft**。R29の「③=inflation」は誤り（③のhonest寄与とleakを混同）と確定。**エンドゲーム robust floor = checkpoint-only(8.160, runtime-safe ~10min)** が sub-v8 を置換。
- **R33 blend ❌却下**: 0.3×③ + 0.7×我々stack v2.6（ref 54750899）= **LB 8.502 > checkpoint-only 8.160**（我々GBDT stackと③GBDTが相関し多様性乏、koolbox tracker の方が③と脱相関）。→ **robust floor = checkpoint-only 8.160 で確定、我々stackは足さない**。
- **R32 確定（メダル圏外）**: eval は near-flat でない（誤差=極端ドリフト尾18%井戸=SSE46%）、9.0は我々手法(PF/beam)天井でhonest天井でない（40+チームhonest 5-6実在）、**全候補oracle尾10.84=我々primitiveはメダル(5-6)到達不能**。honest fork床~8.0-8.2 > bronze~7.17。メダルは根本primitive発明=数週間R&D(範囲外)。
- **hedge=pilkwang(7.074)**: ②④(+0.95)=LB-fit=private崩落見込み→~8へ減衰。aggressive枠。

## 現ベスト（履歴: robust 枠の旧既定値）
- **sub-v8 = stack v2.6**: 56+H3=59特徴・5-seed LGBM 平均。CV 9.1456 / LB 9.014。（R33で checkpoint-only 8.160 に floor 更新）
- 来歴監査 ✅（2026-07-11）: ローカル
  `notebooks/submission/kernel_stack_v2_blend/stack_v2_blend_submission.py`
  （commit `43e6649`）は Kaggle `taichiiiii/rogii-stack-v2-blend` 最新版と
  **バイト同一**。⚠️ 紛らわしい点2つ: (1) kernel 名は "blend" だが実体は
  v2.6 特徴注入（blend は不使用、pf_bma はライブラリコードとして残置・未呼出）
  (2) `kernel_stack_v2/` ディレクトリは**旧v2**（H3なし・単発LGBM）— 最終週に
  間違えてこちらを触らないこと。
- ランタイムゲート ✅（実測、sub-v8 の実ラン log）: 可視ラン（776 well）
  tracker/spatial 6119s（7.885s/well・失敗0）+ 学習/推論 ≈ 550s = **総計 6667s
  (111分)**。隠し再実行（773+~200=973 well）推定 ≈ **137分 ≈ 2.3h → 9h 予算に
  約4倍の余裕**。
  - ⚠️**誤警報防止（2026-07-15 daily watch, topic 725205）**: フォーラムで「提出Notebookが5–7h」報告が複数出るが、**これは我々には非該当**。Tucker Arrants(rank5)が明示＝「visible test/ は3 well見本、提出時は隠しtest ~200 well に差し替え再実行される（仕様）」。OP の爆発は *per-well 限界コストが重いパイプライン特有*（3well→20分が200wellで数十h）。我々の sub-v8 は可視111分の**92%(6119/6667s)が773 train well の固定 tracker/spatial コスト**で test 側限界コストは小（同rate 7.885s/well を隠し~200well に外挿しても+約26分）。∴他者の5-7h報告を見ても137分見積りは揺るがない。ホスト側の系統的スコアリング遅延も未確認。
- 安全機構 ✅: 全トラッカー never-raise（flat-anchor fallback）+ per-well
  carry_last fallback + リーク境界（TVT/Geology/formation列は特徴に不使用）を
  ファイル冒頭 docstring が宣言、実装済み。

## 最終2枠の設計
| 枠 | 内容 | 選択基準 |
|---|---|---|
| **robust** | その時点の **5-seed CV ベスト**（現状 sub-v8。R24 等が二重ゲート
  =CV−0.10＋転移実績/probe を通せば後継に差し替え） | CV。public LB は±0.4
  ノイズ帯として無視 |
| **aggressive = 公開fork hedge（2026-07-15 確定）** | **pilkwang fork**
  （ref 54715010, kernel `taichiiiii/rogii-pilkwang-hedge` v2 = T4 完走・valid
  14151）。2026-07-15 の徹底調査で **自作 medal shot は物理の壁で閉**（GR形状選択=
  eval near-flat/level生成=field-CV崩壊/self-ref=oracle 10.35/δ_w=null/外部最新手法
  も全滅）、honest 変種も marginal（R24 −0.034・16seed stack CV −0.05〜−0.09 は
  CV-only・**dup-overlay は private で no-op=ケースA**）と確定。→ aggressive 枠を
  「**private 崩落モデルが外れた場合**」への**無コストヘッジ**に転換。pilkwang=robust
  forkable な公開7.x（LB-probe hack無・guardrail健全・sub-v8 と脱相関）。private 崩落
  なら ~9.2（sub-v8 が max で拾う）/崩れなければ 7.x=メダル。⚠️未解明の2ftギャップ
  （honest床9.2 vs 観測7.0-7.3=事前学習checkpointの実力不明）がこの option を正当化 |
  2枠 max の good side を取る。**bank/pf_bma 系は LB 毒で絶対除外**。旧 R24/16seed
  は marginal で不採用。fork は再取得可（`kaggle kernels pull pilkwang/...`）、
  最終週に v2 を再実行して最新化 |

## 最終週チェックリスト（7/29 開始）
1. [ ] develop の全採用変更が submission kernel に反映されているか diff 監査
   （特に R24 以降の採用分。kernel は `kernel_stack_v2_blend/` が正）
2. [ ] ローカル `ROGII_SMOKE=20` スモーク → Kaggle push → 可視ラン完走確認
   （111分基準から大幅増なら原因調査。9h の 50% を超えたら構成を疑う）
3. [ ] submission.csv の id 集合が sample_submission と完全一致（既存 assert）
4. [ ] 提出 → LB 記録（較正点として。選択判断には使わない）
5. [ ] **最終2枠を手動選択**（robust + aggressive）→ 選択スクリーンショット
6. [ ] sub-vN タグ + ledger 記録（kernel version ↔ commit SHA 対応を明記）

## 既知リスク
- 並行セッションと同一リポジトリで作業中 — kernel ファイル編集は台帳で宣言
  してから（WIP 衝突防止）。
- shake-up は我々に有利のはず（正直 CV↔LB 較正、フォーク一族の public 過適合）
  だが、CV<6 帯で相関崩壊の報告（rank 2/3）は我々の帯（9.1）には未適用。
- 提出枠は 5/日 — 最終日にまとめて依存しない（8/4 までに両枠を検証済みに）。
