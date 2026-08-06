# 最終2枠 再設計（R52後・事前登録版）

作成: 2026-07-25（janson probe ref 54971967 の採点前に決定規則を固定し、
スコアを見てから理由を作る過度な楽観バイアス（3回記録済み）を構造的に排除する）。

## 前提の変化（R52, 07-25 実測）

- 公開LB 5671チーム。メダル線: **gold 5.899 (top22) / silver 6.454 (top284) / bronze 6.495 (top568)**。
  旧「銀7.047/銅7.093」は陳腐化。[6.3, 6.594) に780チーム密集。
- 我々（Latte）= rank 1231 @ 7.003。bronze まで −0.51ft / 663順位。
- 公開フォーク天井 = johnjanson `hahaha-nondet-agi` V1（自己採点 6.594）。
  muelsyse111 の監査付きフォーク5変種 = 6.638–6.667 実測（全て源流より悪い＝knob探索は枯れ気味、
  家族の再実行分散は小さい ~0.07ft）。
- **採点機構の重要な帰結**: 採点rerunは隠し~200 wellsを一度に計算し、public(~52固定)/private(残り)は
  同一runの部分集合。**publicスコアが付いた提出は private スコアも確定済み**（後日開示されるだけ）。
  ∴「privateでrerunが落ちる」シナリオは存在しない。slot2の役割は
  **public→private の井戸抽選ギャップ + public-fit knobの逆風** のヘッジのみ。
- private ギャップの規模感: souldrive #728477 = public 52-well で SE(RMSE)~0.78ft、
  private ~150-well 側は縮む（~0.4ft級）。提出間の相対ギャップは相関 ρ でさらに縮む
  （σ_rel ≈ σ·√(2(1−ρ))）。

## 目的関数の転換

R46（bronze帯の維持 = robust床）→ **期待privateランクの最大化**。
メダルは「密集帯のpublic-fit勢がprivateで沈む × 我々が線に近い」の抽選のみ＝約束不能・約束しない。

## 候補プールと相関構造（可視3-well, 14151行, 07-25実測）

| 候補 | public | 補正RMS(vs janson sp45基底) | janson補正との相関 |
|---|---|---|---|
| janson-fork | ref 54971967 採点待ち（期待 6.59–6.70） | 2.820 ft | 1.0 |
| canqiang | 7.003 | 2.430 ft | 0.952 |
| pilkwang | 7.074 | 2.344 ft | 0.952 |
| checkpoint | 8.160 | 4.721 ft | **0.115** |

- canqiang↔pilkwang = **1.0000**（R50の双子確定を再現）。
- janson↔{canqiang, pilkwang} = **0.952**、最終予測の対差 1.11–1.38 ft
  （双子の0.41ftより大）＝**同族だが補正層に~9%の固有分散**（PF seed-branch midpoint hedge +
  balanced visible-prefix profile が固有部分）。
- checkpoint = 唯一の脱相関候補（0.02–0.11）。ただし public 1.5ft ハンデ。

## 事前登録の決定規則（janson スコア開示前に固定・変更禁止）

1. **janson ≤ 6.85** → 最終2枠 = **{janson-fork, pilkwang 7.074}**。
   - slot1: 最良の再現可能 generalizer。
   - slot2 が checkpoint でなく pilkwang な理由:
     (a) checkpoint が private で janson を逆転するには >1.5ft の相対スイング
     （ρ=0.11 でも σ_rel≈0.67ft ⇒ >2σ ≈ 数%）＝ほぼ死枠。
     (b) pilkwang は同族の**最薄補正**版＝「janson固有knobのpublic-fit逆風」という
     最も確率の高い失敗モードを、家族共通baseを保ったままカバー（逆転条件 ~0.5ft）。
     (c) 無メダル前提の期待ランクも pilkwang(~1348) ≫ checkpoint(~2600)。
2. **6.85 < janson**（家族実測帯 6.59–6.67 から外れ＝双子クラスタ帯以下の価値しかない）
   → **R46 現状維持 {pilkwang 7.074, checkpoint 8.160}**。
   janson はプール入りのみ。（canqiang 7.003 が pilkwang より上に見えるのは
   ノイズ帯内の双子＝R50 の結論を維持し置換しない。）
3. スコアが付かない（採点エラー）場合のみ翌日1回だけ再試行。それでも失敗なら規則2。
4. ~~knob探索はしない~~ → **user改訂（07-25「提出したものをさらに高精度化したい」）**:
   高精度化トラックを追加する。ただし**盲目的な public-LB probe は引き続き禁止**
   （muelsyse111 の5変種全滅・780チームと同土俵・public-fit knobはprivate毒 R33 — この根拠は不変）。
   許可されるのは次の2経路のみ:
   (a) **公開実測で 6.594 を下回る自己完結・recompute型 source の忠実フォーク**
       （実測の出所と再現機構を検証してから）。
   (b) **773 train well の paired CV ハーネスでゲートを通った knob 変種**
       （固定 well 集合で baseline V1 と paired 比較、well単位RMSE、CV改善が
       ノイズ帯を超えた変種のみ提出）。52-well public でのチューニングは何であれ不可。

   **ハーネスの制約（Opusレビュー 07-25, 初版FAIL→修正で確定した規律）**:
   - 事前学習資産（ravaghi boosters / fleongg models / pilkwang model-package）は
     全773 wellで学習済み＝held-out井に対しin-sample → **HARNESS_MODEでは全て無効化し
     from-scratch経路で走らせる**。∴ **modelpkg系プロファイルのA/BはCVでは測定不能＝計画から除外**。
   - パイプライン定数自体が773-well tunedのため、**ハーネスの絶対値はLBと比較禁止・
     用途は固定well集合でのpaired deltaのみ**（K=10のpooled値単体では快晴/リークを判別できない、
     per-well分布の裾の有無で判定）。
   - CV測定可能なknob: GS×1.3（PF内部）、SP45重み、BH cap、VP profile
     （conservative/balanced/aggressive — prefix情報のみ使用でhonest）。
5. 高精度化トラックで janson run より良い public スコアの提出が生まれた場合、
   規則1の slot1 をそれに置換（slot2=pilkwang は不変）。事前登録の趣旨（スコアを見てから
   理由を作らない）は各変種の**提出前に CV 判定を記録**することで維持する。
6. 08-04 に Kaggle UI で**明示 select**（デフォルト latest-2 は誤り）。

## slot2 見直し（07-27 追記・GSラダー成立後の再分析、最終決定は08-04）

GSラダー成立により slot2 の最適解が変わった可能性がある。ヘッジ階層（public / 補正の薄さ）:
GS1.45系 6.411 < GS1.0-janson 6.671 < pilkwang 7.074（< checkpoint 8.160 = 唯一の脱相関）。

- slot2 の役割 = slot1（GS1.45系）の private 失敗モードの保険。
- **失敗モードA「gs増幅がpublic部分集合の癖だった」→ janson run1 (ref54971967, 6.671, GS1.0)** が
  pilkwang より 0.4ft 良い位置で同じ保険を提供（gs増幅なし・同familyベース）。
- 失敗モードB「contact-gated family全体の崩落」→ pilkwang も同family（相関0.95）のため保険にならず、
  真の保険は checkpoint 8.160 だが1.7ftのコストで期待順位を大きく損なう。
- ∴ **暫定推奨: slot2 = janson run1 6.671（pilkwang 7.074 から置換）**。
  モードBは受容リスクとする（silver帯の期待値最大化を優先）。
  08-04 の選択時に全採点データ（宝くじ結果・w系knob）で最終確認する。

## slot2 再々検討（07-28・R73 後。07-27 の janson 推奨を撤回）

**前提（08-04 に UI で要確認）**: Kaggle の最終順位は**選択した提出のうち private が良い方**で決まる。
∴ 最終2枠の目的関数は **E[max(P1, P2)]** の最大化であり、「保険」ではなく「2枚の抽選券」。

07-27 は slot2 = janson run1 (6.671, GS1.0) を推した。根拠は失敗モードA「gs 増幅は public 部分集合の癖」
だったが、**R73 でこのモードは概ね潰れた**（GS1.45 vs GS1.0 は z=−3.64、隠し ~52 井での 4σ 級の差。
public/private は同一 ~200 井のランダム分割なので系統差は無くサンプリング誤差のみ）。
失敗モードB（family 全体の崩落）は janson も同 family（ρ=0.952）で**元々ヘッジになっていない**。
∴ **janson を slot2 に置く理由は消え、期待値で 0.19 ft 損なだけ**。

残る実リスクは **winner's curse（seed 選択が public 固有の運を拾う分は private に転移しない）**。
これに対する正しいヘッジは**同一 config の独立した2ドロー目**である。

- **推奨（更新）: slot1 = GS1.45 の public 最良ドロー / slot2 = GS1.45 の public 次点ドロー**。
- 理由: 良い config からの2ドローは、「良い config 1本 + 期待値が 0.19 悪い config 1本」を支配する。
  public→private 相関が強ければ選択が効き、弱ければ2本とも同分布からの独立サンプルとなり
  E[max] は単独より上がる。**どちらに転んでも混ぜるより良い**。
- 唯一この推奨が崩れるのは「family 全体が private で崩落」だが、その真のヘッジは checkpoint 8.160
  のみで 1.7 ft のコスト。silver 帯の期待値最大化を優先し、引き続き**受容リスク**とする。

## 選択操作のリスク評価（07-29 に公式ルールで確認・重要）

**規約 §3.18.c の原文**: *"A 'Final Submission' is the Submission selected by the user, **or automatically selected by Kaggle in the event not selected by the user**, that is/are used for final placement."*
＝**未選択でも Kaggle が自動選択する**（Kaggle の標準動作は public スコア最良のものを選ぶ）。
§2.2.b: *"You may select up to two (2) Final Submissions for judging."*

### ⚠️ 07-30 訂正: 上の「belt-and-braces に格下げ」は **R102 で無効になった。明示選択は必須**

この節は 07-29 時点、**slot2 が「GS1.45 の2本目のドロー（6.447）」だった前提**で書かれていた。
その前提なら推奨＝public 最良の2件で、自動選択と一致していた。

**しかし R102 で slot2 を「iaztec GS1.3 の 6.464」に変更した**（同一 config の2ドローは private 水準を
共有するので2回目の試行にならないため）。**6.464 は public 全体で4番目**であり、
**自動選択（public 最良2件 = 6.411 と 6.447、どちらも GS1.45）では絶対に到達できない**。

∴ **「08-04 に UI で明示選択」は必須作業**。これを飛ばすと R100/R102/R107 の脱相関の論拠が丸ごと消える。

- **書き込みは Kaggle UI 専用**（`kagglesdk/competitions/services/competition_api_service.py` の全 RPC を
  列挙して確認済み。選択用エンドポイントは存在しない）。
- **★ただし読み出しは API で可能**: `competition_submissions(group=SUBMISSION_GROUP_SELECTED)`。
  **∴ クリックが実際に保存されたかを機械的に検証できる**。
  `uv run python scripts/submission_status.py --final-check` が
  `CONFIRMED: the selection on Kaggle matches the plan exactly` を出し **exit 0** になるまで、
  クリックは成立していないものとして扱う（現状は 0件選択＝exit 1）。
- **単一障害点**: このエージェントから到達できないブラウザ上の2つのチェックボックス。
  自動化は不可能で、上の read-back だけが唯一のガード。

## 正直な期待値

- 規則1成立でも public rank ~850–950 = **メダル圏外**。これは柱1のupgrade（1231→~900）+
  private抽選で線に最接近する施策であり、メダルの約束ではない。
- janson の knob は public フィードバックで選ばれており、private では +0.1〜0.3ft の
  逆風があり得る。それでも 6.7–6.9 < pilkwang 7.07 が維持される見込み（これが規則1の余裕幅）。

## 追記（07-25, 両runのスコア開示前）: 2-run 時の適用規則

同一アーティファクト（kernel v1）を2回提出した（run1 = ref 54971967, run2 = ref 54972786）。
目的: (a) nondet 一族の再実行ノイズ帯の自前実測（souldrive #728477 の方法論・R44 外部実測 0.089–0.381 の同型）、
(b) 単発の不運 seed のヘッジ — 同一 rerun 内では public/private が同じ seed 実現を共有するため、
**良い方の public を選ぶことは private でも期待値プラス**。
- 規則1の適用: min(run1, run2) ≤ 6.85 → slot1 = **public が良い方の run**、slot2 = pilkwang 7.074。
- 規則2の適用: min(run1, run2) > 6.85 → R46 維持。
- |run1 − run2| を本コンペの nondet 帯の実測値として台帳に記録する。
- これは knob 探索ではない（バイト同一・パラメータ変更ゼロ）。追加の変種提出はしない。

## 実行ログ

- 07-25: janson V1 忠実フォーク（V1仕様をセル単位検証: profile modelpkg_005 =
  gate 0.00425/scale6, BH cap 2.00 — muelsyse111 の監査コピーとの差分は当該2knobのみ）。
  kernel `taichiiiii/rogii-janson-v1-fork-probe` v1 COMPLETE →
  submission.csv 検証（14151行/有限/mean 11904.2=家族フィンガープリント一致）→
  **提出 ref 54971967**（採点待ち、重量パイプラインは最大~16h）。
- 07-25: 偏差相関分析（上表）実施。スクリプト: scratchpad watch0725（一次データは本文の表）。
- 07-25: run2（バイト同一・seed-band測定）提出 **ref 54972786**。両run採点PENDING、30分poll×2稼働。
- 07-25: **採点前の忠実性検証PASS**: (a) kernelログ走査=フォールバック/欠損データセット警告ゼロ、
  全補正層発火確認（GUARDED override 3/3・visible-prefix校正 cal_seeds24/final_seeds48/particles350・
  gold contact override・PF seed-branch hedge applied=1）、可視3-well実行~13分。
  (b) **我々のfork可視出力 = janson本人の可視出力と完全一致**（14151行 RMS diff 0.0000 / max|d| 0.0000 /
  移動行0%）＝可視スケールでは決定論的・完全忠実再現。"nondet" はhidden規模でのみ発現の可能性、
  run2の|run1−run2|で実測される。
