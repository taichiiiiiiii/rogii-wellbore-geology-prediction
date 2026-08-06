# private-robustness 分析（public追従からの転換）※Opusレビュー反映版

## 0. 前提の明確化（最重要）
**private には直接フィットできない**——private LB はフィードバックゼロの隠しsplit（それが定義）。「private にフィット」= **public LB を最適化基準にせず、honest な汎化を優先し public-fit（過学習）を避ける** の意。上位者証言（Deotte「大shake-up」/Tucker「public rank無意味・CVを信じろ」）と整合。

## 1. private-robustness の測定手段＝fork剥落（ただし分解能に限界）
公開kernelを fork して隠し~200 wellで再実行すると、元提出にkeyしていた LB-fit 層（overlap/gold-visible-prefix/LB-probe定数）が非転移で剥がれる。**剥落量 = 元public − fork実測 は public-fit の間接推定**（測定でなく機構推論, R44）。

| fork | 元public | fork実測 | 剥落 | 判定 |
|---|---|---|---|---|
| pilkwang | ~7.03-7.07※ | 7.074 | ≈0 | honest（R35精読と整合、ただし下記ノイズ床内） |
| iaztec | 7.032**(team)** | 7.126 | +0.09 | **ノイズ帯内=判別不能** |
| ayodeji | 6.768**(team)** | 7.160 | +0.39 | **ほぼノイズ帯内=判別不能** |
| **低public-scoreフォーク** | 7.016 | 7.742 | **+0.73** | 🔴 **唯一の有意信号=public-fit（機構推論として強く支持）** |

### ★剥落の分解能限界（レビュー指摘C1・重要）
- **byte同一コード再ランでLB は 0.089〜0.381 揺れる**（nvidia-kaggle teardown, ledger項目13）→ **±0.4未満はノイズ帯**。
- ∴ iaztec(+0.09)/ayodeji(+0.39) の剥落は**rerunノイズと区別不能**＝「public-fit度でnear-7を細かく順序付ける」ことは統計的に不可。
- さらに iaztec/ayodeji の「元public」は**team最良スコア**（別メンバー/別提出かもしれない）で、pilkwang/prvsiyan の**当該notebook個別スコア**とは定義が違う（交絡）。
- **有意に言えるのは1点のみ**: **低public-scoreのフォーク（7.016）は剥落0.73で public-fit と強く推定される**＝「公開スコアが低いforkほど厚いLB-fit」の実例。私が追っていたそのフォーク（7.016）は罠だった。

## 2. R33/R35 への含意（「解決」ではなく public側の傍証）
- 剥落は**public側の隠しtest再実行**の値であり、**private崩落を直接測っていない**。R33/R35（privateで崩落するか維持か）は**依然 測定不能**。
- 言えること: 低public-scoreのforkは public-fit（剥落大）の傾向→ **枠に入れない**。pilkwang は剥落小＝相対的に honest だが、**private は依然 coin-flip（~30-55%, R39確定）**。「bronze手中」でも「最大確率」でもない。

## 3. 最終2枠：private-robust な選択

### 前提となる2つの数理（レビューで確立）
1. **medal確率は候補2案でほぼ同一 ≈ P(pilkwang が private で bronze<7.093 維持)**:
   - pilkwang+iaztec: iaztec は plateau系統の相関コピー（hidden 0.05差、公開で既に非bronze 7.126>7.093）→ iaztec が独立にpilkwangを跨いでbronze入りする確率≈0。
   - pilkwang+checkpoint: checkpoint 8.160 は bronze線から遠く P(<7.093)≈0。
   - ∴ **第2枠が iaztec でも checkpoint でも medal確率は変わらない**（iaztecの「medal狙い」利点は幻）。
2. **best-of-2 の分散ヘッジは脱相関の時だけ効く**:
   - pilkwang−iaztec = hidden 0.05差（相関≈1）→ **best-of-2 ≈ pilkwang、iaztecの限界寄与≈0**。
   - pilkwang−checkpoint = RMSE 5.32（実測脱相関, R38）→ 唯一 floor 保険になり得る。
3. **honest でもドメインシフトで共倒れする**（無根拠の「honestは崩落しない」を撤回）:
   - well-CV 9.2309 → **field-CV 10.6886（+1.46ft）**＝honestなstackですら未見fieldで系統劣化。
   - plateau系統（pilkwang/iaztec）は隠しwellが未見fieldなら**共に**被る。checkpoint（別出力=脱相関）だけが独立に持ちこたえ得る。

### ★推奨（修正後）: **pilkwang 7.074 + checkpoint-only 8.160**
| | 候補 | 役割 | 根拠 |
|---|---|---|---|
| 柱1 | pilkwang 7.074（ref 54715010） | bronze bet | 最良honest単体・shakeup有利（LB-fitters沈む） |
| 柱2 | checkpoint-only 8.160（ref 54727116） | **脱相関 floor 保険** | 唯一の真の脱相関（plateauとRMSE 5.32）。plateau系統がドメインシフトで系統崩落した場合の独立アンカー |

- **medal確率は pilkwang単独とほぼ同じ**（どちらの第2枠でも）。差が出るのは**worst-case（private-robustness そのもの）**で、脱相関 checkpoint が優る。
- 「private-robustnessを目的にする」なら、robustnessに効かない相関コピー(iaztec)でなく、robustnessに効く脱相関floor(checkpoint)を選ぶのが枠組みと整合。

### 代替（medal-max に全振りする場合のみ）: pilkwang + iaztec
- ドメインシフトが軽微で private≈public 分布なら checkpoint の floor は使われず、iaztec の僅かなnear-7宝くじ券が僅かに優り得る（ただし iaztec は公開で既に pilkwang 後方=薄いスライス）。
- ただし field-CV +1.46ft と上位者の「大shake-up」証言はシフト有意側＝この賭けは弱い。**タイトルが private-robustness なら不整合**。

## 4. なぜ「新しい提出」を作らないか
- private-robust な submission は**既にプールに存在**（pilkwang/checkpoint = 採点済み honest fork）。fix は「public追従でなく honest+脱相関で**選ぶ**」こと。
- 脱相関 honest blend（pilkwang+checkpoint を recompute で in-kernel blend）は理論上さらに分散減だが: (a)recompute-ens は**資源壁で採点空欄**（実装問題であって脱相関価値の否定ではない）, (b)checkpoint(8.16)を混ぜると bias増が分散減を上回る公算（TVT誤差=per-well系統オフセット支配, R26）。**2枠選択としての脱相関(checkpoint単体)は有効だが、blendは非推奨**。
- 自作honest手法(sub-v8 9.014)は100% honestだが弱すぎ（bronze遠い）。

## 5. アクション（08-04 最終確定に向け）
- [x] 剥落による public-fit 推定（有意信号は低public-scoreフォークのみ、near-7はノイズ帯内と確認）
- [x] best-of-2/相関/ドメインシフトの数理でヘッジ効果を再評価
- [ ] 08-04: **pilkwang 7.074 + checkpoint-only 8.160** を明示 select（デフォルト最新2＝prvsiyan-lite/mid の危険→**必ず手動select**）
- [ ] （任意）private≈public を示唆する新情報が出れば柱2を iaztec に振替検討（medal-max）

## 注記（一次出所）
- ※pilkwang「元 ~7.03-7.07」は R44 記載の伝聞、一次LB未固定（結論は7.05でもノイズ内で頑健）。
- checkpoint「別系統③」は不正確: ③(ravaghi/fleongg checkpoint)は pilkwang も共有。脱相関を担保するのは③でなく**出力実測 RMSE 5.32**（R38）。
- private は依然 coin-flip（~30-55%, R39）。本分析は「public追従を止め honest+脱相関で選ぶ」ための整理であり、bronze を保証しない。

検証: `analysis/experiment_ledger.md` R33/R35/R38/R39/R44、提出 ref 54715010/54727116/54761607/54808746
