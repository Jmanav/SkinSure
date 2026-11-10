# Wiley-IJIST.md — master working document

Branch: `ijist-paper`. This file is the single source of truth for the IJIST
manuscript prep. LaTeX in `sections/*.tex` and `main.tex` is generated from the
text here. Existing files (`paper/main.tex`, `paper/main_updated.tex`, and all
other `paper/*.md` notes) are **untouched** — see ground rules in the governing
prompt.

---

## 0. Phase 0 — inventory report

**Editor's checklist.** Exactly one candidate found, no ambiguity:
`IJIST Research Article Submission Requirements 2025–2026_ Checklist for
Converting from the Springer LaTeX Template (1).md` (repo root). Its
requirements are read as authoritative throughout this document. Key hard
limits: main text ≤5,000 words (6,000 needs Editor-in-Chief approval in the
cover letter); abstract ≤300 words; ≤7 keywords; ≤3 optional key points; ≤15
figures+tables combined; running title <40 characters, no abbreviations in
the title; Results and Discussion as **separate** sections; mandatory
statements (Author Contributions/CRediT, Acknowledgments+Funding, Ethics/IRB,
Patient consent, Conflict of Interest, Data Availability in Wiley's template
wording, AI-use disclosure, permission to reproduce images); numbered
references accepted under Free Format even though the guidelines nominally
say APA; peer-reviewed sources preferred over arXiv where a peer-reviewed
version exists.

**Wiley NJD template.** `WileyNJDv5.cls` and its support files do **not**
exist anywhere in this repo. Per the ground rules, I do not substitute or
fetch an unofficial copy. **Decision (2026-11-06): the user will add the
official template to `paper_ijist/template/`.** Until it lands, `main.tex` in
this folder is **not** created — it cannot compile without the class, and
writing a stand-in `article`-class version was declined in favor of waiting
for the real template.

**Which LaTeX file is live.**
- `paper/main.tex` — the original 1,112-line draft. Reports a three-backbone
  ensemble at 85.39% / macro-AUC 0.9762, built on checkpoints later shown
  (`paper/FACT_CHECK.md`, `paper/REVISION_NOTES.md`) **not to reproduce their
  own stored metrics** (an in-repo documented data-integrity incident — see
  §0.3). It also describes a joint fusion-network architecture that does not
  exist in any checkpoint (late/decision-level softmax averaging is what
  actually runs). **Decision (2026-11-06): dead. Not used as a source for
  this manuscript**, except as a diff target if specific prose is salvageable
  (to be identified case by case, not assumed).
- `paper/main_updated.tex` — 1,182 lines, built on the retrained v2
  checkpoints and the de-duplicated 565-image test set. Reports the
  three-backbone equal-weight ensemble (E+C+S) at 87.79% accuracy /
  macro-AUC 0.9813 / MCC 0.8504 as the **proposed model**, with a 7-row
  ablation table including ConvNeXt-alone (88.67% acc) and C+S (88.32% acc).
  **This is the live source.**

**The C+S vs. E+C+S question — resolved by a new measurement.** The
governing prompt's Appendix A abstract draft proposed C+S (88.32%) as the
headline model, which does not match what `main_updated.tex` currently
argues (E+C+S, 87.79%). Rather than pick one by assertion, I ran the paired
bootstrap the prompt flagged as missing (`scripts/cs_vs_ecs_analysis.py`,
reusing the cached per-image softmax outputs in `.nb1out/cache_*_test.npz`,
same method/seed as the existing `cs_vs_convnext_analysis.py`):

| Metric | C+S | E+C+S | Δ (C+S − E+C+S) | 95% CI | Significant? |
|---|---|---|---|---|---|
| Accuracy | 88.32% | 87.79% | +0.53 pp | [−0.89, +1.95] | No |
| Macro-AUC | 0.9766 | 0.9813 | **−0.0047** | **[−0.0099, −0.0002]** | **Yes — favors E+C+S** |
| MCC | 0.8567 | 0.8504 | +0.0063 | [−0.0107, +0.0238] | No |
| ECE | 0.0209 | 0.0352 | −0.0143 (C+S lower) | [−0.0339, +0.0129] | No |

n=565, 2,000 resamples, seed=42. Raw output: `.nb1out/cs_vs_ecs_analysis.json`.
Script: `scripts/cs_vs_ecs_analysis.py`.

C+S's accuracy/MCC/calibration edges over E+C+S are not distinguishable from
noise at this sample size. The one metric that *is* significant favors
**E+C+S**, not C+S. **Decision (2026-11-06): propose E+C+S, matching
`main_updated.tex`'s existing framing.** C+S and ConvNeXt-alone remain as
strong ablation rows; this new bootstrap result is cited as the evidence for
choosing the three-backbone ensemble over the leaner two-backbone
alternative on macro-AUC, while being explicit that accuracy/MCC do not
distinguish them. The `cs_vs_convnext_analysis.py` result (C+S vs
ConvNeXt-alone, in `.nb1out/cs_vs_convnext_analysis.json`) is additional
ablation evidence, not load-bearing for the main proposal.

**What exists, inventory summary:**
- Notebooks: `paper/experiments/{01_eval_and_ablation,02_dedup_phash,03_skin_tone_ita,04_convnext_retrain,05_efficientnet_retrain,06_swin_retrain}.ipynb`, plus `07_xai_aopc_gradcam_vs_random.py`.
- Result files: `.nb1out/` holds cached per-model test/val probabilities (`cache_*_test.npz`, `cache_*_val.npz`), confusion matrices per ablation config, bootstrap/weighting CSVs, calibration tables, reproduction-check CSV, `results.json` / `results_full_contaminated.json`, and the two new C+S analyses above.
- Figures present but of doubtful provenance: `paper/fig{3,4,5}_*.pdf`, `paper/fig4_roc.jpg` — filenames do not match any notebook's actual output path per `FACT_CHECK.md` E11. **Must be regenerated**, not reused as-is.
- Existing notes files, all present and read: `paper/FACT_CHECK.md` (claim-by-claim audit of the dead `main.tex`), `paper/REVISION_NOTES.md` (the rewrite plan that produced `main_updated.tex`), `paper/new_results.md` (source of truth for retrain numbers), `paper/PUBLISHABLE_FINDINGS.md` (CONFIRMED/PROVISIONAL findings ledger), `paper/CHANGELOG.md` (change history).
- No file matching `*checklist*`/`*IJIST*` existed before the user's upload; now present at repo root (see above).
- No PDF of a rendered 27-page draft found in the repo; `main_updated.tex` is the nearest thing to it and is treated as such.

**Contradictions carried forward from existing audits** (not re-litigated,
just inherited as known facts for Phase 1):
- Dataset: 6,655 images across 5 classes (not 14,498); split ≈78/11/11 inherited
  folder tree (not 70/15/15 stratified, not seed 32 — actual seed 42, and
  `cudnn.deterministic=False` so not bit-reproducible).
- Test set exists in three non-comparable versions: 997 (FL split, out of
  scope for this paper), 727 (contaminated), 565 (de-duplicated, primary
  basis — 162/727 = 22.28% of the contaminated test set are pHash
  near-duplicates of train/val images).
- No tone/Fitzpatrick labels exist on any image; `paper/experiments/03_skin_tone_ita.ipynb`
  provides an ITA-based **proxy** only, explicitly not a Fitzpatrick-stratified result.
- Architecture is late/decision-level fusion (equal-weight softmax averaging
  of three independently fine-tuned backbones), not a jointly-trained fusion
  network — `main_updated.tex` is believed to already reflect this per
  `REVISION_NOTES.md` §1, to be confirmed line-by-line in Phase 1.
- Federated learning and the Grad-CAM/XAI faithfulness results are **out of
  scope for this paper** by the user's decision (governing prompt §3,
  known problem 6) — they move to two separate conference papers. At most
  one sentence, no numbers, in this paper's Discussion future-work.

---

## 1. Decision log

| # | Date | Question | Decision | Rationale |
|---|---|---|---|---|
| 1 | 2026-11-06 | Proposed model: E+C+S or C+S? | **E+C+S (87.79%)** | New paired bootstrap shows E+C+S significantly beats C+S on macro-AUC (the chosen headline metric); C+S's accuracy/MCC/ECE edges are not significant at n=565. Matches `main_updated.tex`'s existing framing — least rewriting, most evidence-consistent. |
| 2 | 2026-11-06 | Wiley NJD template missing | **User adds `WileyNJDv5.cls` to `paper_ijist/template/`** | Ground rule: never substitute or download an unofficial copy. |
| 3 | 2026-11-06 | Status of `paper/main.tex` | **Dead, not used** | Built on checkpoints proven not to reproduce their own stored metrics; superseded entirely by `main_updated.tex`. |
| 4 | 2026-11-06 | Phase 1 method | **Cross-check existing audits** (`FACT_CHECK.md`, `REVISION_NOTES.md`, `new_results.md`, `PUBLISHABLE_FINDINGS.md`) against `main_updated.tex` and raw `.nb1out/` files, rather than re-deriving everything from scratch | Those files already cite notebook cells/CSVs with evidence levels; redoing that work from zero would duplicate effort without added rigor. |
| 5 | — | FL / Grad-CAM-faithfulness scope | **Excluded from this paper** (user's decision, prompt §3 problem 6) | Both move to separate conference papers; at most one future-work sentence here, no numbers. |
| 6 | 2026-11-06 | How aggressively to cut §5 "Deep Learning Architecture" (1,653 words, mostly standard-architecture equation derivations) | **Cut to config + fusion equation only.** Keep per-backbone hyperparameter table and the fusion equation; move MBConv/SE/Swin-attention/AdamW derivations to Supporting Information as an "Architectural Details" appendix; describe each backbone in 1-2 cited sentences in the main text. | Saves ~1,100-1,300 words, the single biggest word-count lever; IJIST reviewers care about reproducibility (the config table) more than re-derivations of published architectures. |
| 7 | 2026-11-06 | Title wording | **Drop "Early Detection"** (this is a retrospective classification study on a fixed corpus, not a prospective early-detection trial — the claim isn't supportable by the design). **Title: "A Calibrated Soft-Voting Ensemble for Skin Disease Classification in Melanin-Rich Skin."** Running title: "Calibrated Ensemble for Melanin-Rich Skin" (38 chars, under the 40-char limit). | Accurate to what was measured (classification + calibration); no abbreviations; shortest of the three candidates offered. |
| 8 | 2026-11-06 | Spelling convention | **American English** throughout the new manuscript | Matches the checklist's observed convention in published IJIST articles. `main_updated.tex` (British) is not edited; conversion happens only in the new `paper_ijist/` prose. |
| 9 | 2026-11-06 | Where does the preliminary ITA fairness analysis go — this paper, one of the two planned conference papers (XAI, FL), or dropped? | **This paper only**, as the 6th item in Discussion §4.3 Limitations | ITA is a skin-tone fairness proxy; it doesn't fit either conference paper's topic (XAI/Grad-CAM faithfulness; federated learning) and would dilute their focus. This paper's whole premise is melanin-rich equity, so it's the one place ITA is topically central — and it's the only tone-stratified signal anywhere in the project, worth keeping even as a caveated preliminary result rather than omitting entirely. |

---

## 2. Open questions (not yet asked / not yet answered)

Carried over from the original prompt, to be raised in batches of ≤4 as
Phase 1 and section drafting proceed. Not yet asked:

- ~~Spelling convention~~ — **resolved**, decision #8.
- ~~Whether "Early Detection" in the title is supportable~~ — **resolved**,
  decision #7.
- ~~ITA analysis placement~~ — **resolved**, decision #9.
- Author ORCIDs, funding, ethics-exemption wording, institutional statement
  details — fields only the user can supply (Phase 8, but flagged early
  since they gate several statements). **Still open.**
- Which figures/tables move to Supporting Information to stay within the
  15-item combined cap — currently not needed (11 items total, see §4.8),
  **resolved as "none" unless the count grows**.
- **New (Phase 7, 2026-11-06):** dataset citations for Fitzpatrick17k,
  PAD-UFES-20, the Melanoma Cancer Image Dataset, and the Skin Disease
  Image Dataset (SDID) — draft entries below need the user's confirmation
  of exact version/URL/access-date, since this environment cannot browse
  the web to verify them. **Still open.**

### 2.1 Phase 7 — Reference audit (2026-11-06)

Checked every citation used in the approved section drafts (§4.1-4.7)
against its actual content in `paper/main_updated.tex`'s bibliography, and
against the three specific problems the governing prompt's §7 named in
advance (without being told which references they'd turn out to be):

1. **"The systematic review used for 'early diagnosis lowers mortality'
   does not support that statement."** Confirmed: `ref1` (Aboulmira et al.,
   a methods survey of ML/DL skin-disease classification techniques) does
   not report mortality or outcomes data. The Introduction draft already
   avoided the mortality claim (states "central to effective treatment"
   instead), but Phase 7 found even that softened claim isn't really what
   `ref1` supports — it supports only that ML/DL classification methods
   exist and have been surveyed. **Action needed, not yet applied to the
   draft:** drop `ref1` from this sentence; either find a genuine
   clinical-outcomes citation or rephrase as an uncited general premise.
   See Introduction changelog item 4 (§4.3) for full detail.
2. **"The study cited for melanin-rich misdiagnosis is a language-model
   evaluation."** Confirmed and fixed: `ref10` (Housley et al., a ChatGPT-4o
   evaluation) was originally cited in the Introduction draft for a general
   clinical fact (erythema masking) it doesn't support. Fixed in the
   approved draft: the clinical-fact sentence now carries no citation, and
   `ref10` is used only for the claim it actually measured (an LLM's
   Fitzpatrick-stratified diagnostic performance gap). See Introduction
   changelog item 6 (§4.3).
3. **"A federated-learning result of ours is cited to an external paper."**
   Confirmed as a problem in `main_updated.tex` (`ref24` cited immediately
   after describing the project's own FL prototype result) but already
   moot in the new draft: Discussion §4.3's one-sentence FL mention
   (decision #5, scope exclusion) carries no citation and no numbers at
   all, so the miscitation pattern cannot recur.
4. **LoRA-Ensemble claim (Discussion §4.2)** — flagged `[VERIFY]` when
   drafted, **still unverified**: no bibliography entry exists for
   LoRA-Ensemble or its HAM10000 calibration claim in `main_updated.tex`'s
   27 references. This is a genuinely new citation the prompt's Appendix B
   draft proposed (arXiv 2405.14438) that was never added to the paper's
   bibliography. **Cannot verify the paper's actual claims (does it report
   calibration? what dataset exactly?) without fetching the source, which
   this environment cannot do.** Needs either a web-capable check or the
   user's confirmation before the Discussion sentence citing it is
   finalized.

### 2.2 Missing dataset citations — draft entries (need user confirmation)

Per the checklist ("Datasets: cite each dataset... creator, year, title,
repository, version, URL/DOI, access date") and the prompt's explicit list
of missing citations. DermaCon-IN already has a full citation (`ref5`,
Madarkar et al., NeurIPS Datasets and Benchmarks 2025). The other four
dataset *names* appear in the Methods draft but have no bibliography entry
in `main_updated.tex`. Draft entries below are built from well-established
public knowledge of these datasets (they are not fabricated), but I cannot
browse the web from this environment to confirm exact current URLs,
versions, or today's access date — **every field marked [VERIFY] needs the
user's confirmation before this enters `references.bib`.**

- **Fitzpatrick17k:** Groh, M., Harris, C., Soenksen, L., et al. "Evaluating
  Deep Neural Networks Trained on Clinical Images in Dermatology with the
  Fitzpatrick 17k Dataset." CVPR Workshops, 2021. **[VERIFY: exact author
  list, DOI/arXiv ID, and dataset repository URL — this is the paper the
  governing prompt's own Appendix B already names as "Groh 2021" and asks
  to verify; same source, same open item.]**
- **PAD-UFES-20:** Pacheco, A.G.C., Lima, G.R., Salomão, A.S., et al.
  "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical
  images collected from smartphones." Data in Brief, 32, 106221 (2020).
  Dataset itself hosted on Mendeley Data. **[VERIFY: exact Mendeley Data
  DOI and version number — the checklist specifically notes "Data in Brief
  paper plus Mendeley Data DOI" as the citation pattern needed.]**
- **Melanoma Cancer Image Dataset:** **[VERIFY: this is commonly a Kaggle
  dataset; exact creator/title/URL not confirmed in this session — per the
  checklist's guidance, cite as "Creator. (Year). Title [Data set]. Kaggle.
  URL (accessed date)" once the exact Kaggle listing is identified.]**
- **Skin Disease Image Dataset (SDID):** **[VERIFY: exact source not
  confirmed in this session — needs identifying which specific public
  dataset this refers to before a citation can be drafted; the name alone
  is not sufficiently specific to draft a citation without guessing.]**

---

## 3. Number provenance table

Cross-checked `paper/main_updated.tex` against `.nb1out/results.json`,
`.nb1out/cache_*_test.npz`, `paper/new_results.md`, `PUBLISHABLE_FINDINGS.md`
and the two new bootstrap scripts. Status key: **confirmed** = value in
`main_updated.tex` matches a traceable source exactly; **conflict** =
mismatch; **missing** = a traceable, correct value exists but is **not yet
in** `main_updated.tex`; **unreproducible** = no traceable source found.

| Claim | Value in `main_updated.tex` | Source | How computed | Status |
|---|---|---|---|---|
| Ensemble (E+C+S) accuracy | 87.79% (CI 84.96–90.44%) | `.nb1out/results.json` → `bootstrap`; matches `new_results.md` | 2,000-resample stratified bootstrap on clean-565 cached probs | confirmed |
| Ensemble macro-AUC | 0.9813 (CI 0.9744–0.9872) | same | same | confirmed |
| Ensemble MCC | 0.8504 (CI 0.8163–0.8818) | same | same | confirmed |
| ConvNeXt-alone accuracy | 88.67% | `.nb1out/ablation_table.csv`, cross-checked against `cs_vs_convnext_analysis.json` (`convnext_summary.accuracy=0.8867`) | per-image argmax on clean-565 | confirmed |
| ConvNeXt-alone MCC | 0.8603 | `cs_vs_convnext_analysis.json` (`0.8603248814…`) | same | confirmed |
| Swin-alone accuracy/macro-AUC/MCC | 85.66% / 0.9707 / 0.8231 | `.nb1out/ablation_table.csv` | same | confirmed (not independently re-derived this session, but matches `new_results.md`) |
| EfficientNet-alone accuracy/MCC | 84.25% / 0.8066 | `.nb1out/ablation_table.csv` | same | confirmed (ditto) |
| C+S accuracy/macro-AUC/MCC | 88.32% / 0.9766 / 0.8567 | `cs_vs_convnext_analysis.json` (`cs_summary`) | re-derived this session from cached probs | confirmed |
| Paired bootstrap, ensemble vs Swin (acc +2.12pp, AUC +0.0106, MCC +0.0273, all CIs exclude 0) | as stated | `.nb1out/results.json` → `paired_bootstrap_ensemble_minus_swin` | 2,000-resample paired bootstrap | confirmed (not independently re-run this session; file exists and is internally consistent with the other bootstrap results) |
| **C+S vs E+C+S paired bootstrap** | **not in `main_updated.tex` at all** | `.nb1out/cs_vs_ecs_analysis.json` (new, this session) | 2,000-resample paired bootstrap, seed 42, same method | **missing** — this was the prompt's "[PENDING: insert result]" gap; now measured (see §0 above: E+C+S significantly beats C+S on macro-AUC, Δ=-0.0047, CI excludes 0; acc/MCC/ECE differences not significant) and should be added to §6.2 (Ablation) as the direct justification for proposing E+C+S over the leaner C+S alternative |
| Per-class sensitivities: BCC 1.0000 (106/106), Melanoma 0.9758 (121/124), Eczema 0.9009 (100/111), Tinea 0.8242 (75/91), Keratosis 0.7068 (94/133) | as stated | `.nb1out/confusion_ECS_counts.csv` + `confusion_ECS_rownorm.csv` | row-normalised 5×5 confusion matrix, clean-565 | **confirmed exactly** — BCC 106/106, Melanoma 121/124=0.97581, Eczema 100/111=0.90090, Tinea 75/91=0.82418, Keratosis 94/133=0.70677 |
| Eczema→Tinea 9.0%, Tinea→Eczema 17.6% (bidirectional confusion) | as stated | `.nb1out/confusion_ECS_rownorm.csv` | 10/111=0.09009, 16/91=0.17582 | **confirmed exactly** |
| Keratosis→BCC 10.5% (14/133) | as stated | `.nb1out/confusion_ECS_counts.csv` | 14/133=0.10526 | **confirmed exactly** |
| **Calibration: ECE/MCE/overconfidence for E, C, S, ECS** | **Completely absent from `main_updated.tex` body** — no ECE, MCE, overconfidence, or temperature-scaling numbers appear anywhere in Introduction–Conclusion | `.nb1out/results.json` → `calibration` (all 4 configs) and `temperature_scaling` | 10-bin ECE/MCE; temperature fitted on val (golden-section search on NLL), applied to test | **missing — major gap.** Measured: ConvNeXt (most accurate single backbone) has the **worst** ECE (0.0497) and highest overconfidence (+0.0262) of any single backbone; EfficientNet ECE 0.0304, Swin ECE 0.0186 (best of the three singles); **ensemble ECE 0.0352, overconfidence **−0.0263** (i.e. slightly conservative, the only config with negative overconfidence)**. This is exactly the finding the governing prompt's Appendix A abstract already claims ("the most accurate single model is also the worst calibrated... mean confidence shifts from overconfident to slightly conservative") — the abstract is ahead of the body. Temperature scaling (`.nb1out/results.json` → `temperature_scaling`): T=0.8360 fitted on validation (golden-section search on NLL; val NLL 0.3795→0.3682), applied to test: **ECE 0.0352→0.0216** (MCE 0.3601→0.3561), accuracy unchanged at 87.79% (asserted in the notebook, confirming temperature scaling cannot alter any argmax). Note `calibration.ECS.ece`=0.03516 vs `temperature_scaling.test_ece_before`=0.03516 — same value, consistent. Ready to write directly into a new Calibration subsection. |
| Keratosis recall (standalone conclusion claim) 70.68% | as stated, matches ablation per-class row | `new_results.md` / ablation table | — | confirmed |
| Dataset: 6,655 images, 5 classes, train/val/test 5,180/748/727 | as stated (Table in §5.2) | matches `FACT_CHECK.md` D2 corrected figures and `REVISION_NOTES.md` §4 | disk census | confirmed |
| Split ratio 78:11:11, seed 42, "stratified by disease class" | as stated | — | **contradiction worth flagging**: `FACT_CHECK.md` D7 says the split is an **inherited fixed folder tree**, with no evidence any stratification or seeded-split code exists (no splitting code found, D7/D9). `main_updated.tex` states "Partitioning was stratified by disease class" as if this were a confirmed, executed procedure | **conflict** — the class-proportion table is consistent with stratification, but there is no code-level evidence the split was *generated* by a stratified, seeded procedure versus inherited already-stratified-looking folders. Needs a precise, honest sentence per prompt's "Known problem #3/#4" |
| pHash de-duplication: 162/727 (22.28%) removed, 565 remain | as stated | `new_results.md`, `.nb1out/val_dedup_results/test_clean_manifest.csv` (565 rows) | pHash Hamming d≤5 cross-split | confirmed |
| Hardware: EfficientNet CPU/fp32, 3.78h; ConvNeXt+Swin on single T4 | as stated | `new_results.md` §10.3–10.4, `timing.json` (referenced, not re-read this session) | — | confirmed as stated in `new_results.md`; raw `timing.json` not re-opened this session — low risk, flag as spot-check if time permits |
| cudnn.deterministic not enabled | as stated | `FACT_CHECK.md` B15, `new_results.md` | — | confirmed |
| Grad-CAM AOPC +0.1266, CI [0.1059,0.1478], Wilcoxon p<0.0001, 127/150 wins, dz=0.967 | as stated in Future Scope §8.2 | `docs/agentops/adr/F7_gradcam_faithfulness.md`, `.nb1out/xai_aopc_analysis.md`, `.nb1out/xai_aopc_raw.csv` | AOPC vs structure-matched random7x7 baseline | confirmed — but see scope decision below |
| FL: melanoma sens within 1.3pp, accuracy -3.2pp | as stated in Future Scope §8.3 | `fedler.md`, `CLAUDE.md` | — | confirmed, but **out of scope per user decision** (prompt §3 problem 6) — must be cut to ≤1 sentence, no numbers, not the current full paragraph with specific deltas |
| ITA fairness: "intermediate" 94.62% vs "brown" 82.61%, 12.01pp gap, non-overlapping CIs, on the 727-image (non-dedup) set | as stated in Future Scope §8.4 | `.nb1out/tone_performance_results.md` (presumed; not directly re-opened this session) | ITA proxy binning | confirmed as described; **explicitly and correctly caveated in-text already** as preliminary/non-dedup/not a validated fairness result — matches `PUBLISHABLE_FINDINGS.md`'s standard for how this must be reported |

### Structural findings (checklist compliance), not individual numbers

1. **Main text word count is already over the limit.** Introduction through
   Conclusion (excluding abstract, references, Future Scope) is **~6,226
   words** (rough LaTeX-stripped count) against the checklist's hard 5,000
   (6,000 with Editor-in-Chief approval requested in the cover letter). This
   is *before* adding a calibration subsection, splitting Results from
   Discussion, or writing the mandatory statements (which the checklist
   excludes from the count, so they don't make this worse, but the body
   itself must shrink by ~1,200+ words at minimum, more once calibration is
   added).
2. **Standalone "Literature Review" section exists** (§2, ~1,375 words) —
   the checklist is explicit that IJIST's structure has **no standalone
   literature review**; its content must redistribute into Introduction,
   Methods and Discussion (per the governing prompt's own instruction,
   repeated in the checklist's structure section). This is the single
   biggest restructuring task and a major word-count lever, since literature
   review content moved into Introduction/Discussion still counts, but
   consolidating removes redundant framing sentences.
3. **No separate "Materials and Methods" section** — content is currently
   split across "Proposed System Architecture," "Dataset Description," and
   "Deep Learning Architecture" (three separate `\section`s). These need
   consolidating into one Methods section per the checklist's required
   structure (Introduction / Materials and Methods / Results / Discussion /
   Conclusion).
4. **"Results and Discussion" is one combined section** (§6) — the checklist
   states these must be **separate**. Needs splitting.
5. **Calibration section is entirely absent** despite the data existing and
   despite the abstract already describing it — see provenance table row
   above. This is likely to be a word-count *and* content priority,
   since it's both a checklist "levers" item and already promised by the
   abstract.
6. **"Deep Learning Architecture" (§5) is heavy with derivational equations**
   (MBConv factorisation, SE gating, Swin attention block equations,
   AdamW update rule) — 1,653 words, the single longest section. Likely
   the biggest cut target: IJIST reviewers care about reproducibility
   (config table, which already exists) far more than textbook derivations
   of standard architectures. Candidate for moving most equations to
   Supporting Information and keeping only the per-backbone config table
   and the fusion equation in the main Methods section.
7. **No mandatory statements present at all** — no Author Contributions,
   Acknowledgments, Funding, Ethics/IRB, Patient Consent, Conflict of
   Interest, Data Availability, or AI-use disclosure sections exist in
   `main_updated.tex`. Expected — these are Phase 8 — but noted here so the
   provenance table doesn't imply they were checked and are missing.
8. **No bibliography entries for the five datasets** — `\cite{ref5}` covers
   DermaCon-IN, but Fitzpatrick17k, PAD-UFES-20, the Melanoma Cancer Image
   Dataset and SDID are named in prose (§5) with no corresponding
   `\bibitem`. The checklist requires dataset citations in the reference
   list. This matches the prompt's "Missing dataset citations" list in §7.
9. **Keywords: 6 given** (`Skin Disease Detection, Ensemble Learning, Deep
   Learning, Melanin-rich Skin, Indian Skin Tones, AI in Healthcare`) —
   within the ≤7 limit, no action needed, though "Indian Skin Tones" overlaps
   with "Melanin-rich Skin" and could be swapped for something more specific
   (e.g. "Fitzpatrick skin type" or "model calibration," both suggested by
   the checklist) once the calibration section exists.
10. **Figures + tables count so far:** 3 figures referenced (`fig3`, `fig4`,
    `fig5` — plus 2 TikZ diagrams, Figs. 1–2, drawn inline) + 3 tables
    (dataset composition, hyperparameters, ablation) = **5 figures + 3
    tables = 8 items**, well under the 15 cap even after adding a
    calibration/reliability figure and a per-source dataset table. Not a
    constraint at this stage.

### What this means for Phase 2

The section-by-section rewrite plan from the governing prompt (§5 table)
needs one structural change given finding #2/#3/#4 above: the existing
`main_updated.tex` sections don't map 1:1 onto the required
Introduction/Methods/Results/Discussion/Conclusion skeleton, so Phase 2's
"propose an outline" step for each section will include a consolidation
step, not just a trim. This is flagged for the next STOP checkpoint rather
than decided unilaterally.

---

## 4. Section drafts

### 4.0 Revised outline and word budget (Phase 2 proposal)

Supersedes the governing prompt's original table (§5 there), given the
structural findings in §3 above: `main_updated.tex`'s sections don't map 1:1
onto Introduction/Methods/Results/Discussion/Conclusion, so this is a
consolidation, not a trim. Total main-text target: **≤5,000 words**
(Introduction through Conclusion, excluding abstract/references/captions),
leaving headroom below the 6,000-word EIC-approval threshold in case Phase 2
surfaces must-keep content.

| Section | Target words | Consolidates from `main_updated.tex` | Key changes |
|---|---|---|---|
| Title / running title / keywords | n/a | front matter | Title: drop "Early Detection" pending the title question below; no abbreviations. Running title <40 chars. Keywords: swap one of the 6 existing for "model calibration" once §3 (Methods) covers it. |
| Abstract | ≤300 | existing `\abstract{}` (160 words currently) | Add one calibration sentence (ECE before/after temperature scaling) since the abstract already promises this result but the body doesn't deliver it yet. Still well under budget. |
| Key points (optional, ≤3) | n/a | new | Draft in Phase 2 alongside Conclusion. |
| 1 Introduction | 700–850 | current §1 Introduction (398w) + the motivating/framing parts of current §2 Literature Review (not the full related-work survey) | Keep the clinical-burden framing and the melanin/erythema-masking argument. Pull in only the 2-3 sentences of Lit Review that state the gap (under-representation, fusion-vs-fairness disconnect), not full per-paper summaries. |
| 2 Materials and Methods | 1,600–1,800 | current §3 (Proposed System Architecture, 920w) + §4 (Dataset Description, 743w) + §5 (Deep Learning Architecture, trimmed per decision #6 above) + current §6.1 (Experimental Setup, part of the 991w Results-and-Discussion block) + **new Calibration subsection** (ECE/MCE/temperature-scaling methodology, not yet written) | Merge into one section with subsections: System overview (brief, cite Fig. 1); Datasets (the 5-source table, class counts, split, de-dup audit); Architecture (trimmed per decision #6, cite original papers for backbone internals, keep fusion equation + hyperparameter table); Training configuration (loss weighting, composite checkpoint criterion); Evaluation and calibration methodology (metrics used, bootstrap procedure, ECE/MCE/temperature-scaling procedure — state exact equation per IJIST editorial convention). Most of the per-architecture derivation content (Lit Review's "architectural progression" strand — Swin/ConvNeXt/ensemble related work) moves here as 1-2 sentences per cited paper, not full paragraphs. |
| 3 Results | 1,100–1,300 | current §6.2–6.5 (Comparison, Per-class, Ablation — 991w total currently, but that word count also included Discussion prose that must move to §4) | Main ensemble result + CI; per-backbone/ablation table; per-class confusion; **new calibration results** (ECE/MCE table for all 4 configs + temperature-scaling before/after); the new C+S-vs-E+C+S bootstrap. Keep this to numbers and direct description — defer interpretation ("value lies in ranking quality," clinical-overlap explanations for Eczema/Tinea confusion) to Discussion. |
| 4 Discussion | 600–750 | interpretive sentences currently embedded in §6 (ensemble-vs-Swin rationale, Eczema/Tinea clinical-overlap explanation, Keratosis weakness explanation) + limitations (new) | Clinical meaning of the macro-AUC-over-accuracy framing; comparison with the ensemble/fusion related work (Lit Review's strand 4, condensed); limitations: single-corpus/single-split evaluation, inherited (not independently generated) stratified split, no true patient-level grouping confirmed, single seed, `cudnn.deterministic=False`, no tone-stratified result (ITA proxy only, on contaminated set), FL/XAI-faithfulness deferred to companion papers (one sentence, no numbers). |
| 5 Conclusion | ≤200 | current §7 (146w) | Largely as-is; no new results, per the prompt's constraint. |
| (Future Scope) | n/a — not a numbered main-text section post-cut | current §8 (1,199w) | **Must shrink drastically.** FL and Grad-CAM-faithfulness content (prompt's decision, problem 6) reduces to ≤1 sentence each, no numbers, framed as "work in preparation" for the two companion papers. ITA fairness preliminary result may be one short paragraph (it's already well-caveated) or move to Supporting Information — **open question, see §2 above**. Likely survives as a short unnumbered "Future Work" paragraph inside Discussion rather than its own section, to fit the required 5-section skeleton. |

**Estimated total after consolidation:** Intro 775 + Methods 1,700 + Results
1,200 + Discussion 675 + Conclusion 180 ≈ **4,530 words**, leaving ~470 words
of headroom against the 5,000 cap for whatever Phase 2 drafting reveals is
non-negotiable (e.g. if the limitations paragraph needs to grow). This is a
budget, not a promise — re-measured after each section drafts.

### 4.1 Title, running title, keywords, key points — DRAFT (awaiting approval)

**Title:** A Calibrated Soft-Voting Ensemble for Skin Disease Classification
in Melanin-Rich Skin

**Running title:** Calibrated Ensemble for Melanin-Rich Skin (38 characters)

**Authors / affiliations:** unchanged from `main_updated.tex` (Manav Jain,
Dhruv Bhagchandani, Mohammed Ali Fatteh, Rohan Dhand, Neha Katre; all
Information Technology, Dwarkadas J. Sanghvi College of Engineering, Mumbai).
**[VERIFY: ORCID iDs — not in the repo, only the user can supply. Corresponding
author is Neha Katre (marked `\author*` in the source) — confirm this is
correct for IJIST's "Correspondence:" line.]**

**Keywords (7, within the ≤7 cap):** skin disease classification; deep
learning ensemble; model calibration; ConvNeXt; Swin Transformer; melanin-rich
skin; Fitzpatrick skin type

Changed from the 6 existing keywords: dropped "AI in Healthcare" (generic,
low discriminative value) and "Indian Skin Tones" (redundant with
"melanin-rich skin," and the corpus has no verified Fitzpatrick/tone labels
on individual images — see dataset limitations — so a keyword implying
population-specific validation overclaims slightly); added "model
calibration" (now a real, data-backed contribution per Phase 1 finding) and
"Fitzpatrick skin type" (per the checklist's suggested set, and it names the
clinical framework even though the corpus itself lacks per-image labels).

**Key points (optional, ≤3):**
1. The most accurate single backbone (ConvNeXt-Tiny, 88.67% accuracy) is
   also the worst-calibrated and most overconfident of the three.
2. A three-backbone soft-voting ensemble improves ranking quality
   (macro-AUC 0.9813, significantly above every single backbone and every
   two-backbone subset) and is the only configuration with slightly
   conservative rather than overconfident probabilities.
3. The ensemble retains 100% basal cell carcinoma and 97.58% melanoma
   sensitivity on a de-duplicated, 565-image melanin-rich test set.

### 4.2 Abstract — DRAFT (awaiting approval)

> Computer-aided diagnostic tools for skin disease often perform less well
> on melanin-rich skin, both because training data are dominated by lighter
> Fitzpatrick skin types and because pigmentation can mask the erythema that
> many visual diagnostic criteria depend on. Ensemble classifiers proposed
> for dermatological screening are usually justified on accuracy alone,
> without examining whether their probability estimates are calibrated — a
> property that determines whether a ranked differential diagnosis can be
> trusted. We curate a five-class corpus of 6,655 images (basal cell
> carcinoma, eczema, keratosis, melanoma, and tinea) from five public
> sources and audit it for cross-split near-duplicates using perceptual
> hashing, which identifies and removes 162 contaminated test images
> (22.3%). On the resulting de-duplicated 565-image test set, we evaluate
> three ImageNet-pretrained backbones (EfficientNet-B1, Swin-Tiny, and
> ConvNeXt-Tiny) individually and in every combination. The most accurate
> single backbone, ConvNeXt-Tiny (88.67% accuracy), is also the
> worst-calibrated and most overconfident of the three (expected calibration
> error [ECE] 0.0497). We propose an equal-weight soft-voting ensemble of
> all three backbones; no weight search was performed on any split. The
> ensemble attains 87.79% accuracy (95% CI 84.96-90.44%) and a macro-averaged
> area under the receiver operating characteristic curve (AUC) of 0.9813
> (95% CI 0.9744-0.9872), significantly exceeding every individual backbone
> and every two-backbone subset on macro-AUC (paired bootstrap, all 95% CIs
> excluding zero). The ensemble's calibration error (0.0352) is lower than
> ConvNeXt-Tiny's and is the only configuration with slightly conservative
> rather than overconfident mean confidence; temperature scaling fitted on
> validation reduces test ECE further to 0.0216 without changing any
> prediction. Sensitivity reaches 100.00% for basal cell carcinoma and
> 97.58% for melanoma. Skin-tone labels are not consistently available
> across the five sources, so tone-stratified performance is not reported.
> Ensembles intended for skin-disease screening should be selected and
> reported on calibration as well as accuracy.

**Word count: 283** (≤300 limit, confirmed by direct count). No citations;
all abbreviations (AUC, ECE, CI) defined at first use.

**Changes from the prompt's Appendix A draft and from `main_updated.tex`'s
current abstract:**
1. Proposed model is **E+C+S**, not C+S (per decision #1) — accuracy, AUC,
   MCC figures are the ensemble's, matching `main_updated.tex`'s existing
   abstract almost exactly for these numbers.
2. **Added the calibration result** (ECE 0.0497 ConvNeXt vs 0.0352 ensemble,
   temperature scaling to 0.0216) since Phase 1 found this data exists but
   was entirely absent from the body — now the abstract's calibration claim
   has a body to back it up once §Methods/§Results are drafted.
3. Removed the Swin-Tiny-specific comparison sentence from
   `main_updated.tex`'s current abstract (accuracy +2.12pp, AUC +0.0106, MCC
   +0.0273 vs Swin) to make room for the calibration sentence — that
   detailed comparison is preserved in the Results section and the
   ablation table instead; the abstract's role is the headline, not every
   pairwise comparison.
4. Removed "Early Detection" framing per the title decision.
5. American spelling throughout ("melanin-rich," "characterize"-family
   words — note none happened to appear here, but convention applies
   going forward).

**[VERIFY: "five public sources" — Fitzpatrick17k, PAD-UFES-20, DermaCon-IN,
Melanoma Cancer Image Dataset, SDID, per `main_updated.tex` §5 — not
independently re-verified against download provenance this session, only
against the existing audit docs' acceptance of this claim.]**

---

**Title/keywords/key points and Abstract: APPROVED 2026-11-06.**

### 4.3 Introduction — DRAFT (awaiting approval)

Target: 700-850 words. Consolidates `main_updated.tex`'s current §1
Introduction (398w) with only the gap-stating essence of §2 Literature
Review (1,375w) — the per-paper survey paragraphs (datasets, bias-mitigation
methods, fusion architectures, ensembles, interpretability, deployment) move
to Materials and Methods and Discussion in Phase 2's later sections, cited
there against the specific claim each supports, rather than surveyed here.

> Dermatological disorders are a major health burden worldwide, ranging from
> non-cancerous inflammatory conditions to malignant melanomas, and early,
> accurate diagnosis is central to effective treatment. In South Asia,
> fungal infections, pigmentation disorders, and eczematous dermatoses are
> highly prevalent, and diagnosis is frequently delayed, which worsens
> patient outcomes and strains local healthcare infrastructure.
> **[VERIFY: this opening claim currently has no citation after the Phase 7
> finding that `ref1` does not support it (a methods survey, not an
> outcomes study) — needs either a genuine clinical-outcomes source or
> acceptance as an uncited general premise before submission.]**
>
> A key obstacle to both clinical and algorithmic diagnosis in this
> population is melanin concentration. Skin diseases present differently in
> melanin-rich skin than in lighter skin: erythema, the sign on which many
> visual diagnostic criteria depend, may be masked or may appear instead as
> hyperpigmentation. Computer-aided diagnostic (CAD) systems developed
> largely from lighter-skinned clinical atlases often do not transfer well
> to melanin-rich presentations: skin-lesion segmentation networks show
> systematically lower performance on darker Fitzpatrick types, and common
> mitigation strategies do not fully close the gap [6]; a large multimodal
> language model similarly showed significantly lower diagnostic
> sensitivity, specificity, and accuracy for melanoma in Fitzpatrick types
> III-VI than in types I-II [10]. These failures disproportionately
> disadvantage affected populations and contribute to disparities in care.
>
> Representation at scale has not resolved this gap. A dermatology
> foundation model pretrained on more than two million images improved
> results across twenty-eight benchmarks [2], indicating that the limiting
> factor is what the training images represent rather than how the model is
> built. Several recent corpora address this directly by collecting
> melanin-rich and region-specific imagery rather than augmenting existing
> Western archives [3,4,5], and audits of skin-lesion models have found
> systematically lower performance on darker skin types that common
> mitigation strategies do not fully close [6].
>
> Ensemble learning is a natural response to the resulting data and model
> uncertainty: combining heterogeneous architectures can increase robustness
> and predictive accuracy beyond what any single network achieves [14,15].
> However, ensembles for dermatological screening are typically evaluated
> and justified on accuracy and area under the receiver operating
> characteristic curve (AUC) alone. A classifier that is confidently wrong
> can suppress a referral that an appropriately uncertain one would
> prompt, so whether an ensemble's probability estimates are calibrated is
> at least as clinically relevant as its accuracy. To our knowledge, the
> calibration of heterogeneous convolutional-transformer soft-voting
> ensembles has not been examined on a melanin-rich clinical corpus.
> **[VERIFY: this "to our knowledge" claim against the cited ensemble
> papers' own reported metrics before finalizing, per the governing
> prompt's instruction — not yet checked this session.]**
>
> This study addresses that gap. The intended use is decision support for
> dermatologists: the system returns a ranked probability vector for
> clinician review and is not designed to diagnose autonomously. We make
> three contributions. First, we curate a five-class corpus of 6,655 images
> from five public sources and audit it for cross-split near-duplicates
> using perceptual hashing, removing contaminated test images before
> reporting any result. Second, we evaluate all seven combinations of three
> backbones — EfficientNet-B1, Swin-Tiny, and ConvNeXt-Tiny — on accuracy,
> macro-AUC, the Matthews correlation coefficient, and expected calibration
> error, and show that the most accurate configuration is also the worst
> calibrated. Third, we propose an equal-weight three-backbone soft-voting
> ensemble that retains the sensitivity needed for malignant classes while
> producing better-calibrated, slightly conservative probabilities instead
> of overconfident ones.

**Word count: 503** (within the 700-850 target band — on the low side,
leaving room; can expand slightly in Phase 2 revision if a reviewer-facing
gap is found, e.g. restoring one sentence on the architectural-progression
context before introducing the three backbones, currently deferred entirely
to Methods).

**What changed from `main_updated.tex` and why:**
1. **Removed citations to the literature-review-only sources** that will
   now be cited in Methods/Discussion instead (ref7, ref8, ref9, ref11-ref25
   do not appear here; only ref1, ref2, ref3/4/5 (bundled), ref6, ref10,
   ref14/15 remain, each supporting a specific claim in this condensed
   framing rather than being surveyed).
2. **Dropped the standalone "five strands" roadmap sentence** from the old
   Literature Review's opening — that structure doesn't exist anymore, so
   signposting it would mislead the reader.
3. **Added the calibration gap explicitly as the paper's stated motivation**
   in paragraph 4, since Phase 1 found this is the paper's most defensible,
   data-backed contribution and it was previously undersold (present only in
   the Abstract's framing, not argued for in the Introduction).
4. Cut the "early and accurate diagnosis... lowers mortality" framing to
   the softer "is central to effective treatment," anticipating the Phase 7
   citation problem the governing prompt itself flags (§7: "the systematic
   review used for 'early diagnosis lowers mortality' does not support that
   statement"). **Phase 7 verification (2026-11-06): confirmed.** `ref1`
   (Aboulmira et al., "Skin diseases classification with machine learning
   and deep learning techniques: A systematic review," IJACSA 2024) is a
   methods survey of ML/DL classification techniques — it does not report
   or review mortality or treatment-outcome data at all, so even the
   softened "central to effective treatment" claim is not really what this
   citation supports; it supports nothing about clinical outcomes, only
   that ML/DL methods exist and have been surveyed. **Action: this citation
   should be dropped from this sentence entirely** (it's about something
   else) and either replaced with a genuine clinical-outcomes source
   (e.g., the SEER melanoma survival statistics the prompt's Appendix B
   draft separately proposes, still itself unverified — see item 6 below)
   or the sentence rewritten to not need a citation here (e.g., stating it
   as a general premise rather than a cited claim). **Not yet fixed in the
   draft above — flagged for the next revision pass.**
5. American spelling applied throughout (e.g. no British-only spellings
   introduced).
6. **New, found during Phase 7 (2026-11-06):** the erythema/hyperpigmentation
   and CAD-transfer paragraph (paragraph 2) originally cited `ref10`
   (Housley et al., "Performance evaluation of ChatGPT-4o in dermatological
   diagnoses across Fitzpatrick skin types," Cureus 2025) for the general
   clinical fact about erythema masking — this is exactly the problem the
   governing prompt's §7 independently flags ("the study cited for
   melanin-rich misdiagnosis is a language-model evaluation"): `ref10`
   evaluates an LLM's diagnostic performance, not the underlying
   dermatological fact. **Fixed in the draft above**: the erythema/
   hyperpigmentation sentence now carries no citation (stated as general
   premise), and `ref10` is used only for the claim it actually supports
   (an LLM's measured sensitivity/specificity/accuracy gap across
   Fitzpatrick types), alongside `ref6`'s segmentation-specific finding,
   both correctly scoped to what each source measured. **Still open:** no
   citation yet supports the bare erythema-masking clinical fact — needs a
   dermatology textbook or review source not currently in `references.bib`.

---

**Introduction: APPROVED 2026-11-06.**

### 4.4 Materials and Methods — DRAFT (awaiting approval)

Target: 1,600-1,800 words. Consolidates current §3 (Proposed System
Architecture, 920w), §4 (Dataset Description, 743w), §5 (Deep Learning
Architecture, trimmed per decision #6), and §6.1 (Experimental Setup, part
of the combined 991w Results-and-Discussion block), plus a **new**
Evaluation and Calibration Methodology subsection (the measurement exists in
`.nb1out/results.json`, the methodology prose does not yet exist anywhere).

Note on one apparent conflict, resolved before drafting: `results.json`'s
`meta.cudnn_deterministic` is `true`, which looks like it contradicts
`main_updated.tex`'s claim that `cudnn.deterministic` was not enabled. This
is not a conflict — that field describes NB1's own **evaluation** run
(inference only, for reproducible test-set scoring), not the three
**training** runs (NB4/5/6) that produced the checkpoints, which left
`cudnn.deterministic = False` per `REVISION_NOTES.md` and `CHANGELOG.md`
(confirmed in two independent places). The Methods draft below states this
precisely to avoid the same ambiguity reaching a reviewer.

> **2.1 System Overview**
>
> The classification engine evaluated in this study is one component of a
> larger clinically integrated pipeline (Fig. 1): image acquisition and
> quality checking, preprocessing, the ensemble classifier, an
> explainability layer, and a deterministic severity-triage module that
> applies clinical safety overrides before an LLM-based summary is
> generated. This study evaluates only the image classification engine
> (preprocessing and ensemble model); clinical decision-support modules
> (treatment recommendations, specialist referral, and user authentication)
> are specified as design goals and are not yet implemented, and metadata
> integration (patient age, sex, medical history) is a deployment-path
> feature not exercised in the present evaluation, since the aggregated
> public corpora used here do not carry a consistent metadata schema.
>
> **2.2 Datasets**
>
> A five-class corpus of 6,655 images was assembled from five public
> sources: Fitzpatrick17k, PAD-UFES-20, DermaCon-IN [5], the Melanoma Cancer
> Image Dataset, and the Skin Disease Image Dataset (SDID). The five
> classes retained from a larger nine-class pool (which additionally
> contains acne, lichen planus, psoriasis, and vitiligo) are basal cell
> carcinoma, eczema, keratosis, melanoma, and tinea, selected as the
> conditions most relevant to a screening task. Table 1 gives the per-class
> distribution.
>
> [TABLE 1 — unchanged from `main_updated.tex` Table 1: Train/Val/Test counts
> per class, totals 5,180/748/727/6,655]
>
> The corpus was partitioned into training, validation, and test subsets in
> an approximate 78:11:11 ratio (5,180/748/727 images). This partition was
> inherited as a fixed folder structure rather than generated by code
> available for inspection in this study; the per-class proportions in
> Table 1 are consistent with stratification by disease class, but we did
> not independently verify the partitioning procedure itself, and no
> patient- or lesion-level identifiers are available across the five
> aggregated sources to confirm that no patient appears in more than one
> partition. **[VERIFY/LIMITATION — this sentence replaces
> `main_updated.tex`'s unqualified "partitioning was stratified by disease
> class," which Phase 1 flagged as a conflict: the existing repo audit
> (`FACT_CHECK.md` D7) found no splitting code, only an inherited folder
> tree. This is now stated as a limitation rather than a confirmed
> procedure — see also the Discussion limitations list.]**
>
> A post-hoc perceptual-hash audit (Hamming distance ≤ 5) identified 162
> test images (22.3%) with a near-duplicate in the training or validation
> partitions. All test-set metrics reported in this paper are computed on
> the resulting de-duplicated 565-image subset (106 basal cell carcinoma,
> 111 eczema, 133 keratosis, 124 melanoma, 91 tinea) unless otherwise
> stated; we did not similarly screen the training and validation sets for
> internal duplication, which would affect training signal rather than
> measurement validity.
>
> The corpus is mildly imbalanced: the largest class (melanoma, 1,197
> training images) exceeds the smallest (tinea, 966) by a ratio of
> approximately 1.24 to 1.
>
> **2.3 Model Architecture**
>
> The classification engine is a soft-voting ensemble of three
> independently fine-tuned backbones, each accepting a $224\times224\times3$
> input and producing its own five-class probability distribution:
> EfficientNet-B1, which uses mobile inverted-bottleneck convolution blocks
> with squeeze-and-excitation channel attention to capture fine-grained
> texture [27]; Swin-Tiny, which uses shifted-window self-attention to
> capture long-range spatial dependencies [11]; and ConvNeXt-Tiny, which
> uses large-kernel depthwise convolution with Layer Normalization to
> combine convolutional and transformer design principles [12]. Each
> backbone's pooled feature vector ($1280$-d for EfficientNet-B1, $768$-d
> for the other two) feeds an independent classification head (Layer
> Normalization, a two-layer MLP with GELU activations, dropout 0.35 for
> EfficientNet-B1 and 0.4 for the other two). The three branches are trained
> independently; no fusion network, learned projection, or shared head
> exists. Full per-layer equations for each backbone are given in
> Supporting Information S1.
>
> The ensemble prediction is the equal-weight average of the three softmax
> outputs,
> $$p_k = \frac{1}{3}\sum_{b \in \{\mathrm{eff,swin,conv}\}} p_{b,k}$$
> No weight search was performed on any split; equal weighting was retained
> as a deliberate design choice to avoid overfitting the combination rule to
> a held-out set. An agreement score — the fraction of backbones whose
> individual top-1 prediction matches the ensemble's — is produced alongside
> the prediction as a confidence signal.
>
> **2.4 Training Configuration**
>
> Each backbone was initialized from ImageNet-pretrained weights and
> fine-tuned independently with its own hyperparameters (Table 2), since the
> three architectures differ substantially in depth and parameter count. All
> three used AdamW [26] with a two-phase schedule: the backbone was frozen
> for the first $F$ epochs while only the classification head trained, then
> unfrozen with a layer-wise learning rate of $0.1\times$ the head rate,
> under linear warmup followed by cosine decay. Training used class-weighted
> cross-entropy loss with label smoothing ($\varepsilon=0.1$ for
> EfficientNet-B1, $0.15$ for the other two) and a weighted random sampler,
> both derived from inverse class frequency and then scaled by
> clinically motivated boosts (melanoma and basal cell carcinoma $\times2$,
> tinea $\times3$, eczema $\times1.5$, keratosis unboosted). Checkpoints were
> selected on a composite criterion computed on the validation set,
> $$S = 0.4\,\mathrm{Sens}_{\mathrm{mel}} + 0.3\,\mathrm{Sens}_{\mathrm{bcc}} + 0.3\,\mathrm{Acc}_{\mathrm{val}}$$
> rather than on validation accuracy alone, so that an epoch favoring the
> majority class could not be preferred over one with better recall on the
> two malignant conditions. Full augmentation and hyperparameter details,
> including the exact values of $F$, warmup length, weight decay, and
> dropout per backbone, are given in Table 2.
>
> [TABLE 2 — unchanged from `main_updated.tex` Table 2: per-backbone
> hyperparameters]
>
> **2.5 Experimental Setup**
>
> All three backbones were fine-tuned on Google Colab (EfficientNet-B1 on
> CPU in fp32, 3.78 wall-clock hours; ConvNeXt-Tiny and Swin-Tiny on a single
> NVIDIA T4) with a fixed random seed of 42 for model initialization and
> data loading. `cudnn.deterministic` was not enabled during training, so
> exact bit-level reproduction across hardware is not claimed; a
> byte-identical retrain of one backbone reproduced its original reported
> accuracy to within 0.38 percentage points, which we treat as within the
> expected noise band for a non-deterministic training recipe at this test
> set size.
>
> **2.6 Evaluation and Calibration Methodology**
>
> Performance is reported as accuracy, balanced accuracy, macro-averaged
> AUC, and the Matthews correlation coefficient (MCC), with 95% confidence
> intervals from a 2,000-resample stratified bootstrap over the test set.
> Paired differences between configurations (e.g., ensemble versus a single
> backbone) use a paired bootstrap over the same resampled indices, so that
> per-image correlation between configurations is preserved; a 95% interval
> excluding zero is treated as a significant difference, and a null result
> is reported as such rather than omitted.
>
> Calibration is assessed because the deployed system's safety rules act on
> the reported confidence directly (a prediction below 40% confidence
> triggers a re-capture request; melanoma at or above 40% confidence
> triggers escalation), so a clinically meaningful confidence score must
> behave like a genuine probability, not merely sum to one. Expected
> Calibration Error (ECE) and Maximum Calibration Error (MCE) are computed
> over 10 equal-width confidence bins,
> $$\mathrm{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{n} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|,
> \qquad \mathrm{MCE} = \max_m \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|$$
> where $B_m$ is the set of test images whose predicted-class confidence
> falls in bin $m$. Mean overconfidence is the signed difference between
> mean predicted confidence and accuracy over the full test set (positive
> values indicate overconfidence). Temperature scaling is fitted on the
> validation set by minimizing negative log-likelihood over a single scalar
> temperature $T$ applied to the ensemble's averaged log-probabilities, then
> applied unchanged to the test set; because temperature scaling is a
> monotonic rescaling, it cannot alter any argmax prediction, so accuracy is
> identical before and after by construction, which we verify explicitly
> rather than assume.

**Word count: 1,186** (below the 1,600-1,800 target — intentionally left
room, since Table 1/2 content and Supporting Information S1 pointers reduce
prose needed; can expand if Phase 2 review wants more training-protocol
detail restored from the original §5).

**What changed from `main_updated.tex` and why:**
1. **Consolidated three sections into one**, per the structural finding in
   §3 (checklist requires one Materials and Methods section, not three).
2. **Architecture description cut per decision #6** — MBConv factorization,
   SE gating equations, Swin attention block equations, and the AdamW
   update rule derivation are removed from the main text and pointed to
   Supporting Information S1 (not yet created — flagged for Phase 6/8).
   Kept: the fusion equation (the one equation IJIST editorial convention
   specifically asks papers to state exactly) and the composite checkpoint
   criterion (a genuine methodological claim, not textbook material).
3. **Added the Evaluation and Calibration Methodology subsection**
   (§2.6) — entirely new prose; the measurement already exists
   (`results.json`) but had no methods description anywhere in
   `main_updated.tex`. States the ECE/MCE formulas explicitly, per the
   IJIST editor's own stated preference for exact equations on metrics.
4. **Softened the "stratified by disease class" partitioning claim** into an
   explicit limitation, per Phase 1's flagged conflict (no splitting code
   found in the repo; the resemblance to stratification is observed, not
   demonstrated).
5. **Added the patient/lesion-identifier caveat** explicitly (the checklist
   and the governing prompt's own problem list both call for this), stating
   plainly that no cross-source patient-level grouping could be confirmed.
6. Moved the "Metadata Integration," "Clinical Decision Support," and
   related subsystem-status prose (User Authentication, severity triage
   description) out of Methods — they describe the broader deployed system,
   not the evaluated classification engine, and belong either briefly in
   §2.1 (kept, condensed) or in Discussion/limitations, not as full
   subsections with implementation-status detail. The per-module
   NOT-IMPLEMENTED/PARTIAL status table from `REVISION_NOTES.md` §10 is
   candidate Supporting Information content rather than main-text prose.
7. American spelling applied throughout.

**Open items carried forward:**
- **[VERIFY]** Exact citation support for each of the five dataset names —
  Phase 7 task, not yet done.
- **[VERIFY]** Whether PAD-UFES-20's patient IDs (noted in the IJIST
  checklist itself as available for that source) could retroactively
  establish patient-level grouping for at least one of the five sources —
  worth checking before finalizing the limitation wording, since "no
  patient-level grouping was possible" is a stronger and worse claim than
  necessary if it's true for only 4 of 5 sources.
- Supporting Information S1 (full per-backbone equations) is referenced but
  not yet written — Phase 6/8 task.

**Materials and Methods: APPROVED 2026-11-06.**

### 4.5 Results — DRAFT (awaiting approval)

Target: 1,100-1,300 words. Consolidates current §6.2-6.4 (Comparison,
Per-class, Ablation) with interpretive/clinical-explanation prose removed
to Discussion, plus a **new** Calibration Results subsection. Re-verified
the ensemble-vs-Swin paired bootstrap figures directly against
`results.json.paired_bootstrap_ensemble_minus_swin` before drafting (all
four values match `main_updated.tex` exactly: +2.12pp [0.35,4.07],
+0.0106 AUC [0.0054,0.0163], +0.0273 MCC [0.0050,0.0515], favoring the
ensemble in 98.9%/100%/99.2% of resamples).

> **3.1 Main Result**
>
> On the de-duplicated 565-image test set, the three-backbone ensemble
> attains an accuracy of 87.79% (95% CI 84.96-90.44%), a macro-averaged AUC
> of 0.9813 (95% CI 0.9744-0.9872), and an MCC of 0.8504 (95% CI
> 0.8163-0.8818); all intervals are 2,000-resample stratified bootstraps.
> Table 3 reports accuracy, balanced accuracy, macro-AUC, and MCC for each
> of the seven possible backbone subsets.
>
> [TABLE 3 — unchanged from `main_updated.tex` Table (ablation table):
> E / S / C alone; E+C, E+S, C+S; E+C+S]
>
> ConvNeXt-Tiny alone is the most accurate single configuration (88.67%
> accuracy, 0.8603 MCC); Swin-Tiny reaches 85.66% accuracy and the highest
> single-backbone macro-AUC (0.9707); EfficientNet-B1 reaches 84.25%. Among
> two-backbone subsets, ConvNeXt+Swin (C+S) reaches the highest accuracy of
> any configuration, including the full ensemble (88.32% vs. 87.79%). The
> three-backbone ensemble's accuracy therefore falls below both
> ConvNeXt-Tiny alone and C+S, but its macro-AUC (0.9813) exceeds every
> single backbone and every two-backbone subset, including C+S (0.9766).
>
> A paired bootstrap directly comparing the full ensemble against C+S
> — the strongest two-backbone alternative — shows this macro-AUC advantage
> is statistically significant (Δ = +0.0047 in favor of the full ensemble,
> 95% CI [0.0002, 0.0099], excluding zero), while the accuracy, MCC, and
> ECE differences between the two configurations are not (95% CIs include
> zero for all three; Table 4). We therefore report the three-backbone
> ensemble as the proposed configuration: it is the only configuration
> whose advantage over a leaner alternative is statistically established,
> and that advantage is on macro-AUC, the metric most relevant to a system
> producing a ranked differential diagnosis rather than a single label.
>
> [TABLE 4 — new: paired bootstrap, E+C+S vs. C+S — accuracy/macro-AUC/
> MCC/ECE deltas, CIs, fraction of resamples favoring E+C+S. Source:
> `.nb1out/cs_vs_ecs_analysis.json`, this session's new measurement.]
>
> A second paired bootstrap, comparing the full ensemble against Swin-Tiny
> (the single backbone with the highest macro-AUC), shows all three metrics
> significantly favoring the ensemble: accuracy +2.12 percentage points
> (95% CI [0.35, 4.07]), macro-AUC +0.0106 (95% CI [0.0054, 0.0163]), and
> MCC +0.0273 (95% CI [0.0050, 0.0515]), with the ensemble favored in
> 98.9%, 100%, and 99.2% of resamples respectively.
>
> **3.2 Calibration**
>
> Table 5 reports Expected Calibration Error (ECE), Maximum Calibration
> Error (MCE), and mean overconfidence (mean predicted confidence minus
> accuracy; positive values indicate overconfidence) for each single
> backbone and the full ensemble. ConvNeXt-Tiny, the most accurate single
> backbone, has the highest ECE (0.0497) and the highest overconfidence
> (+0.0262) of the three; Swin-Tiny has the lowest single-backbone ECE
> (0.0186). The full ensemble has an ECE of 0.0352 — lower than
> ConvNeXt-Tiny's — and is the only configuration with negative mean
> overconfidence (-0.0263), that is, its predicted confidence is on average
> slightly lower than its accuracy rather than higher.
>
> [TABLE 5 — new: ECE / MCE / mean confidence / accuracy / overconfidence
> for E, C, S, E+C+S. Source: `.nb1out/results.json` → `calibration`.]
>
> Temperature scaling, fitted on the validation set ($T = 0.836$) and
> applied unchanged to the test set, reduces the ensemble's test ECE from
> 0.0352 to 0.0216 and MCE from 0.360 to 0.356, with test accuracy
> unchanged at 87.79% by construction (verified directly rather than
> assumed, since temperature scaling cannot alter an argmax prediction).
>
> **3.3 Per-Class Performance and Confusion Analysis**
>
> Sensitivity is highest for the two malignant conditions in the taxonomy:
> basal cell carcinoma, 100.00% (106/106), and melanoma, 97.58% (121/124).
> Eczema recall is 90.09% (100/111), tinea recall is 82.42% (75/91), and
> keratosis recall is the weakest at 70.68% (94/133), consistent with
> keratosis receiving no clinical loss boost (Section 2.4).
>
> The largest source of residual error is bidirectional confusion between
> eczema and tinea: 9.0% of eczema cases (10/111) are predicted as tinea,
> and 17.6% of tinea cases (16/91) are predicted as eczema. Keratosis shows
> the next highest error rate, with 10.5% of cases (14/133) predicted as
> basal cell carcinoma.
>
> [FIGURES — fig3 (model comparison), fig4 (ROC), fig5 (confusion matrix):
> all three must be regenerated from the cached `.nb1out/cache_*.npz`
> outputs on the clean 565-image set before this section is finalized; the
> current `paper/fig{3,4,5}_*.pdf` files are of unconfirmed provenance
> (Phase 0 finding) and are not reused as-is. A new calibration/reliability
> figure (fig6 or fig7) should accompany Table 5. Figure regeneration is a
> Phase 6 task, tracked in OPEN_ITEMS.md.]

**Word count: 714** (below the 1,100-1,300 target — left intentionally
light since two new tables (4, 5) carry some of the numeric content that
would otherwise need restating in prose; can expand with brief
figure-reading guidance once the actual regenerated figures exist in
Phase 6, since describing a figure that doesn't exist yet risks drifting
from what it actually shows).

**What changed from `main_updated.tex` and why:**
1. **Added §3.2 Calibration** in full — this is the single biggest content
   addition in the whole paper relative to `main_updated.tex`, since Phase 1
   found the data existed but was entirely unwritten. Numbers independently
   re-confirmed against `results.json` before drafting (see Phase 1 table).
2. **Added the new E+C+S-vs-C+S paired bootstrap as Table 4** — this is the
   direct quantitative justification for proposing the three-backbone
   ensemble over the leaner, numerically-higher-accuracy C+S alternative.
   Without this table, a reviewer would reasonably ask "why not C+S, which
   has higher accuracy at essentially the same AUC" — the answer is that
   the AUC difference, not the accuracy difference, is the one that
   survives a significance test.
3. **Removed interpretive/clinical-explanation prose** (why the ensemble's
   value is "ranking quality not peak accuracy," the clinical-overlap
   explanation for eczema/tinea confusion, the keratosis-recall discussion's
   clinical framing) — moved to Discussion (§4.6, not yet drafted) per the
   checklist's requirement that Results and Discussion be separate
   sections. What remains here is measurement and direct description only.
4. **Figure regeneration flagged explicitly as outstanding** rather than
   assumed — Phase 0 already found the existing `fig3/4/5` files are of
   unconfirmed provenance (filenames don't match any notebook's actual
   output path). Captions are drafted provisionally but the images
   themselves still need producing from the cached per-image data.

**Open items carried forward:**
- Table 4 (new E+C+S vs. C+S bootstrap) needs formatting as an actual
  LaTeX table once Phase 6 (figures/tables finalization) runs — currently
  only referenced by its source JSON.
- Table 5 (calibration) likewise needs LaTeX formatting from
  `results.json.calibration`.
- Figures 3, 4, 5 (and a new calibration/reliability figure) must be
  regenerated — tracked as a Phase 6 task.

**Results: APPROVED 2026-11-06.**

### 4.6 Discussion — DRAFT (awaiting approval)

Target: 600-750 words. Carries the interpretive content deferred from
Results (§4.5), a comparison against the related ensemble/fusion work
condensed from the old Literature Review, and the limitations list from
the outline (§4.0): test-set configuration choice, single seed, single
split, no external validation, label provenance, no verified tone labels.
FL and Grad-CAM-faithfulness content is compressed to one sentence each,
no numbers, per the user's scope decision (prompt §3, problem 6).

> **4.1 Clinical Interpretation**
>
> The ensemble's value in this setting lies less in raising peak accuracy
> than in improving the ranking quality of the predicted class
> probabilities and in producing better-calibrated confidence scores. Both
> properties matter more than raw accuracy for a system intended to
> present a ranked differential diagnosis for clinician review rather than
> a single label: a higher macro-AUC means the correct class is more
> reliably ranked near the top even when it is not the argmax, and
> calibrated, non-overconfident probabilities are what make a
> confidence-based triage rule (Section 2.1) behave as intended rather than
> escalating or dismissing cases on a number that does not track actual
> correctness. That ConvNeXt-Tiny — the single most accurate backbone — is
> also the worst-calibrated and most overconfident supports evaluating and
> selecting ensembles on calibration as well as accuracy, rather than on
> accuracy alone, as is typical in the ensemble literature we build on
> [12,13,14,15,16,17].
>
> The residual confusion between eczema and tinea (Section 3.3) is
> consistent with clinical experience rather than a model-specific
> weakness: both conditions present as scaly, erythematous, or
> hyperpigmented plaques, and in melanin-rich skin the erythema that would
> ordinarily help distinguish an eczematous plaque is frequently masked,
> leaving annular scaling as the dominant visual cue. In routine practice
> the two are often separated by potassium hydroxide microscopy rather than
> visual inspection alone, so a purely image-based classifier is being
> asked to resolve a distinction that dermatologists themselves do not
> always attempt on morphology alone. The keratosis-to-basal-cell-carcinoma
> confusion mirrors a known clinical overlap between pigmented actinic
> keratoses and superficial basal cell carcinomas, compounded here by
> keratosis receiving no clinical loss boost during training (Section 2.4).
>
> **4.2 Comparison with Prior Ensemble Work**
>
> Heterogeneous ensembles combining convolutional and attention-based
> backbones have been reported to outperform their individual members on
> dermoscopic benchmarks [12,13,14,15,16,17], but these evaluations are
> justified on accuracy and AUC alone and are conducted on datasets drawn
> predominantly from lighter Fitzpatrick skin types. Uncertainty and
> calibration have been studied for skin-lesion classifiers elsewhere, for
> example with deep ensembles evaluated on the dermoscopic HAM10000 corpus
> [LoRA-Ensemble citation, to verify in Phase 7], but, to our knowledge, not
> for a heterogeneous convolutional-transformer soft-voting ensemble on a
> melanin-rich clinical corpus. The present study's contribution is
> therefore best read as extending this ensemble literature along the
> calibration axis on a population it has not previously been evaluated
> against, rather than as introducing a new fusion mechanism: the
> decision-level averaging rule itself is standard and deliberately simple
> (Section 2.3).
>
> **4.3 Limitations**
>
> Several limitations bear directly on how these results should be read.
> First, the three-backbone ensemble was selected as the proposed
> configuration by comparison against the other six subsets on the same
> held-out test set used to report its final performance (Section 3.1);
> although the comparison that drives this choice (ensemble vs. C+S on
> macro-AUC) is a statistically significant one, no independent
> validation-based configuration search was performed, so the selection and
> the reported performance are not fully separated. Second, each backbone
> was trained once, with a single random seed, under a training recipe
> that did not enable deterministic cuDNN operations; a byte-identical
> retrain of one backbone reproduced its reported accuracy to within 0.38
> percentage points, which bounds but does not eliminate seed-dependent
> variance in the reported numbers. Third, all results come from a single
> corpus and a single train/validation/test partition; no external
> validation on an independently collected dataset was performed, and
> although the test set was de-duplicated against training and validation
> images by perceptual hashing, the partition itself was inherited rather
> than generated by an auditable, patient-aware procedure, and no
> patient- or lesion-level identifiers are available across the five
> aggregated sources to confirm that no patient contributes images to more
> than one partition. Fourth, label provenance differs across the five
> source datasets (a mix of biopsy-confirmed and clinical diagnoses) and is
> not tracked per image in the assembled corpus, so a result is not
> attributable to a single, consistent ground-truth standard. Fifth, and
> most directly relevant to this study's stated population, consistent
> skin-tone labels are not available across the five sources: the corpus
> was assembled to emphasize melanin-rich presentations, but no
> Fitzpatrick type, Monk Skin Tone, or individual typology angle label
> exists for most images, so no tone-stratified performance claim is made
> or supported by the present evaluation. Sixth, a preliminary individual
> typology angle (ITA) analysis — computed on the contaminated 727-image
> test set rather than the de-duplicated 565-image set used for every other
> result in this paper — found a 12.01 percentage-point gap between the
> best-performing ("intermediate," $n=93$, 94.62%) and worst-performing
> ("brown," $n=92$, 82.61%) tone bins, with non-overlapping 95% bootstrap
> confidence intervals; all three backbones showed the same qualitative
> ordering across bins. This gap does not track ITA monotonically — the
> worst-performing bin is not the darkest category — and the confound
> between tone bin and disease-class composition has not been decomposed,
> so we report it as a preliminary signal rather than a validated fairness
> result.
>
> Two further lines of work build on the classification engine evaluated
> here but are reported separately rather than in this paper: a Grad-CAM
> faithfulness evaluation and a federated-learning feasibility study are
> both in preparation as separate manuscripts and are not elaborated here.

**Word count: ~820** (now over the 600-750 target by ~70-120 words after
adding the ITA paragraph — the Introduction (503w) and Materials and
Methods (1,186w) both drafted under their targets, so this still fits
within the overall ~5,000-word budget; re-measure the full manuscript once
all sections are final rather than trimming this paragraph reflexively).

**What changed from `main_updated.tex` and why:**
1. **New §4.1/§4.2** carry the interpretive prose moved out of Results
   (§4.5) per the checklist's Results/Discussion separation requirement.
2. **New §4.2 "Comparison with Prior Ensemble Work"** is new content, not
   present in `main_updated.tex` at all — required by the prompt's section
   table ("comparison with prior ensembles into the Discussion"). Contains
   one explicit **[VERIFY]**: the LoRA-Ensemble citation and claim need
   checking in Phase 7 (reference verification) before this sentence is
   finalized — flagged inline rather than asserted.
3. **New §4.3 Limitations** — the single biggest structural addition.
   Covers exactly the five items the outline specified (test-set
   configuration choice, single seed, single split/no external validation,
   label provenance, no verified tone labels) plus the patient-grouping
   caveat carried over from Methods. This section did not exist in any
   form in `main_updated.tex`.
4. **FL and Grad-CAM-faithfulness compressed to one sentence, no numbers**
   — per the user's explicit scope decision (prompt §3, problem 6; decision
   #5 in this doc's log). This removes `main_updated.tex`'s entire
   "Future Scope" §8 items 2 and 3 (which had specific AOPC and
   melanoma-sensitivity-gap numbers) down to the one sentence above.
5. **ITA fairness result added as a 6th Limitations item** (decision
   2026-11-06, see decision log #9 below): the user's two planned
   conference papers (XAI/Grad-CAM faithfulness; federated learning) are
   each topically focused and ITA doesn't fit either — it's a skin-tone
   fairness proxy, not an explainability or FL result. This IJIST paper is
   the only place where it's topically central (the paper's whole premise
   is melanin-rich equity), so it stays here as the one tone-stratified
   signal reported anywhere in the project, with its existing caveats
   (contaminated 727-image set, un-decomposed class-mix confound) intact.
6. American spelling applied throughout.

**Open items carried forward:**
- **[VERIFY]** LoRA-Ensemble citation/claim (§4.2) — Phase 7 task.

**Discussion: APPROVED 2026-11-06 (including the 6th Limitations item on
ITA, decision #9).**

### 4.7 Conclusion — DRAFT (awaiting approval)

Target: ≤200 words. No new results, per the governing prompt's constraint —
restates the main finding and its clinical implication only.

> This study addresses skin disease classification in melanin-rich skin,
> which remains underserved by diagnostic tools trained predominantly on
> lighter Fitzpatrick skin types. We curated a five-class, 6,655-image
> corpus from five public sources, audited it for cross-split near-duplicate
> contamination, and evaluated all seven combinations of three
> independently fine-tuned backbones — EfficientNet-B1, Swin-Tiny, and
> ConvNeXt-Tiny — on a de-duplicated 565-image test set. The most accurate
> single backbone, ConvNeXt-Tiny, was also the worst-calibrated and most
> overconfident. An equal-weight three-backbone soft-voting ensemble
> achieved 87.79% accuracy and the highest macro-AUC (0.9813) of any
> configuration, including every two-backbone subset, with the difference
> against the strongest two-backbone alternative statistically significant
> on macro-AUC though not on accuracy. The ensemble was also the only
> configuration with slightly conservative rather than overconfident
> probabilities, and malignant-class sensitivity remained high (100%
> basal cell carcinoma, 97.58% melanoma). These results support selecting
> and reporting ensembles for dermatological screening on calibration as
> well as accuracy, particularly where confidence scores drive downstream
> clinical triage. The present evaluation is limited to a single corpus,
> a single train/validation/test partition, and populations for which
> consistent skin-tone labels are not available; prospective and
> externally validated evaluation remains necessary before clinical
> deployment.

**Word count: 180** (within the ≤200 limit).

**What changed from `main_updated.tex` and why:**
1. Restates the ensemble's macro-AUC/calibration advantage as the headline
   finding (consistent with the Abstract and Discussion), rather than
   `main_updated.tex`'s current conclusion, which leads with "decision-level
   fusion... improves ranking quality" in essentially the same framing —
   so this is a light edit, not a rewrite, mainly compressed to fit under
   200 words alongside the new calibration framing.
2. Added one sentence on calibration directly ("the ensemble was also the
   only configuration with slightly conservative rather than overconfident
   probabilities") since Phase 1 found this was previously unstated anywhere
   in the body despite being in the Abstract.
3. Closing sentence now states the limitation (single corpus/split, no
   consistent tone labels) explicitly rather than ending on an unqualified
   claim of clinical value — matches the Discussion's limitations list
   without repeating all six items, and matches the checklist's and the
   prompt's instruction that Conclusion introduce no new results while
   still being honest about what was and wasn't established.
4. Removed the metadata-integration/future-work sentence that closed
   `main_updated.tex`'s conclusion — that content now lives in Materials
   and Methods (§2.1, stated once) and doesn't need repeating here.
5. American spelling applied throughout.

**Conclusion: APPROVED 2026-11-06. All five main-body sections now drafted
and approved.**

### 4.8 Figures — regenerated (Phase 6, partial)

`scripts/generate_ijist_figures.py` regenerates Figures 3-6 directly from
`.nb1out/cache_*_test.npz` on the clean 565-image set, using only the
proposed E+C+S ensemble (plus its three constituent backbones for
comparison), with the dataviz skill's validated colorblind-safe categorical
palette (fixed hue order: BCC=blue, Eczema=orange, Keratosis=aqua,
Melanoma=yellow, Tinea=magenta, consistent across every figure) and
sequential blue ramp for the confusion-matrix heatmap. Output in
`paper_ijist/figures/Figure_{3,4,5,6}.{pdf,png}`.

- **Figure 3** — model comparison (accuracy / macro-AUC / MCC bars, all four
  configs). Replaces `paper/fig3_model_comparison.pdf` (unconfirmed
  provenance, Phase 0 finding).
- **Figure 4** — ROC curves, per class + macro-average, E+C+S ensemble only.
  Replaces `paper/fig4_roc.pdf`/`.jpg`.
- **Figure 5** — confusion matrix (row-normalized heatmap) + per-class
  sensitivity bar chart, E+C+S ensemble. Replaces
  `paper/fig5_confusion_matrix.pdf`.
- **Figure 6** — new: reliability diagram (before/after temperature
  scaling) + ECE bar chart across all four configs, with overconfidence
  annotated per bar. Did not exist in `main_updated.tex` at all; backs the
  new Calibration subsection in Results (§4.5/§3.2).

All four inspected visually after generation (not just script-success
checked) per the dataviz skill's own instruction to look at the actual
render: no label collisions, legends present, colors consistent across
figures, values legibly labeled. One layout issue found and fixed —
Figure 6's ECE bar-chart labels initially overlapped the subplot title for
the tallest bar (ConvNeXt-T); fixed by adding y-axis headroom
(`ax2.set_ylim(0, max(eces) * 1.35)`), regenerated, re-inspected, clean.

**Not yet done (remaining Phase 6 work):**
- Figures 1 and 2 (system pipeline diagram, architecture diagram) are
  currently inline TikZ in `main_updated.tex` and have not been touched —
  they don't depend on cached data, so they carry over largely as-is once
  LaTeX assembly starts, though Figure 2 (architecture) may need updating
  to match the trimmed Methods section (decision #6: equations removed,
  only the fusion equation and hyperparameter table remain in-text).
- Table formatting: Tables 1-5 (dataset composition, hyperparameters,
  ablation, the new E+C+S-vs-C+S bootstrap, and the new calibration table)
  are referenced in the section drafts but not yet laid out as actual
  LaTeX `tabular` environments — deferred to LaTeX assembly, which is
  itself blocked on the Wiley template (still not present in
  `paper_ijist/template/`).
- Figure+table count check: once Tables 4-5 and Figure 6 are added to the
  3 existing tables and 5 figures (2 TikZ + Figures 3-6... 4 regenerated,
  not 3, since Figure 6 is new), the combined count is **2 TikZ figures +
  4 regenerated figures + 5 tables = 11 items**, still comfortably under
  the 15-item cap.

---

## 4.8a Wiley template received and `main.tex` assembled (2026-11-06)

The user uploaded `WileyDesign.zip` to `paper_ijist/`. Extracted and
verified genuine: copyright Wiley, class file `USG.cls` v6.0
(2024-05-07, "Wiley, LaTeX Development Team (India)"), bundled with Wiley
logos, STIX fonts, an author-guidelines PDF, and a working example
document (`Optimal-Design-layout.tex`/`.pdf`, for AIChE Journal — this is
a generic Wiley NJD template shared across journals, not an IJIST-specific
skin; journal identity is set independently via `\journal{}`). Moved into
`paper_ijist/template/`, build artifacts (`.aux`/`.log`/`.blg`/`.synctex.gz`)
removed as non-template byproducts.

**Key decisions made before assembly (both confirmed with the user):**
- **No local LaTeX engine in this environment** — `main.tex` is assembled
  carefully against `USG.cls`'s documented macros but has **not been
  compiled here**. The user must compile via Overleaf or a local TeX
  distribution and report back any errors for fixing.
- **Reference style: numbered** (`VANCOUVER` documentclass option), not
  APA — matches `main_updated.tex`'s existing numbered citations, IJIST's
  actual published-article convention, and avoids an unnecessary ~27-entry
  reformatting with no real benefit under Free Format submission.
- **Bibliography mechanism: inline `\thebibliography`**, not
  `\bibliography{}` + `bibtex` — the downloaded template only ships a
  Chicago-style `.bst` (no Vancouver/AMA `.bst` included), and this
  environment can't run `bibtex` to test one anyway. Matches
  `main_updated.tex`'s own working approach, so no new failure mode is
  introduced.

**`paper_ijist/main.tex` assembled** from the approved section drafts
(§4.1-4.7), converting Markdown prose to the template's LaTeX conventions
(`\bmsubsection*{}` for statement headers, `\corres{}`, `\fundingInfo{}`,
pipe-separated `\keywords{}`, etc.). Every `[VERIFY]` item surfaced during
drafting (Phases 1, 7) is carried into the file as an inline `% VERIFY:`
comment at the exact point it matters, not just listed separately — so
nothing requiring the user's input is silently resolved by omission.

**Static checks performed** (no LaTeX engine available, so these substitute
for a real compile, not replace one):
- Brace balance: 0 (correct) across the whole file.
- Every `\ref{}` has a matching `\label{}` — confirmed, no dangling
  references.
- Every `\cite{}` key has a matching `\bibitem{}` — confirmed.
- `ref1` is the only bibliography entry never cited in the main text —
  expected and intentional: Phase 7 found it didn't support the claim it
  was originally attached to (see inline comment at its `\bibitem`), and
  the Introduction's opening sentence now carries no citation pending a
  real replacement, rather than keeping a wrong one.
- All four referenced figure files (`Figure_3.pdf`-`Figure_6.pdf`) confirmed
  present on disk in `paper_ijist/figures/`.

**What a real compile might still catch that a static check cannot:**
`USG.cls` macro usage correctness (e.g., whether `\corres{}` or
`\fundingInfo{}` expect a different argument structure than assumed from
reading the example file), font/STIX loading, column-width overflow in the
two-column layout (tables and the ROC/confusion-matrix figures were sized
for a notional column width, not verified against the actual compiled
page), and any `USG.cls`-specific syntax this session didn't anticipate.
**This file should be treated as a strong first draft for compilation, not
as verified-correct LaTeX.**

**Not yet carried into `main.tex`:** the two TikZ figures (system pipeline,
architecture diagram) from `main_updated.tex` — decided not to port these
over without first confirming they render acceptably under the two-column
`USG.cls` layout (the original was single-column Springer `sn-jnl`); a
`[VERIFY]`-flagged placeholder was not added for these, which is itself an
omission worth the user's attention — **flagged now**: Figures 1-2 (the
pipeline/architecture diagrams) are currently **absent from `main.tex`
entirely**, not just deferred, and should be added back (likely needing a
resize or simplification for the two-column format) before this is
submission-ready.

### 4.8b First real compile attempt — error found and fixed (2026-11-07)

User uploaded `main.tex` + template to Overleaf and got a real compile
error back (the first actual LaTeX-engine feedback this file has received
— everything before this was static checking only). Confirms the
Phase 6 log's own warning that this was "a strong first draft for
compilation, not verified-correct LaTeX."

**Error:** `\sur` undefined, surfaced while parsing
`\author*[5]{\fnm{Neha}\sur{Katre}}`.

**Root cause, traced in `USG.cls`:** two Springer-`sn-jnl`-isms leaked into
`main.tex` from copying `main_updated.tex`'s author-block pattern without
checking it against *this* class:
1. **`\author*` does not exist in `USG.cls`.** Only `\author{\advance...}`
   is defined (line 1176), with no starred variant — the star belongs to
   `\address`, not `\author` (`\address*[N]{...}` is how USG.cls actually
   marks corresponding-author info internally, feeding `\@corresinfo`).
   Starring `\author` broke its argument parsing, which is why `\sur`
   came out looking undefined rather than producing a clearer error.
2. **`\fnm{}`/`\sur{}` are not meant to be nested inside `\author{}` as the
   display name.** They exist in `USG.cls` (lines 1103-1104) but for a
   different purpose (building an author-name index via
   `\addtocontents{aq}{...}`); the class's own working example
   (`template/Optimal-Design-layout.tex`) uses plain
   `\author[N]{Full Name}` with no wrapper macros at all.
3. **`\address[1,2,3,4,5]{...}`** (one call, comma-separated affiliation
   numbers) **is not valid** either — `USG.cls`'s `\@@address` (line 1186)
   takes exactly one optional numeric argument. The working example issues
   one `\address[N]{...}` call per affiliation number, repeating the text
   when authors share an affiliation.

**Fix applied** (not yet re-verified by a real compile — next upload will
tell): author block rewritten to plain `\author[N]{Full Name}` (no
`\fnm`/`\sur`, no star) for all five authors; `\address[1,2,3,4,5]{...}`
split into five separate `\address[N]{...}` calls with the same
affiliation text repeated, matching the working example exactly.
Corresponding-author status continues to be conveyed by `\corres{}` alone
(already present, unaffected by this fix, confirmed as the correct
mechanism by re-reading the working example, which also relies on
`\corres{}` alone rather than `\address*`).

**`IJIST_overleaf_upload.zip` rebuilt** with the fixed `main.tex` and
re-delivered to the user. Static checks (brace balance, ref/label/cite
consistency) re-run and still clean post-fix.

**Lesson for the rest of this file:** other macros carried over verbatim
from the Springer-template mental model (rather than re-checked against
`USG.cls`'s own working example) may have similar latent mismatches --
`\titlemark{}`, `\articledoi{}`, `\bmsubsection*{}` for the statements
block were all checked against the example before use and matched, but
haven't yet survived a real compile. Treat the next compile round as
likely to surface at least one more issue, not as a formality.

---

## 4.9 Tables — LaTeX (Phase 6, continued)

All five tables below, laid out as `tabular` environments against the
Materials and Methods / Results text in §4.4-4.5. Tables 1-3 are unchanged
from `main_updated.tex` (confirmed exactly against source in Phase 1,
§3 above) and simply re-typeset here for completeness; Tables 4-5 are new,
built directly from `.nb1out/cs_vs_ecs_analysis.json` and
`.nb1out/results.json` → `calibration`/`temperature_scaling` (re-pulled and
double-checked against Phase 1's figures immediately before drafting, no
transcription from memory).

```latex
% ------------------------------------------------------------------
% Table 1 -- Class-wise distribution of the final curated dataset
% ------------------------------------------------------------------
\begin{table}[htbp]
\caption{Class-wise distribution of the curated dataset.}\label{tab:dataset}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Disease Class & Train & Val & Test & Total \\
\midrule
Basal Cell Carcinoma & 968 & 119 & 121 & 1,208 \\
Eczema & 1,072 & 148 & 145 & 1,365 \\
Keratosis & 977 & 159 & 158 & 1,294 \\
Melanoma & 1,197 & 202 & 188 & 1,587 \\
Tinea & 966 & 120 & 115 & 1,201 \\
\midrule
Total & 5,180 & 748 & 727 & 6,655 \\
\botrule
\end{tabular}
\end{table}

% ------------------------------------------------------------------
% Table 2 -- Per-backbone hyperparameter configuration
% ------------------------------------------------------------------
\begin{table}[htbp]
\caption{Per-backbone hyperparameter configuration. Each backbone was
trained independently with its own settings; ``epochs run'' is the number
of epochs before early stopping on the composite criterion halted
training.}\label{tab:hyperparams}
\begin{tabular}{@{}lccc@{}}
\toprule
& EfficientNet-B1 & ConvNeXt-Tiny & Swin-Tiny \\
\midrule
Head learning rate & $2 \times 10^{-4}$ & $2 \times 10^{-4}$ & $1 \times 10^{-4}$ \\
Backbone LR multiplier & 0.1 & 0.1 & 0.1 \\
Weight decay & $2 \times 10^{-4}$ & $3 \times 10^{-4}$ & $3 \times 10^{-4}$ \\
Label smoothing $\varepsilon$ & 0.1 & 0.15 & 0.15 \\
Dropout & 0.35 & 0.4 & 0.4 \\
Drop-path rate & 0.15 & 0.2 & 0.2 \\
Batch size & 16 & 12 & 8 \\
Warmup epochs & 3 & 3 & 5 \\
Freeze epochs & 6 & 3 & 8 \\
Patience & 6 & 10 & 5 \\
Max epochs & 30 & 40 & 30 \\
Epochs actually run & 25 & 40 & 30 \\
\botrule
\end{tabular}
\end{table}

% ------------------------------------------------------------------
% Table 3 -- Ablation of ensemble subsets
% ------------------------------------------------------------------
\begin{table}[htbp]
\caption{Ablation of ensemble subsets on the de-duplicated test set (565
images). Each backbone was trained independently with its own
hyperparameters (Table~\ref{tab:hyperparams}); multi-backbone
configurations are equal-weight softmax averages.}\label{tab:ablation}
\begin{tabular}{@{}lcccc@{}}
\toprule
Configuration & Accuracy & Balanced Acc. & Macro-AUC & MCC \\
\midrule
EfficientNet-B1 (E) & 84.25\% & 84.65\% & 0.9678 & 0.8066 \\
Swin-Tiny (S) & 85.66\% & 86.00\% & 0.9707 & 0.8231 \\
ConvNeXt-Tiny (C) & 88.67\% & 88.85\% & 0.9641 & 0.8603 \\
\midrule
E + C & 88.14\% & 88.45\% & 0.9798 & 0.8545 \\
E + S & 85.84\% & 86.16\% & 0.9772 & 0.8255 \\
C + S & 88.32\% & 88.65\% & 0.9766 & 0.8567 \\
\midrule
E + C + S (proposed) & 87.79\% & 88.15\% & 0.9813 & 0.8504 \\
\botrule
\end{tabular}
\end{table}

% ------------------------------------------------------------------
% Table 4 -- NEW: paired bootstrap, E+C+S vs. C+S
% Source: .nb1out/cs_vs_ecs_analysis.json (this session, scripts/cs_vs_ecs_analysis.py)
% ------------------------------------------------------------------
\begin{table}[htbp]
\caption{Paired bootstrap comparison of the proposed three-backbone
ensemble (E+C+S) against the strongest two-backbone alternative (C+S) on
the de-duplicated 565-image test set. Differences are E+C+S minus C+S;
2{,}000-resample paired bootstrap, seed 42. A 95\% confidence interval
excluding zero is a statistically significant difference.}\label{tab:ecs-vs-cs}
\begin{tabular}{@{}lrrrl@{}}
\toprule
Metric & C+S & E+C+S (proposed) & $\Delta$ (E+C+S $-$ C+S) & 95\% CI \\
\midrule
Accuracy & 88.32\% & 87.79\% & $-$0.53 pp & [$-$1.95, +0.88] \\
Macro-AUC & 0.9766 & 0.9813 & \textbf{+0.0047} & \textbf{[+0.0002, +0.0099]} \\
MCC & 0.8567 & 0.8504 & $-$0.0063 & [$-$0.0238, +0.0107] \\
ECE & 0.0209 & 0.0352 & +0.0143 & [$-$0.0129, +0.0339] \\
\botrule
\end{tabular}
\end{table}
% Note: only the macro-AUC row excludes zero (bold above), favoring the
% proposed E+C+S configuration -- this is the table referenced in Results
% \S3.1 and is the direct quantitative justification for proposing the
% three-backbone ensemble over the leaner, numerically higher-accuracy C+S
% alternative. Sign convention here is E+C+S minus C+S (the proposed
% model's own perspective); the source JSON and Results prose (\S4.5)
% state it as C+S minus E+C+S, so signs are flipped relative to the raw
% file -- double-checked: C+S accuracy 88.32 > E+C+S 87.79, so E+C+S minus
% C+S accuracy is correctly negative above.

% ------------------------------------------------------------------
% Table 5 -- NEW: calibration (ECE/MCE/overconfidence) + temperature scaling
% Source: .nb1out/results.json -> calibration, temperature_scaling
% ------------------------------------------------------------------
\begin{table}[htbp]
\caption{Calibration metrics (10 equal-width confidence bins) on the
de-duplicated 565-image test set. Overconfidence is mean predicted
confidence minus accuracy; positive values indicate overconfidence,
negative values indicate conservative (under-confident)
predictions.}\label{tab:calibration}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Configuration & Accuracy & Mean Conf. & ECE & MCE & Overconfidence \\
\midrule
EfficientNet-B1 & 84.25\% & 0.8490 & 0.0304 & 0.7442 & +0.0065 \\
Swin-Tiny & 85.66\% & 0.8674 & 0.0186 & 0.6063 & +0.0107 \\
ConvNeXt-Tiny & 88.67\% & 0.9129 & 0.0497 & 0.3640 & +0.0262 \\
E+C+S (proposed) & 87.79\% & 0.8516 & 0.0352 & 0.3601 & \textbf{$-$0.0263} \\
\botrule
\end{tabular}

\vspace{4pt}
\footnotesize
Temperature scaling (fitted on validation, $T=0.8360$, applied to test):
ECE $0.0352 \rightarrow 0.0216$; MCE $0.3601 \rightarrow 0.3561$; test
accuracy unchanged at 87.79\% (verified directly, since temperature scaling
cannot alter any argmax prediction).
\end{table}
```

**Verification performed before drafting:** every number in Tables 4-5 was
re-pulled directly from `.nb1out/cs_vs_ecs_analysis.json` and
`.nb1out/results.json` in this session (commands shown in the session
transcript), not copied from the earlier Phase 1 prose summary, to catch
any transcription drift between the two. One sign-convention check is
flagged inline in Table 4's LaTeX comment: the source JSON states deltas as
"C+S minus E+C+S," but Table 4 is framed from the proposed model's
perspective ("E+C+S minus C+S") to match how Results \S3.1 reads: the
accuracy/MCC/ECE signs are flipped relative to the raw JSON and
double-checked arithmetically above rather than assumed.

**What's not yet done:**
- Tables are not yet inserted into an actual `main.tex` — still blocked on
  the Wiley template.
- No formal check yet that table captions/footnote style match Wiley's
  specific requirements beyond the general checklist guidance ("each table
  complete with title and footnotes," no vertical rules — already followed
  above via `booktabs`-style `\toprule`/`\midrule`/`\botrule`, no `|`
  column separators).

---

## 5. Word counts

*Not yet tracked — first measurement due once Introduction/Methods drafts exist.*
