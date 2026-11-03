# CLAUDE.md

---

# SkinSure — AI-Powered Skin Disease Screening for Indian Skin Tones

> A clinical decision support system built specifically for Fitzpatrick IV–VI skin types, combining a deep learning ensemble, explainability tooling, and an agentic layer to guide patients from image capture to care.

---

## Documentation map (keep these consistent)

| File | Role |
|---|---|
| `CLAUDE.md` (this file) | **Source of truth** for current state, conventions and design invariants |
| `docs/architecture.md` | Target folder layout and import map |
| `docs/ROADMAP.md` | Phase-by-phase build plan (phases 0–12) |
| `fedler.md` | **Source of truth for federated learning and differential privacy** (overrides any other doc). **Parked 2026-11-02** pending a re-run on a different dataset |
| `paper/PUBLISHABLE_FINDINGS.md` | Results flagged as paper material, each marked PROVISIONAL or CONFIRMED. Nothing marked PROVISIONAL may be written up as an effect |
| `src/models/CLAUDE.md` | Model notebooks, training decisions, checkpoint format |
| `QnA.md` | Learning log; not normative |

If a doc disagrees with this file or with `fedler.md`, fix the doc. When code changes the state below, update this file in the same change.

---

## Current Implementation State

Last audited 2026-10-31 against the files on disk.

**Implemented:**
- Schemas — `api/schemas/patient.py` (`PatientMetadata`, Fitzpatrick IV–VI gate) and `api/schemas/predictions.py` (`Predictions`, `XAISummary`, `Severity`, `ConfidenceLevel`, `ClinicalSummary`; validator enforces `urgent → immediate_referral`)
- Model layer — `src/models/ensemble.py` (`SkinEnsemble`, `CheckpointPaths`), `efficientnet.py`, `convnext.py`, `swin.py`; training/eval notebooks in `src/models/`. **The app loads the v2 retrains** (`models/checkpoints/{efficientNet,convNext,swin}/*_5cls_skin_v2.pth`, set in `configs/models.yaml`) and **generations must never be mixed across backbones**. **There is no fallback and never was** — `get_checkpoint_paths()` returns exactly those three paths, so a missing checkpoint raises rather than degrading. The v1 `*_5cls_skin.pth` files and an older `best_*_5cls.pth` set were **deleted on 2026-11-02** (~479 MB): no code, config or test referenced them, and v1 is the generation that fails the save-then-verify gate, so keeping it as a "safety net" meant holding a known-defective copy. They were gitignored, so this is not recoverable from git — retrain if ever needed. Note `paper/experiments/01_eval_and_ablation.ipynb` does resolve v1 filenames as a last resort, but only under its Kaggle-side `CKPT_ROOT`, never these local paths, and it tries v2 first precisely so a stale original cannot reinstate the unreproducible paper numbers. Switching to v2 is mostly a ConvNeXt fix — on the contaminated 727-image paper test set, EfficientNet 85.11% → 84.73%, Swin 85.39% → 85.97%, **ConvNeXt 71.35% → 89.13%**, which makes ConvNeXt the strongest single backbone rather than the weakest. On the de-duplicated 565-image set those become 84.25% / 85.66% / 88.67%, with the 3-model ensemble at 87.79% (macro-AUC 0.9813). **Three different test sets are in play and must never be compared with each other:** 997 images (the federated-learning split, `fedler.md`), 727 (the paper's contaminated test set) and 565 (the same set after pHash de-duplication removed 162, the primary basis for reported metrics). Always state which one a figure comes from.
- Preprocessing — `src/preprocessing/`: `inference_transform.py` (224px resize + ImageNet normalisation, `preprocess_image`), `quality_check.py`, `hair_removal.py`, `artifact_removal.py`, `color_normalization.py`, `pipeline.py` (`preprocess()`), thresholds and step switches in `configs/preprocessing.yaml`
- `scripts/predict_one.py` — single-image inference through the preprocessing pipeline (works)
- Clinical summarizer — `src/agents/clinical_summarizer.py` (`build_context`, async `summarize`, post-processing); demo in `scripts/summarize_one.py`
- Severity triage — `src/agents/severity_triage.py` (`triage(predictions, summary, patient) -> TriageResult`; all three hard overrides, plus **Rule F4**, a config-gated prototype that is OFF by default — see "Severity triage hard overrides")
- Orchestrator and API — `src/agents/orchestrator.py` (`run_screening(image, patient, *, ensemble, explainer, client) -> ScreeningResult`: quality gate → ensemble → explainer → summarizer → triage; statuses `completed` / `rejected` / `blocked`; `treatment_advice_allowed` is False for urgent, referral and re-capture cases) and `api/main.py` (`create_app(ensemble, explainer, client)`; `GET /health`, `POST /predict` taking a multipart `image` + `metadata` JSON string). The explainer is injected; when the ensemble is a real `SkinEnsemble` and none is passed, `create_app` defaults to `GradCamExplainer`. With no explainer available `/predict` returns 503 rather than invent explainability data.
- Explainability — `src/explainability/`: `gradcam.py` (`grad_cam`, `ensemble_heatmap`; per-backbone target layers `bn2` / `stages[-1]` / `layers[-1]`, Swin is NHWC), `xai_summary.py` (`summarize_heatmap` heatmap → `XAISummary`, `estimate_lesion_mask`), `explainer.py` (`GradCamExplainer`, the app's default), `evaluation.py` (deletion curve / AOPC for any heatmap); `scripts/evaluate_xai.py` compares Grad-CAM with external heatmaps (e.g. SHAP `.npy`) and a random baseline under one procedure
- Config — `src/utils/config.py` (`get_settings()` for `configs/agents.yaml`; `get_model_config()` / `get_checkpoint_paths()` for `configs/models.yaml`; `get_preprocessing_config()` for `configs/preprocessing.yaml`), `configs/agents.yaml`, `configs/models.yaml` (checkpoint paths and ensemble weights)
- Cost accounting and spend gate — `src/utils/costs.py` (`cost_usd()` prices one response's `usage` from the table in `configs/llm.yaml`; Haiku 5.5 is tiered by prompt length, so a prompt over `long_prompt_threshold_tokens` prices 5× on every category, and cache reads/writes count toward that length) and `scripts/_guard.py` (`SpendGuard`: a script that calls a paid API refuses to run without both `SKINSURE_ALLOW_SPEND=1` and an explicit `--max-usd`; neither has a money-spending default)
- Packaging — `pyproject.toml`, `.env.example`, `.gitignore`
- **Federated learning (Phases 0–2)** — `src/training/train_utils.py` (reusable training loop: `train_one_epoch`, `evaluate`, `create_model`, transforms, dataloaders, loss weighting), `src/federated/client.py` (`SkinSureClient` NumPyClient + `ClientApp`), `src/federated/server.py` (`weighted_average` aggregation + `ServerApp`), `scripts/run_fl_simulation.py` (FL simulation via `start_simulation()` with `FedAvgWithCheckpoint`), `scripts/train_baseline.py` (centralized baseline trainer), `scripts/make_federated_shards.py` (IID/non-IID shard generation). Data splits in `data/splits/` (`global_test.csv`, `trainable_pool.csv`, `shards/iid/`, `shards/noniid/`). Dataset: 6,655 images, 5 classes. **Both measured on the same 997-image global test set:** centralized baseline 83.15% acc / 98.32% melanoma sens / 0.9162 composite; FL (FedAvg, 10 rounds, non-IID) 79.94% acc / 97.06% melanoma sens / 0.8999 composite — privacy costs ~3.2 points of accuracy, concentrated in Keratosis (-13.4), while melanoma sensitivity holds within 1.3 points. See `fedler.md` for full per-class results.

**Known gaps:**
1. `tests/unit/` covers schemas, severity triage (every override branch), the clinical summarizer (fake client), the orchestrator, preprocessing and explainability (toy models and synthetic scenes); `tests/integration/` covers `/predict` and `/health` with fakes, plus one real-checkpoint run (skipped when checkpoints are absent). 250 tests pass (run with `./venv/Scripts/python.exe -m pytest tests/`, ~5 min; the system Python lacks torch/cv2/anthropic/fastapi). `tests/unit/test_xai_evaluation.py` additionally pins the AOPC experiment's batched deletion scorer against the reference `deletion_curve` on a fake ensemble, so the speed optimisation in `scripts/xai_aopc_experiment.py` cannot silently diverge. No `tests/fairness/` yet. Nothing in the suite touches the real LLM, a real lesion photo, or real-data accuracy. `FakeClient` accepts `**kwargs` and ignores them, so it cannot catch a request parameter the real API would reject — `test_request_carries_system_prompt_and_context` asserts on the request shape (no `temperature`, `output_config` present) to cover that blind spot.
2. ~~`summarize_one.py` has not been run against the real Anthropic API.~~ **Done 2026-10-31**: all three `--case` values ran against Claude Haiku 5.5 and returned valid, parseable JSON. Confirmed live: `output_config={"effort": "low"}` is accepted; `max_tokens: 1000` is enough for thinking plus the JSON body (~500 output tokens in practice); the disagreement caveat and the artifact `recapture_needed` force both fire; and on melanoma at 0.91 the LLM returned `urgent` **with** `immediate_referral: true` unprompted, so the schema validator never had to reject a response. Cost ~$0.0005 per call. It stays gated by `scripts/_guard.py` (`SKINSURE_ALLOW_SPEND=1` **and** `--max-usd`, neither defaulted). Two residual unknowns: the artifact case's final `confidence_level: low` is ambiguous because `_TIER_DOWN` maps both `moderate→low` and `low→low`, so the live run proves the branch executed (`recapture_needed` was forced) but not the size of the downgrade — the parametrized unit test covers all three tiers instead; and nothing has yet been run against a real lesion photo (see gap 6).
3. **Hair removal is measured and stays off — it hurts (2026-11-02).** Paired A/B over the whole clean 565 set, each image scored twice through the same v2 ensemble (`scripts/hair_ab_experiment.py`, `scripts/hair_ab_analyze.py`, raw rows cached in `.nb1out/hair_ab_raw.csv`). It **never corrected a single image and broke two**, both true BCC: `ISIC_0054063` BCC→Keratosis with BCC confidence collapsing 0.9314→0.3366 (a confident malignant call turned benign, the worst available outcome), and `ISIC_0073116` BCC→Melanoma. Whole-set accuracy −0.35 pp, macro AUC −0.0011, **BCC sensitivity 100% → 98.11%**; on the 41 images where it fired, accuracy −4.88 pp and BCC sensitivity −7.69 pp. Melanoma sensitivity is **exactly unchanged** (121/124 both ways, including all 7 melanomas where it fired) — the cost falls on BCC. McNemar is b=2, c=0 (exact p=0.50): not significant on 2 discordant pairs, but the sign is unanimous — nothing was ever fixed. **Root cause: the step is mistuned, not merely unhelpful.** At `threshold=10` the black-hat filter is non-selective on this dataset — of the 524 images it skipped, the mask covered a median 58.9% of the frame and ≥75% on 185 of them, so `MAX_MASK_FRACTION` is the only thing preventing wholesale inpainting over lesions. The 7.26% that slip under the guard inpainted a median 6.9% of pixels, i.e. they sit just below the cutoff rather than being precise masks, which is why firing concentrates on BCC (24.5%): BCC lesions are dark and textured and respond to black-hat like hair does, so the step preferentially damages the class it fires on. Re-tuning would need a much higher threshold and a thinner, elongation-aware kernel validated against hair-annotated images before another test is worth running. A control check on the 524 skipped images found zero change on every metric and a maximum probability shift of 0.00000, confirming the scoring path is deterministic and both flips are genuine inpainting effects.
4. **Artifact removal is measured and stays off — it is harmless but does nothing (2026-11-02).** Paired A/B over the whole clean 565 set, same harness and statistics as hair removal (`scripts/artifact_ab_experiment.py`, `scripts/artifact_ab_analyze.py`, raw rows in `.nb1out/artifact_ab_raw.csv`, report in `.nb1out/artifact_ab_analysis.md`). **It fired on 335 of 565 images (59.3%) and changed exactly one label — a fix, not a break.** Whole-set accuracy +0.18 pp (87.61% → 87.79%, 495→496/565), macro AUC −0.0001. **Both safety-critical sensitivities are bit-identical: melanoma 121/124 (97.58%) and BCC 106/106 (100%) in both arms, McNemar b=0 c=0 on each.** The single flip is `v-tinea-face-4`, true Tinea, Keratosis(0.4655)→Tinea(0.4882) after inpainting 0.4% of pixels — a genuine fix but a coin-flip-margin case. McNemar overall is b=0, c=1, p=1.0000. **This is the opposite failure mode to hair removal: that step was mistuned and damaging, this one is well-behaved and inert.** The masks are small and plausible — fired images inpaint a median 0.31% of the frame, 220 of 335 under 1%, median 0.03× the 10% cutoff (hair's fired images sat at a median 6.9%, just under their guard). Glare dominates (252 of 335 fired masks), then ink (57) and ruler (26); 206 images have an empty mask. The `MAX_MASK_FRACTION=0.10` guard stops 24 images (4.25%, median mask 14.7%, max 47.9%) and is doing real work — without it 63.5% of the set would be inpainted — but it is no longer the *only* thing preventing wholesale damage, unlike in the hair case. Guard trips concentrate in Eczema (9.9%) and Tinea (7.7%) and **never** in BCC (0/106), so the detector is not systematically attacking a malignant class. **It stays off**: a single discordant pair cannot justify adding a pixel-mutating step to the live path when the models were trained without one, and the earlier "47% of 100 BCC images" figure is superseded (true BCC firing rate is 43.4%). The determinism control is clean — 230 skipped images, max probability shift 0.00000000, so the one flip is a real inpainting effect. If it is ever revisited, the case to beat is +1 image on an 0.47-margin Tinea.

5. Preprocessing is unit-tested on synthetic images only (gate boundaries, guards, step order, defaults off). Those tests pin behaviour; they say nothing about tuning on real images (see gap 3).
6. ~~`predict_one.py` and the Grad-CAM explainer were run on a synthetic image only.~~ **Done 2026-11-02**: run on 15 real test photos, 3 from each class in `data/raw/.../test/`. The pipeline works end to end — preprocessing, ensemble, Grad-CAM and the `XAISummary` bridge all produce valid output on real skin, no crashes, and `artifact_flag` fired on 0 of 15 (no spurious retakes). Explanation takes **~2.6–6.3 s mean, up to 15.7 s** across runs on the dev CPU, not the 2.3 s measured on one synthetic image — a synchronous `/predict` needs a timeout and probably a progress response at that latency. Re-run on the v2 checkpoints: 11 of 15 correct versus 12 of 15 on v1, which at n=15 is noise and not evidence either way, but confidence rose markedly (BCC 0.72 → 0.958, Eczema ~0.88 → 0.93) and the three remaining class errors are Keratosis→Eczema/Tinea and Tinea→Eczema, consistent with Keratosis being the weakest class in the NB1 ablation. Two further observations, both pointing at gap 7 rather than at Grad-CAM itself: `boundary_alignment` came back `poor` on 9 of 15 and never `good`, which suggests the threshold is mis-tuned rather than that the model ignores every lesion border; and `focus_description` was `background_skin` on 4 of 15, including 2 correct high-confidence Melanoma calls (0.957 and 0.853) — either the brightness segmentation is wrong about where the lesion is, or the models genuinely use perilesional skin. Neither can be settled without annotated masks, which are **out of scope (decided 2026-11-03)**; treat both as advisory and see gap 7 for the threshold re-tuning that is still tractable. `focus`/`boundary` fell back to `unknown` on 2 of 15.
7. The heatmap → `XAISummary` bridge is **heuristic and unvalidated**: thresholds are first guesses and the lesion outline is a brightness segmentation, not a trained segmenter. The first real-photo run (gap 6) gives it some shape: over 15 images `boundary_alignment` was `poor` 9 times, `partial` 4, `unknown` 2 and **never `good`**, and `focus_description` was `background_skin` 4 times including on two correct, high-confidence Melanoma predictions. A field that never reaches its best value across a whole class spread is more likely mis-thresholded than right. **Annotated lesion masks are out of scope (decided 2026-11-03)**, so the original validation route is closed and these fields cannot be verified as *correct about the lesion*. What is still available, and what gap 10's result makes worthwhile: Grad-CAM provably beats a structure-matched random baseline (+0.1266 AOPC, p < 0.0001), so the heatmaps do track something the model uses — which locates the fault in the **bridge's thresholds**, not in the heatmaps. The tractable work is therefore to re-tune the thresholds so the output distribution is plausible (a field that is never `good` across all five classes is self-evidently mis-cut) and to treat `boundary_alignment` / `focus_description` as **advisory, never evidence**, in any patient-facing or paper text. Fields fall back to `unknown` when evidence is missing. The artifact flag forces re-capture, so false flags still cost patients a retake — that one is worth bounding empirically even without masks.
8. **Federated learning is parked as of 2026-11-02** — it will be re-run on a different dataset than the EfficientNet results below were measured on, handled separately from the product work. Treat every FL number here and in `fedler.md` as pending re-measurement, and fix this buffer bug before that re-run rather than carrying it forward. FL averages EfficientNet's 69 int64 `num_batches_tracked` buffers as floats and truncates them back (`src/federated/client.py:76`). It did not corrupt the Phase 2 results — only `running_mean`/`running_var` reach inference and those are floats — but it is wrong and should be fixed before Phase 3/4. ConvNeXt and Swin have no non-float buffers. See `fedler.md` Phase 4 notes. **A second pre-re-run fix, found 2026-11-03:** the split manifests on disk do not match the documented FL corpus. `data/splits/global_test.csv` holds 904 rows and `trainable_pool.csv` 5,131 (total 6,035), not the 997 / 5,658 quoted here and in `fedler.md` — 620 train images in exactly three classes (Eczema 189, Keratosis 151, Melanoma 280) appear in no manifest. Both trainers default to those CSVs, so the FL and baseline metrics were probably measured on 904 test images, not 997. Regenerate the splits and confirm 997 / 5,658 before re-measuring; full mechanism in `fedler.md` beside the dataset line.
9. `.nb1out/agreement_tiers.csv` is **mis-binned and should not be quoted**: it reports n=0 for the 0.33 and 0.67 tiers and n=484 for 1.0, dropping 81 of the clean 565 images (14.3%). NB1 compared the float agreement score against the literals `0.33` / `0.67`, but the score can only be 0, 1/3, 2/3 or 1, and `1/3 = 0.3333… ≠ 0.33`, so both mixed tiers matched nothing. `.nb1out/_agree.py` re-derives the correct tiers from the cached logits. **The shipped code is not affected** — `Predictions.confidence_tier()` (`api/schemas/predictions.py:35`) uses `>=` thresholds (0.99 / 0.66), not equality, so 2/3 correctly maps to `moderate`; the gap 6 run confirmed live that partial-agreement images reach the right tiers. Only the notebook's table and any paper text quoting it need fixing.
10. **Grad-CAM is measurably better than chance — but it has still never been compared with SHAP (2026-11-03).** ~~The cheapest useful next step is a Grad-CAM-vs-random AOPC number.~~ **Done**, on a Kaggle T4 via `paper/experiments/07_xai_aopc_gradcam_vs_random.py`: 150 images from the clean 565 set, 30 per class, all five classes (raw rows `.nb1out/xai_aopc_raw.csv`, report `.nb1out/xai_aopc_analysis.md`, explainer `docs/agentops/aopc_explained.md`). Grad-CAM AOPC **0.3099** [0.2831, 0.3376] vs the structure-matched `random7x7` baseline **0.1834** [0.1678, 0.1994]: paired difference **+0.1266** [+0.1059, +0.1478], Wilcoxon **p < 0.0001**, **127/150 wins (84.7%)**, Cohen's dz **0.967**. Significant in all five per-class rows (p ≤ 0.0012), and it holds on the images the ensemble got **wrong** (+0.1479, p=0.0006) as well as right (+0.1229) — faithfulness is a property of the explanation, not of being correct. **Read `gradcam - random7x7` only.** The `randompixel` arm posts the *highest* AOPC of the three (0.4887) while having zero localisation content by construction: i.i.d. speckle deletion shreds texture globally and pushes the image off-distribution, so it measures model fragility to high-frequency corruption, not explanation quality. It is a sanity check that AOPC is OOD-sensitive, never a rival method. **What this does and does not settle:** Grad-CAM beats chance on these models, so the heatmaps are not decorative — which also shifts gap 7 toward "the XAISummary bridge is mis-thresholded" rather than "the model ignores the lesion". It says nothing about SHAP, and **that comparison is deliberately deferred (decided 2026-11-03)** — not a blocker: Grad-CAM now has a positive faithfulness result, nothing in the product waits on SHAP, and `/explain` (where SHAP would live) is not built. Revisit only if `/explain` gets scheduled or a reviewer asks. **The scope limit is accepted, not open:** AOPC measures whether the map found pixels *the model used*, which is the question faithfulness asks; whether those pixels are *the lesion* would need annotated masks, which this project does not have and will not acquire, so it is **out of scope** rather than a pending task. One bug was found and fixed in the same change: the notebook used the raw `(1,1,1)` weights from `configs/models.yaml`, but `SkinEnsemble` normalises them, so every probability was 3× too large (p(class) read 2.6 at 0% deletion). `aopc()` is a linear functional of the curve, so the constant factor rescaled every value identically — win count and dz are bit-identical before and after, and only the magnitudes were wrong. The script now normalises and asserts the probability vector sums to 1. The teammate's SHAP work is on branch `xai-shap` of `MohammedAli-Fatteh/SkinSure` (two Colab notebooks, not importable, not merged here); its AOPC numbers used a different blanking baseline in each notebook, so they are not directly comparable until re-scored with `evaluate_xai.py`.

**Empty stubs (0 bytes):** `treatment_advisor.py`, `doctor_referral.py`, `pharmacy_locator.py`; `src/CLAUDE.md`, `src/agents/CLAUDE.md`; `docs/indian_skin_tones.md` (domain knowledge still to be written). `src/agents/tools/` is an empty directory.

**Not yet created:**
- `GET /history`, `POST /history`, `POST /explain`; auth, rate limiting and middleware
- Preprocessing: `augmentation.py` (training only)
- `docker/`, `.github/`
- `src/federated/privacy.py` (differential privacy — fedler Phase 3)

**Future scope (deliberately not planned for the core product):**
- `skin_tone_classifier.py` — automatic Fitzpatrick detection from the image. Today the IV–VI gate relies only on the Fitzpatrick type in `PatientMetadata`.
- Differential privacy on top of FL — next phase; see `fedler.md` Phase 3.
- FL for ConvNeXt and Swin backbones — see `fedler.md` Phase 4.
- **SHAP vs Grad-CAM faithfulness — deferred 2026-11-03, nothing blocks it.** Grad-CAM has a measured faithfulness result (gap 10: +0.1266 AOPC over a structure-matched random baseline, p < 0.0001), SHAP is not installed, and `/explain` is not built. `scripts/evaluate_xai.py` is the harness if it is ever picked up. Pick it up only if `/explain` is scheduled or a paper reviewer asks for the comparison.
- **Annotated lesion masks — out of scope 2026-11-03.** Would be needed to verify that Grad-CAM's focus is *the lesion* (rather than pixels the model uses) and to validate the `XAISummary` bridge's `boundary_alignment` / `focus_description`. The project does not have them and will not acquire them, so both fields stay **advisory, never evidence**, and the tractable work is threshold re-tuning (gap 7). Do not write either field up as a validated measurement.

---

## Commands

`pyproject.toml` exists (install with `pip install -e .[dev]`), and `tests/unit/` + `tests/integration/` run today. Commands (some not available yet):

**Use the project venv, not the system Python** — `torch`, `cv2`, `anthropic` and `fastapi` are installed only in `venv/`, so the system interpreter fails at collection with `ModuleNotFoundError`. On Windows the interpreter is `./venv/Scripts/python.exe`.

```bash
# Run all tests
./venv/Scripts/python.exe -m pytest tests/

# Run a single test file
./venv/Scripts/python.exe -m pytest tests/unit/test_clinical_summarizer.py

# Run with fairness audit
./venv/Scripts/python.exe -m pytest tests/fairness/test_skin_tone_bias.py

# Start API dev server (loads the real checkpoints at startup)
./venv/Scripts/python.exe -m uvicorn api.main:app --reload

# Training (src/training/, once created)
python -m src.training.train --config src/training/configs/ensemble.yaml
python -m src.training.evaluate --fairness-audit
```

Works today:

```bash
./venv/Scripts/python.exe scripts/predict_one.py <image>     # ensemble prediction
# Real API; both the env var and the cap are required, and --case picks which
# post-processing rule to exercise (disagreement / artifact / urgent).
# summarize_one.py calls load_dotenv(override=True), so .env wins over any
# ANTHROPIC_API_KEY already exported in the shell.
SKINSURE_ALLOW_SPEND=1 ./venv/Scripts/python.exe scripts/summarize_one.py --case disagreement --max-usd 0.05
./venv/Scripts/python.exe scripts/evaluate_xai.py <image_dir> [--heatmap-dir DIR]   # deletion/AOPC: Grad-CAM vs external heatmaps vs random

# Paired preprocessing A/B experiments (hours on CPU; both write raw rows to .nb1out/
# and are re-runnable. The *_analyze.py scripts are pure-stdlib+numpy and instant.)
./venv/Scripts/python.exe scripts/hair_ab_experiment.py      && ./venv/Scripts/python.exe scripts/hair_ab_analyze.py
./venv/Scripts/python.exe scripts/artifact_ab_experiment.py  && ./venv/Scripts/python.exe scripts/artifact_ab_analyze.py

# Grad-CAM vs random-baseline AOPC on a stratified sample (--resume picks up a
# partial .nb1out/xai_aopc_raw.csv; --sample-csv reuses the recorded manifest so the
# image set cannot drift between runs).
./venv/Scripts/python.exe scripts/xai_aopc_experiment.py --resume --sample-csv .nb1out/xai_aopc_sample.csv
./venv/Scripts/python.exe scripts/xai_aopc_analyze.py
```

---

## Architecture and Key Patterns

### Schema layer (`api/schemas/`)

`PatientMetadata` enforces the Fitzpatrick IV–VI gate at the boundary — any image from a Fitzpatrick I–III patient is rejected with a clear validator error before reaching the model. Fields: `patient_id`, `patient_name`, `patient_age` (0–120), `patient_sex`, `fitzpatrick_type`, `patient_location` (optional), `current_season` (optional), `body_area`, `symptom_duration_days`, `medical_history` (list), `is_spreading_rapidly` (bool, default False). Enums: `Sex`, `Season`, `BodyArea`, `FitzpatrickType`.

`ClinicalSummary` fields: `severity` (`routine` / `monitor` / `urgent`), `summary`, `recommended_action`, `confidence_level` (`high` / `moderate` / `low`), `confidence_caveats`, `recapture_needed`, `immediate_referral`. Its `@model_validator` enforces `urgent → immediate_referral` at the Pydantic level, so the safety constraint is checked in two places: the LLM system prompt and the schema validator.

`Predictions` labels are the `DiseaseClass` enum (exact checkpoint strings: `Basal_Cell_Carcinoma`, `Eczema`, `Keratosis`, `Melanoma`, `Tinea`); confidences and `agreement_score` are bounded 0–1, and the secondary label must differ from the primary with confidence ≤ primary. `SkinEnsemble` refuses checkpoints whose classes are not exactly that set.

`XAISummary` fields: `focus_description` (`FocusRegion` enum: `lesion_centre`, `lesion_border`, `diffuse`, `background_skin`, `artifact`, `unknown`), `boundary_alignment` (`BoundaryAlignment`: `good`, `partial`, `poor`, `unknown`), `artifact_flag`. A validator requires `artifact_flag=True` when focus is `artifact`.

`Predictions.class_probabilities` is an **optional** `dict[str, float] | None` holding the full 5-class ensemble distribution (populated by `SkinEnsemble.predict()`; validated to be exactly the 5 `DiseaseClass` keys with values in [0,1] when present). It exists because the top-2 fields cannot express p(melanoma) when melanoma ranks third or lower. Read it through `Predictions.probability_of(label)`, which falls back to the top-2 fields and returns `None` when the probability is genuinely unavailable — so callers degrade gracefully rather than guessing.

`Predictions.confidence_tier()` maps `agreement_score` (a fraction of 3 models: 0 / 0.33 / 0.67 / 1.0) to `"high"` (all three agree) / `"moderate"` (two agree) / `"low"` (otherwise) — use this method rather than replicating the threshold logic.

### Clinical summarizer (`src/agents/clinical_summarizer.py`)

Two-phase design — `build_context()` is pure Python (no LLM, and it deliberately omits `patient_name`), `summarize()` is async and calls the Anthropic API (model, `max_tokens` and `effort` come from `configs/agents.yaml`; the client is injectable for tests). **No `temperature`:** sampling params are rejected on Haiku 5.5 and are a 400 on every model from API version `2026-10-21`, so `output_config={"effort": ...}` is the depth/cost control instead. Thinking is on by default on this model, so `max_tokens` has to cover thinking plus the JSON body — too low a value truncates the response and fails parsing. Post-processing in `summarize()` runs after the LLM response:
1. If `xai_summary.artifact_flag`: downgrade `confidence_level` one tier via `_TIER_DOWN` and force `recapture_needed = True`.
2. If `confidence_tier() == "low"` and no caveat already mentions "model disagreement": append the standard caveat string.

The LLM is instructed to return **JSON only** — no markdown, no preamble. Parsing or schema failures raise `ValueError` with the raw response included for debugging.

### Config system (`src/utils/config.py`)

`get_settings()` and `get_model_config()` are `lru_cache`-wrapped — call them freely; each reads its YAML only once. `get_checkpoint_paths()` returns a `CheckpointPaths` resolved against the project root (it imports `ensemble` lazily because that pulls in torch). `Settings` is a `BaseSettings` subclass, so environment variables can override YAML values if needed.

### Design invariants

- **LLM interprets; Python decides.** The LLM proposes severity; deterministic rules in `severity_triage.py` override it. Never let the LLM have the final word on urgent/non-urgent.
- **Context builder is deterministic; summarizer is not.** All clinical reasoning lives inside the LLM with a constrained system prompt. `build_context()` only formats — no interpretation. **Spell out counts the LLM would otherwise guess:** given a bare `agreement_score` of `0.33` it stated "the two models agree" (wrong in both live runs that exposed it), so the line now reads `"1 of 3 models agree with the primary label (score 0.33, low tier)"`. A derived figure in the context needs its denominator, not just its value.
- **Fairness is a first-class CI check.** Per-Fitzpatrick-subtype accuracy must be tested on every model update.
- **Uploaded images are never used for training.** They are kept only for the patient's own history (`GET /history`).
- **Secrets never touch git or images.** `.env` locally (gitignored), a secret manager in the cloud.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Agent | Anthropic Claude (via `anthropic` async SDK) |
| API | FastAPI + Pydantic v2 |
| Config | YAML + `pydantic-settings` |
| ML Frameworks | PyTorch, timm |
| Model Backbones | EfficientNet-B1, ConvNeXt-Tiny, Swin-Tiny |
| xAI | Grad-CAM first; SHAP, LIME later (planned) |
| Federated learning | Flower (`flwr`) FedAvg — Phases 0–2 done; see `fedler.md` |
| Differential privacy | Deferred to Phase 3 — see `fedler.md` |
| Containerisation | Docker + docker-compose (planned) |
| CI/CD | GitHub Actions (planned) |

---

## Full Architecture Reference

### System Architecture

```
User (Web App)
      │
      ▼
[FastAPI Backend]
      │
      ├── POST /predict ──► [Preprocessing Pipeline]
      │                           │
      │                    [Ensemble Model]
      │                    EfficientNet + ConvNeXt + Swin
      │                           │
      │                    [xAI Layer]
      │                    Grad-CAM / SHAP / LIME
      │                           │
      │                    [Orchestrator Agent]
      │                           │
      │              ┌────────────┼────────────┐
      │              ▼            ▼            ▼
      │     [Clinical        [Severity     [Treatment
      │      Summarizer]      Triage]       Advisor]
      │              │            │            │
      │              └────────────┼────────────┘
      │                           ▼
      │                   [Router]
      │              ┌────────────┴────────────┐
      │              ▼                         ▼
      │     [Pharmacy Locator]        [Doctor Referral]
      │
      └── GET /history ──► [Patient Records]
```

### ML Pipeline

| Step | Module | Purpose |
|---|---|---|
| Quality check | `quality_check.py` | Reject images under 200px (short side) or nearly featureless; warn on borderline blur |
| Hair removal | `hair_removal.py` | DullRazor-style black-hat + inpainting (off by default — measured harmful, gap 3) |
| Artifact removal | `artifact_removal.py` | Inpaint pen ink, bubbles/glare, ruler lines (off by default — measured inert on the 565 set, gap 4) |
| Color normalization | `color_normalization.py` | Shades-of-Gray (off by default) |
| Augmentation | `augmentation.py` | Tone-aware augmentation for training |

`preprocess(image)` (in `pipeline.py`) orients the image (EXIF), runs the quality gate, then the enabled pixel-changing steps in the order hair → artifacts → colour, and returns a `PreprocessResult` (`image`, `quality`, `steps_applied`, `notes`). A rejected image comes back with `quality.ok == False` and patient-readable `quality.reasons`; borderline blur adds a `quality.warnings` entry instead. **The models were trained with only resize + ImageNet normalisation, so all three pixel-changing steps are off by default** (`configs/preprocessing.yaml`); enable one only after comparing predictions with and without it on a held-out set. Thresholds come from a 1,500-image sample of the raw dataset: short side ≥ 200px; blur (Laplacian variance) rejects below 5 and warns below 15. A higher blur cut-off would reject BCC and Melanoma images far more often than Eczema or Tinea (about 18% / 9% vs 1% / 0.3% at 15), so it is kept low. Shades-of-Gray can flatten genuine skin-tone colour, which matters for Fitzpatrick IV–VI.

The Fitzpatrick IV–VI gate is enforced by `PatientMetadata`; an image-based `skin_tone_classifier.py` is future scope.

Ensemble voting: weighted average across EfficientNet (texture), ConvNeXt (varied lighting), Swin Transformer (global lesion context), with weights from `configs/models.yaml`. `agreement_score` = fraction of the three models whose top-1 matches the ensemble top-1; it is passed to the summarizer, and low agreement triggers explicit uncertainty flagging.

Classes (5), in model index order (alphabetical, as stored in each checkpoint's `classes`): Basal_Cell_Carcinoma, Eczema, Keratosis, Melanoma, Tinea. See `src/models/CLAUDE.md` for training details and sensitivity targets.

### Agentic Layer

| Agent | File | Status |
|---|---|---|
| Orchestrator | `src/agents/orchestrator.py` | **Implemented** (`run_screening`; router to later agents still to come) |
| Clinical Summarizer | `src/agents/clinical_summarizer.py` | **Implemented** |
| Severity Triage | `src/agents/severity_triage.py` | **Implemented** |
| Treatment Advisor | `src/agents/treatment_advisor.py` | Stub |
| Doctor Referral | `src/agents/doctor_referral.py` | Stub |
| Pharmacy Locator | `src/agents/pharmacy_locator.py` | Stub |

Severity triage hard overrides (`triage(predictions, summary, patient) -> TriageResult`; thresholds 0.40 in `severity_triage.py`): melanoma at any moderate+ confidence → `urgent`; model confidence < 40% → block, request re-capture; rapidly spreading infection (`patient.is_spreading_rapidly` and an infection class — currently Tinea) → `urgent`. Rules only escalate or block, never downgrade; escalation and the infection rule are skipped when the result is blocked. Urgent cases must never receive self-treatment advice (the router suppresses the treatment advisor).

**Rule F4 (prototype, OFF by default)** — Rule 2 gates on `primary_label`, the argmax, so a melanoma ranked second can never reach it *at any threshold*; all three melanomas the shipped ensemble misses are exactly that shape (`.nb1out/weighting_melanoma_misses.csv`: p(mel) = 0.343, 0.369, 0.064). Rule F4 escalates when melanoma is **not** top-1 but `p(melanoma) >= ranked_melanoma_min_confidence`, reading the new optional `Predictions.class_probabilities` via `probability_of()`; with that field absent the rule falls back to the top-2 fields and simply does not fire for a melanoma ranked third or lower, i.e. today's behaviour. It only escalates, is skipped when blocked, and logs a distinct `reasons` entry ("melanoma ranked below top-1 at probability ...").

Threshold **0.31**, chosen on the 748-image **validation** split (202 melanoma): the lowest threshold recovering all 3 non-argmax validation melanomas at **zero** false escalations on the 546 non-melanoma validation images (the zero-cost plateau is 0.31–0.35; at 0.36 recall drops to 1). Frozen, then applied to test (565 images, 124 melanoma): recovers **2 of 3** known misses for **1** false escalation (0.23% of non-melanoma). The p=0.064 case is unreachable at any usable threshold. Note 0.30 was fitted to those 3 test cases; validation agrees it is roughly right but puts the clean cut one step higher (0.30 costs 1 false flag on validation). **Effect size is tiny — 3 validation and 3 test melanomas cannot establish this is a net safety gain**, which is why it ships off. Sweep: `.nb1out/f4_threshold_sweep_val.csv`. Enable with `triage.escalate_on_ranked_melanoma: true` in `configs/agents.yaml`.

### xAI (`src/explainability/`)

**Why Grad-CAM runs on `/predict` and SHAP does not.** The choice is about latency and output shape, **not** about which method explains these models better — that has never been measured (gap 10), so this is a design decision held under review, not a finding.

1. **Latency.** Grad-CAM is one forward + one backward pass per backbone: ~5.0 s mean for the whole ensemble on the dev CPU over 15 real photos (up to 15.7 s; an earlier 2.3 s figure came from a single synthetic image). SHAP's patch/kernel approaches need hundreds to thousands of forward passes per image, which is minutes on CPU. `/predict` is synchronous and a patient waits on it, so the live path takes the cheap method.
2. **Output shape.** `XAISummary` needs a spatial map ("where did the model look") reducible to focus region, boundary alignment and artifact presence. Grad-CAM yields a dense 7×7 spatial map natively; SHAP yields per-patch attributions that would need their own conversion to a spatial map first.
3. **Already hooked into these backbones.** Per-architecture target layers are solved in `gradcam.py`, including Swin's NHWC transpose — work that any replacement method would have to redo.

So SHAP is deferred to `/explain` (not yet built): the slow, thorough method for a second opinion, the fast one in the live path. **Do not read this as Grad-CAM being the more faithful method.** `scripts/evaluate_xai.py` is the harness that would settle it — it scores Grad-CAM, any external heatmap (e.g. SHAP `.npy`) and a random baseline under one identical procedure, on the principle that a method which does not beat random explains nothing. It has never been run (gap 10). **The Grad-CAM-vs-random half of that question is now answered by a different harness** (`paper/experiments/07_xai_aopc_gradcam_vs_random.py`, run on a T4): Grad-CAM beats the structure-matched random baseline by +0.1266 AOPC, p < 0.0001, 127/150 wins. So Grad-CAM does explain something. **The SHAP comparison is deferred, by decision (2026-11-03), not pending.** It would be the thing that could overturn the `/predict` choice, but nothing is blocked on it: Grad-CAM has a positive faithfulness result, SHAP is not installed, and `/explain` — the endpoint it would serve — does not exist. The work if it is ever picked up: re-score the `xai-shap` notebooks' maps with `evaluate_xai.py` so both methods are measured under one identical procedure, then decide whether SHAP earns its integration cost. Until then this section stands as a **latency and output-shape decision with a measured faithfulness floor**, which is a materially stronger position than it was before gap 10 was measured. SHAP is not installed (absent from `pyproject.toml` and the venv), and the `xai-shap` notebooks used a different blanking baseline in each, so their AOPC figures are not comparable until re-scored with this harness.

- **Grad-CAM** — **implemented**: per-backbone maps averaged with the ensemble weights, converted to `XAISummary` (focus, boundary alignment, artifact flag) before reaching the summarizer. Grad-CAM++ is not implemented.
- **SHAP** — feature-level patch attribution (planned in-repo; prototype notebooks exist on the teammate's `xai-shap` branch; intended for `/explain`, not the `/predict` path)
- **LIME** — superpixel local explanations; validates model attends to lesion, not background

### API endpoints (planned)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Image + metadata → `ClinicalSummary` |
| `POST` | `/explain` | Prediction → xAI heatmap report |
| `GET` | `/history` | Retrieve patient history |
| `POST` | `/history` | Save consultation record |
| `GET` | `/health` | Service health check |

### Federated learning — Phases 0–2 complete

`src/training/train_utils.py` provides the reusable training loop. `src/federated/client.py` and `server.py` implement the Flower `NumPyClient`/`ClientApp`/`ServerApp`. `scripts/run_fl_simulation.py` runs the full FL simulation using `fl.simulation.start_simulation()` with a `FedAvgWithCheckpoint` strategy (subclass that captures aggregated parameters). Canonical data splits live in `data/splits/` (global test set, trainable pool, IID and non-IID hospital shards for 3 clients).

**Results (EfficientNet-B1 only), both on the held-out 997-image global test set:**

| Metric | Centralized | Federated | Gap |
|---|---|---|---|
| Test accuracy | 83.15% | 79.94% | -3.21% |
| Melanoma sensitivity | 98.32% | 97.06% | -1.26% |
| BCC sensitivity | 91.16% | 90.61% | -0.55% |
| Composite | 0.9162 | 0.8999 | -0.0163 |

The accuracy loss concentrates in Keratosis (70.10% → 56.70%); safety-critical classes hold. Per-client validation metrics run ~2 points optimistic versus this test set — always quote the global test numbers.

**Remaining:** Keratosis regression, differential privacy (Phase 3), ConvNeXt/Swin FL (Phase 4), Fitzpatrick fairness eval (Phase 5). Full details and results in `fedler.md`, the source of truth.

### Testing (`tests/` — planned)

```bash
pytest tests/unit/          # Preprocessing, model, agent, explainability
pytest tests/integration/   # End-to-end: image in → ClinicalSummary out
pytest tests/fairness/test_skin_tone_bias.py  # Per-Fitzpatrick accuracy audit
```

Fairness suite runs automatically on every PR — any regression in per-tone accuracy fails the check.

---

*For the target layout see `docs/architecture.md`; for the build plan see `docs/ROADMAP.md`; for federated learning see `fedler.md`. Domain knowledge on Indian skin tones and regional conditions is still to be written in `docs/indian_skin_tones.md`.*
