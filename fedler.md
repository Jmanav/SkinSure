# Federated Learning for SkinSure — Learn-and-Build Plan

## Why this exists

Right now SkinSure's three models (EfficientNet-B1, ConvNeXt-Tiny, Swin-Tiny — see
`src/models/`) were trained the normal, centralized way: all training images sat in
one place (Colab / Google Drive), one training loop saw all of them.

Federated learning (FL) flips this. Imagine 3 hospitals, each with their own
patients' skin images, and **none of them are allowed to send those images
anywhere** (DPDP Act — India's data protection law — plus basic patient privacy).
FL lets all 3 hospitals contribute to one shared model without any hospital's raw
images ever leaving its building. Only *model weight updates* (numbers describing
how the model changed after training on local data) get sent to a central
coordinator, which blends them together and sends back an improved shared model.
Repeat many rounds → you get a model as good as if all data had been pooled,
without ever pooling it.

This doc is organized as **phases**, each phase pairing the minimum concept you
need with the piece of code that concept unlocks. Learn it, then immediately
build it on real (simulated) SkinSure data — don't stockpile theory before
touching code.

## SkinSure's two-section architecture

SkinSure is built as two distinct but connected sections:

### Section 1 — B2B: Federated Training Pipeline

Hospitals and clinics are FL clients. Each hospital:
1. Has its own local dataset of labeled skin disease images (doctors label via
   diagnosis, dataset grows as more patients visit).
2. Trains a **local model** on its own server using its own data.
3. Sends only **model weight updates** (never raw images) to SkinSure's central
   FL coordinator.
4. Receives back an improved global model after FedAvg aggregation.

No raw patient images ever cross organizational boundaries. Hospitals get access
to the improved global model for free — this is their incentive. They contribute
data (indirectly, via weight updates) and receive a model trained on effectively
everyone's data without anyone sharing theirs.

### Section 2 — B2C: Inference Pipeline

A patient or doctor uploads an image of a skin disease on the SkinSure website →
the ensemble model (served from the latest FL-trained global checkpoint) runs
inference → severity assessment + confidence level + XAI explanation →
agentic AI layer generates a clinical summary.

This is the **inference/serving** side. Already partially built (`ensemble.py`,
`src/agents/`, `api/`).

### How the two sections connect

Section 1 (B2B) **produces** the model. Section 2 (B2C) **serves** it. The flow
is one-way: nothing a patient uploads is ever used for training. Uploaded images
are kept only for that patient's own history (`GET /history`) and never enter any
training pipeline.

### Design decision: hospital-level clients, not per-patient

Each SkinSure patient only contributes 5–10 images. Treating each patient as
their own FL client would mean a full communication round (send global
weights out, train, send weights back, average) to absorb a training signal so
small it barely moves the model — expensive in round-trips, cheap in benefit.

The design uses **hospitals/clinics as FL clients** instead. This is the
canonical FL use case: data genuinely can't leave the institution, FL is the
only way to train across institutions without moving data, and the privacy
story under the DPDP Act is real and defensible.

### Client structure for simulation

Since no real hospital partnerships exist yet, the FL pipeline is built as a
**simulation** on the existing curated dataset (6,655 images from the IPD
dataset, 5 diseases; see "Ground truth" section below for per-class counts).
The simulation is designed so the same code plugs into real hospital clients
later.

**3 simulated clients:**

| Client | Role | Shard character (non-IID) |
|--------|------|--------------------------|
| Hospital A | Large urban hospital | Skewed toward Melanoma, BCC |
| Hospital B | Clinic in a tropical area | Skewed toward Tinea, Eczema |
| Hospital C | Multi-specialty hospital | Mixed distribution |

Each client needs **≥200 images per disease** for local training to be
meaningful. Tinea is the smallest class (1,150 images in total), so it is the
binding constraint when splitting across clients.

### DPDP Act compliance summary

| Data flow | Privacy mechanism | DPDP compliance |
|-----------|-------------------|-----------------|
| Hospital → FL coordinator | Only weight updates transmitted, never raw images | ✓ No personal data leaves the institution |
| FL coordinator → all clients | Aggregated global model weights | ⚠ Averaging reduces but does not eliminate leakage risk; Phase 5 (differential privacy) addresses this |

> **Phase 5 adds differential privacy** on top of this: even if someone
> intercepted the weight updates, calibrated noise makes it mathematically
> impossible to reverse-engineer any individual patient's data from them.

---

## Phase 0 — Training loop fundamentals → extract a trainable module

FL is just "run this training loop, but somewhere else, then average the
results." So the training loop itself has to click first, and there's currently
no reusable training code in the repo to build on — training only exists in
Colab notebooks (`src/models/*.ipynb`).

### Learn
- **What a training step actually does**: forward pass → loss → `loss.backward()`
  → optimizer `.step()` → `.zero_grad()`. You've only seen `@torch.no_grad()`
  inference in `ensemble.py`; training is the same forward pass minus `no_grad`,
  plus these extra steps.
- **What a "weight" / "parameter" is concretely**: `model.state_dict()` — an
  ordered dict of tensor name → tensor values. This is the *exact* thing that
  gets sent over the network in FL. Open a Python shell, load one of your
  checkpoints, and inspect `ckpt["model_state"].keys()` and shapes — this single
  exercise demystifies most of FL.
- **Epochs vs. steps vs. batches**: you've seen `freeze_epochs` in the notebooks;
  make sure you understand what one epoch means and how a batch and a step
  relate.
- Time: 2–3 hours if you skim your own notebooks with this lens.

### Pre-requisite: canonical data splits

Before extracting the training loop, create canonical train/val/test split
manifests in `data/splits/`. The Colab notebooks used their own internal splits
that aren't recorded in the repo. To reproduce the centralized baseline and
later compare it fairly against federated training, you need a single fixed
split:

1. Hold out **~15% of the full dataset as a global test set** (stratified by
   class). This test set is never seen during any training — centralized or
   federated — and is the common benchmark for comparing them.
2. The remaining ~85% is the trainable pool. In centralized mode it splits
   further into train/val; in federated mode Phase 1 shards it across hospitals
   (each hospital gets its own local train/val split from its shard).
3. Save as `data/splits/global_test.csv`, `data/splits/trainable_pool.csv`.

### Build
Port **one** backbone's training loop out of its notebook into a plain module,
`src/training/train_utils.py`:

```python
def train_one_epoch(model, dataloader, optimizer, criterion, device) -> None: ...
def evaluate(model, dataloader, device) -> dict:  # val_acc, melanoma_sensitivity, etc.
```

Reuse the exact loss weighting (`Melanoma 2×, BCC 2×, Tinea 3×, Eczema 1.5×`) and
composite-score logic already documented in `src/models/CLAUDE.md` — FL doesn't
change *how* a model should be trained, only *where*.

Start with **EfficientNet-B1 only** (smallest/fastest). ConvNeXt and Swin come
later (Phase 4) — don't parallelize all three backbones through FL on your
first pass.

**Checkpoint before moving on:** `train_one_epoch` + `evaluate` run standalone
against the trainable pool (with a local train/val split) and roughly reproduce
the centrally-trained checkpoint's val_acc. Evaluate on the global test set to
establish the **centralized baseline** — this is the number federated training
must approach.

---

## Phase 1 — FL core concepts → simulate hospital data shards

### Learn
- **The basic FL loop (FedAvg — the algorithm everything below builds on)**:
  1. Server sends current global model weights to each client (hospital).
  2. Each client trains that model on its own local data for a few epochs.
  3. Each client sends back only its updated weights (or the *delta* — the
     difference from what it received).
  4. Server averages all clients' weights (weighted by how much data each
     client had) → new global model.
  5. Repeat for N "rounds."
- **Client vs. server vs. round** — vocabulary you'll see everywhere.
- **Non-IID data**: in real life, Hospital A might see mostly Melanoma cases and
  Hospital B mostly Tinea — the data isn't identically distributed across
  clients, which makes FL harder than averaging models trained on random splits
  of the same distribution. This directly matters for you, since your dataset
  is already class-imbalanced (Melanoma skewed) — splitting it into hospital
  shards will make each shard's imbalance different, and worth reasoning about
  deliberately.
- **Communication cost**: why you can't just run FedAvg every single training
  step (network round-trips are the bottleneck, not compute) — motivates "a few
  local epochs per round" instead of 1 step per round.
- **Resource**: Google's original FedAvg paper (McMahan et al., 2017,
  "Communication-Efficient Learning of Deep Networks from Decentralized Data")
  — just the first 4 pages, skip the theory-heavy proofs.
- Time: half a day.

### Build
Split the **trainable pool** (from Phase 0's `data/splits/trainable_pool.csv`)
into N=3 shards (one per simulated hospital). The global test set is never
included in any shard — it stays untouched for evaluation.

Write this as `scripts/make_federated_shards.py`, outputting shard manifests
(lists of image paths + labels) — not new copies of the image files. Build
**two** sharding strategies:

1. **IID shard** (random split, same class balance in each shard) — sanity-check
   FedAvg works at all before introducing the harder problem.
2. **Non-IID shard** (skew by hospital specialization):
   - Hospital A: heavy Melanoma + BCC (urban dermatology center)
   - Hospital B: heavy Tinea + Eczema (clinic in a tropical area)
   - Hospital C: mixed (multi-specialty hospital)

   Non-IID skew should be calibrated so every shard stays above the **200
   images per disease** floor. For the smallest classes (Tinea at ~978 in the
   trainable pool, Eczema at ~1,522), the skew between hospitals will be mild —
   accept this rather than forcing dramatic skew that starves a shard. The
   non-IID character comes primarily from the larger classes (Melanoma, BCC).

Each shard is a stand-in for one hospital's local dataset. The simulation does
not model how hospitals acquire data (patient intake, doctor labeling, dataset
growth over time). That's real-world operations, not FL mechanics.

**Checkpoint before moving on:** shard manifests exist for 3 clients under
both IID and non-IID strategies, and you can load any one of them through
Phase 0's `train_one_epoch`/`evaluate` as if it were a single hospital's
local dataset.

---

## Phase 2 — Flower framework basics → single-machine simulation, one backbone

You decided to use Flower (`flwr`) rather than hand-rolling FedAvg — good choice
for learning, since it removes the networking/serialization boilerplate and lets
you focus on the ML.

### Learn
- **Concepts specific to Flower's current API** (1.x+): `ClientApp`, `ServerApp`,
  `Strategy` (FedAvg is one built-in `Strategy` class), `flwr.client.NumPyClient`
  (the class you subclass to wrap your PyTorch model), and `flwr run` (the CLI
  entry point that replaces the old `start_server`/`start_client` calls).
  Simulation mode runs many virtual clients in one Python process, which is
  exactly what you want to start with.
- **Use the new Flower API throughout** (`ClientApp`/`ServerApp` + `flwr run`),
  not the legacy `start_server`/`start_client` pattern. The legacy API is being
  deprecated and most new tutorials target the new one.
- **Do the official Flower PyTorch quickstart tutorial** end to end, unmodified,
  before touching SkinSure code. It trains a small CNN on CIFAR-10 across
  simulated clients — the point is to see a *working* FL loop before adapting
  your own models into it, so when something breaks in your version you know
  whether it's "my model" or "my FL wiring."
- Time: half a day to a day, including running the tutorial.

### Build
1. `pip install flwr` (add to `pyproject.toml` under a new
   `[project.optional-dependencies].federated` group — keep it optional since
   it's not needed for API/inference).
2. Run the official Flower quickstart-pytorch tutorial unmodified — don't skip
   this step even though it feels like a detour.
3. Build `src/federated/client.py`: a `flwr.client.NumPyClient` subclass wrapping
   `EfficientSkinClassifier`, representing **one hospital** —
   `get_parameters`/`set_parameters` convert between `state_dict` and Flower's
   `List[np.ndarray]` format, `fit()` calls Phase 0's `train_one_epoch` on that
   client's shard, `evaluate()` calls Phase 0's `evaluate`.
4. Build `src/federated/server.py`: configure the `FedAvg` strategy (num rounds,
   fraction of clients per round, local epochs per round) and wrap it in a
   `ServerApp`.
5. Build a `ClientApp` in `client.py` that instantiates the `NumPyClient`.
6. Run with `flwr run` in simulation mode with N=3 virtual clients (one per
   hospital) on your Phase 1 shards, on your one machine.

**Success criterion:** after several rounds, the federated-trained model's
val_acc and melanoma_sensitivity **on the global test set** (held out in
Phase 0, never seen by any client) approach the centralized baseline from
Phase 0. This gap is the real quantity of interest — federated will likely be
slightly worse; the question is how much.

> **Met.** 10 rounds on non-IID shards: 79.94% test accuracy vs 83.15% centralized
> (-3.21%), melanoma sensitivity 97.06% vs 98.32% (-1.26%). Full results in
> "Ground truth" below. Measure on the global test set, not the per-client
> validation splits — the latter run ~2 points optimistic because each client
> validates on its own shard's distribution.

**Key metric to track per round:**
- Global model val_acc and melanoma_sensitivity (the numbers that matter)
- Per-client local val_acc before and after each round (shows whether FL helps
  smaller/skewed hospitals — if Hospital B's Melanoma accuracy improves from
  FL even though it has few local Melanoma samples, FL is working as intended)

**Optional extra validation:** the synthetic shards above test whether your
FedAvg *code* works, but the non-IID skew is one you engineered yourself.
**Fed-ISIC2019** (part of the FLamby benchmark suite) is a public skin lesion
dataset already partitioned by real hospital/source, with genuine non-IID
structure. Once your client/server code runs cleanly on your own shards,
optionally point the same `client.py`/`server.py` at Fed-ISIC2019 as a sanity
check that your pipeline holds up against non-IID skew you didn't design —
not required to proceed, just a stronger signal before Phase 3.

---

## Phase 3 — Operational reality → multi-process, localhost networking

Once Phase 2 works, graduate to real client/server processes talking over
localhost instead of one in-process simulation. There isn't much new theory
here — the ML logic from Phase 2 doesn't change — so this phase is almost pure
build.

### Learn
- What actually differs between simulation mode and real multi-process
  deployment (mainly: processes start independently and can drop mid-round,
  instead of everything living in one Python process you fully control).
- How `flwr run` with the `--superlink` flag (or Flower's deployment engine)
  manages multi-process topologies.

### Build
- Run the `ServerApp` in one process/terminal.
- Each simulated hospital as its own process running its `ClientApp`, connecting
  to the server via Flower's gRPC transport (built into `flwr run`).
- No custom `communication.py` is needed — Flower handles gRPC transport out of
  the box. Only create a communication module if a specific networking
  requirement surfaces later.
- Add config for how many clients must be online to start a round, and confirm
  the system tolerates a client dropping mid-round (a hospital's server going
  offline temporarily should not break the training round for everyone else).

---

## Phase 4 — Extend to all three backbones

Repeat Phase 2–3 for ConvNeXt-Tiny and Swin-Tiny once EfficientNet's federated
pipeline is proven.

### Design decision: independent FL per backbone

Each backbone (EfficientNet, ConvNeXt, Swin) is a standalone classifier with
its own `state_dict`. The ensemble is just a weighted average of their softmax
outputs at inference time — there are no shared parameters and no joint training.
This means each backbone federates independently: 3 separate FL jobs (one per
backbone), each running FedAvg over its own model weights. The ensemble weights
in `configs/models.yaml` remain an inference-time concern and do not participate
in FL aggregation.

### Build
- Reuse Phase 0's approach to port each backbone's training loop into
  `src/training/train_utils.py`.
- Reuse Phase 2/3's `ClientApp`/`ServerApp` code, parameterized by backbone
  (pass the backbone name as config so the same client code loads the right
  model class).
- Run all three FL jobs and evaluate each backbone's global test set performance
  independently, then combine via the existing ensemble at inference.

---

## Phase 5 — Differential privacy (deferred)

Deferred deliberately — get plain FedAvg working first (Phases 0–4). DP adds
noise that will obscure whether bugs are "my FL code" or "DP epsilon too
aggressive," so a noise-free baseline needs to exist first.

### Learn
- **Epsilon/delta intuition**: DP adds calibrated noise to updates so no single
  patient's data can be reverse-engineered from the model, at a cost of some
  accuracy. Epsilon is the "privacy budget" — smaller epsilon = more private,
  more noise, worse accuracy.
- **DP-SGD**: the standard way to add DP to neural net training (clip
  per-example gradients, add noise, then average).

### Build
- Wrap the working `FedAvg` strategy with Flower's built-in DP strategy classes
  (`DifferentialPrivacyServerSideFixedClipping`, etc.) — this is additive, not a
  rewrite. Flower's built-in DP covers the standard use case; only introduce an
  external library (e.g. OpenDP) if a specific DPDP Act compliance requirement
  surfaces that Flower's DP cannot satisfy.
- `src/federated/privacy.py` wraps the DP configuration and epsilon tracking.
- **Critical experiment:** measure melanoma_sensitivity at different epsilon
  values. If DP noise at reasonable epsilon (ε ≤ 8) causes melanoma_sensitivity
  to drop below clinically acceptable thresholds, document this tradeoff
  explicitly — this is the real cost of privacy for medical AI, and worth
  reporting honestly.

---

## Ground truth — current state (updated 2026-10-27)

### What exists and works

- **Phases 0–2 fully implemented and committed** on the `federated-learning` branch.
- `src/training/train_utils.py` — reusable training loop: `train_one_epoch`, `evaluate`, `create_model`, `build_loss_weights`, `build_optimizer`, `build_scheduler`, transforms, dataloaders. Supports all three backbones (`efficientnet`, `convnext`, `swin`).
- `src/federated/client.py` — `SkinSureClient` (`NumPyClient` subclass) representing one hospital; `ClientApp` for `flwr run` CLI.
- `src/federated/server.py` — `weighted_average` metric aggregation; `ServerApp` with configurable `FedAvg` strategy for `flwr run` CLI.
- `scripts/run_fl_simulation.py` — standalone FL simulation using `fl.simulation.start_simulation()` with `FedAvgWithCheckpoint` (custom subclass that captures aggregated parameters for post-training evaluation). Evaluates the final global model on the held-out global test set.
- `scripts/make_federated_shards.py` — IID and non-IID shard generation for 3 hospitals.
- `scripts/train_baseline.py` — centralized baseline trainer with early stopping and composite-score tracking.
- `data/splits/` — canonical splits: `global_test.csv`, `trainable_pool.csv`, and hospital shards under `shards/iid/` and `shards/noniid/` (3 CSVs each).
- `flwr` added to `pyproject.toml` under `[project.optional-dependencies].federated`.
- **Dataset:** 6,655 images across 5 target classes (from the 9-class IPD dataset). Per-class: Melanoma 1,587, Eczema 1,365, Keratosis 1,294, BCC 1,208, Tinea 1,201. Trainable pool: 5,658; Global test: 997.
- 179 tests pass.

### Centralized baseline results (EfficientNet-B1, 30 epochs, early stopped at 28)

| Metric | Value |
|---|---|
| Test accuracy | 83.15% |
| Melanoma sensitivity | 98.32% |
| BCC sensitivity | 91.16% |
| Composite score | 0.9162 |
| Eczema sensitivity | 73.04% |
| Keratosis sensitivity | 70.10% |
| Tinea sensitivity | 80.56% |

### Federated learning results (FedAvg, 10 rounds, 3 hospitals, non-IID, EfficientNet-B1)

Trained on Kaggle with 2x T4 GPUs. Checkpoint: `models/federated/efficientnet_federated.pth`
(config confirmed in the checkpoint: 10 rounds, 2 local epochs, FedAvg, 3 clients,
`data/splits/shards/noniid`).

**Global test set — the authoritative numbers (997 held-out images, never seen by any client):**

| Metric | Value |
|---|---|
| Test accuracy | 79.94% |
| Melanoma sensitivity | 97.06% |
| BCC sensitivity | 90.61% |
| Composite score | 0.8999 |
| Test loss | 0.5312 |

Per-class sensitivity / specificity on the global test set:

| Class | Sensitivity | Specificity |
|---|---|---|
| Melanoma | 97.06% | 99.60% |
| Basal_Cell_Carcinoma | 90.61% | 96.81% |
| Tinea | 83.33% | 88.74% |
| Eczema | 69.61% | 93.44% |
| Keratosis | 56.70% | 96.64% |

**Round-by-round distributed validation** (each client's own local val split — useful for
watching convergence, but *not* comparable to the global test numbers above, since each
client validates on its own shard):

| Metric | Round 1 | Round 5 | Round 10 |
|---|---|---|---|
| Val accuracy | 64.56% | 79.01% | 82.32% |
| Melanoma sensitivity | 84.70% | 94.18% | 95.26% |
| BCC sensitivity | 92.54% | 94.80% | 96.04% |
| Composite | 0.8101 | 0.8981 | 0.9161 |

Distributed loss convergence: 1.056 → 0.857 over 10 rounds.

### Centralized vs Federated — global test set (apples to apples)

| Metric | Centralized | Federated | Gap |
|---|---|---|---|
| Test accuracy | 83.15% | 79.94% | **-3.21%** |
| Melanoma sensitivity | 98.32% | 97.06% | -1.26% |
| BCC sensitivity | 91.16% | 90.61% | -0.55% |
| Composite | 0.9162 | 0.8999 | -0.0163 |

Per-class sensitivity:

| Class | Centralized | Federated | Gap |
|---|---|---|---|
| Melanoma | 98.32% | 97.06% | -1.26% |
| Basal_Cell_Carcinoma | 91.16% | 90.61% | -0.55% |
| Tinea | 80.56% | 83.33% | **+2.77%** |
| Eczema | 73.04% | 69.61% | -3.43% |
| Keratosis | 70.10% | 56.70% | **-13.40%** |

**Interpretation.** Federated training costs ~3.2 points of overall accuracy, but the loss is
not spread evenly — it concentrates almost entirely in Keratosis (-13.4), the class that was
already weakest centrally. The safety-critical classes hold: Melanoma sensitivity 97.06%
(within 1.3 points of centralized) at 99.60% specificity, so the privacy-preserving model
still catches nearly every melanoma without flooding clinicians with false alarms. Tinea
actually improved under FL, which is consistent with the non-IID design (Hospital B is
Tinea/Eczema-heavy, so Tinea gets concentrated local training signal).

Keratosis is the thing to fix in later phases — likely more rounds, or a sharding strategy
that does not leave any single hospital with too few Keratosis examples to learn from.

### What's next

- **Phase 3:** Differential privacy (Flower DP strategies, epsilon experiments). Note the
  DP accuracy cost will stack on top of the -3.21% FL cost measured above, and the critical
  experiment is whether melanoma sensitivity stays clinically acceptable.
- **Phase 4:** Repeat FL for ConvNeXt-Tiny and Swin-Tiny backbones.
- **Phase 5:** Fairness evaluation across Fitzpatrick subtypes.
- **Keratosis regression:** investigate the -13.4 point drop before adding DP noise on top of it.

### No hospital partnerships yet
The FL pipeline is a simulation designed to convince hospitals to participate. The code is structured so real hospital clients can replace simulated shards without architectural changes. Client granularity is **hospital/clinic** (B2B). Patient uploads are never used for training.

## Out of scope for this plan

- **Hospital partnership logistics** — onboarding, legal agreements, compute
  requirements at hospital sites. This plan builds the FL code; hospital
  partnerships are a business development track.
- **Real multi-hospital deployment / auth** between real hospital servers —
  everything here is simulated on your machine, by design.
- Changes to `api/`, `src/agents/` — federated learning only affects *how*
  models are trained, not the inference/serving path, which stays exactly as
  `ensemble.py` does today.

---

## Suggested order of work from here

1. Phase 0: create canonical data splits (`data/splits/`), learn training-loop
   fundamentals, then extract EfficientNet-B1's training loop into
   `src/training/train_utils.py` and establish the centralized baseline.
2. Phase 1: learn FL/FedAvg core concepts, then write `make_federated_shards.py`
   with both IID and non-IID strategies for 3 hospitals.
3. Phase 2: learn Flower basics (run the official quickstart first), then build
   `src/federated/client.py` + `server.py` and run a single-machine simulation.
4. Check back in — decide whether to proceed to Phase 3 (multi-process) or
   Phase 4 (more backbones) first, based on what Phase 2 reveals.
5. Phase 5 (differential privacy) only after Phases 0–4 are solid.
