# Architecture Decision Records

Decisions with full evidence trails, written up for interview, paper, and audit use.

| ADR | Decision | Date | Status |
|---|---|---|---|
| [F1](F1_ensemble_weights.md) | Keep equal ensemble weights — melanoma sensitivity gap is not significant | 2026-11-02 | Accepted |
| [F4](F4_ranked_melanoma_escalation.md) | Ranked-melanoma escalation rule: prototype OFF by default | 2026-11-02 | Accepted |
| [F5](F5_hair_removal_off.md) | Hair removal stays off — measured harmful to BCC | 2026-11-02 | Accepted |
| [F6](F6_artifact_removal_off.md) | Artifact removal stays off — measured inert | 2026-11-02 | Accepted |
| [F7](F7_gradcam_faithfulness.md) | Grad-CAM is measurably faithful; SHAP comparison deferred | 2026-11-03 | Accepted |

Each ADR names its evidence files. Raw data lives in `.nb1out/`; experiment scripts in `scripts/`.
