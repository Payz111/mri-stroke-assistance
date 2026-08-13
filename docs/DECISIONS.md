# Architecture Decision Records

Every significant technical decision in this project, with the alternatives that were
considered and the reason one was chosen. Each record also carries its **outcome** —
what actually happened once the decision met reality — because several of them did not
survive contact with the data.

Format: problem → options → decision → rationale → outcome.

---

## ADR-001: Dataset for V1

**Date:** 2026-02-04 · **Status:** Accepted · **Outcome:** Held up

**Problem:** Which dataset should back the stroke segmentation MVP?

**Options:**
1. ISLES 2022 — DWI/ADC/FLAIR, 250 cases, multi-centre
2. Acute Stroke 2,888 — more data, but requires ICPSR access approval
3. ISLES 2015 — older, smaller

**Decision:** ISLES 2022

**Rationale:** A current benchmark with open Zenodo access, multi-centre and
multi-vendor acquisition (a genuine generalisation test), exactly the modalities the
clinical question needs, and an active community to compare results against.

**Outcome:** Used throughout. Later supplemented with SOOP (OpenNeuro ds004889,
~1121 subjects) for training only — ISLES 2022 remains the validation set, so the
reported numbers stay comparable across experiments.

---

## ADR-002: Dataset for V2 (perfusion)

**Date:** 2026-02-04 · **Status:** Accepted · **Outcome:** Pending data

**Problem:** MRI PWI or CT perfusion for the V2 perfusion pipeline?

**Options:**
1. ISLES 2015/16/17 — MRI PWI, but dated and limited
2. ISLES 2024 — CTP, current and well structured

**Decision:** ISLES 2024 (CTP)

**Rationale:** Better and more accessible data; CTP is what is actually used for triage
in clinical practice; the core/penumbra logic is identical for MRI PWI and CTP; the
follow-up infarct mask makes an ideal ground truth for evaluation.

**Outcome:** The threshold-based pipeline is implemented and unit-tested against
synthetic volumes, but the 99 GB dataset has not been downloaded, so **none of it has
been validated on real perfusion data**. Treat V2 as a prototype.

---

## ADR-003: Segmentation architecture

**Date:** 2026-02-04 · **Status:** Accepted · **Outcome:** Superseded by ADR-011

**Problem:** Which architecture for the baseline?

**Options:**
1. nnU-Net v2 — state of the art, but heavy to configure
2. MONAI 3D U-Net — simpler, integrates cleanly
3. Attention U-Net — adds attention gates
4. Swin UNETR — transformer-based

**Decision:** MONAI 3D U-Net as the baseline, with nnU-Net as a possible follow-up

**Rationale:** Fastest path to a first result, full control over the code, and good
enough to establish a baseline worth beating.

**Outcome:** Correct as a starting point — the baseline reached val_dice 0.606. It was
then replaced; see ADR-011.

---

## ADR-004: Report generation approach

**Date:** 2026-02-04 · **Status:** Accepted · **Outcome:** Implemented

**Problem:** How do you generate report prose without hallucinating?

**Options:**
1. End-to-end image → text (VLM) — modern, but hallucinates
2. Template-based from JSON — controllable, less natural-sounding
3. LLM paraphrase of the JSON behind a validator — a compromise

**Decision:** Template-based plus a validator for the MVP; LLM paraphrase considered
for a later version

**Rationale:** Zero hallucinations is a hard requirement, not a preference. Every
statement in the text must be traceable to a field in the findings JSON, and a template
guarantees that by construction. An LLM can be introduced later for more fluent prose,
but only behind the same validator.

**Outcome:** Implemented in `src/report/`. The validator runs five cross-checks (lesion
count, total volume, laterality, impression class, and no unverified numbers). The LLM
paraphrase has not been built.

---

## ADR-005: Vascular territory assignment

**Date:** 2026-02-04 · **Status:** Accepted · **Outcome:** Implemented, still a heuristic

**Problem:** How should the vascular territory (MCA/ACA/PCA/vertebrobasilar) be assigned?

**Options:**
1. Atlas-based mapping — register to a territory atlas
2. Rule-based on anatomy — centroid position → coarse rule
3. Learned classifier — train one on the data

**Decision:** Rule-based for the MVP, atlas-based later

**Rationale:** Far quicker to implement and adequate for a coarse classification, with
room to refine using an atlas once the rest of the pipeline is proven.

**Outcome:** Implemented in `src/findings/territory.py` as centroid heuristics. It
remains a heuristic — no atlas registration — and the report reflects that by carrying a
`territory_confidence` field rather than asserting the territory outright.

---

## ADR-006: Quality control and the refusal policy

**Date:** 2026-02-04 · **Status:** Accepted · **Outcome:** Implemented 2026-08-12

**Problem:** What should happen when the input data is of poor quality?

**Options:**
1. Always produce a prediction, flagged with low confidence
2. No-score when QC fails
3. Partial output — return whatever can be computed

**Decision:** No-score on a critical QC failure

**Rationale:** Safety outweighs completeness. Saying "I cannot read this" honestly is
better than guessing, and in a clinical setting a confident wrong answer is more
dangerous than no answer at all. The QC report is informative on its own.

**Outcome:** Implemented in `src/qc/qc_pipeline.py` and enforced in
`run_inference`, which now runs the gate *before* the model. Four checks:

| Check | Blocks on |
|-------|-----------|
| `modalities` | a required series missing, empty, all-zero or all-NaN |
| `spacing` | missing, non-numeric, zero/NaN, or outside 0.3–6.0 mm |
| `coverage` | volume below 32×32×8, not 3-D, or under 2% non-zero voxels |
| `dwi_adc_consistency` | DWI and ADC identical, near-identical (r > 0.98), or shape-mismatched |

On a critical failure the pipeline returns `pred_mask`, `findings` and `validation` as
`None` with a report explaining which gate failed — no mask, no numbers, nothing to
misread. The REST API answers `status="qc_failed"`, the Gradio demo shows the refusal,
and `scripts/infer_single.py` exits with code 2.

The fourth check earns its place: submitting the same series as both DWI and ADC is a
realistic operator error that would otherwise produce a confident, entirely wrong
ADC-restriction finding.

Validated on all 250 ISLES 2022 subjects — a gate that rejects valid studies would be
worse than no gate:

| | |
|---|---|
| Subjects checked | 250 |
| Blocked | **0** |
| False-positive rate | **0.0%** |
| DWI/ADC correlation on real data | min −0.835, median −0.644, max −0.038 |

The correlation is negative on every real study, as the physics implies: restricted
diffusion is bright on DWI and dark on ADC. The worst real case sits at −0.038 against
a +0.98 threshold, so the duplicate-series check has roughly a full unit of headroom
and cannot plausibly fire on a genuine pair.

---

## ADR-007: Storage formats

**Date:** 2026-02-04 · **Status:** Accepted · **Outcome:** Implemented

**Problem:** How should masks and structured findings be stored?

**Decision:**
- Masks: NIfTI (`.nii.gz`) — the medical imaging standard
- Findings: JSON, typed via a Pydantic schema
- Report: plain text
- Preview: PNG with an overlay

**Outcome:** All four in use. One subtlety cost real debugging time: the extractors
return numpy `float32` scalars, which are not JSON-serialisable, and every caller
originally hid this with `json.dumps(..., default=str)` — silently emitting volumes as
strings. The builder now coerces to plain Python floats so the schema holds.

---

## ADR-008: Experiment tracking

**Date:** 2026-02-04 · **Status:** Accepted · **Outcome:** NOT IMPLEMENTED

**Problem:** How should experiments be tracked?

**Options:**
1. MLflow — self-hosted, open source
2. Weights & Biases — cloud, better UI
3. TensorBoard — simple, ships with PyTorch
4. Plain logs — minimal

**Decision:** MLflow as the primary tracker, TensorBoard optional

**Rationale:** Free and self-hosted, keeps artifacts, parameters and metrics together,
makes runs easy to compare, and provides a model registry for versioning.

**Outcome:** **Never wired up.** Training reads YAML directly and writes a
`training_history.json` plus an `experiment_meta.json` per run; those artifacts are
committed under `Training_results/`. MLflow was removed from the dependencies rather
than left as an unused import. In hindsight, for a single-developer project with seven
runs, flat JSON artifacts in git have been sufficient and are easier for a reader to
inspect than an MLflow database.

---

## ADR-009: V2 perfusion thresholds

**Date:** 2026-02-04 · **Status:** Accepted · **Outcome:** Implemented

**Problem:** Which thresholds define core and penumbra?

**Decision:** Configurable, with these defaults:
- Hypoperfusion: Tmax ≥ 6 s
- Core: rCBF ≤ 30%
- Target mismatch: core < 70 mL, mismatch ratio > 1.8, mismatch volume > 15 mL

**Rationale:** These are the thresholds used in the DEFUSE-3 and EXTEND-IA trials and in
routine practice, so the output speaks the language a stroke clinician already uses.
Keeping them in config rather than hard-coded means they can be adjusted per site.

**Outcome:** Implemented in `src/v2_perfusion/`, configurable via `configs/v2_ctp.yaml`,
and echoed into the findings JSON as `thresholds_used` so any report states the criteria
it applied. Not yet validated on real CTP data — see ADR-002.

---

## ADR-010: Product positioning and safety

**Date:** 2026-02-04 · **Status:** Accepted · **Outcome:** Held up

**Problem:** How should the system be positioned with respect to safety?

**Decision:** Clinical decision support, explicitly not autonomous diagnosis

**Principles:**
- Human-in-the-loop is mandatory
- Output is a draft for review, never a final diagnosis
- No treatment recommendations
- Evidence-linked: every claim carries the mask and slice indices behind it
- Honest disclaimers and stated limitations

**Outcome:** Reflected throughout the README, the model card and the report templates.
Every lesion in the findings JSON carries a `mask_ref` and `evidence_slices`, so each
sentence can be traced back to voxels.

---

## ADR-011: Attention U-Net over the plain U-Net

**Date:** 2026-03-24 · **Status:** Accepted · **Outcome:** Implemented, current model

**Problem:** The combined-data U-Net plateaued at a per-subject Dice of 0.567, and tiny
lesions were far worse at 0.193. What is the cheapest change with the largest gain?

**Options:**
1. Higher input resolution — directly targets tiny lesions, but costs a lot of memory
2. Attention gates on the skip connections — small parameter increase
3. nnU-Net — likely the strongest, but a rewrite
4. Ensemble over 5 folds — reliable gain, 5x the training time

**Decision:** Attention U-Net (5.86M parameters), plus mild augmentation, AMP,
test-time augmentation and small-component filtering

**Rationale:** Attention gates suppress irrelevant background activation at the skip
connections, which is precisely the failure mode on small lesions in a large field of
view. It is a contained change to the architecture rather than to the whole pipeline.

**Outcome:** Per-subject Dice 0.567 → 0.691, median 0.629 → 0.772, HD95 19.1 → 13.4 mm,
and tiny-lesion Dice 0.193 → 0.377. Augmentation also cut the overfitting gap from 0.136
to 0.045. Tiny lesions remain the weakest category by a wide margin.

---

## ADR-012: Augmentation strength

**Date:** 2026-03-05 · **Status:** Accepted after a failure · **Outcome:** Implemented

**Problem:** The baseline overfitted (train/val Dice gap 0.136), so augmentation was
added — and the first attempt made things dramatically worse.

**Decision:** Mild augmentation only — `RandFlip` (LR 0.5, AP 0.3), Gaussian noise 0.05,
contrast 0.7–1.5

**Rationale:** The first configuration used aggressive affine and elastic deformation
and dropped val_dice from 0.606 to 0.405. Stroke lesions are small, irregular and
defined by subtle intensity differences; deformation destroys exactly the signal the
model needs. Flips are safe because brain anatomy is near-symmetric, and mild noise and
contrast changes mimic real scanner variation.

**Outcome:** Overfitting gap fell from 0.136 to 0.045 with no loss of accuracy. The
failed run is kept under `Training_results/Train_03_01_2026/` rather than deleted.

---

## Template for new records

```markdown
## ADR-XXX: [Title]

**Date:** YYYY-MM-DD · **Status:** Proposed / Accepted / Rejected / Superseded

**Problem:** [What is being decided?]

**Options:**
1. ...
2. ...

**Decision:** [What was chosen]

**Rationale:** [Why]

**Outcome:** [What actually happened once it met reality]
```
