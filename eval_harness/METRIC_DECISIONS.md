# BraTS-METS 2025 — Local Eval Harness: Metric Decision Log

_Purpose: a single reference for every metric definition, constant, and gap-resolution
choice behind the local evaluation harness, so local numbers can be reconciled with
the official Synapse leaderboard. Each entry records **what we chose**, **why**, and
**provenance** (verified-from-source vs. our-decision-pending-anchor)._

_Last updated: 2026-06-16_

---

## Provenance legend
- ✅ **VERIFIED** — taken verbatim from official source (challenge paper or official scorer code).
- 🟡 **OUR DECISION** — chosen by us where the public sources are silent; to be confirmed
  against ONE real leaderboard score ("the anchor"). Marked `ANCHOR TODO` in code.
- 🔴 **OPEN** — not yet resolved.

## Sources
- **[PAPER]** BraTS-METS 2025 Lighthouse Challenge analysis, arXiv:2504.12527
  (downloaded; text at `/tmp/brats_mets_2025.txt` during build).
- **[SCORER-2023]** Official BraTS lesion-wise metrics code, `rachitsaluja/BraTS-2023-Metrics`
  `metrics.py` — same lineage as the 2025 MLCube scorer.
- **[PANOPTICA]** `BrainLesion/panoptica` `metrics.md` — instance detection engine.
- **[SYNAPSE]** Leaderboard page `syn74274097` (source of the 24 column names; JS-gated, scorer is a Docker MLCube — not readable as plain source).
- **[DATASET]** This project's own label scheme (SESSION_STATE.md, convert_to_nnunet.py).

---

## 0. Dataset label scheme (this project's .nii.gz files) — ✅ VERIFIED [DATASET]
```
0 = background
1 = NETC  (non-enhancing tumor core)
2 = SNFH  (edema / FLAIR hyperintensity)
3 = ET    (enhancing tumor)
4 = RC    (resection cavity; rare, post-treatment only)
```
> NOTE: The PAPER's Table 1 uses a DIFFERENT integer convention (ET=2, SNFH=3) and even
> mislabels TC as "ET+SNFH". We do NOT follow the paper's integers or its TC text. The
> **official scorer code** [SCORER-2023] is authoritative and matches OUR dataset integers.

---

## 1. Region (ROI) definitions — ✅ VERIFIED [SCORER-2023]
Built from the official `get_TissueWiseSeg()` unions, mapped onto our label integers:

| Region | Label union (our integers) | Meaning |
|--------|----------------------------|---------|
| **ET** | {3}                        | enhancing tumor |
| **TC** | {1, 3}                     | tumor core = NETC + ET |
| **WT** | {1, 2, 3}                  | whole tumor = NETC + SNFH + ET |
| **RC** | {4}                        | resection cavity (standalone — see §2) |

> TC = ET + NETC (classic BraTS), **NOT** ET + SNFH. The paper's Table 1 wording was a typo;
> the scorer code settles it. ET and WT are evaluated as distinct entities [PAPER].

---

## 2. RC (resection cavity) handling — 🟡 OUR DECISION (Gap 2)
- RC = label {4}, scored as its **own standalone region**. NOT merged into TC or WT.
- The leaderboard reports `*_rc` columns, so we compute all 4 RC metrics.
- **Cases without RC** (the majority [PAPER]): RC is treated like any absent class —
  **excluded** from that case's RC scores and from the RC aggregate denominator
  (i.e. RC mean is over RC-present cases only). See §6 aggregation.
- **Why:** consistent with how an absent class is handled everywhere; avoids diluting the
  RC mean with structural zeros. `ANCHOR TODO`: confirm leaderboard doesn't instead
  count RC-absent cases as perfect (1.0) or as 0.

---

## 3. Lesion-wise DSC — ✅ VERIFIED constants [SCORER-2023], BraTS-MET variant
Per-region, per-lesion Dice then averaged.
- **Dilation factor = 1** (MET-specific; GLI/PED/SSA use 3). ✅
- **Connectivity = 26** for connected components. ✅
- **Lesion volume threshold = 2 mm³** — GT lesions below this are excluded from
  lesion-wise calc (noise floor). ✅ (MET-specific; other tasks use 50.)
- **Matching:** dilate each GT connected component by factor 1, intersect with prediction;
  overlapping predicted components are the match. ✅
- **False-negative / false-positive lesion penalty: DSC = 0.** ✅
- Data are resampled to **1 mm³ isotropic** before scoring [PAPER §509] — so 1 voxel ≈ 1 mm³.

---

## 4. Lesion-wise NSD (Normalized Surface Dice) — ✅ tolerances VERIFIED [PAPER + external]
- NSD replaces the older lesion-wise HD95 for 2025. Same lesion matching as §3.
- **Tolerance τ: compute BOTH 0.5 mm AND 1.0 mm** — these are the documented BraTS-METS
  NSD tolerances. ✅ Carry both columns side-by-side now; drop one once the anchor tells
  us which the leaderboard reports. (Gap 1 — matches the user's "use both, prune later".)
- **FN/FP lesion penalty for NSD = 0** — 🟡 OUR DECISION (Gap 4). NSD is a 0–1 ratio so a
  miss → 0 is the sensible analogue of the DSC=0 rule. (Old HD95 used 374; not applicable
  to a ratio metric.) `ANCHOR TODO`: confirm 0 (vs. excluding the lesion).

---

## 5. Small-instance detection (TP / FN / FP / F1) — 🟡 OUR DECISION (Gap 3) + ✅ formula [PANOPTICA]
A pure **detection** track (did we find the lesion? — not how well we traced it), per region.
- **"Small" cutoff: GT lesions with volume < 27 mm³** are the small-instance population.
  🟡 OUR DECISION — the user set 27 mm³ (this project's documented sub-detection band).
  `ANCHOR TODO`: confirm whether it's <27, a 27–275 band, or another cutoff.
- **Matching = ANY overlap** (overlap threshold → ~0). A predicted component touching a
  GT lesion counts as detected. 🟡 OUR DECISION — detection-first, per user.
  `ANCHOR TODO`: confirm the leaderboard's IoU/overlap threshold.
- **TP** = small GT lesion that was detected; **FN** = small GT lesion missed;
  **FP** = predicted lesion with no GT match (in the small population).
- **F1 = TP / (TP + 0.5·(FP + FN))** — this is panoptica's Recognition Quality (RQ),
  which IS the F1 score. ✅ VERIFIED [PANOPTICA].
- Engine: `panoptica` instance matcher configured for any-overlap.

---

## 6. Aggregation across cases — 🟡 OUR DECISION (Gap 5)
- Per-case: compute the 24 values; absent regions in that case → NaN for that region.
- Aggregate ("..._mean_..."): mean over cases **where that region is present** (NaN-skipping).
  i.e. a region's mean denominator = number of cases that actually contain that region.
- **Why:** avoids penalizing/rewarding the model for a structure that doesn't exist in a
  case (e.g. RC in pre-treatment, ET in the 46 zero-ET cases). Consistent with §2.
- `ANCHOR TODO`: confirm leaderboard denominator (all cases vs. region-present cases).

---

## 7. The 24 leaderboard columns → engine map — ✅ structure VERIFIED
| Columns | Engine | Section |
|---------|--------|---------|
| `Lesionwise_dsc_mean_{et,tc,wt,rc}` | lesion-wise DSC | §3 |
| `Lesionwise_nsd_mean_{et,tc,wt,rc}` | lesion-wise NSD (τ=0.5 & 1.0) | §4 |
| `Small_instance_{tp,fn,fp,f1}_{et,tc,wt,rc}` | panoptica detection | §5 |

4 regions × (2 lesionwise + 4 small-instance) = **24**. ✅

---

## 8. The hard limit on "identical"
The official scorer is a **Docker MLCube** [SYNAPSE] — we cannot read it as source.
Everything ✅ above is mirrored from the public paper + same-lineage scorer code and is
identical-by-construction. Everything 🟡 is a documented best-effort default that we will
LOCK with **one** real leaderboard score (the "anchor"). Until that anchor exists, the
harness is "spec-faithful", not "proven bit-identical". Every 🟡 is marked `ANCHOR TODO`
in code so reconciliation is a quick parameter check, not a rebuild.

### Open `ANCHOR TODO` list (resolve when first leaderboard score arrives)
1. NSD τ: keep 0.5 or 1.0 (or both)? — §4
2. NSD miss penalty = 0? — §4
3. Small-instance cutoff = <27 mm³? — §5
4. Small-instance overlap threshold = any-overlap? — §5
5. Aggregation denominator = region-present cases? — §6
6. RC-absent cases: excluded vs. 1.0 vs. 0? — §2
