# BraTS 2026 MET — Daily Paper Review Log

## 2026-06-16

First run of this log (no prior entries to dedupe against). Found 10 papers spanning BraTS-METS-specific challenge/method papers, small-lesion-sensitive architectures (Mamba variants), and direct nnU-Net extensions. Best picks: the two-stage modality-impact metastases paper (directly targets small-lesion sensitivity, our weak NETC class) and ++nnU-Net's registration-based augmentation module (drop-in extension of our existing baseline).

### 1. [Analysis of the MICCAI BraTS-METS 2025 Lighthouse Challenge: Brain Metastasis Segmentation on Pre- and Post-treatment MRI](https://arxiv.org/abs/2504.12527)
- **Technique/architecture:** Not a method paper — describes the 2025 Lighthouse Challenge design: inter/intra-rater annotation variability study (4 independent segmentation passes per case, 2 from scratch + 2 with AI pre-segmentation, recorded on video), and release of the 2023/2024 pre- and post-treatment datasets plus a new high-quality annotated test set.
- **Results:** No model results; describes dataset/annotation protocol only (the same dataset family our 1296-case training set comes from).
- **Verdict:** Interesting but low priority — no algorithmic content to implement, but useful for understanding annotation noise/variability in our own labels (could inform loss weighting or label smoothing decisions).

### 2. [Improving Generalization of Deep Learning for Brain Metastases Segmentation Across Institutions](https://arxiv.org/abs/2604.00397)
- **Technique/architecture:** Domain-adaptation/generalization framework for brain-metastases segmentation trained at one institution and evaluated at others, addressing scanner/protocol/demographic shift (abstract-level detail only; PDF text was not extractable).
- **Results:** Not confirmed from abstract — could not retrieve full results text.
- **Verdict:** Worth testing/ablating — our dataset mixes SRI24-atlas and native UCSD clinical-space cases, which is itself a domain-shift problem; techniques here may transfer to harmonizing across our two coordinate spaces.

### 3. [Segmentation of Brain Metastases in MRI: A Two-Stage Deep Learning Approach with Modality Impact Study](https://arxiv.org/abs/2407.14011)
- **Technique/architecture:** Systematically studies which MRI modality combination helps most (finds T1c+T1+FLAIR beats using all modalities), then proposes a two-stage detection-then-segmentation 3D U-Net pipeline explicitly designed to catch small metastases that single-pass models miss.
- **Results:** Reports the targeted 3-modality + two-stage combo "significantly" outperforms single-pass models on small lesions; sets a new benchmark on their brain-met cohort (exact Dice not surfaced in abstract, not directly comparable to BraTS-MET).
- **Verdict:** Strong candidate — implement. Directly attacks small/multi-focal lesion sensitivity, which is exactly our NETC weak spot (0.460 Dice); the two-stage detect-then-segment pattern is a realistic single-GPU addition on top of nnU-Net.

### 4. [Adaptable Segmentation Pipeline for Diverse Brain Tumors with Radiomic-guided Subtyping and Lesion-Wise Model Ensemble](https://arxiv.org/html/2512.14648v1)
- **Technique/architecture:** BraTS 2025 Lighthouse pipeline (covers PED, MEN, MEN-RT, and MET tracks) that uses radiomic features to detect tumor subtype for balanced training, ensembles multiple SOTA models, and applies custom lesion-level metrics to weight ensemble members and tune lesion-wise post-processing.
- **Results:** Reports performance "comparable to top-ranked algorithms" across the BraTS test sets; not broken out by class for MET specifically in the abstract.
- **Verdict:** Worth testing/ablating — lesion-wise post-processing and ensemble weighting could help with our 75% multi-focal-lesion cases, but the radiomic-subtyping pipeline is a heavier lift than a single-GPU setup easily supports.

### 5. [S³-Mamba: Small-Size-Sensitive Mamba for Lesion Segmentation](https://arxiv.org/abs/2412.14546)
- **Technique/architecture:** Enhanced Visual State Space block with multiple residual connections and channel-wise attention to preserve small-lesion features through downsampling, plus a Tensor-based Cross-feature Multi-scale Attention module and a regularized curriculum-learning schedule that orders training samples from easy (large lesions) to hard (small lesions).
- **Results:** Reports superiority over baselines on three medical segmentation datasets, particularly for small lesions (exact Dice numbers not surfaced; datasets are not BraTS-MET).
- **Verdict:** Strong candidate — implement. The curriculum-by-lesion-size idea and the small-lesion-preserving SSM block are directly aimed at the same failure mode driving our low NETC Dice, and the curriculum-learning piece especially is a cheap addition to an existing nnU-Net training loop.

### 6. [MHMamba: Multi-Head Mamba for 3D Brain Tumor Segmentation](https://arxiv.org/abs/2605.16464)
- **Technique/architecture:** U-shaped 3D architecture with a multi-head state-space (Mamba) backbone — splits channels into parallel SSM heads with residual aggregation, adds a channel-space calibration module and adaptive skip-connection fusion, aiming to fix Transformer-style long-range modeling cost in 3D MRI.
- **Results:** Reports improvements in accuracy, boundary smoothness, and sensitivity specifically to "tumor core and small-volume enhancement areas" while keeping Mamba's linear complexity (no BraTS-MET numbers; likely BraTS-glioma benchmarks).
- **Verdict:** Worth testing/ablating — relevant to small enhancing-tumor sensitivity (our ET/NETC classes), but swapping the whole backbone from nnU-Net's CNN to a Mamba U-Net is a much bigger single-GPU engineering investment than the module-level papers above.

### 7. [++nnU-Net: Scaling nnU-Net with Prefix-Based Data Augmentation](https://arxiv.org/abs/2606.10713)
- **Technique/architecture:** Registration-based data augmentation module that runs before nnU-Net's standard preprocessing/training pipeline, generating anatomically plausible augmented training pairs via image registration ("prefix" stage prepended to the normal nnU-Net workflow).
- **Results:** Outperforms vanilla nnU-Net across five 2D datasets, with up to ~22% Dice improvement in the most prominent cases; not yet validated on 3D or on BraTS-style multi-class tumor data.
- **Verdict:** Strong candidate — implement. This is a literal drop-in extension to the exact nnU-Net v2 3d_fullres baseline we're already running, so integration cost is low; main risk is that reported gains are 2D-only and may not transfer directly to our 3D multi-focal-lesion setting, so worth a quick ablation before committing.

### 8. [Conformal Lesion Segmentation for 3D Medical Images](https://arxiv.org/abs/2510.17897)
- **Technique/architecture:** Post-hoc, training-free framework (Conformal Lesion Segmentation) that calibrates per-sample decision thresholds via conformal prediction on a held-out calibration set, giving a statistical guarantee that test-time false-negative rate stays below a target tolerance — replacing the usual fixed 0.5 threshold.
- **Results:** Demonstrated on kidney-lesion datasets reducing false-negative rates versus fixed-threshold baselines; not benchmarked on brain tumor data.
- **Verdict:** Worth testing/ablating — since NETC's low Dice is partly a missed-small-lesion problem, a calibrated per-class threshold (instead of nnU-Net's default 0.5 argmax) is a near-zero-cost post-processing experiment worth running on our existing fold-0 predictions before touching the architecture.

### 9. [DRBD-Mamba for Robust and Efficient Brain Tumor Segmentation with Analytical Insights](https://arxiv.org/abs/2510.14383)
- **Technique/architecture:** Dual-resolution bi-directional Mamba block that captures multi-scale long-range dependencies while cutting the computational overhead of standard multi-axial Mamba scanning, using a space-filling curve for the 3D-to-1D feature mapping to preserve spatial locality; also includes an analysis of Mamba robustness across different BraTS data splits.
- **Results:** Reported as more computationally efficient than prior Mamba 3D segmentation models at comparable accuracy on BraTS-glioma splits (no BraTS-MET numbers).
- **Verdict:** Worth testing/ablating — the efficiency focus makes it the most single-GPU-realistic of the Mamba options here, but it's still a backbone swap with non-trivial reimplementation risk versus the nnU-Net-compatible papers above.

### 10. [Postoperative glioblastoma segmentation: Development of a fully automated pipeline using deep convolutional neural networks and comparison with currently available models](https://arxiv.org/abs/2404.11725)
- **Technique/architecture:** MONAI- and nnU-Net-trained pipeline specifically segmenting tumor subregions plus the surgical resection cavity on postoperative glioma MRI, validated externally across two Spanish centers and a public dataset using Dice, Jaccard, and volumetric similarity.
- **Results:** Reported strong performance in classifying extent of resection; exact Dice for the cavity class not surfaced in the abstract, and glioma (not metastasis) postoperative anatomy may differ somewhat from our RC class.
- **Verdict:** Worth testing/ablating — most directly relevant prior work on segmenting a resection-cavity label with nnU-Net specifically; worth reading the full method for cavity-specific preprocessing/postprocessing tricks even though it's not metastasis-specific and not numerically comparable to BraTS-MET.

---
