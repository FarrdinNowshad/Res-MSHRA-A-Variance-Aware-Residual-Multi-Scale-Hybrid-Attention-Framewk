# Res-MSHRA-A-Variance-Aware-Residual-Multi-Scale-Hybrid-Attention-Framewk

**A Variance-Aware Residual Multi-Scale Hybrid Attention Framework for Cross-Dataset Histopathology Classification.**

Farrdin Nowshad¹, Sabit Ahamed Preanto², Hasan Imam Bijoy²,³

¹ Department of Electrical and Electronic Engineering, BRAC University, Dhaka, Bangladesh
² Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh
³ Department of Computing and Information System, Daffodil International University, Dhaka, Bangladesh

---

Res-MSHRA stacks two Multi-Scale Hybrid Residual Attention (MS-HRA) blocks on top of ResNet50 (truncated at `layer3`) and trains them with a two-phase schedule, focal loss with label smoothing, multi-seed ensembling, and 11-view test-time augmentation. The **same pipeline** is applied to three structurally distinct H&E datasets — **BreaKHis** (four magnifications), **BACH** (binary-collapsed), and **MHIST** (HP vs SSA, official split) — to test how well the recipe generalises across binary histopathology tasks.

> **Scope.** The principal contribution is the **methodological discipline** of the evaluation — variance-aware reporting, protocol transparency, and paired component ablation across magnifications. The architecture itself is essentially CBAM applied to ResNet50, and the paper treats it as the experimental vehicle through which that discipline was tested rather than as a novel design.

> **Status.** Manuscript under revision, targeting *Scientific Reports*. The results in this repository correspond to the version currently under revision.

---

## Contributions (from the paper)

1. A variance-aware, protocol-transparent evaluation of Res-MSHRA across BreaKHis, BACH, and MHIST under a single fixed training and inference pipeline, with per-seed AUC, ensemble lift, and dataset-specific hyperparameters reported alongside every headline result.
2. A decomposition of the sensitivity–specificity asymmetry on imbalanced histopathology data into two mechanisms — **post-hoc threshold selection** (determines magnitude) and **minority-class augmentation** (determines sign) — demonstrated through the contrasting behaviour of BreaKHis (Sens > Spec) and MHIST (Spec > Sens).
3. A paired component ablation across six configurations and four BreaKHis magnifications, using paired ΔAUC with 95% CIs, identifying components that consistently help (TTA, and the full MS-HRA block at intermediate magnifications) and those whose effects are not separable from seed variability (the individual CBAM modules, focal loss, and two-phase training). Negative results from SWA and post-hoc temperature scaling are also reported.

---

## Results

| Dataset           | Seeds | AUC   | Sens. | Spec. | Acc.  | Notes |
|-------------------|:-----:|:-----:|:-----:|:-----:|:-----:|---|
| BreaKHis 40×      | 3     | 0.961 | 0.943 | 0.889 | 92.7% | n = 344 test images |
| BreaKHis 100×     | 3     | 0.973 | 0.977 | 0.852 | 93.8% | best magnification |
| BreaKHis 200×     | 3     | 0.963 | 0.960 | 0.815 | 91.9% | |
| BreaKHis 400×     | 5     | 0.914 | 0.948 | 0.743 | 88.1% | 320 px input, strong aug, highest variance |
| BACH (binary)     | 5     | 0.986 | 0.950 | 1.000 | 97.5% | F1 = 0.974, TP/TN/FP/FN = 38/40/0/2 |
| MHIST (HP vs SSA) | 5     | 0.944 | 0.822 | 0.922 | 87.2% | official 977-image test split, F1 = 0.841 |

All numbers are seed-ensembled with 11-view TTA at the accuracy-optimal threshold. BreaKHis uses patient-stratified 80/20 splits; BACH uses a stratified random 80/20 split (n = 400); MHIST uses the official 2175/977 split.

**Honest comparison note.** On MHIST under the same official split, the original ResNet-18 baseline reports AUC 0.927 and HGPT (Liu et al.) reports 91.19% accuracy — the latter exceeds Res-MSHRA's 88.5% accuracy on the same split. The paper discusses this in §5.3.

---

## Repository contents

| Notebook | Purpose |
|---|---|
| `RES-MSHRA_BreaKHis.ipynb` | Trains and evaluates Res-MSHRA on BreaKHis across all four magnifications (40×/100×/200×/400×). |
| `RES-MSHRA_BACH.ipynb`     | Trains and evaluates on BACH, reformulated as a binary task (Normal+Benign vs InSitu+Invasive). |
| `RES-MSHRA_MHIST.ipynb`    | Trains and evaluates on MHIST (HP vs SSA) using the official train/test split. |
| `ablation_on_BreaKHis.ipynb` | Six-configuration ablation (A1–A6) on BreaKHis: removes TTA, both MS-HRA blocks, spatial attention, channel attention, focal loss, and the two-phase training schedule respectively. |

---

## Datasets

Download each dataset separately and place it where the relevant notebook expects it (paths are set at the top of each `CFG` class, defaulting to standard Kaggle input locations).

- **BreaKHis** — Breast Cancer Histopathological Database, 7909 H&E images at 40×/100×/200×/400× magnification.
  <https://www.kaggle.com/datasets/ambarish/breakhis>
- **BACH** — Breast Cancer Histology Challenge, 400 H&E images. We collapse the four-class labels into binary (Normal+Benign vs InSitu+Invasive).
  <https://www.kaggle.com/datasets/truthisneverlinear/bach-breast-cancer-histology-images>
- **MHIST** — Minimalist Histopathology image classification dataset, 3152 colorectal polyp images (HP vs SSA), with an official 2175/977 split.
  <https://bmirds.github.io/MHIST/>

Each dataset is governed by its own license — review the source pages before redistributing data.

---

## Architecture

```
Input (3 × 224 × 224)            [400× uses 320 × 320]
   │
   ▼
ResNet50 (ImageNet1K_V2 weights), truncated after layer3   → 1024 channels
   │
   ▼
MS-HRA block 1   (1024 → 1024, stride 1)
   │   ├─ residual conv pair
   │   ├─ CBAM-style channel attention (reduction ratio 16)
   │   ├─ spatial attention (7×7)
   │   └─ residual add
   ▼
MS-HRA block 2   (1024 → 2048, stride 2)   [same structure]
   │
   ▼
GAP → BN → Dropout → Linear(2048→512) → ReLU → BN → Dropout(0.3) → Linear(512→1)
```

**Total parameters: 87,916,677 (~88 M).**

---

## Training recipe

- **Two-phase schedule.** Phase 1: backbone frozen, head trained with cosine LR. Phase 2: backbone unfrozen with a 3-epoch warmup + cosine schedule, with separate (lower) backbone LR and head LR.
- **Loss.** Focal loss (γ = 2) with label smoothing (ε = 0.05). α = 0.33 on BreaKHis (benign upweighted against the ≈1:2 imbalance); α = 0.5 on BACH and MHIST (training sets already class-balanced).
- **Optimisation.** AdamW with weight decay 5e-3 (6e-3 at BreaKHis 400×), mixed precision (AMP), gradient clipping at 1.0, `drop_last`, `WeightedRandomSampler` for class balance.
- **Seed ensembling.** 3 seeds {42, 123, 2024} for BreaKHis 40×/100×/200×; 5 seeds {42, 123, 2024, 7, 1337} for BreaKHis 400×, BACH, and MHIST.
- **TTA.** 11 views — 6 rotation/flip variants + 5-crop.

### Hardware and software

- Single **NVIDIA Tesla T4** GPU (15 GB VRAM, CUDA Compute Capability 7.5)
- 30 GB system RAM (Kaggle online notebook environment)
- **PyTorch** (CUDA-enabled) with `torch.cuda.amp` for mixed precision
- `cudnn.benchmark = True`

---

## Reproducing the results

1. Download the three datasets from the links above and place them where each notebook's `CFG.DATA_ROOT` expects them. Defaults assume the Kaggle layout:
   - BreaKHis → `/kaggle/input/breast-histopathology-images/`
   - BACH → `../input/bach-binary/bach/{train,test}/{benign,malignant}/`
   - MHIST → `mhist_preprocessed/{train,test}/{benign,malignant}/` (HP → benign, SSA → malignant)
2. Open the relevant notebook in Kaggle (or any Jupyter environment with a CUDA-capable GPU).
3. Run all cells. Each notebook is self-contained — model definition, training, multi-seed loop, and evaluation are all in one file.

### Environment

- Python 3.10+
- PyTorch with CUDA, torchvision
- scikit-learn, numpy, PIL, matplotlib, seaborn, tqdm

A standard Kaggle Python image satisfies all of these out of the box.

---

## Ablation study (`ablation_on_BreaKHis.ipynb`)

Six configurations are compared against the full model across BreaKHis magnifications, using the same seeds as the full model (N = 3 at 40×/100×/200×, N = 5 at 400×). Paired ΔAUC with 95% confidence intervals isolate each component's contribution.

| Config | What is removed |
|---|---|
| A1 | Test-time augmentation (single-view inference instead of 11-view TTA) |
| A2 | Both MS-HRA blocks (head attached directly to ResNet50 `layer3`) |
| A3 | Spatial attention (channel attention + residual pathway retained) |
| A4 | Channel attention (spatial attention + residual pathway retained) |
| A5 | Focal loss → standard BCE (class-balanced sampling retained) |
| A6 | Two-phase training → single-stage fine-tuning |

**Headline findings.** TTA is the only universally beneficial component (its 95% CI excludes zero at 200× and 400×, directionally consistent at 40× and 100×). The MS-HRA block as a whole contributes at 100× (CI excludes zero); its effect is negligible at 40× and 400×. Individual CBAM channel and spatial attention modules do not show a separable AUC contribution. Focal loss vs BCE and two-phase vs single-stage training also fall within seed variability.

---

## Explainability

Section 3.11 / 4.7 of the paper apply four established post-hoc explanation methods alongside the model's native spatial-attention maps:

- **Grad-CAM** (final conv layer of MS-HRA block 2)
- **Grad-CAM++** (same target layer; pixel-wise weights from higher-order partial derivatives)
- **LIME** (500 perturbed samples, default Quickshift segmentation, top-8 positive superpixels)
- **GradientSHAP** (via `shap.GradientExplainer`, 4-image zero-tensor background)
- **Native spatial-attention maps** extracted via forward hooks on the post-sigmoid 14×14 attention output of each MS-HRA block

Implementation uses the public `pytorch_grad_cam` library, `lime`, and `shap`. The relationship between attention maps and the ablation findings is discussed in §4.7.

---

## Reported negative results

In the spirit of the paper's "methodological discipline" framing, two interventions that did **not** help are reported alongside the headline metrics:

- **Stochastic Weight Averaging (SWA)** — best-validation checkpoint outperformed SWA in 32 of 33 runs across the magnification × seed grid.
- **Post-hoc temperature scaling** — fitted temperatures of T = 9–52 indicated a degenerate logit space; calibration is therefore excluded from headline metrics.

---

## Citation

The manuscript is under revision and not yet published. A citation entry will be added once a DOI is available. If you use this code in the meantime, please open an issue so the appropriate preprint version can be pointed to.

---

## License

Code: to be added (likely MIT). Datasets are governed by their original licenses — see each dataset's source page.
