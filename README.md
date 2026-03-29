# Derm7pt+: A Concept-Consistent Dermoscopy Benchmark

---

## Overview

Concept Bottleneck Models route all predictions through a layer of clinically interpretable concepts. When a dataset contains concept-level inconsistencies — where identical concept profiles map to conflicting diagnosis labels — the bottleneck becomes unresolvable, imposing a hard accuracy ceiling regardless of backbone architecture or training strategy.

We apply **rough set theory** to the Derm7pt dermoscopy benchmark and find that:

- **305** unique concept profiles are formed by the 7 dermoscopic criteria of the 7-point melanoma checklist
- **50 profiles (16.4%)** are inconsistent, spanning **306 images (30.3%)** of the dataset
- This yields a theoretical accuracy ceiling of **92.1%** for any hard CBM trained on the raw data

**Derm7pt+** is a concept-consistent benchmark derived from Derm7pt via rough set analysis and filtering of all boundary-region images. Two filtered variants are provided.

| Property | Asymmetric | Symmetric |
|---|---|---|
| Total images | 841 | 705 |
| Melanoma samples | 252 | 116 |
| Class ratio (M:NM) | 1:2.3 | 1:5.1 |
| Quality of classification γ | >0.697 | 1.000 |
| Accuracy ceiling | 83.2% | 100% |
| Full concept consistency | Partial | Yes |

---

## What Is Provided

```
derm7pt-plus/
├── images/                                  # Cropped dermoscopic images (383x384 px)
├── meta/
│   └── meta.csv                             # Concept annotations and diagnosis labels
├── splits/
│   ├── symmetric_train_indexes_filtered.csv
│   ├── symmetric_valid_indexes_filtered.csv
│   ├── symmetric_test_indexes_filtered.csv
│   ├── asymmetric_train_indexes_filtered.csv
│   ├── asymmetric_valid_indexes_filtered.csv
│   └── asymmetric_test_indexes_filtered.csv
├── Derm7pt_CBM.ipynb
└── README.md
```

All images have been manually cropped to remove border artifacts and center the lesion within the frame. They are saved at **383×384 pixels**. Each case has two paired images: a clinical photograph (`clinic`) and a dermoscopic image (`derm`).

---

## Filtering Strategies

**Symmetric removal** restricts the dataset to images whose concept profile maps unambiguously to a single diagnosis label. This yields 705 cases, a perfect quality of classification (γ = 1.0), and no hard accuracy ceiling. The cost is the removal of 136 melanoma cases (54.0% of all melanoma cases), worsening class imbalance from 1:3.0 to 1:5.1.

**Asymmetric removal** retains all melanoma cases from the boundary region while removing only the conflicting non-melanoma samples. This embeds the clinical prior that false negatives carry a higher cost than false positives in a screening setting. All 252 melanoma cases are preserved and the class ratio improves to 1:2.3. A residual accuracy ceiling of 83.2% applies.

Neither strategy is universally optimal. The choice should be explicitly documented in any publication that uses Derm7pt+ for CBM evaluation.

---

## meta.csv Format

The file contains 1,011 rows (one per case) and 16 columns. Rows are zero-indexed and correspond directly to the index values in the split files.

| Column | Description |
|---|---|
| `diagnosis` | Full diagnostic label (e.g., `melanoma`, `clark nevus`, `seborrheic keratosis`). For binary melanoma/non-melanoma classification, all values starting with `melanoma` map to the positive class. |
| `pigment_network` | A mesh-like grid of pigmented lines overlying a lighter background, formed by melanin distributed along the rete ridges. A typical network shows regular meshwork and uniform color, while an atypical network is irregular in shape or color and abruptly terminates at the lesion periphery. Values: `typical`, `atypical`, `absent` |
| `streaks` | Radial projections at the lesion border arising from confluent junctional nests or radial melanocyte growth. Regular streaks are uniformly distributed around the periphery, while irregular streaks are focal or asymmetrically distributed and carry higher diagnostic significance for melanoma. Values: `regular`, `irregular`, `absent` |
| `pigmentation` | The distribution and uniformity of coloration within the lesion, reflecting melanin density across the epidermis and dermis. Regular pigmentation is homogeneous and symmetrically distributed, while irregular pigmentation involves abrupt color variations or eccentric darkening associated with atypical melanocytic proliferation. Values: `diffuse_regular`, `localized_regular`, `localized_irregular`, `diffuse_irregular`, `absent` |
| `regression_structures` | Areas where melanocytic proliferation has undergone partial or complete regression, replaced by fibrosis or melanophages. Blue areas correspond to dermal melanophages, white areas to fibrosis, and combinations indicate the co-occurrence of both. Values: `blue_areas`, `white_areas`, `combinations`, `absent` |
| `dots_and_globules` | Small round to oval structures corresponding to pigment aggregates in the epidermis or dermis. Regular variants are uniformly sized and symmetrically distributed, while irregular variants differ in size, shape, or distribution and are associated with melanoma. Values: `regular`, `irregular`, `absent` |
| `blue_whitish_veil` | A confluent structureless blue-white area overlying a raised portion of the lesion, corresponding histopathologically to compact orthokeratosis above a hyperpigmented epidermis with dermal melanophages. It is one of the most specific dermoscopic features of invasive melanoma. Values: `present`, `absent` |
| `vascular_structures` | Blood vessel morphologies visible under dermoscopy, whose patterns reflect the vascular architecture of different lesion types. Comma vessels are curved and associated with dermal nevi, arborizing vessels are branching and typical of basal cell carcinoma, hairpin vessels are looped and seen in keratinizing tumors, dotted vessels appear as small red dots common in melanoma and Spitz nevi, and linear irregular vessels are poorly organized and associated with amelanotic melanoma. Values: `comma`, `wreath`, `arborizing`, `hairpin`, `dotted`, `within_regression`, `linear_irregular`, `absent` |
| `derm` | Filename of the dermoscopic image (relative to `images/`). |

```

## Split File Format

Each split file contains a single column named `indexes` with integer values. Each value is a zero-based row index into `meta.csv`.

**Example** (`symmetric_train_indexes_filtered.csv`):

```
indexes
16
17
27
...
```

---

## Loading a Split

The following code loads the training, validation, and test sets for the symmetric strategy. Each resulting dataframe contains the dermoscopic image path, the 7 concept columns, and the diagnosis label.

```python
import pandas as pd

meta = pd.read_csv("meta/meta.csv")

concept_cols = [
    "pigment_network", "streaks", "pigmentation",
    "regression_structures", "dots_and_globules",
    "blue_whitish_veil", "vascular_structures"
]
selected_cols = ["derm"] + concept_cols + ["diagnosis"]

train_idx = pd.read_csv("splits/symmetric_train_indexes_filtered.csv")["indexes"]
val_idx   = pd.read_csv("splits/symmetric_valid_indexes_filtered.csv")["indexes"]
test_idx  = pd.read_csv("splits/symmetric_test_indexes_filtered.csv")["indexes"]

train_df = meta.iloc[train_idx][selected_cols].reset_index(drop=True)
val_df   = meta.iloc[val_idx][selected_cols].reset_index(drop=True)
test_df  = meta.iloc[test_idx][selected_cols].reset_index(drop=True)
```

For binary melanoma/non-melanoma classification, derive the binary label as follows:

```python
for df in [train_df, val_df, test_df]:
    df["label"] = df["diagnosis"].str.startswith("melanoma").astype(int)
```

To switch to the asymmetric strategy, replace the split filenames with the `asymmetric_*` equivalents. The `meta.csv` file is shared across both strategies.

---

## Citation

If you use Derm7pt+ in your research, please cite our paper:

```bibtex
@article{napoles2026derm7pt,
  title     = {Concept Inconsistency in Dermoscopic Concept Bottleneck Models:
               A Rough-Set Analysis of the Derm7pt Dataset},
  author    = {N\'apoles, Gonzalo and Grau, Isel and Salgueiro, Yamisleydi},
  journal   = {[TBA]},
  year      = {2026}
}
```

Derm7pt+ is derived from the original Derm7pt dataset. If you use the images or metadata, we would kindly ask you to cite the original dataset paper:

```bibtex
@article{Kawahara2018-7pt,
  author    = {Kawahara, Jeremy and Daneshvar, Sara and Argenziano, Giuseppe and Hamarneh, Ghassan},
  title     = {Seven-Point Checklist and Skin Lesion Classification Using Multitask Multimodal Neural Nets},
  journal   = {IEEE Journal of Biomedical and Health Informatics},
  volume    = {23},
  number    = {2},
  pages     = {538--546},
  year      = {2019},
  doi       = {10.1109/JBHI.2018.2824327}
}
```

The original Derm7pt images are distributed under a CC BY-NC-ND 4.0 license and can be downloaded from [derm.cs.sfu.ca](http://derm.cs.sfu.ca).

---

## License

The split files and metadata in this repository are released under the MIT License. The dermoscopic images are derived from the original Derm7pt dataset and are subject to its CC BY-NC-ND 4.0 license. Users are responsible for ensuring their use complies with the terms of that license and any applicable institutional or ethical requirements for dermoscopic image data.
