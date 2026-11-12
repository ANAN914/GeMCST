# GeMCST

Gene-anchored multi-view graph contrastive learning for spatial domain identification in cancer spatial transcriptomics.

Spatial proximity and morphological similarity can coexist with meaningful local differences in gene expression. GeMCST addresses this challenge by using gene expression relationships to guide how spatial and histological evidence contributes to spatial domain learning. The framework filters redundant morphological information, encodes RNA, spatial and morphology graph views separately, and applies RNA community informed contrastive constraints to preserve local transcriptional differences while identifying spatially coherent tissue regions.

In a human breast cancer benchmark, the manuscript reports that GeMCST ranked first by average rank across four metrics among 15 methods, including 14 baselines. It also examines local agreement with reference annotations, region-specific functions, continuous expression gradients within domains and communication between domains. The downstream gradient and communication analyses described in the manuscript are outside the scope of the `all` command in this release.

**Associated manuscript:** *GeMCST: Gene-anchored Multi-view Graph Contrastive Learning for Spatial Domain Identification in Cancer Spatial Transcriptomics*.

The framework comprises:

- **Morphological feature extraction:** UNI2-h extracts local H&E image features, followed by token pruning for diversity to reduce redundant morphological information.
- **Multiview graph learning:** separate RNA, spatial and morphology graphs are encoded by independent Center-Adaptive Residual Graph Encoder (CARE) branches and integrated through spot-specific attention.
- **RNA-informed representation learning:** Spatial RNA-community Informed Contrastive Learning (SRI-CLR) uses RNA communities and spatial relationships to constrain the joint representation.
- **Spatial domain identification:** Soft-Potts soft clustering and Cancer Boundary Refinement (CBR) combine neighbourhood consistency with conservative boundary correction.

# Installation

GeMCST is implemented in Python. A CUDA-capable GPU is recommended for UNI2-h feature extraction and model training. The manuscript reports Python 3.10 and PyTorch 2.5.1; the dependency list in this repository is not a fully pinned reproduction environment.

## Step 1: Obtain the source code

Download or copy this release directory, then open a terminal in it:

```sh
cd GeMCST
```

## Step 2: Create a Conda environment

```sh
conda create -n env_GeMCST python=3.10
conda activate env_GeMCST
```

## Step 3: Install required packages

Install mutually compatible versions of PyTorch, torchvision, PyTorch Geometric and torch-sparse for your Python and CUDA environment. Then install the remaining requirements:

```sh
python -m pip install -r requirements.txt
```

The image feature workflow imports `get_encoder` from the UNI package and requests the `uni2-h` encoder. Install the UNI package separately and obtain access to its model weights under the provider's terms. Authenticate outside the source code. Datasets and pretrained weights are not included in this release.

## Step 4: Install GeMCST

```sh
python -m pip install .
```

# Usage and reproducibility

## Input data

For a Visium sample, arrange the input files as follows:

```text
DATA_DIR/
└── SAMPLE_ID/
    ├── filtered_feature_bc_matrix.h5
    ├── metadata.tsv
    └── spatial/
        ├── tissue_hires_image.png
        ├── tissue_lowres_image.png
        ├── tissue_positions_list.csv
        └── scalefactors_json.json
```

`metadata.tsv` is optional for training but required for evaluation against reference annotations. Its index should contain spot barcodes matching the expression matrix; reference labels are read from `Classification`. Supported 10x MTX layouts can also be used by the data loader.

## Run the pipeline

For the H1 HER2-positive breast cancer sample:

```sh
python -m GeMCST all --data-dir /path/to/data --sample-id H1 --results-dir /path/to/results --metadata-path /path/to/data/H1/metadata.tsv --dataset-profile her2_breast --seed 0 --n-domains-mode auto
```

Replace the paths with your local directories. For annotated samples, pass `--metadata-path` explicitly so the evaluation stage can find the reference labels and compute ARI/NMI. The intended installed entry points are `gemcst` and `python -m GeMCST`.

| Stage | Operation |
| --- | --- |
| `image` | Extract morphological features and prepare RNA features and multiview graphs. |
| `train` | Train the model, identify spatial domains and apply configured boundary refinement. |
| `evaluate` | Evaluate predictions against metadata and generate evaluation plots. |
| `all` | Run `image`, `train` and `evaluate` in sequence. |

To run an individual stage, replace `all` in the command above. Run `image` before `train` when precomputed inputs are unavailable. For an unannotated sample, run `image` and `train` separately rather than the supervised evaluation workflow.

```sh
python -m GeMCST --help
```

Intermediate features, graphs, model outputs and evaluation files are written beneath the results directory. Retain the saved configuration with the results to record the settings used for each run.

## Scope of reproduction

This release provides the feature extraction, graph preparation, domain training and evaluation workflow. The manuscript also describes intradomain expression-gradient and interdomain communication analyses; these downstream analyses are not executed by the `all` command.

Report the random seed, dataset profile, domain-count selection, model settings, CBR settings and final CBR prediction field used for evaluation. If annotations inform the number of domains or a refinement procedure, report that use explicitly. Package construction and entry-point checks do not establish numerical reproduction of the manuscript results.

# Parameter Settings

The command line sets data paths, sample ID, dataset profile, seed and domain-count selection mode. Additional model parameters are configured through environment variables.

## Parameter definitions

| Parameter | Description | Current default or behaviour |
| --- | --- | --- |
| `--seed` | Random seed. | `0` |
| `--dataset-profile` | Select sample-specific configuration. | `auto` |
| `--n-domains-mode` | Select how the domain count is determined. | `auto` |
| `N_DOMAINS` | Explicit number of spatial domains. | Overrides inference where supported; DLPFC has additional profile logic. |
| `PRETRAIN_EPOCHS` | Number of pretraining epochs. | `500` |
| `TRAIN_EPOCHS` | Number of joint-training epochs. | `350` for H1 and V1 breast profiles; `460` otherwise. |
| `DIVPRUNE_KEEP_RATIO` | Fraction of morphological patch tokens retained by diversity pruning. | `0.75` |
| `SRI_CLR_WEIGHT` | Weight of the SRI-CLR loss. | `0.06` |
| `SRI_CLR_MARGIN` | Contrastive separation margin. | `0.75` |
| `SRI_CLR_WARMUP_EPOCH` | Epoch at which SRI-CLR is enabled. | Profile-dependent; H1 uses the warm-up ratio unless overridden. |
| `SOFT_POTTS_ETA` | Soft-Potts configuration coefficient. | `0.25` |
| `SOFT_POTTS_LAMBDA` | Soft-Potts configuration coefficient. | `0.05` |
| `SOFT_POTTS_ITERS` | Number of Soft-Potts update iterations. | `6` |
| `EXPRESSION_GRAPH_K` | Number of expression-graph neighbours. | Profile-dependent. |
| `SPATIAL_GRAPH_K` | Number of spatial-graph neighbours. | Profile-dependent. |
| `EDGE_GRAPH_K` | Number of morphology-graph neighbours. | Profile-dependent. |

Profile settings and environment-variable overrides should be checked together when reproducing a specific experiment.

## Manuscript training settings

The manuscript reports 500 pretraining epochs followed by 350 joint-training epochs, an SRI-CLR loss weight of 0.06, SRI-CLR activation at epoch 105 and Soft-Potts auxiliary-label updates every 20 epochs. These are reported experimental settings, not a claim that every dataset profile uses identical defaults.

For example, set the training duration and SRI-CLR start explicitly before running the pipeline.

Linux shell:

```sh
export PRETRAIN_EPOCHS=500
export TRAIN_EPOCHS=350
export SRI_CLR_WARMUP_EPOCH=105
```

Windows PowerShell:

```powershell
$env:PRETRAIN_EPOCHS = "500"
$env:TRAIN_EPOCHS = "350"
$env:SRI_CLR_WARMUP_EPOCH = "105"
```

# Hardware specifications

The manuscript reports experiments in an Ubuntu 20.04 environment using computing resources that included a virtual GPU with 32 GB of memory (vGPU-32GB). This describes the reported experimental environment rather than a measured minimum hardware requirement.

# Code structure

```text
GeMCST/
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── setup.py                    # Package installation
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── GeMCST/
    ├── __init__.py             # Package interface
    ├── __main__.py             # python -m GeMCST entry point
    ├── _version.py             # Package version
    ├── main.py                 # Arguments and stage dispatch
    ├── project.py              # GeMCSTRunner and workflow stages
    ├── train.py                # Training and optimization
    ├── networks.py             # Model definitions
    ├── graph_function.py       # Multiview graph construction
    ├── load_data_preprocess.py # Image features and diversity pruning
    └── utils.py                # Data loading and analysis helpers
```

# License

This project is licensed under the [MIT License](LICENSE). Third-party code, datasets and pretrained model weights remain subject to their respective licenses.
