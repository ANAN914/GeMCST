import os
import sys
import numpy as np
import scanpy as sc
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from scipy.sparse import issparse,csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler

from pathlib import Path, PurePath
from typing import Optional, Union
from anndata import AnnData
from PIL import Image
import pandas as pd
from typing import Literal
import scipy
import matplotlib.pyplot as plt

_QUALITY = Literal["fulres", "hires", "lowres"]
_background = ["black", "white"]

def load_spatial_data(data_path, genome=None, library_id=None, load_images=True, quality='hires'):

    if data_path.endswith('.h5'):
        print(f"H5 file detected; loading: {data_path}")

        adata = sc.read_10x_h5(data_path)
        
    elif os.path.isdir(data_path):
        print(f"MTX directory detected; loading: {data_path}")

        mtx_files = [f for f in os.listdir(data_path) if f.endswith('.mtx')]
        if not mtx_files:

            print(f"No .mtx file found; trying to load in standard Visium format")
            adata = sc.read_visium(data_path, 
                                genome=genome,
                                library_id=library_id,
                                load_images=load_images)
        else:

            adata = sc.read_10x_mtx(data_path, var_names='gene_symbols', make_unique=True)
    
    else:
        raise ValueError(f"Unrecognized path format: {data_path}")

    adata.var_names_make_unique()

    if 'spatial' not in adata.uns:
        print("MTX data is missing spatial coordinates; please add them manually")

    return adata

def read_10X_Visium(path, 
                    genome = None,
                    count_file ='filtered_feature_bc_matrix.h5', 
                    library_id = None, 
                    load_images =True, 
                    quality ='hires',
                    image_path = None
                ):

    full_path = os.path.join(path, count_file) if count_file else path

    adata = load_spatial_data(full_path, genome, library_id, load_images, quality)

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    
    return adata

def read_SlideSeq(path, 
                library_id = None,
                scale = None,
                quality = "hires",
                spot_diameter_fullres= 50,
                background_color = "white",
                ):

    count = pd.read_csv(os.path.join(path, "count_matrix.count"))
    meta = pd.read_csv(os.path.join(path, "spatial.idx"))

    adata = AnnData(count.iloc[:, 1:].set_index("gene").T)

    adata.var["ENSEMBL"] = count["ENSEMBL"].values

    adata.obs["index"] = meta["index"].values

    if scale == None:
        max_coor = np.max(meta[["x", "y"]].values)
        scale = 2000 / max_coor

    adata.obs["imagecol"] = meta["x"].values * scale
    adata.obs["imagerow"] = meta["y"].values * scale

    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "Slide-seq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"] = scale

    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres
    adata.obsm["spatial"] = meta[["x", "y"]].values

    return adata

def read_merfish(path, 
                library_id=None,
                scale=None,
                quality="hires",
                spot_diameter_fullres=50,
                background_color="white",
                ):

    counts = sc.read_csv(os.path.join(path, 'counts.csv')).transpose()
    locations = pd.read_excel(os.path.join(path, 'spatial.xlsx'), index_col=0)
    if locations.min().min() < 0:
        locations = locations + np.abs(locations.min().min()) + 100
    adata = counts[locations.index, :]
    adata.obsm["spatial"] = locations.to_numpy()

    if scale == None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "MERSEQ"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    return adata

def read_seqfish(path,
                library_id= None,
                scale= 1.0,
                quality= "hires",
                field = 0,
                spot_diameter_fullres = 50,
                background_color = "white",
                ):

    count = pd.read_table(os.path.join(path, 'counts.matrix'), header=None)
    spatial = pd.read_table(os.path.join(path, 'spatial.csv'), index_col=False)

    count = count.T
    count.columns = count.iloc[0]
    count = count.drop(count.index[0]).reset_index(drop=True)
    count = count[count["Field_of_View"] == field].drop(count.columns[[0, 1]], axis=1)
    spatial = spatial[spatial["Field_of_View"] == field]

    adata = AnnData(count)

    if scale == None:
        max_coor = np.max(spatial[["X", "Y"]])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = spatial["X"].values * scale
    adata.obs["imagerow"] = spatial["Y"].values * scale

    adata.obsm["spatial"] = spatial[["X", "Y"]].values

    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "SeqFish"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    return adata

def read_stereoSeq(path,
                bin_size=100,
                is_sparse=True,
                library_id=None,
                scale=None,
                quality="hires",
                spot_diameter_fullres=1,
                background_color="white",
                ):
    from scipy import sparse
    count = pd.read_csv(os.path.join(path, "count.txt"), sep='\t', comment='#', header=0)
    count.dropna(inplace=True)
    if "MIDCounts" in count.columns:
        count.rename(columns={"MIDCounts": "UMICount"}, inplace=True)
    count['x1'] = (count['x'] / bin_size).astype(np.int32)
    count['y1'] = (count['y'] / bin_size).astype(np.int32)
    count['pos'] = count['x1'].astype(str) + "-" + count['y1'].astype(str)
    bin_data = count.groupby(['pos', 'geneID'])['UMICount'].sum()
    cells = set(x[0] for x in bin_data.index)
    genes = set(x[1] for x in bin_data.index)
    cellsdic = dict(zip(cells, range(0, len(cells))))
    genesdic = dict(zip(genes, range(0, len(genes))))
    rows = [cellsdic[x[0]] for x in bin_data.index]
    cols = [genesdic[x[1]] for x in bin_data.index]
    exp_matrix = sparse.csr_matrix((bin_data.values, (rows, cols))) if is_sparse else \
                sparse.csr_matrix((bin_data.values, (rows, cols))).toarray()
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame(index=genes)
    adata = AnnData(X=exp_matrix, obs=obs, var=var)
    pos = np.array(list(adata.obs.index.str.split('-', expand=True)), dtype=np.int)
    adata.obsm['spatial'] = pos

    if scale == None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 20 / max_coor

    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "StereoSeq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    return adata

def refine(
    sample_id, 
    pred, 
    dis, 
    shape="hexagon"
    ):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6 
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
    return refined_pred

def refine_knn(sample_id, pred, spatial, shape="hexagon"):
    """Refine labels from local spatial neighbors without an NxN distance matrix."""
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        raise ValueError("shape must be 'hexagon' or 'square'")

    pred = np.asarray(pred, dtype=object)
    spatial = np.asarray(spatial, dtype=np.float64)
    if len(pred) != len(sample_id) or spatial.shape[0] != len(sample_id):
        raise ValueError("sample_id, pred, and spatial must contain the same number of spots")
    if len(pred) <= 1:
        return pred.tolist()

    effective_nbs = min(num_nbs, len(pred) - 1)
    neighbor_idx = NearestNeighbors(
        n_neighbors=effective_nbs + 1,
        algorithm="kd_tree",
        n_jobs=-1,
    ).fit(spatial).kneighbors(spatial, return_distance=False)

    refined_pred = []
    for i, neighbors in enumerate(neighbor_idx):
        neighbor_labels = pd.Series(pred[neighbors])
        counts = neighbor_labels.value_counts()
        self_pred = pred[i]
        if (
            counts.get(self_pred, 0) < effective_nbs / 2
            and counts.iloc[0] > effective_nbs / 2
        ):
            refined_pred.append(counts.index[0])
        else:
            refined_pred.append(self_pred)
    return refined_pred

def seed_torch(seed=1029):   
    random.seed(seed)    
    os.environ['PYTHONHASHSEED'] = str(seed)    
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 
                                    
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

"""Dataset-specific domain color palettes used by spatial and UMAP plots.

Colors are ordered by the sorted domain categories used by the plotting code.
The mapping changes presentation only; it does not change clustering labels or
the model/training objective.
"""

PALETTE_COLON = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4c78a4", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#2f4b7c",
]

PALETTE_LUNG = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4c78a4", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#2f4b7c",
]

PALETTE_HER2 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4c78a4", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#2f4b7c",
]

PALETTE_V1_BREAST = [
    "#1976B5", "#F28E2B", "#159A72", "#D7192C", "#7A2CBF",
    "#8C564B", "#D85AA6", "#B5B43C", "#159AC0", "#A9C7E6",
    "#F4B183", "#65B96E", "#EF8E8E", "#B29BC8", "#B88779",
    "#E7A0B9", "#C7C66B", "#8AC6D9", "#B84C4C", "#9A6500",
]

PALETTE_DEFAULT = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4c78a4", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#2f4b7c",
]

def palette_for_dataset(profile=None, sample_id=None, data_dir=None):
    """Return the requested palette, falling back to the original palette."""
    text = " ".join(str(x or "") for x in (profile, sample_id, data_dir)).lower()
    if "colon_cancer_p1_032um" in text or "colon" in text:
                                                                        
        return list(PALETTE_V1_BREAST)
    if "human_lung_cancer" in text or "lung" in text:
        return list(PALETTE_LUNG)
    if "v1_breast" in text or "v1_breast_cancer" in text:
        return list(PALETTE_V1_BREAST)
    if "her2" in text or "breast" in text or str(sample_id or "").upper() == "H1":
        return list(PALETTE_HER2)
    return list(PALETTE_DEFAULT)

def colors_for_categories(categories, profile=None, sample_id=None, data_dir=None):
    """Map sorted category values to stable colors, cycling only if necessary."""
    palette = palette_for_dataset(profile, sample_id, data_dir)
    return [palette[i % len(palette)] for i in range(len(categories))]

"""Representation contracts shared by visualization and result export."""

import numpy as np

PRECLUSTER_LATENT_KEY = "GeMCST_precluster_latent"

def select_precluster_umap_input(adata):
    """Return only the continuous latent captured before clustering optimization."""
    if PRECLUSTER_LATENT_KEY not in adata.obsm:
        raise KeyError(
            f"Missing adata.obsm['{PRECLUSTER_LATENT_KEY}']; UMAP must be computed "
            "from the pre-clustering continuous latent, not a final embedding or "
            "Soft-Potts probabilities."
        )

    latent = np.asarray(adata.obsm[PRECLUSTER_LATENT_KEY])
    if latent.ndim != 2 or latent.shape[0] != adata.n_obs:
        raise ValueError(
            f"adata.obsm['{PRECLUSTER_LATENT_KEY}'] must have shape "
            f"(n_obs, n_features), got {latent.shape}."
        )
    if not np.isfinite(latent).all():
        raise ValueError(f"adata.obsm['{PRECLUSTER_LATENT_KEY}'] contains non-finite values.")
    return latent

"""Frozen post-hoc expression decoder.

This module is deliberately independent from the GeMCST/Soft-Potts training
graph. It fits a deterministic ridge mapping from the final spot embedding to
the already-preprocessed expression matrix and returns an extra expression
view without modifying the input matrix.
"""

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

def _dense_float32(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("embedding and expression must be two-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("embedding and expression must contain finite values")
    return array

def fit_posthoc_expression_decoder(
    embedding: Any,
    expression: Any,
    *,
    alpha: float = 1.0,
    batch_size: int = 2048,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Decode an additional expression matrix from a frozen embedding.

    The target is the expression representation already stored in ``adata.X``;
    no normalization or mutation is performed here. Predictions are clipped
    at zero because the downstream spatial expression plots expect
    non-negative values.
    """
    z = _dense_float32(embedding)
    x = _dense_float32(expression)
    if z.shape[0] != x.shape[0]:
        raise ValueError("embedding and expression must have the same number of spots")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(z, x)
    decoded_chunks = []
    for start in range(0, z.shape[0], int(batch_size)):
        decoded_chunks.append(model.predict(z[start : start + int(batch_size)]))
    decoded = np.vstack(decoded_chunks).astype(np.float32, copy=False)
    decoded = np.maximum(decoded, 0.0)

    residual = decoded.astype(np.float64) - x.astype(np.float64)
    metrics: dict[str, float | int] = {
        "n_spots": int(x.shape[0]),
        "n_genes": int(x.shape[1]),
        "embedding_dim": int(z.shape[1]),
        "alpha": float(alpha),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "observed_mean": float(np.mean(x)),
        "enhanced_mean": float(np.mean(decoded)),
        "observed_zero_fraction": float(np.mean(x == 0)),
        "enhanced_zero_fraction": float(np.mean(decoded == 0)),
    }
    return decoded, metrics

"""Evaluation plot styling (restored missing helper)."""
def configure_evaluation_style(plot_module=None):
    if plot_module is None:
        import matplotlib.pyplot as plot_module
    plot_module.rcParams.update({"font.family": "Times New Roman", "pdf.fonttype": 42, "ps.fonttype": 42})

def should_show_h1_umap_metrics(sample_id):
    return str(sample_id).strip().upper() == "H1"

def spatial_panel_title(model_name, metrics=None, show_metrics=False):
    if not show_metrics or not metrics:
        return model_name
    values = []
    for name in ("ARI", "NMI"):
        value = metrics.get(name, metrics.get(name.lower()))
        if value is not None:
            values.append(f"{name}={value:.3f}")
    return model_name + ("\n" + ", ".join(values) if values else "")
