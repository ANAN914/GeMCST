"""
Diversity-based pruning for UNI image features.

The implementation follows the official DivPrune token-selection logic:
compute the full cosine-distance matrix, initialize with the token whose
nearest-neighbor distance is largest, then greedily add the token whose minimum
distance to the selected set is largest.  For spatial transcriptomics we usually
must keep the AnnData spot axis unchanged, so the helper can also map each spot
to its nearest selected diverse prototype.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

@dataclass(frozen=True)
class DiversityPruneResult:
    """Container returned by apply_diversity_prune."""

    features: np.ndarray
    selected_indices: np.ndarray
    selected_mask: np.ndarray
    assignment: Optional[np.ndarray]
    keep_count: int
    keep_ratio: float

def _validate_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    if features.shape[0] == 0:
        raise ValueError("features must contain at least one token/spot")
    if not np.all(np.isfinite(features)):
        raise ValueError("features contain NaN or Inf values")
    return features

def _resolve_keep_count(n_tokens: int, keep_ratio: Optional[float], keep_count: Optional[int]) -> int:
    if keep_count is not None:
        keep_count = int(keep_count)
    else:
        ratio = 1.0 if keep_ratio is None else float(keep_ratio)
        if ratio <= 0:
            raise ValueError("keep_ratio must be > 0")
        if ratio <= 1:
            keep_count = int(np.ceil(n_tokens * ratio))
        else:
            keep_count = int(ratio)

    if keep_count <= 0:
        raise ValueError("keep_count must be > 0")
    return min(keep_count, n_tokens)

def _l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)

def select_diverse_indices(
    features: np.ndarray,
    keep_ratio: Optional[float] = 0.5,
    keep_count: Optional[int] = None,
    start: str = "maximin",
    random_state: int = 0,
) -> np.ndarray:
    """
    Select diverse rows using greedy farthest-first traversal.

    Parameters
    ----------
    features
        Feature matrix with shape (n_tokens, n_features).
    keep_ratio
        Fraction of rows to keep. Values > 1 are interpreted as a count.
    keep_count
        Explicit number of rows to keep. Overrides keep_ratio.
    start
        Initial token strategy: "maximin" (official DivPrune), "centroid",
        "random", or "first".
    random_state
        Seed used only when start="random".

    Returns
    -------
    np.ndarray
        Sorted selected row indices.
    """
    features = _validate_features(features)
    n_tokens = features.shape[0]
    keep_count = _resolve_keep_count(n_tokens, keep_ratio, keep_count)

    if keep_count == n_tokens:
        return np.arange(n_tokens, dtype=np.int64)

    normalized = _l2_normalize(features)
    cosine_distance = 1.0 - normalized @ normalized.T
    cosine_distance = np.maximum(cosine_distance, 0.0)

    if start == "maximin":
        if n_tokens == 1:
            first = 0
        else:
            second_smallest = np.partition(cosine_distance, kth=1, axis=0)[1, :]
            first = int(np.argmax(second_smallest))
    elif start == "centroid":
        centroid = _l2_normalize(normalized.mean(axis=0, keepdims=True))[0]
        first = int(np.argmax(1.0 - normalized @ centroid))
    elif start == "random":
        first = int(np.random.default_rng(random_state).integers(n_tokens))
    elif start == "first":
        first = 0
    else:
        raise ValueError("start must be one of: 'maximin', 'centroid', 'random', 'first'")

    selected = np.empty(keep_count, dtype=np.int64)
    selected[0] = first
    is_selected = np.zeros(n_tokens, dtype=bool)
    is_selected[first] = True

    min_distance = cosine_distance[first].copy()
    min_distance[first] = -np.inf

    for step in range(1, keep_count):
        next_index = int(np.argmax(min_distance))
        selected[step] = next_index
        is_selected[next_index] = True

        distance_to_new = cosine_distance[next_index]
        min_distance = np.minimum(min_distance, distance_to_new)
        min_distance[is_selected] = -np.inf

    return np.sort(selected)

def apply_diversity_prune(
    features: np.ndarray,
    keep_ratio: Optional[float] = 0.5,
    keep_count: Optional[int] = None,
    preserve_shape: bool = True,
    start: str = "maximin",
    random_state: int = 0,
) -> DiversityPruneResult:
    """
    Apply diversity pruning to a feature matrix.

    If preserve_shape=True, every original row is replaced by its nearest
    selected diverse prototype, keeping shape (n_spots, n_features).  This is
    the safest mode for AnnData pipelines because gene, spatial, and image rows
    stay aligned.  If False, only selected rows are returned.
    """
    features = _validate_features(features)
    selected_indices = select_diverse_indices(
        features,
        keep_ratio=keep_ratio,
        keep_count=keep_count,
        start=start,
        random_state=random_state,
    )

    selected_features = features[selected_indices]
    selected_mask = np.zeros(features.shape[0], dtype=bool)
    selected_mask[selected_indices] = True

    if preserve_shape:
        normalized = _l2_normalize(features)
        selected_normalized = _l2_normalize(selected_features)
        assignment = np.argmax(normalized @ selected_normalized.T, axis=1).astype(np.int64)
        pruned_features = selected_features[assignment]
    else:
        assignment = None
        pruned_features = selected_features

    return DiversityPruneResult(
        features=np.asarray(pruned_features, dtype=np.float32),
        selected_indices=selected_indices,
        selected_mask=selected_mask,
        assignment=assignment,
        keep_count=int(selected_indices.size),
        keep_ratio=float(selected_indices.size) / float(features.shape[0]),
    )

def diversity_prune(
    features: np.ndarray,
    keep_ratio: Optional[float] = 0.5,
    keep_count: Optional[int] = None,
    preserve_shape: bool = True,
    start: str = "maximin",
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-friendly functional API.

    Returns the pruned/mapped features and the selected source indices.
    """
    result = apply_diversity_prune(
        features,
        keep_ratio=keep_ratio,
        keep_count=keep_count,
        preserve_shape=preserve_shape,
        start=start,
        random_state=random_state,
    )
    return result.features, result.selected_indices

import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

try:
    from uni import get_encoder
except ImportError as e:
    print(f"UNI module import failed: {e}")
    raise

class image_feature:
    def __init__(
        self,
        adata,
        pca_components=128,
        cnnType='UNI2-h',
        verbose=True,
        seeds=88,
        use_diversity_prune=True,
        divprune_keep_ratio=0.75,
        divprune_keep_count=None,
        divprune_start=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        self.seeds = seeds
        self.cnnType = cnnType
        self.use_diversity_prune = use_diversity_prune
        self.divprune_keep_ratio = divprune_keep_ratio
        self.divprune_keep_count = divprune_keep_count
        self.divprune_start = divprune_start or os.environ.get("DIVPRUNE_START", "centroid")

    def load_cnn_model(self):
        if self.cnnType == 'UNI2-h':
            if self.verbose:
                print("Loading UNI 2-h model from HuggingFace...")

            model, transform = get_encoder(enc_name='uni2-h', device=self.device)
            if self.verbose:
                model_name = model.__class__.__name__
                prefix_count = self._get_prefix_token_count(model)
                embed_dim = getattr(model, "embed_dim", "unknown")
                patch_embed = getattr(model, "patch_embed", None)
                patch_size = getattr(patch_embed, "patch_size", "unknown")
                print(
                    "UNI encoder loaded: "
                    f"class={model_name}, embed_dim={embed_dim}, "
                    f"prefix_tokens={prefix_count}, patch_size={patch_size}, device={self.device}"
                )

            manual_uni_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

            if transform is None:
                if self.verbose:
                    print("[WARN] Using fallback manual transform")
                return model, manual_uni_transform
            
            if self.verbose:
                print("[OK] Using official UNI transform")
            return model, transform
        else:

            import torchvision.models as models
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)

            return model, None

    def _get_prefix_token_count(self, model):
        """Infer how many non-patch prefix tokens a timm ViT sequence contains."""
        if hasattr(model, "num_prefix_tokens"):
            try:
                return int(model.num_prefix_tokens)
            except Exception:
                pass

        count = 0
        if getattr(model, "cls_token", None) is not None:
            count += 1
        if getattr(model, "reg_token", None) is not None:
            reg_token = getattr(model, "reg_token")
            if hasattr(reg_token, "shape") and len(reg_token.shape) >= 2:
                count += int(reg_token.shape[1])
        if getattr(model, "reg_tokens", None) is not None:
            reg_tokens = getattr(model, "reg_tokens")
            if hasattr(reg_tokens, "shape") and len(reg_tokens.shape) >= 2:
                count += int(reg_tokens.shape[1])
        return count

    def _tokens_from_forward_features(self, model, tensor):
        """
        Return UNI patch tokens when the timm/UNI model exposes them.

        UNI2-h is loaded through timm. Depending on timm version, forward_features
        can return either a dictionary with x_norm_patchtokens or a token sequence.
        """
        if not hasattr(model, "forward_features"):
            return None

        output = model.forward_features(tensor)
        if isinstance(output, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "tokens"):
                value = output.get(key)
                if value is not None:
                    return value
            for key in ("last_hidden_state", "x_norm"):
                value = output.get(key)
                if value is not None and getattr(value, "ndim", 0) == 3:
                    prefix_count = self._get_prefix_token_count(model)
                    return value[:, prefix_count:, :] if prefix_count > 0 else value
            return None

        if isinstance(output, (tuple, list)):
            output = output[0]

        if getattr(output, "ndim", 0) == 3:
            prefix_count = self._get_prefix_token_count(model)
            return output[:, prefix_count:, :] if prefix_count > 0 else output

        return None

    def _pool_pruned_patch_tokens(self, patch_tokens):
        """Prune patch tokens with diversity_prune(), then mean-pool to one spot embedding."""
        tokens_np = patch_tokens.squeeze(0).detach().cpu().numpy()
        if tokens_np.ndim != 2 or tokens_np.shape[0] == 0:
            raise ValueError(f"Invalid patch token shape: {tokens_np.shape}")

        if self.use_diversity_prune:
            prune_result = apply_diversity_prune(
                tokens_np,
                keep_ratio=self.divprune_keep_ratio,
                keep_count=self.divprune_keep_count,
                preserve_shape=False,
                start=self.divprune_start,
                random_state=self.seeds,
            )
            selected_tokens = prune_result.features
            selected_indices = prune_result.selected_indices
        else:
            selected_tokens = tokens_np
            selected_indices = np.arange(tokens_np.shape[0], dtype=np.int64)

        return selected_tokens.mean(axis=0), selected_indices, tokens_np.shape[0]

    def extract_image_feat(self):

        model, uni_transform = self.load_cnn_model()
        model.eval()

        if model is None:
            raise ValueError("Failed to load UNI2-h model")
        if uni_transform is None:
            raise ValueError("Failed to load UNI2-h transform")
        
        if self.verbose:
            print("[OK] UNI2-h model and transform loaded successfully")

        features_list = []
        spot_names = []
        patch_token_counts = []
        selected_token_counts = []

        if "slices_path" not in self.adata.obs.keys():
            raise ValueError("Please run image_crop() first to generate spot images")

        if len(self.adata.obs['slices_path']) == 0:
            raise ValueError("No slice paths found in adata.obs['slices_path']")
        
        if self.verbose:
            print(f"[OK] Found {len(self.adata.obs['slices_path'])} slice paths")

        with tqdm(total=len(self.adata),
                desc=f"Extracting {self.cnnType} features",
                bar_format="{l_bar}{bar} [ time left: {remaining} ]", 
                ncols=80, dynamic_ncols=True, leave=True) as pbar:
            
            for i, (spot, slice_path) in enumerate(self.adata.obs['slices_path'].items()):

                if not os.path.exists(slice_path):
                    raise FileNotFoundError(f"Slice file not found: {slice_path}")

                try:
                    spot_slice = Image.open(slice_path).convert('RGB')
                    if self.verbose and i == 0:
                        print(f"[OK] First image loaded successfully: {slice_path}")
                        print(f"   Image size: {spot_slice.size}")
                except Exception as e:
                    raise IOError(f"Failed to load image {slice_path}: {e}")

                try:
                    tensor = uni_transform(spot_slice).unsqueeze(0).to(self.device)
                    if self.verbose and i == 0:
                        print("[OK] Transform applied successfully")
                        print(f"   Tensor shape: {tensor.shape}")
                except Exception as e:
                    raise RuntimeError(f"Failed to apply transform: {e}")
                
                try:
                    with torch.no_grad():
                        patch_tokens = self._tokens_from_forward_features(model, tensor)
                        if patch_tokens is not None:
                            features, selected_indices, token_count = self._pool_pruned_patch_tokens(patch_tokens)
                            selected_count = int(len(selected_indices))
                        else:
                            output = model(tensor)
                            if hasattr(output, 'last_hidden_state'):
                                features = output.last_hidden_state.mean(dim=1).cpu().numpy().ravel()
                            else:
                                features = output.cpu().numpy().ravel()
                            token_count = 0
                            selected_count = 0

                    if features.shape[0] != 1536:
                        raise ValueError(f"Expected UNI2-h feature dimension 1536, got {features.shape[0]}")

                    if self.verbose and i == 0:
                        print("UNI feature extraction path: patch_tokens" if token_count > 0 else "UNI feature extraction path: pooled_embedding")
                        print(f"   Feature shape: {features.shape}")
                        if token_count > 0:
                            print(f"   Patch tokens: {token_count}")
                            print(f"   Selected tokens: {selected_count}")
                        print(f"   Feature range: [{features.min():.4f}, {features.max():.4f}]")
                        print(f"   Feature mean: {features.mean():.4f}")
                        print(f"   Feature std: {features.std():.4f}")
                except Exception as e:
                    print(f"Failed to extract UNI features from: {slice_path}")
                    raise RuntimeError(f"UNI feature extraction failed: {e}")

                features_list.append(features)
                spot_names.append(spot)
                patch_token_counts.append(token_count)
                selected_token_counts.append(selected_count)
                pbar.update(1)

        if len(features_list) == 0:
            raise ValueError("No features extracted")

        feature_dim = features_list[0].shape[0]
        if feature_dim != 1536:
            raise ValueError(f"Expected feature dimension 1536, got {feature_dim}")
        
        if self.verbose:
            print("[OK] All features extracted successfully")
            print(f"   Total spots: {len(features_list)}")
            print(f"   Feature dimension: {feature_dim}")

        feat_df = pd.DataFrame(features_list, index=spot_names)
        self.adata.obsm["image_feat"] = feat_df.to_numpy()
        feature_for_pca = self.adata.obsm["image_feat"]
        self.adata.obs["uni_patch_token_count"] = patch_token_counts
        self.adata.obs["uni_selected_token_count"] = selected_token_counts
        if self.use_diversity_prune:
            self.adata.obsm["image_feat_divprune"] = self.adata.obsm["image_feat"].copy()
            feature_for_pca = self.adata.obsm["image_feat_divprune"]
        self.adata.uns["divprune"] = {
            "enabled": bool(self.use_diversity_prune),
            "level": "uni_patch_tokens",
            "selection": self.divprune_start,
            "keep_ratio": float(self.divprune_keep_ratio) if self.divprune_keep_count is None else None,
            "keep_count": None if self.divprune_keep_count is None else int(self.divprune_keep_count),
            "pooling": "mean",
            "source": "UNI forward_features patch tokens",
            "output_key": "image_feat",
            "divprune_output_key": "image_feat_divprune" if self.use_diversity_prune else None,
            "image_feat_source": "image_feat_divprune" if self.use_diversity_prune else "image_feat",
            "fallback": "pooled_embedding" if max(patch_token_counts or [0]) == 0 else None,
        }

        if self.verbose:
            print("Performing StandardScaler normalization...")
        
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_for_pca)
        
        if self.verbose:
            print("[OK] StandardScaler normalization completed")

        if self.verbose:
            print(f"Performing PCA: 1536 -> {self.pca_components}")
        
        try:
            pca = PCA(n_components=self.pca_components, random_state=self.seeds)
            self.adata.obsm["image_feat_pca"] = pca.fit_transform(normalized_features)
            
            if self.verbose:
                print("[OK] PCA completed successfully")
                print(f"   Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
        except Exception as e:
            raise RuntimeError(f"Failed to perform PCA: {e}")

        if "image_feat" not in self.adata.obsm:
            raise ValueError("Failed to add image_feat to adata.obsm")
        if "image_feat_pca" not in self.adata.obsm:
            raise ValueError("Failed to add image_feat_pca to adata.obsm")
        
        if self.verbose:
            print(f"[OK] Image features added to adata.obsm['image_feat'] (Dim: {self.adata.obsm['image_feat'].shape})")
            print(f"[OK] PCA-reduced features added to adata.obsm['image_feat_pca'] (Dim: {self.adata.obsm['image_feat_pca'].shape})")
            print("UNI2-h feature extraction completed successfully.")

        return self.adata

def image_crop(
    adata,
    save_path,
    library_id=None,
    crop_size=60,
    target_size=224,
    verbose=False,
):
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    os.makedirs(save_path, exist_ok=True)

    adata.uns['image_crop_dir'] = str(save_path)

    if ("imagerow" not in adata.obs.columns or "imagecol" not in adata.obs.columns) and "spatial" in adata.obsm:
        adata.obs["imagecol"] = adata.obsm["spatial"][:, 0]
        adata.obs["imagerow"] = adata.obsm["spatial"][:, 1]
        if verbose:
            print("Mapped obsm['spatial'] pixel coordinates to obs['imagecol/imagerow']")

    if 'imagerow' in adata.obs.columns and 'imagecol' in adata.obs.columns:
        imagerow_col = 'imagerow'
        imagecol_col = 'imagecol'
        if verbose:
            print("[OK] Using 'imagerow' and 'imagecol' columns for image cropping")
    elif 'array_row' in adata.obs.columns and 'array_col' in adata.obs.columns:
        imagerow_col = 'array_row'
        imagecol_col = 'array_col'
        if verbose:
            print("[OK] Using 'array_row' and 'array_col' columns for image cropping")
    else:
        raise KeyError(
            "Spatial coordinate columns were not found. Expected 'imagerow'/'imagecol' "
            "or 'array_row'/'array_col'.\n"
            f"Current obs columns: {list(adata.obs.columns)}"
        )
    
    spatial_meta = adata.uns["spatial"][library_id]
    image_key = spatial_meta.get("use_quality", "hires")
    scalefactors = spatial_meta.get("scalefactors", {})
    coord_scale = float(scalefactors.get(f"tissue_{image_key}_scalef", 1.0))
    if coord_scale <= 0:
        coord_scale = 1.0

    tile_names = []
    existing_count = 0
    generated_count = 0

    existing_files = set()
    if os.path.exists(save_path):
        for f in os.listdir(save_path):
            if f.endswith('.png'):
                existing_files.add(f)

    img_pillow = None
    need_generate = False
    
    with tqdm(total=len(adata),
            desc="Cropping spot images",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]", ncols=80, dynamic_ncols=True, leave=True) as pbar:
        for imagerow, imagecol in zip(adata.obs[imagerow_col], adata.obs[imagecol_col]):
            scaled_imagecol = float(imagecol) * coord_scale
            scaled_imagerow = float(imagerow) * coord_scale
            tile_name = f"{image_key}-s{coord_scale:.6g}-{scaled_imagecol:.2f}-{scaled_imagerow:.2f}-{crop_size}.png"
            out_path = Path(save_path) / tile_name

            if tile_name in existing_files:
                tile_names.append(str(out_path))
                existing_count += 1
                pbar.update(1)
                continue

            if img_pillow is None:

                if "use_quality" in adata.uns["spatial"][library_id]:
                    image_key = adata.uns["spatial"][library_id]["use_quality"]
                else:
                    image_key = "hires"
                image = spatial_meta["images"][image_key]
                if image.dtype == np.float32 or image.dtype == np.float64:
                    image = (image * 255).astype(np.uint8)
                img_pillow = Image.fromarray(image)
            
            left = scaled_imagecol - crop_size / 2
            upper = scaled_imagerow - crop_size / 2
            right = scaled_imagecol + crop_size / 2
            lower = scaled_imagerow + crop_size / 2

            tile = img_pillow.crop((left, upper, right, lower))
            tile = tile.resize((target_size, target_size))
            tile.save(out_path, "PNG")
            tile_names.append(str(out_path))
            generated_count += 1
            
            if verbose:
                print(f"Generated tile at scaled location ({scaled_imagecol}, {scaled_imagerow})")
            pbar.update(1)

    adata.obs["slices_path"] = tile_names

    if verbose or generated_count > 0 or existing_count > 0:
        print("[OK] Spot image cropping completed")
        print(f"   - Existing tiles reused: {existing_count}")
        print(f"   - Newly generated tiles: {generated_count}")
    
    if verbose:
        print("Spot image paths added to adata.obs['slices_path']")

    return adata
