import os
import time
import json
import psutil
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.sparse import issparse
from typing import Optional, List, Union
import anndata
import matplotlib.pyplot as plt
from .utils import read_10X_Visium, read_merfish, read_SlideSeq, read_seqfish, read_stereoSeq, refine, refine_knn
from .load_data_preprocess import image_feature, image_crop
from .graph_function import graph, combine_graph_dict, build_triview_graph_dict
from .networks import GeMCSTModel, AdversarialNetwork
from .train import train

LEGACY_DENSE_MAX_SPOTS = int(os.environ.get("LEGACY_DENSE_MAX_SPOTS", "5000"))

def _use_legacy_dense_refinement(n_spots: int) -> bool:
    return LEGACY_DENSE_MAX_SPOTS > 0 and int(n_spots) <= LEGACY_DENSE_MAX_SPOTS

class GeMCSTRunner:
    def __init__(
        self,
        save_path: str = "./",
        task: str = "Identify_Domain",
        pre_epochs: int = 1000,
        epochs: int = 500,
        use_gpu: bool = True
    ):
        """
        Initialize GeMCST pipeline runner
        
        Parameters:
        -----------
        save_path : str
            Path to save results (default: "./")
        task : str
            Analysis task type (default: "Identify_Domain")
            Options: ["Identify_Domain", "Integration"]
        pre_epochs : int
            Number of pretraining epochs (default: 1000)
        epochs : int
            Number of training epochs (default: 500)
        use_gpu : bool
            Whether to use GPU if available (default: True)
        """
        self.save_path = Path(save_path)
        self.task = task
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.use_gpu = use_gpu
        self.seed = int(os.environ.get("SEED", "0"))

    def _get_adata(
        self,
        platform: str,
        data_path: str,
        data_name: str,
        verbose: bool = True
    ) -> sc.AnnData:
        """
        Load spatial transcriptomics data with enhanced compatibility
        
        Parameters:
        -----------
        platform : str
            Technology platform 
            Options: ['Visium', 'ST', 'MERFISH', 'slideSeq', 'stereoSeq']
        data_path : str
            Path to data directory
        data_name : str
            Name of dataset
        verbose : bool
            Whether to save raw data (default: True)
            
        Returns:
        --------
        sc.AnnData
            Loaded spatial transcriptomics data
        """
        platform_options = ['Visium', 'ST', 'MERFISH', 'slideSeq', 'stereoSeq']
        if platform not in platform_options:
            raise ValueError(f"Platform must be one of {platform_options}")

        full_data_path = os.path.join(data_path, data_name)

        if platform == 'Visium':
            print(f"Checking Visium data format: {full_data_path}")
            adata = self._load_visium_data(full_data_path)
                
        else:

            reader_map = {
                'MERFISH': read_merfish,
                'slideSeq': read_SlideSeq,
                'stereoSeq': read_stereoSeq
            }
            
            if platform in reader_map:
                adata = reader_map[platform](full_data_path)
            else:
                raise ValueError(f"Unsupported platform: {platform}")

        adata.var_names_make_unique()
        
        if verbose:
            save_dir = self.save_path / "Data" / data_name
            save_dir.mkdir(parents=True, exist_ok=True)
            adata.write(save_dir / f'{data_name}_raw.h5ad', compression="gzip")
            print(f"Raw data saved to: {save_dir / f'{data_name}_raw.h5ad'}")
            
        return adata
    
    def _load_visium_data(self, data_path: str) -> sc.AnnData:

        spatial_dir = os.path.join(data_path, 'spatial')
        if os.path.isdir(spatial_dir):
            print(f"Detected spatial directory; loading with scanpy.read_visium: {data_path}")
            try:
                adata = sc.read_visium(data_path)
                adata.var_names_make_unique()
                return adata
            except Exception as e:
                print(f"scanpy.read_visium failed: {e}; trying fallback loaders.")

        if os.path.isfile(data_path) and data_path.endswith('.h5'):
            print(f"Detected H5 matrix file; loading with scanpy.read_10x_h5: {data_path}")
            adata = sc.read_10x_h5(data_path)
            adata.var_names_make_unique()
            return adata

        if os.path.isdir(data_path):

            files = os.listdir(data_path)

            h5_files = [f for f in files if f.endswith('.h5') and 'matrix' in f.lower()]
            if h5_files:
                h5_file = os.path.join(data_path, h5_files[0])
                print(f"Detected H5 matrix file: {h5_file}")
                adata = sc.read_10x_h5(h5_file)
                adata.var_names_make_unique()
                return adata

            mtx_dirs = [d for d in files if os.path.isdir(os.path.join(data_path, d)) 
                    and any('matrix' in f.lower() for f in os.listdir(os.path.join(data_path, d)))]
            if mtx_dirs:
                mtx_dir = os.path.join(data_path, mtx_dirs[0])
                print(f"Detected MTX matrix directory: {mtx_dir}")
                return self._load_mtx_data(data_path)

            metadata_files = [f for f in files if f.startswith('metadata.') and f.endswith(('.tsv', '.csv'))]

            if metadata_files or any('spatial' in f.lower() for f in files):
                print("Detected standard Visium-like structure; loading with scanpy.read_visium.")
                try:
                    adata = sc.read_visium(data_path); adata.var_names_make_unique(); return adata
                except Exception as e:
                    print(f"scanpy.read_visium failed: {e}")

                    return self._load_custom_visium_data(data_path)

            print("Standard format not detected; trying MTX loader.")
            try:
                return self._load_mtx_data(data_path)
            except Exception as e:
                print(f"MTX loader failed: {e}")

        print("Path format not recognized; trying fallback loaders.")

        try:
            adata = sc.read_visium(data_path); adata.var_names_make_unique(); return adata
        except Exception as e:
            print(f"scanpy.read_visium failed: {e}")

        try:
            from .utils import read_10X_Visium
            return read_10X_Visium(data_path)
        except Exception as e:
            print(f"legacy Visium reader failed: {e}")

        raise ValueError(f"Could not load Visium data: {data_path}")
    
    def _load_mtx_data(self, sample_path: str) -> sc.AnnData:

        library_id = os.path.basename(sample_path)
        print(f"Processing sample: {library_id}")

        h5_file = os.path.join(sample_path, "filtered_feature_bc_matrix.h5")
        if os.path.exists(h5_file):
            print("Detected standard 10x H5 format")
            adata = sc.read_10x_h5(h5_file)

            spatial_dir = os.path.join(sample_path, "spatial")
            if os.path.exists(spatial_dir):
                adata = self._fill_10x_spatial(adata, spatial_dir, library_id)
            
            return adata

        mtx_dir = self._find_mtx_dir(sample_path)
        spatial_dir = os.path.join(sample_path, "spatial")
        meta_path = self._find_metadata_file(sample_path)
        
        if mtx_dir is not None and os.path.exists(spatial_dir):
            print("Detected standard 10x MTX format")

            print("  -> Reading expression matrix...")
            adata = sc.read_mtx(os.path.join(mtx_dir, 'matrix.mtx')).T
            feature_file = "features.tsv" if os.path.exists(os.path.join(mtx_dir, "features.tsv")) else "genes.tsv"
            genes = pd.read_csv(os.path.join(mtx_dir, feature_file), header=None, sep='\t')
            barcodes = pd.read_csv(os.path.join(mtx_dir, 'barcodes.tsv'), header=None, sep='\t')
            
            adata.var_names = genes[1].values if genes.shape[1] > 1 else genes[0].values
            adata.var['gene_ids'] = genes[0].values
            adata.obs_names = barcodes[0].values
            adata.var_names_make_unique()

            print("  -> Adding spatial metadata...")

            positions = pd.read_csv(os.path.join(spatial_dir, "tissue_positions_list.csv"), header=None, index_col=0)
            positions = positions.loc[adata.obs_names]

            adata.obsm['spatial'] = positions.iloc[:, [4, 3]].values

            with open(os.path.join(spatial_dir, "scalefactors_json.json")) as f:
                scalefactors = json.load(f)

            adata.uns["spatial"] = {
                library_id: {
                    "images": {
                        "hires": self._read_hires_image(self._find_hires_image(sample_path, spatial_dir, library_id)),
                    },
                    "scalefactors": scalefactors,
                    "metadata": {},
                    "use_quality": "hires"
                }
            }

            if meta_path is not None:
                self._load_metadata(adata, meta_path)

            self._map_coordinates_to_obs(adata)
            
            print(f"[OK] Data loaded successfully: {adata.shape}")
            return adata

        count_file = os.path.join(sample_path, "count-matrices", f"{library_id}.tsv.gz")
        coord_file = os.path.join(sample_path, "meta", f"{library_id}_labeled_coordinates.tsv")
        img_file = os.path.join(sample_path, "images", "HE", f"{library_id}.jpg")

        if os.path.exists(count_file) and os.path.exists(coord_file):
            print("Detected HER2 breast cancer dataset format")

            adata = pd.read_csv(count_file, sep='\t', index_col=0).T
            adata = sc.AnnData(adata)

            coords = pd.read_csv(coord_file, sep='\t', index_col=0)

            common_indices = adata.obs_names.intersection(coords.index)
            adata = adata[common_indices].copy()
            coords = coords.loc[common_indices]

            adata.obs = pd.concat([adata.obs, coords], axis=1)

            if os.path.exists(img_file):

                coord_columns = self._detect_coordinate_columns(coords)
                
                adata.uns["spatial"] = {
                    library_id: {
                        "images": {
                            "hires": self._read_hires_image(img_file),
                        },
                        "scalefactors": {
                            "tissue_hires_scalef": 1.0,
                            "spot_diameter_fullres": 150.0
                        },
                        "metadata": {},
                        "use_quality": "hires"
                    }
                }

                if coord_columns:
                    adata.obsm['spatial'] = coords[coord_columns].values.astype(float)

                    adata.obs["imagerow"] = adata.obsm['spatial'][:, 1]
                    adata.obs["imagecol"] = adata.obsm['spatial'][:, 0]
            
            print(f"[OK] HER2 breast cancer data loaded successfully: {adata.shape}")
            return adata

        raise FileNotFoundError(f"Could not identify a supported data format in {sample_path}")
    
    def _find_mtx_dir(self, sample_path: str) -> Optional[str]:
        """Find a 10x-style MTX directory across common naming variants."""
        candidates = [
            os.path.join(sample_path, "filtered_feature_bc_matrix"),
            os.path.join(sample_path, "filters_count_matrices"),
            os.path.join(sample_path, "filtered_count_matrices"),
            os.path.join(sample_path, "count_matrices"),
            sample_path,
        ]
        if os.path.isdir(sample_path):
            for root, dirs, _files in os.walk(sample_path):
                rel_depth = os.path.relpath(root, sample_path).count(os.sep)
                if rel_depth >= 2:
                    dirs[:] = []
                    continue
                candidates.append(root)
        for candidate in candidates:
            has_matrix = os.path.exists(os.path.join(candidate, "matrix.mtx"))
            has_barcodes = os.path.exists(os.path.join(candidate, "barcodes.tsv"))
            has_features = (
                os.path.exists(os.path.join(candidate, "features.tsv"))
                or os.path.exists(os.path.join(candidate, "genes.tsv"))
            )
            if has_matrix and has_barcodes and has_features:
                return candidate
        return None

    def _find_metadata_file(self, sample_path: str) -> Optional[str]:
        """Find optional sample metadata in TSV or CSV format."""
        for name in ["metadata.tsv", "metadata.csv"]:
            path = os.path.join(sample_path, name)
            if os.path.exists(path):
                return path
        return None

    def _find_hires_image(self, sample_path: str, spatial_dir: str, library_id: str) -> str:
        """Find H&E image across Visium-like and HER2/ST-like naming variants."""
        candidates = [
            os.path.join(spatial_dir, "tissue_hires_image.png"),
            os.path.join(spatial_dir, "tissue_hires_images.png"),
            os.path.join(sample_path, f"{library_id}.jpg"),
            os.path.join(sample_path, f"{library_id}.png"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No hires tissue image found for {library_id}: {candidates}")

    def _read_hires_image(self, img_path: str) -> np.ndarray:
        try:
            return plt.imread(img_path)
        except (OSError, ValueError, SyntaxError):
            from PIL import Image

            with Image.open(img_path) as img:
                return np.asarray(img.convert("RGB"))

    def _detect_metadata_index(self, metadata: pd.DataFrame, obs_names: pd.Index) -> Optional[int]:

        common_index_names = ['Row.names', 'barcode', 'spot_id', 'cell_id', 'index', 'ID']
        
        for i, col_name in enumerate(metadata.columns):

            if col_name in common_index_names:
                return i

            if metadata[col_name].isin(obs_names).sum() > len(obs_names) * 0.5:
                return i

        first_col = metadata.columns[0]
        if metadata[first_col].isin(obs_names).sum() > len(obs_names) * 0.3:
            return 0
        
        return None
    
    def _fill_10x_spatial(self, adata: sc.AnnData, spatial_dir: str, library_id: str) -> sc.AnnData:

        positions = pd.read_csv(os.path.join(spatial_dir, "tissue_positions_list.csv"), header=None, index_col=0)
        positions = positions.loc[adata.obs_names]

        adata.obsm['spatial'] = positions.iloc[:, [4, 3]].values

        with open(os.path.join(spatial_dir, "scalefactors_json.json")) as f:
            scalefactors = json.load(f)

        adata.uns["spatial"] = {
            library_id: {
                "images": {
                    "hires": self._read_hires_image(self._find_hires_image(os.path.dirname(spatial_dir), spatial_dir, library_id)),
                },
                "scalefactors": scalefactors,
                "metadata": {},
                "use_quality": "hires"
            }
        }

        self._map_coordinates_to_obs(adata)
        
        return adata
    
    def _load_metadata(self, adata: sc.AnnData, meta_path: str) -> None:

        if meta_path is not None and os.path.exists(meta_path):
            print(f"  -> Loading metadata file: {os.path.basename(meta_path)}")

            sep = '\t' if meta_path.endswith('.tsv') else ','
            
            try:

                metadata = pd.read_csv(meta_path, sep=sep)

                index_col = self._detect_metadata_index(metadata, adata.obs_names)
                
                if index_col is not None:

                    metadata = pd.read_csv(meta_path, sep=sep, index_col=index_col)

                    common_indices = metadata.index.intersection(adata.obs_names)
                    if len(common_indices) > 0:
                        metadata = metadata.loc[common_indices]
                        adata.obs = adata.obs.join(metadata, how='left')
                        print(f"    [OK] Loaded {len(common_indices)} metadata records")
                    else:
                        print("    [WARN] Metadata index does not match adata; skipping metadata load")
                else:
                    print("    [WARN] Could not detect metadata index column; skipping metadata load")
                    
            except Exception as e:
                print(f"    [ERROR] Metadata loading failed: {e}")
    
    def _map_coordinates_to_obs(self, adata: sc.AnnData) -> None:

        adata.obs["imagerow"] = adata.obsm['spatial'][:, 1]
        adata.obs["imagecol"] = adata.obsm['spatial'][:, 0]
        
        print("[OK] Spatial coordinates mapped to adata.obs['imagerow/col']")
    
    def _detect_coordinate_columns(self, coords: pd.DataFrame) -> List[str]:

        coord_patterns = [
            ['x', 'y'],
            ['pixel_x', 'pixel_y'],
            ['X', 'Y'],
            ['coord_x', 'coord_y']
        ]
        
        for pattern in coord_patterns:
            if all(col in coords.columns for col in pattern):
                return pattern

        numeric_cols = coords.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            return numeric_cols[:2]
        
        return []
    
    def _smart_barcode_matching(self, tsv_barcodes, spatial_barcodes):

        common_barcodes = tsv_barcodes.intersection(spatial_barcodes)
        if len(common_barcodes) > 0:
            return common_barcodes
        
        print("Trying smart barcode matching...")

        matching_strategies = [

            lambda x: x.split('-')[0] if '-' in x else x,

            lambda x: x[1:] if x.startswith('X') else x,

            lambda x: x.lower(),

            lambda x: ''.join(c for c in x if c.isalnum())
        ]
        
        for strategy in matching_strategies:
            try:

                tsv_transformed = tsv_barcodes.map(strategy)
                spatial_transformed = spatial_barcodes.map(strategy)

                common_barcodes = tsv_transformed.intersection(spatial_transformed)
                
                if len(common_barcodes) > 0:
                    print(f"[OK] Strategy found {len(common_barcodes)} matching barcodes")

                    matched_indices = []
                    for transformed_bc in common_barcodes:
                        tsv_match = tsv_barcodes[tsv_transformed == transformed_bc]
                        spatial_match = spatial_barcodes[spatial_transformed == transformed_bc]
                        
                        if len(tsv_match) == 1 and len(spatial_match) == 1:
                            matched_indices.append(tsv_match[0])
                    
                    return pd.Index(matched_indices)
                    
            except Exception as e:
                print(f"[WARN] Barcode matching strategy failed: {e}")
                continue

        print("Trying partial barcode matching...")

        tsv_set = set(tsv_barcodes)
        spatial_set = set(spatial_barcodes)

        partial_matches = []
        for tsv_bc in tsv_set:
            for spatial_bc in spatial_set:

                common_chars = set(tsv_bc) & set(spatial_bc)
                if len(common_chars) >= min(8, len(tsv_bc), len(spatial_bc)):
                    partial_matches.append(tsv_bc)
                    break
        
        if partial_matches:
            print(f"[OK] Found {len(partial_matches)} partial barcode matches")
            return pd.Index(partial_matches)
        
        print("[ERROR] All barcode matching strategies failed")
        return pd.Index([])
    
    def _load_custom_visium_data(self, data_path: str) -> sc.AnnData:

        print(f"Trying to load custom Visium data: {data_path}")

        files = os.listdir(data_path)

        h5_files = [f for f in files if f.endswith('.h5')]
        if h5_files:
            h5_file = os.path.join(data_path, h5_files[0])
            print(f"[OK] Found H5 file: {h5_file}")
            return sc.read_10x_h5(h5_file)

        mtx_files = [f for f in files if f.endswith('.mtx')]
        barcode_files = [f for f in files if 'barcode' in f.lower()]
        feature_files = [f for f in files if 'feature' in f.lower() or 'gene' in f.lower()]
        
        if mtx_files and barcode_files and feature_files:
            print("[OK] Found MTX format files")

            return self._load_mtx_data(data_path)

        tsv_gz_files = [f for f in files if f.endswith('.tsv.gz')]
        spatial_dir = os.path.join(data_path, "spatial")
        
        if tsv_gz_files and os.path.exists(spatial_dir):
            print(f"[OK] Found B1 format file: {tsv_gz_files[0]}")
            return self._load_b1_format(data_path)

        print("[WARN] Standard files not found; trying generic reader")
        return sc.read_visium(data_path)

    def _load_b1_format(self, data_path: str) -> sc.AnnData:

        import pandas as pd
        import numpy as np
        import json
        
        print(f"Loading B1 format data: {data_path}")
        sample_id = os.path.basename(data_path)

        count_file = os.path.join(data_path, f"{sample_id}.tsv.gz")
        if not os.path.exists(count_file):
            alt_count_file = os.path.join(data_path, f"{sample_id}.tsv")
            if os.path.exists(alt_count_file):
                count_file = alt_count_file
        if not os.path.exists(count_file):
            tsv_files = [f for f in os.listdir(data_path) if f.endswith(('.tsv.gz', '.tsv'))]
            count_file = os.path.join(data_path, tsv_files[0]) if tsv_files else None
        
        if count_file is None:
            raise FileNotFoundError(f"TSV or TSV.GZ expression file not found: {data_path}")
        
        df = pd.read_csv(count_file, sep='\t', index_col=0)
        if df.shape[0] > df.shape[1]:
            df = df.T
        adata = sc.AnnData(df)

        print("  -> Parsing spatial coordinates and aligning image...")
        spatial_dir = os.path.join(data_path, "spatial")

        selection_path = os.path.join(spatial_dir, f"{sample_id}_selection.tsv.gz")
        if not os.path.exists(selection_path):
            alt_selection_path = os.path.join(spatial_dir, f"{sample_id}_selection.tsv")
            if os.path.exists(alt_selection_path):
                selection_path = alt_selection_path
        
        if os.path.exists(selection_path):
            print(f"  -> Detected {sample_id}_selection.tsv.gz; parsing...")
            spatial_pos = pd.read_csv(selection_path, sep='\t')
            if "selected" in spatial_pos.columns:
                spatial_pos = spatial_pos[spatial_pos["selected"].astype(str).isin(["1", "1.0", "True", "true"])].copy()

            print(f"  Spatial position file shape: {spatial_pos.shape[0]} rows x {spatial_pos.shape[1]} columns")
            print(f"  Columns: {list(spatial_pos.columns)}")
            print("  First 3 rows:")
            print(spatial_pos.head(3))

            if spatial_pos.shape[1] == 7:
                print("  Detected 7-column spatial format")

                spatial_pos.columns = ['x', 'y', 'new_x', 'new_y', 'pixel_x', 'pixel_y', 'selected']

                spatial_pos['coord_id'] = spatial_pos.apply(
                    lambda row: f"{int(row['x'])}x{int(row['y'])}", axis=1
                )

                spatial_pos['pxl_col'] = spatial_pos['pixel_x']
                spatial_pos['pxl_row'] = spatial_pos['pixel_y']
                spatial_pos['row'] = spatial_pos['x']
                spatial_pos['col'] = spatial_pos['y']
                
            elif spatial_pos.shape[1] == 6:
                print("  Detected 6-column spatial format")
                spatial_pos.columns = ['barcode', 'in_tissue', 'row', 'col', 'pxl_row', 'pxl_col']
                spatial_pos['coord_id'] = spatial_pos.apply(
                    lambda x: f"{int(x['row'])}x{int(x['col'])}", axis=1
                )
            else:
                print(f"  [WARN] Unknown column count: {spatial_pos.shape[1]}; trying generic parsing")

                if 'x' in spatial_pos.columns and 'y' in spatial_pos.columns:
                    spatial_pos['coord_id'] = spatial_pos.apply(
                        lambda row: f"{int(row['x'])}x{int(row['y'])}", axis=1
                    )
                    spatial_pos['row'] = spatial_pos['x']
                    spatial_pos['col'] = spatial_pos['y']
                else:

                    spatial_pos.columns = ['col1', 'col2'] + list(spatial_pos.columns[2:])
                    spatial_pos['coord_id'] = spatial_pos.apply(
                        lambda row: f"{int(row['col1'])}x{int(row['col2'])}", axis=1
                    )
                    spatial_pos['row'] = spatial_pos['col1']
                    spatial_pos['col'] = spatial_pos['col2']

                for col in spatial_pos.columns:
                    if 'pixel' in col.lower() or 'pxl' in col.lower():
                        if 'x' in col.lower() or 'col' in col.lower():
                            spatial_pos['pxl_col'] = spatial_pos[col]
                        elif 'y' in col.lower() or 'row' in col.lower():
                            spatial_pos['pxl_row'] = spatial_pos[col]

                if 'pxl_col' not in spatial_pos.columns:
                    spatial_pos['pxl_col'] = spatial_pos['col']
                if 'pxl_row' not in spatial_pos.columns:
                    spatial_pos['pxl_row'] = spatial_pos['row']
            
        else:
            print("  -> Using fallback tissue_positions_list.csv...")
            spatial_pos = pd.read_csv(
                os.path.join(spatial_dir, "tissue_positions_list.csv"),
                header=None
            )

            spatial_pos.columns = ['barcode', 'in_tissue', 'row', 'col', 'pxl_row', 'pxl_col']
            spatial_pos['coord_id'] = spatial_pos.apply(
                lambda x: f"{int(x['row'])}x{int(x['col'])}", axis=1
            )

        spatial_pos = spatial_pos.drop_duplicates(subset='coord_id', keep='first').set_index('coord_id')
        
        common_ids = adata.obs_names.intersection(spatial_pos.index)
        adata = adata[common_ids].copy()
        matched_spatial = spatial_pos.loc[common_ids]

        print("\nCoordinate summary:")
        print(f"  row range: [{matched_spatial['row'].min():.1f}, {matched_spatial['row'].max():.1f}]")
        print(f"  col range: [{matched_spatial['col'].min():.1f}, {matched_spatial['col'].max():.1f}]")
        print(f"  pxl_row range: [{matched_spatial['pxl_row'].min():.1f}, {matched_spatial['pxl_row'].max():.1f}]")
        print(f"  pxl_col range: [{matched_spatial['pxl_col'].min():.1f}, {matched_spatial['pxl_col'].max():.1f}]")
        print(f"  spot count: {len(matched_spatial)}")

        img_path = self._find_hires_image(data_path, spatial_dir, sample_id)
        img = self._read_hires_image(img_path)
        img_h, img_w = img.shape[:2]
        print(f"  image size: {img_w} x {img_h} (width x height)")

        auto_scale = 1.0

        final_x = matched_spatial['pxl_col'].values
        final_y = matched_spatial['pxl_row'].values

        x_offset = 0
        y_offset = 0

        adata.obsm['spatial'] = np.column_stack([final_x + x_offset, final_y + y_offset])
        
        print("[OK] Coordinates aligned with native 1:1 image scale")

        print("\nCoordinate summary:")
        print(f"  spatial[:, 0] (x/col) range: [{adata.obsm['spatial'][:, 0].min():.1f}, {adata.obsm['spatial'][:, 0].max():.1f}]")
        print(f"  spatial[:, 1] (y/row) range: [{adata.obsm['spatial'][:, 1].min():.1f}, {adata.obsm['spatial'][:, 1].max():.1f}]")
        print(f"  expected image range: [0, {img_w}] x [0, {img_h}]")

        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        
        coords = adata.obsm['spatial']

        nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
        distances, _ = nbrs.kneighbors(coords)

        nn_dist = distances[:, 1]
        median_spacing = np.median(nn_dist)

        spot_diameter_fullres = median_spacing * 0.4

        max_spot_size = min(img_w, img_h) / 50
        spot_diameter_fullres = min(spot_diameter_fullres, max_spot_size)

        scalefactors = {
            "tissue_hires_scalef": 1.0,
            "spot_diameter_fullres": spot_diameter_fullres,
            "fiducial_diameter_fullres": spot_diameter_fullres * 2.5,
            "tissue_lowres_scalef": 0.1
        }
        
        print(f"  calculated spot_diameter_fullres: {spot_diameter_fullres:.2f}")
        print(f"  median spot spacing: {median_spacing:.2f}")
        
        adata.uns["spatial"] = {
            sample_id: {
                "images": {"hires": img},
                "scalefactors": scalefactors,
                "use_quality": "hires"
            }
        }

        meta_path = self._find_metadata_file(data_path)
        if meta_path is not None and os.path.exists(meta_path):
            print("  -> Aligning metadata with physical coordinates...")

            meta_sep = "," if meta_path.endswith(".csv") else "\t"
            meta = pd.read_csv(meta_path, sep=meta_sep, index_col=0)

            print(f"  Metadata columns: {list(meta.columns)}")
            print("  Metadata first rows:")
            print(meta.head(3))

            required_cols = ['pixel_x', 'pixel_y', 'x', 'y', 'label']
            available_cols = [col for col in required_cols if col in meta.columns]
            print(f"  Available columns: {available_cols}")

            if 'pixel_x' in meta.columns and 'pixel_y' in meta.columns:
                meta_coords = meta[['pixel_x', 'pixel_y']].values
            else:

                coord_cols = [col for col in ['x', 'y', 'array_col', 'array_row'] if col in meta.columns]
                if len(coord_cols) >= 2:
                    meta_coords = meta[coord_cols[:2]].values
                    print(f"  Using fallback coordinate columns: {coord_cols[:2]}")
                else:
                    print("Could not find suitable coordinate columns.")
                    meta_coords = np.zeros((len(meta), 2))

            cur_coords = adata.obsm['spatial']
            
            print(f"   -> Metadata coordinate range: X[{meta_coords[:,0].min():.1f}-{meta_coords[:,0].max():.1f}], Y[{meta_coords[:,1].min():.1f}-{meta_coords[:,1].max():.1f}]")
            print(f"   -> Current data coordinate range: X[{cur_coords[:,0].min():.1f}-{cur_coords[:,0].max():.1f}], Y[{cur_coords[:,1].min():.1f}-{cur_coords[:,1].max():.1f}]")

            if 'x' in meta.columns and 'y' in meta.columns:
                meta['array_id'] = meta.apply(lambda row: f"{int(round(row['x']))}x{int(round(row['y']))}", axis=1)
            else:

                print("[ERROR] Could not find x/y columns; trying other matching methods")
                return adata

            adata.obs['array_id'] = adata.obs_names

            if 'label' in meta.columns:
                meta_subset = meta[['label', 'array_id']].drop_duplicates('array_id')
            else:

                label_col = meta.columns[0]
                meta_subset = meta[[label_col, 'array_id']].drop_duplicates('array_id')
                meta_subset = meta_subset.rename(columns={label_col: 'label'})
                print(f"  Using '{label_col}' as label column")
            
            adata.obs = adata.obs.merge(meta_subset, on='array_id', how='left').set_index('array_id')
            
            matched_count = adata.obs['label'].notna().sum()
            print(f"Alignment result: matched {matched_count} / {len(adata)} labels")

        adata.obs["imagecol"] = adata.obsm['spatial'][:, 0]
        adata.obs["imagerow"] = adata.obsm['spatial'][:, 1]
        print("[OK] Spatial coordinates mapped to adata.obs['imagerow/col']")
        
        print(f"[OK] B1 data loaded successfully: {adata.shape}")
        return adata

    def _get_image_crop(
        self,
        adata: sc.AnnData,
        data_name: str,
        cnn_type: str = 'UNI2-h',
        pca_n_comps: int = 50,
        use_diversity_prune: bool = True,
        divprune_keep_ratio: float = 0.75,
        divprune_keep_count: Optional[int] = None,
        divprune_start: Optional[str] = None
    ) -> sc.AnnData:
        """
    Extract image features from spatial data
    
    Parameters:
    -----------
    adata : sc.AnnData
        Spatial transcriptomics data
    data_name : str
        Name of dataset
    cnn_type : str
        CNN model type for feature extraction (default: 'UNI2-h')
    pca_n_comps : int
        Number of PCA components (default: 50)
        
    Returns:
    --------
    sc.AnnData
        Data with extracted image features
    """
    
        save_dir = self.save_path / 'Image_crop' / data_name
        save_dir.mkdir(parents=True, exist_ok=True)
    
        adata = image_crop(adata, save_path=save_dir)
        adata = image_feature(
            adata, 
            pca_components=pca_n_comps, 
            cnnType=cnn_type,
            seeds=self.seed,
            use_diversity_prune=use_diversity_prune,
            divprune_keep_ratio=divprune_keep_ratio,
            divprune_keep_count=divprune_keep_count,
            divprune_start=divprune_start,
        ).extract_image_feat()
    
        return adata

    def _get_graph(
        self,
        spatial_coords: np.ndarray,
        distType: str = "BallTree",
        k: int = 12,
        rad_cutoff: float = 150
    ) -> dict:
        """
        Compute spatial graph
        
        Parameters:
        -----------
        spatial_coords : np.ndarray
            Spatial coordinates [n_spots, 2]
        distType : str
            Distance computation method (default: "BallTree")
        k : int
            Number of neighbors (default: 12)
        rad_cutoff : float
            Radius cutoff (default: 150)
            
        Returns:
        --------
        dict
            Graph dictionary containing:
            - adj_norm: Normalized adjacency matrix
            - adj_label: Original adjacency labels
            - norm_value: Normalization factor
        """
        graph_dict = graph(
            spatial_coords,
            distType=distType,
            k=k,
            rad_cutoff=rad_cutoff
        ).main()
        print("Step 2: Spatial graph computation completed")
        return graph_dict

    def _get_triview_graph(
        self,
        expression_data: np.ndarray,
        spatial_coords: np.ndarray,
        image_features: np.ndarray,
        expression_k: int = 12,
        spatial_k: int = 6,
        edge_spatial_k: int = 6,
        image_pca_n: int = 50,
        image_graph_mode: str = "similarity",
        local_subgraph_enabled: bool = True,
        local_expr_keep_ratio: float = 1.0,
        local_spatial_keep_ratio: float = 1.0,
        local_edge_keep_ratio: float = 1.0,
        latgrl_enabled: bool = True,
        latgrl_homo_ratio: float = 0.35,
        latgrl_hetero_ratio: float = 0.20,
    ) -> dict:
        """
        Build expression, spatial and pathology-morphology views for tri-view training.

        Inputs follow the intended flow:
        X: expression/PCA matrix, C: spatial coordinates, I: image patch features.
        The third view uses UNI/DivPrune image features directly.
        """
        graph_kwargs = dict(
            expression=expression_data,
            spatial_coords=spatial_coords,
            image_features=image_features,
            expression_k=expression_k,
            spatial_k=spatial_k,
            edge_spatial_k=edge_spatial_k,
            image_pca_n=image_pca_n,
            image_graph_mode=image_graph_mode,
            local_subgraph_enabled=local_subgraph_enabled,
            local_expr_keep_ratio=local_expr_keep_ratio,
            local_spatial_keep_ratio=local_spatial_keep_ratio,
            local_edge_keep_ratio=local_edge_keep_ratio,
            latgrl_enabled=latgrl_enabled,
            latgrl_homo_ratio=latgrl_homo_ratio,
            latgrl_hetero_ratio=latgrl_hetero_ratio,
        )
        try:
            import inspect
            accepted = set(inspect.signature(build_triview_graph_dict).parameters)
            graph_kwargs = {k: v for k, v in graph_kwargs.items() if k in accepted}
        except (TypeError, ValueError):
            pass
        graph_dict = build_triview_graph_dict(**graph_kwargs)
        print("Step 2: Tri-view graph computation completed")
        print(
            "  Af expression k={expression_k}, As spatial k={spatial_k}, "
            "Ae pathology image k={edge_spatial_k}, image PCA={image_pca_n}".format(
                expression_k=expression_k,
                spatial_k=spatial_k,
                edge_spatial_k=edge_spatial_k,
                image_pca_n=image_pca_n,
            )
        )
        print(f"  image_graph_mode={image_graph_mode}")
        print(
            "  local_subgraph_enabled={local_subgraph_enabled}, "
            "latgrl_enabled={latgrl_enabled}, homo_ratio={latgrl_homo_ratio}, hetero_ratio={latgrl_hetero_ratio}".format(
                local_subgraph_enabled=local_subgraph_enabled,
                latgrl_enabled=latgrl_enabled,
                latgrl_homo_ratio=latgrl_homo_ratio,
                latgrl_hetero_ratio=latgrl_hetero_ratio,
            )
        )
        print("  Tri-view graphs are built here; GNN aggregation is selected during model training")
        print("  Third view uses direct UNI/DivPrune morphology features")
        print("  Tri-view graph-only training will use expression, spatial, and pathology image views")
        return graph_dict

    def _optimize_cluster(
        self,
        adata: sc.AnnData,
        resolution_range: List[float] = None
    ) -> float:
        """
        Optimize clustering resolution
        
        Parameters:
        -----------
        adata : sc.AnnData
            Spatial transcriptomics data
        resolution_range : List[float]
            Range of resolutions to test (default: 0.1-2.5 in 0.01 steps)
            
        Returns:
        --------
        float
            Optimal resolution value
        """
        if resolution_range is None:
            resolution_range = list(np.arange(0.1, 2.5, 0.01))
            
        scores = []
        for r in resolution_range:
            sc.tl.leiden(adata, resolution=r, n_iterations=2, directed=False, random_state=0)
            s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
            scores.append(s)
            
        best_idx = np.argmax(scores)
        best_res = resolution_range[best_idx]
        print(f"Optimal resolution: {best_res:.2f}")
        return best_res

    def _priori_cluster(
        self,
        adata: sc.AnnData,
        n_domains: int,
        return_info: bool = False
    ) -> float:
        """
        Find resolution matching prior domain count
        
        Parameters:
        -----------
        adata : sc.AnnData
            Spatial transcriptomics data
        n_domains : int
            Target number of domains
            
        Returns:
        --------
        float
            Resolution yielding target domain count
        """
        best_res = 1.0
        best_diff = float("inf")
        best_count = None
        for res in sorted(np.arange(0.1, 2.5, 0.01), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res, n_iterations=2, directed=False)
            n_found = len(adata.obs['leiden'].unique())
            diff = abs(n_found - n_domains)
            if n_found == n_domains:
                print(f"Found resolution: {res:.2f} for {n_domains} domains")
                if return_info:
                    return res, n_found, True
                return res
            if diff < best_diff:
                best_diff = diff
                best_res = res
                best_count = n_found
        print(
            f"Warning: exact {n_domains} domains not found; "
            f"closest Leiden resolution {best_res:.2f} gives {best_count} domains"
        )
        if return_info:
            return best_res, best_count, False
        return best_res

    def _get_multiple_adata(
        self,
        adata_list: List[sc.AnnData],
        data_name_list: List[str],
        graph_list: List[dict]
    ) -> tuple:
        """
        Integrate multiple spatial datasets
        
        Parameters:
        -----------
        adata_list : List[sc.AnnData]
            List of spatial datasets
        data_name_list : List[str]
            List of dataset names
        graph_list : List[dict]
            List of spatial graphs
            
        Returns:
        --------
        tuple: (sc.AnnData, dict)
            Integrated AnnData object and combined graph
        """
        multiple_adata = None
        multiple_graph = None
        
        for i, (current_adata, current_name) in enumerate(zip(adata_list, data_name_list)):
            current_adata.obs['batch_name'] = current_name
            current_adata.obs['batch_name'] = current_adata.obs['batch_name'].astype('category')
            
            if i == 0:
                multiple_adata = current_adata
                multiple_graph = graph_list[i]
            else:
                var_names = multiple_adata.var_names.intersection(current_adata.var_names)
                multiple_adata = multiple_adata[:, var_names]
                current_adata = current_adata[:, var_names]
                multiple_adata = anndata.concat([multiple_adata, current_adata])
                multiple_graph = combine_graph_dict(multiple_graph, graph_list[i])

        multiple_adata.obs["batch"] = pd.Categorical(
            multiple_adata.obs['batch_name']
        ).codes.astype(np.int64)
        
        return multiple_adata, multiple_graph

    def _data_process(
        self,
        adata: sc.AnnData,
        pca_n_comps: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess spatial transcriptomics data
        
        Parameters:
        -----------
        adata : sc.AnnData
            Spatial transcriptomics data
        pca_n_comps : int
            Number of PCA components (default: 200)
        Returns:
        --------
        np.ndarray
            Processed data matrix
        """
        adata.raw = adata
        input_matrix = adata.X
        adata.uns["rna_input_source"] = "adata.X"

        X = input_matrix
        if issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        counts = np.asarray(X.sum(axis=1)).reshape(-1)
        counts[counts == 0] = 1.0
        X = X / counts[:, None]
        X = np.log1p(X)

        X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-9)

        pca = PCA(n_components=pca_n_comps, random_state=0, svd_solver="randomized")
        data = pca.fit_transform(X)

        return data

    def _fit(
        self,
        data: np.ndarray,
        graph_dict: dict,
        domains: Optional[np.ndarray] = None,
        n_domains: Optional[int] = None,
        conv_type: str = "GCNConv",
        linear_encoder_hidden: List[int] = [32, 20],
        linear_decoder_hidden: List[int] = [32],
        conv_hidden: List[int] = [32, 8],
        p_drop: float = 0.01,
        dec_cluster_n: int = 20,
        kl_weight: float = 1,
        mse_weight: float = 1,
        bce_kld_weight: float = 1,
        domain_weight: float = 1,
        tri_view: bool = False,
        edge_input_dim: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run GeMCST model training
        
        Parameters:
        -----------
        data : np.ndarray
            Processed input data [n_spots, n_features]
        graph_dict : dict
            Spatial graph dictionary
        domains : np.ndarray, optional
            Domain labels for integration (default: None)
        n_domains : int, optional
            Number of domains for integration (default: None)
        conv_type : str
            Graph convolution type (default: "GCNConv")
        linear_encoder_hidden : List[int]
            Encoder hidden layer sizes (default: [32, 20])
        linear_decoder_hidden : List[int]
            Decoder hidden layer sizes (default: [32])
        conv_hidden : List[int]
            GNN hidden layer sizes (default: [32, 8])
        p_drop : float
            Dropout probability (default: 0.01)
        dec_cluster_n : int
            Number of clusters (default: 20)
        kl_weight : float
            KL divergence weight (default: 1)
        mse_weight : float
            MSE loss weight (default: 1)
        bce_kld_weight : float
            BCE+KLD loss weight (default: 1)
        domain_weight : float
            Domain loss weight (default: 1)
        tri_view : bool
            Whether to use tissue-edge guided tri-view encoding.
        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            Final GeMCST embeddings and the continuous pre-clustering latent
            [n_spots, n_features].
        """
        print("Running GeMCST anchor analysis...")
        start_time = time.time()

        model = GeMCSTModel(
                        input_dim=data.shape[1],
                        Conv_type=conv_type,
                        linear_encoder_hidden=linear_encoder_hidden,
                        linear_decoder_hidden=linear_decoder_hidden,
                        conv_hidden=conv_hidden,
                        p_drop=p_drop,
                        dec_cluster_n=dec_cluster_n,
                        tri_view=tri_view,
                        edge_input_dim=edge_input_dim,
                        )

        if self.use_gpu and torch.cuda.is_available():
            model = model.cuda()

        if self.task == "Identify_Domain":
            trainer = train(
                data, graph_dict, model,
                pre_epochs=self.pre_epochs,
                epochs=self.epochs,
                kl_weight=kl_weight,
                mse_weight=mse_weight,
                bce_kld_weight=bce_kld_weight,
                domain_weight=domain_weight,
                use_gpu=self.use_gpu,
            )
        elif self.task == "Integration":
            if n_domains is None:
                raise ValueError("n_domains must be specified for integration task")
                
            adv_model = AdversarialNetwork(model=model, n_domains=n_domains)
            trainer = train(
                data, graph_dict, adv_model,
                domains=domains,
                pre_epochs=self.pre_epochs,
                epochs=self.epochs,
                kl_weight=kl_weight,
                mse_weight=mse_weight,
                bce_kld_weight=bce_kld_weight,
                domain_weight=domain_weight,
                use_gpu=self.use_gpu,
            )
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        trainer.fit()
        embeddings, _ = trainer.process()

        mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        total_time = (time.time() - start_time) / 60
        print("Step 3: GeMCST anchor training completed")
        print(f"Memory usage: {mem_usage:.2f} GB")
        print(f"Total time: {total_time:.2f} minutes")
        print("Analysis completed successfully")
        
        return embeddings, trainer.precluster_latent

    def _get_cluster_data(
        self,
        adata: sc.AnnData,
        n_domains: int,
        priori: bool = True,
        batch_key: Optional[str] = None,
        use_obsm: str = "GeMCST_embed",
        key_added: str = "GeMCST_domain",
        output_key: str = "GeMCST_refine_domain",
        shape: str = "hexagon",               
        use_cancer_boundary_refine: bool = True,
        cbr_spatial_k: int = 6,
        cbr_max_change_ratio: float = 0.05,
        cbr_boundary_quantile: float = 0.70,
        cbr_low_confidence_quantile: float = 0.35,
        cbr_margin: float = 0.05,
        cbr_lambda_spatial: float = 0.60,
        cbr_lambda_uni: float = 0.30,
        cbr_lambda_anchor: float = 1.00,
        cbr_boundary_loss_weight: float = 0.50,
        cbr_multiscale: bool = True,
        cbr_compartment_aware: bool = False,
        cbr_lambda_compartment: float = 0.25,
        cluster_method: str = "auto",
    ) -> sc.AnnData:
        """
        Perform spatial clustering using GeMCST embeddings and optionally refine results.

        This method applies Leiden clustering to spatial transcriptomics data using the specified
        embedding (`use_obsm`) and optionally refines cluster labels based on spatial proximity.
        It supports both single-slice and multi-slice datasets (via `batch_key`).

        Parameters
        ----------
        adata : sc.AnnData
            Annotated data matrix. Requires:
              - `adata.obsm[use_obsm]`: embedding representation for clustering
              - `adata.obsm['spatial']`: spatial coordinates of each spot/cell

        n_domains : int
            Target number of spatial domains (clusters).

        priori : bool, optional (default: True)
            If True, use `n_domains` directly. If False, optimize number of clusters automatically.

        batch_key : str or None, optional (default: None)
            Key in `adata.obs` that identifies distinct tissue slices. If None, assumes single-slice data.

        use_obsm : str, optional (default: "GeMCST_embed")
            Name of the key in `adata.obsm` containing embedding to use for clustering.

        key_added : str, optional (default: "GeMCST_domain")
            Name of the column in `adata.obs` where Leiden clustering results will be stored.

        output_key : str, optional (default: "GeMCST_refine_domain")
            Name of the column in `adata.obs` to store refined cluster labels based on spatial neighbors.

        shape : str, optional (default: "hexagon")
            Neighborhood structure to use during refinement.
            Choose "hexagon" for Visium-like data or "square" for grid-based ST platforms.

        use_cancer_boundary_refine : bool, optional (default: True)
            If True, run a conservative cancer boundary-aware refinement after the standard
            spatial refinement. The module only updates low-confidence boundary spots using
            GeMCST anchor distance, spatial neighbor support, and UNI morphology support.

        Returns
        -------
        adata : sc.AnnData
            The updated AnnData object with:
            - `adata.obs[key_added]`: initial clustering labels from Leiden
            - `adata.obs[output_key]`: refined domain labels (post-spatial-smoothing)
        """
        sc.pp.neighbors(adata, use_rep=use_obsm, random_state=0)
        
        method = str(cluster_method or "auto").strip().lower()
        if method == "auto":
            method = "leiden_then_kmeans" if priori and n_domains is not None else "leiden"
        valid_methods = {"leiden", "kmeans", "leiden_then_kmeans"}
        if method not in valid_methods:
            raise ValueError(f"Unsupported cluster_method={cluster_method!r}; use one of {sorted(valid_methods)}")

        used_cluster_method = method
        resolution = None
        n_leiden = None
        exact_match = None

        if method == "kmeans":
            if n_domains is None:
                raise ValueError("n_domains is required when cluster_method='kmeans'")
            labels = KMeans(
                n_clusters=n_domains,
                n_init=n_domains * 2,
                random_state=88,
            ).fit_predict(np.asarray(adata.obsm[use_obsm]))
            adata.obs[key_added] = pd.Categorical(labels.astype(str))
        else:
            if priori:
                resolution, n_leiden, exact_match = self._priori_cluster(
                    adata,
                    n_domains,
                    return_info=True,
                )
            else:
                resolution = self._optimize_cluster(adata)
                exact_match = True
            if method == "leiden_then_kmeans" and not exact_match:
                labels = KMeans(
                    n_clusters=n_domains,
                    n_init=n_domains * 2,
                    random_state=88,
                ).fit_predict(np.asarray(adata.obsm[use_obsm]))
                adata.obs[key_added] = pd.Categorical(labels.astype(str))
                used_cluster_method = "kmeans_fallback"
                print(
                    f"Leiden produced {n_leiden} domains instead of {n_domains}; "
                    "using KMeans fallback with fixed domain count"
                )
            else:
                sc.tl.leiden(
                    adata,
                    key_added=key_added,
                    resolution=resolution,
                    n_iterations=2,
                    directed=False,
                    random_state=0,
                )

        clustering_info = {
            "requested_cluster_method": cluster_method,
            "cluster_method": used_cluster_method,
            "n_domains": int(n_domains) if n_domains is not None else None,
            "leiden_resolution": float(resolution) if resolution is not None else None,
            "leiden_domain_count": int(n_leiden) if n_leiden is not None else None,
            "leiden_exact_match": bool(exact_match) if exact_match is not None else None,
            "final_domain_count": int(adata.obs[key_added].nunique()),
        }
        adata.uns["gemcst_clustering"] = {
            key: value for key, value in clustering_info.items() if value is not None
        }

        if batch_key and batch_key in adata.obs.columns:
            result = []
            for b in adata.obs[batch_key].unique():
                sub = adata[adata.obs[batch_key] == b]
                if _use_legacy_dense_refinement(sub.n_obs):
                    adj_2d = distance.cdist(sub.obsm['spatial'], sub.obsm['spatial'], 'euclidean')
                    refined = refine(sub.obs_names.tolist(), sub.obs[key_added].tolist(), adj_2d, shape)
                else:
                    refined = refine_knn(
                        sub.obs_names.tolist(), sub.obs[key_added].tolist(), sub.obsm['spatial'], shape
                    )
                result.extend(zip(sub.obs_names.tolist(), refined))
            adata.obs[output_key] = pd.Series(dict(result))
        else:
            if _use_legacy_dense_refinement(adata.n_obs):
                adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
                refined = refine(adata.obs_names.tolist(), adata.obs[key_added].tolist(), adj_2d, shape)
            else:
                refined = refine_knn(
                    adata.obs_names.tolist(), adata.obs[key_added].tolist(), adata.obsm['spatial'], shape
                )
            adata.obs[output_key] = refined

        adata.uns["gemcst_clustering"]["refined_domain_count"] = int(adata.obs[output_key].nunique())
        adata.obs[f"{output_key}_anchor"] = adata.obs[output_key].astype(str).values

        if use_cancer_boundary_refine:
            adata = self._cancer_boundary_refine(
                adata=adata,
                label_key=output_key,
                embedding_key=use_obsm,
                spatial_k=cbr_spatial_k,
                max_change_ratio=cbr_max_change_ratio,
                boundary_quantile=cbr_boundary_quantile,
                low_confidence_quantile=cbr_low_confidence_quantile,
                margin=cbr_margin,
                lambda_spatial=cbr_lambda_spatial,
                lambda_uni=cbr_lambda_uni,
                lambda_anchor=cbr_lambda_anchor,
                boundary_loss_weight=cbr_boundary_loss_weight,
                multiscale=cbr_multiscale,
                compartment_aware=cbr_compartment_aware,
                lambda_compartment=cbr_lambda_compartment,
            )
            adata.uns["gemcst_clustering"]["post_cbr_domain_count"] = int(adata.obs[output_key].nunique())
        
        return adata

    def _compute_cbr_compartment_scores(
        self,
        adata: sc.AnnData,
        min_genes: int = 2,
    ):
        signatures = {
            "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1"],
            "immune": ["PTPRC", "CD3D", "CD3E", "CD2", "TRAC", "LYZ", "CD68"],
            "stroma": ["COL1A1", "COL1A2", "DCN", "LUM", "ACTA2", "TAGLN"],
            "hypoxia": ["VEGFA", "CA9", "SLC2A1", "LDHA", "PGK1"],
            "invasion": ["VIM", "SNAI2", "ITGA5", "MMP2", "MMP9"],
        }
        n_spots = adata.n_obs
        var_lookup = {}
        for idx, gene in enumerate(adata.var_names.astype(str)):
            key = gene.upper()
            if key not in var_lookup:
                var_lookup[key] = idx

        def _minmax(values):
            values = np.asarray(values, dtype=np.float64)
            lo = np.nanmin(values)
            hi = np.nanmax(values)
            if hi - lo < 1e-12:
                return np.zeros_like(values)
            return (values - lo) / (hi - lo)

        score_names = list(signatures.keys())
        score_matrix = np.zeros((n_spots, len(score_names)), dtype=np.float64)
        matched = {}
        valid_signature = np.zeros(len(score_names), dtype=bool)
        usable = False

        for score_idx, name in enumerate(score_names):
            genes = [gene for gene in signatures[name] if gene.upper() in var_lookup]
            matched[name] = genes
            if len(genes) < min_genes:
                continue
            usable = True
            valid_signature[score_idx] = True
            gene_idx = [var_lookup[gene.upper()] for gene in genes]
            expr = adata.X[:, gene_idx]
            if issparse(expr):
                expr = expr.toarray()
            expr = np.asarray(expr, dtype=np.float64)
            if np.nanmax(expr) > 20:
                expr = np.log1p(expr)
            expr = (expr - np.nanmean(expr, axis=0, keepdims=True)) / (
                np.nanstd(expr, axis=0, keepdims=True) + 1e-9
            )
            score_matrix[:, score_idx] = np.nanmean(expr, axis=1)

        ranked_scores = score_matrix.copy()
        ranked_scores[:, ~valid_signature] = -np.inf
        if valid_signature.sum() >= 2:
            order = np.argsort(ranked_scores, axis=1)
            top_idx = order[:, -1]
            second_idx = order[:, -2]
            top_score = score_matrix[np.arange(n_spots), top_idx]
            second_score = score_matrix[np.arange(n_spots), second_idx]
            closeness = 1.0 - (top_score - second_score) / (
                np.abs(top_score) + np.abs(second_score) + 1e-9
            )
            mixed_score = np.clip(closeness, 0.0, 1.0) * _minmax(second_score)
            top_compartment = np.array(score_names, dtype=object)[top_idx]
        elif valid_signature.sum() == 1:
            top_idx = np.full(n_spots, int(np.where(valid_signature)[0][0]), dtype=np.int64)
            mixed_score = np.zeros(n_spots, dtype=np.float64)
            top_compartment = np.array(score_names, dtype=object)[top_idx]
        else:
            mixed_score = np.zeros(n_spots, dtype=np.float64)
            top_compartment = np.array(["unavailable"] * n_spots, dtype=object)

        score_dict = {name: score_matrix[:, idx] for idx, name in enumerate(score_names)}
        score_dict["mixed_compartment"] = mixed_score
        score_dict["top_compartment"] = top_compartment
        summary = {
            "enabled": bool(usable),
            "min_genes": int(min_genes),
            "matched_genes": matched,
            "valid_signatures": [score_names[idx] for idx in np.where(valid_signature)[0]],
            "signature_order": score_names,
        }
        return score_dict, score_matrix, summary

    def _cancer_boundary_refine(
        self,
        adata: sc.AnnData,
        label_key: str = "GeMCST_refine_domain",
        embedding_key: str = "GeMCST_embed",
        image_key: str = "image_feat_pca",
        spatial_k: int = 6,
        max_change_ratio: float = 0.05,
        boundary_quantile: float = 0.70,
        low_confidence_quantile: float = 0.35,
        margin: float = 0.05,
        lambda_spatial: float = 0.60,
        lambda_uni: float = 0.30,
        lambda_anchor: float = 1.00,
        boundary_loss_weight: float = 0.50,
        multiscale: bool = True,
        compartment_aware: bool = False,
        lambda_compartment: float = 0.25,
    ) -> sc.AnnData:
        """
        Conservative cancer boundary-aware label refinement.

        This is inspired by MRF/Potts spatial label refinement and STAGATE-style
        boundary-aware neighbor weighting. It does not retrain GeMCST or alter
        embeddings; it only updates a small number of uncertain boundary spots when
        RNA anchor, spatial continuity, and UNI morphology jointly support a change.
        """
        required = [label_key]
        if embedding_key not in adata.obsm:
            print(f"CBR skipped: adata.obsm['{embedding_key}'] is missing.")
            return adata
        if image_key not in adata.obsm:
            print(f"CBR skipped: adata.obsm['{image_key}'] is missing.")
            return adata
        if "spatial" not in adata.obsm:
            print("CBR skipped: adata.obsm['spatial'] is missing.")
            return adata
        for key in required:
            if key not in adata.obs:
                print(f"CBR skipped: adata.obs['{key}'] is missing.")
                return adata

        labels = adata.obs[label_key].astype(str).to_numpy()
        n_spots = len(labels)
        if n_spots == 0:
            return adata

        spatial_k = int(max(1, min(spatial_k, n_spots - 1)))
        max_changes = int(np.floor(max_change_ratio * n_spots))
        if max_changes < 1:
            print("CBR skipped: max_change_ratio allows no label changes.")
            return adata

        z = np.asarray(adata.obsm[embedding_key], dtype=np.float64)
        img = np.asarray(adata.obsm[image_key], dtype=np.float64)
        spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)

        label_values = np.array(sorted(pd.unique(labels).astype(str)))
        label_to_idx = {label: idx for idx, label in enumerate(label_values)}
        y = np.array([label_to_idx[label] for label in labels], dtype=np.int64)
        n_labels = len(label_values)
        if n_labels <= 1:
            print("CBR skipped: only one domain is present.")
            return adata

        centroids = np.zeros((n_labels, z.shape[1]), dtype=np.float64)
        for idx in range(n_labels):
            mask = y == idx
            centroids[idx] = z[mask].mean(axis=0)
        anchor_dist = ((z[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
        order = np.argsort(anchor_dist, axis=1)
        best = anchor_dist[np.arange(n_spots), order[:, 0]]
        second = anchor_dist[np.arange(n_spots), order[:, 1]]
        anchor_confidence = (second - best) / (second + 1e-9)

        max_k = spatial_k
        if multiscale:
            max_k = min(n_spots - 1, max(spatial_k, spatial_k * 4))
        if _use_legacy_dense_refinement(n_spots):
            spatial_dist = distance.cdist(spatial, spatial, "euclidean")
            sorted_neighbors = np.argsort(spatial_dist, axis=1)[:, 1:max_k + 1]
        else:
            neighbor_candidates = NearestNeighbors(
                n_neighbors=max_k + 1,
                algorithm="kd_tree",
                n_jobs=-1,
            ).fit(spatial).kneighbors(spatial, return_distance=False)
            self_idx = np.arange(n_spots)[:, None]
            sorted_neighbors = neighbor_candidates[neighbor_candidates != self_idx].reshape(n_spots, max_k)
        neighbor_idx = sorted_neighbors[:, :spatial_k]

        img_norm = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)
        z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
        compartment_scores = None
        compartment_matrix = None
        compartment_summary = {
            "enabled": False,
            "reason": "CBR_COMPARTMENT_AWARE is disabled",
        }
        if compartment_aware:
            compartment_scores, compartment_matrix, compartment_summary = self._compute_cbr_compartment_scores(adata)
            for name in ["epithelial", "immune", "stroma", "hypoxia", "invasion"]:
                adata.obs[f"cbr_{name}_score"] = compartment_scores[name]
            adata.obs["cbr_mixed_compartment_score"] = compartment_scores["mixed_compartment"]
            adata.obs["cbr_top_compartment"] = compartment_scores["top_compartment"]
            adata.uns["cbr_compartment_signatures"] = compartment_summary
            if not compartment_summary["enabled"]:
                compartment_aware = False
                print("CBR compartment-aware evidence skipped: insufficient matched signature genes.")

        compartment_norm = None
        compartment_centroids = None
        if compartment_aware and compartment_matrix is not None:
            compartment_norm = compartment_matrix / (
                np.linalg.norm(compartment_matrix, axis=1, keepdims=True) + 1e-9
            )
            compartment_centroids = np.zeros((n_labels, compartment_matrix.shape[1]), dtype=np.float64)
            for idx in range(n_labels):
                mask = y == idx
                compartment_centroids[idx] = compartment_matrix[mask].mean(axis=0)
            compartment_centroids = compartment_centroids / (
                np.linalg.norm(compartment_centroids, axis=1, keepdims=True) + 1e-9
            )

        boundary_score = np.zeros(n_spots, dtype=np.float64)
        rna_gradient = np.zeros(n_spots, dtype=np.float64)
        morph_gradient = np.zeros(n_spots, dtype=np.float64)
        for i in range(n_spots):
            nb = neighbor_idx[i]
            boundary_score[i] = np.mean(y[nb] != y[i])
            rna_gradient[i] = np.mean(1.0 - np.maximum(z_norm[nb] @ z_norm[i], 0.0))
            morph_gradient[i] = np.mean(1.0 - np.maximum(img_norm[nb] @ img_norm[i], 0.0))

        def _minmax(values):
            values = np.asarray(values, dtype=np.float64)
            lo = np.nanmin(values)
            hi = np.nanmax(values)
            if hi - lo < 1e-12:
                return np.zeros_like(values)
            return (values - lo) / (hi - lo)

        boundary_evidence = (
            _minmax(boundary_score)
            + _minmax(rna_gradient)
            + _minmax(morph_gradient)
        ) / 3.0
        if compartment_aware:
            mixed_evidence = _minmax(compartment_scores["mixed_compartment"])
            boundary_evidence = (
                _minmax(boundary_score)
                + _minmax(rna_gradient)
                + _minmax(morph_gradient)
                + mixed_evidence
            ) / 4.0

        boundary_cut = np.quantile(boundary_score, boundary_quantile)
        confidence_cut = np.quantile(anchor_confidence, low_confidence_quantile)
        candidates = np.where((boundary_score >= boundary_cut) & (anchor_confidence <= confidence_cut))[0]

        changed = np.zeros(n_spots, dtype=bool)
        new_y = y.copy()
        best_gain = np.zeros(n_spots, dtype=np.float64)
        best_reason = np.array(["kept"] * n_spots, dtype=object)
        proposed = []

        for i in candidates:
            nb = neighbor_idx[i]
            if multiscale:
                spatial_support = np.zeros(n_labels, dtype=np.float64)
                scale_defs = [
                    (spatial_k, 0.50),
                    (min(max_k, spatial_k * 2), 0.30),
                    (min(max_k, spatial_k * 4), 0.20),
                ]
                for scale_k, scale_weight in scale_defs:
                    scale_nb = sorted_neighbors[i, :scale_k]
                    support = np.bincount(y[scale_nb], minlength=n_labels).astype(np.float64)
                    support /= (support.sum() + 1e-9)
                    spatial_support += scale_weight * support
            else:
                spatial_support = np.bincount(y[nb], minlength=n_labels).astype(np.float64)
                spatial_support /= (spatial_support.sum() + 1e-9)

            uni_sim = np.maximum(img_norm[nb] @ img_norm[i], 0.0)
            rna_sim = np.maximum(z_norm[nb] @ z_norm[i], 0.0)
            morph_rna_weight = uni_sim * rna_sim
            uni_support = np.zeros(n_labels, dtype=np.float64)
            for local_j, nb_j in enumerate(nb):
                uni_support[y[nb_j]] += morph_rna_weight[local_j]
            uni_support /= (uni_support.sum() + 1e-9)

            anchor_score = -anchor_dist[i]
            anchor_score = (anchor_score - anchor_score.mean()) / (anchor_score.std() + 1e-9)
            compartment_support = np.zeros(n_labels, dtype=np.float64)
            if compartment_aware and compartment_norm is not None and compartment_centroids is not None:
                compartment_support = np.maximum(compartment_centroids @ compartment_norm[i], 0.0)

            boundary_weight = 1.0 + boundary_loss_weight * boundary_evidence[i]
            total_score = (
                lambda_anchor * anchor_score
                + boundary_weight * (
                    lambda_spatial * spatial_support
                    + lambda_uni * uni_support
                    + lambda_compartment * compartment_support
                )
            )
            current = y[i]
            proposal = int(np.argmax(total_score))
            gain = float(total_score[proposal] - total_score[current])
            mixed_penalty = 0.50 * compartment_scores["mixed_compartment"][i] if compartment_aware else 0.0
            local_margin = margin * (1.0 + 0.50 * boundary_evidence[i] + mixed_penalty)
            if proposal != current and gain > local_margin:
                proposed.append((gain, i, proposal))

        proposed.sort(reverse=True, key=lambda x: x[0])
        for gain, i, proposal in proposed[:max_changes]:
            old_label = label_values[y[i]]
            new_label = label_values[proposal]
            new_y[i] = proposal
            changed[i] = True
            best_gain[i] = gain
            best_reason[i] = f"boundary_low_confidence:{old_label}->{new_label}"

        refined_labels = label_values[new_y]
        adata.obs[label_key] = pd.Categorical(refined_labels)
        adata.obs["cbr_changed"] = changed
        adata.obs["cbr_boundary_score"] = boundary_score
        adata.obs["cbr_boundary_evidence"] = boundary_evidence
        adata.obs["cbr_rna_gradient"] = rna_gradient
        adata.obs["cbr_morphology_gradient"] = morph_gradient
        adata.obs["cbr_anchor_confidence"] = anchor_confidence
        adata.obs["cbr_gain"] = best_gain
        adata.obs["cbr_reason"] = best_reason
        if not compartment_aware and "cbr_compartment_signatures" not in adata.uns:
            adata.uns["cbr_compartment_signatures"] = compartment_summary
        adata.uns["cancer_boundary_refinement"] = {
            "enabled": True,
            "label_key": label_key,
            "anchor_label_key": f"{label_key}_anchor",
            "embedding_key": embedding_key,
            "image_key": image_key,
            "spatial_k": spatial_k,
            "multiscale": bool(multiscale),
            "max_change_ratio": max_change_ratio,
            "max_changes_allowed": max_changes,
            "n_candidates": int(len(candidates)),
            "n_changed": int(changed.sum()),
            "boundary_quantile": boundary_quantile,
            "low_confidence_quantile": low_confidence_quantile,
            "margin": margin,
            "lambda_spatial": lambda_spatial,
            "lambda_uni": lambda_uni,
            "lambda_anchor": lambda_anchor,
            "boundary_loss_weight": boundary_loss_weight,
            "compartment_aware": bool(compartment_aware),
            "lambda_compartment": lambda_compartment,
        }
        print(
            "CBR completed: "
            f"{int(changed.sum())}/{n_spots} boundary spots refined "
            f"(candidates={len(candidates)}, max_allowed={max_changes})."
        )
        return adata


def run_image():
    """
    Part 1: load spatial data and extract UNI image features.

    Outputs:
    - {sample}_image_features_ready.h5ad
    - optional image feature .npy files
    """

    import os
    import sys

    import anndata
    import numpy as np

    from .utils import seed_torch

    SEED = int(os.environ.get("SEED", "0"))
    DATA_DIR = os.environ.get("DATA_DIR", "/root/autodl-tmp/HANYIYU/DLPFC")
    SAMPLE_ID = os.environ.get("SAMPLE_ID", "151673")
    RESULTS_DIR = os.environ.get("RESULTS_DIR", "../Results")
    DATASET_PROFILE = os.environ.get("DATASET_PROFILE", "auto")

    REUSE_IMAGE_FEATURES = os.environ.get("REUSE_IMAGE_FEATURES", "0").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    SAVE_IMAGE_NPY = os.environ.get("SAVE_IMAGE_NPY", "0").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    USE_DIVERSITY_PRUNE = os.environ.get("USE_DIVERSITY_PRUNE", "1").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    DIVPRUNE_KEEP_RATIO = float(os.environ.get("DIVPRUNE_KEEP_RATIO", "0.75"))
    _DIVPRUNE_KEEP_COUNT = os.environ.get("DIVPRUNE_KEEP_COUNT", "").strip()
    DIVPRUNE_KEEP_COUNT = int(_DIVPRUNE_KEEP_COUNT) if _DIVPRUNE_KEEP_COUNT else None
    DIVPRUNE_START = os.environ.get("DIVPRUNE_START", "centroid")

    def _ensure_spatial_uns(adata):
        if "spatial" in adata.uns:
            return adata
        if "spatial" not in adata.obsm:
            raise RuntimeError("adata.obsm['spatial'] is missing; cannot extract image features.")
        adata.uns["spatial"] = {
            "V1_Adult_Mouse_Brain": {
                "images": {"hires": None, "lowres": None},
                "scalefactors": {
                    "tissue_hires_scalef": 1.0,
                    "spot_diameter_fullres": 100.0,
                },
            }
        }
        print("Created minimal adata.uns['spatial'] metadata from obsm['spatial'].")
        return adata

    def feature_ready_path(sample_dir):
        return os.path.join(sample_dir, f"{SAMPLE_ID}_image_features_ready.h5ad")

    def load_reusable_features(sample_dir):
        feature_ready_file = feature_ready_path(sample_dir)
        if not os.path.exists(feature_ready_file):
            return None
        adata = anndata.read_h5ad(feature_ready_file)
        if "image_feat_pca" not in adata.obsm:
            raise RuntimeError("Reusable feature file does not contain adata.obsm['image_feat_pca'].")
        if USE_DIVERSITY_PRUNE and "image_feat_divprune" not in adata.obsm:
            print("[FEATURE_DEBUG] reusable_feature_missing_divprune=True")
            return None
        print(f"Loaded existing UNI feature file: {feature_ready_file}")
        return adata

    def extract_uni_features(gemcst):
        print("\nPart 1 Step 1: Loading spatial data...")
        adata = gemcst._get_adata(
            platform="Visium",
            data_path=DATA_DIR,
            data_name=SAMPLE_ID,
        )
        print(f"Loaded raw spatial data: {adata.shape}")

        print("\nPart 1 Step 2: Extracting UNI2-h image features...")
        adata = _ensure_spatial_uns(adata)
        adata = gemcst._get_image_crop(
            adata,
            data_name=SAMPLE_ID,
            cnn_type="UNI2-h",
            use_diversity_prune=USE_DIVERSITY_PRUNE,
            divprune_keep_ratio=DIVPRUNE_KEEP_RATIO,
            divprune_keep_count=DIVPRUNE_KEEP_COUNT,
            divprune_start=DIVPRUNE_START,
        )
        if "image_feat_pca" not in adata.obsm:
            raise RuntimeError("UNI feature extraction finished but image_feat_pca is missing.")
        print(f"UNI image features ready: image_feat_pca={adata.obsm['image_feat_pca'].shape}")
        return adata

    def save_feature_outputs(adata, sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
        feature_ready_file = feature_ready_path(sample_dir)
        adata.write(feature_ready_file)
        print(f"Saved feature-ready h5ad: {feature_ready_file}")

        if SAVE_IMAGE_NPY and "image_feat" in adata.obsm:
            image_feat_file = os.path.join(sample_dir, f"{SAMPLE_ID}_image_features.npy")
            np.save(image_feat_file, adata.obsm["image_feat"])
            print(f"Saved raw image feature npy: {image_feat_file}")
        if SAVE_IMAGE_NPY and "image_feat_pca" in adata.obsm:
            image_pca_file = os.path.join(sample_dir, f"{SAMPLE_ID}_image_pca.npy")
            np.save(image_pca_file, adata.obsm["image_feat_pca"])
            print(f"Saved image PCA npy: {image_pca_file}")

    def main():
        print("Starting image preprocessing part 1: data + UNI features")
        print(f"DATA_DIR={DATA_DIR}")
        print(f"SAMPLE_ID={SAMPLE_ID}")
        print(f"RESULTS_DIR={RESULTS_DIR}")
        print(f"DATASET_PROFILE={DATASET_PROFILE}")
        print(f"REUSE_IMAGE_FEATURES={REUSE_IMAGE_FEATURES}")
        print(f"USE_DIVERSITY_PRUNE={USE_DIVERSITY_PRUNE}")
        print(f"DIVPRUNE_KEEP_RATIO={DIVPRUNE_KEEP_RATIO}")
        print(f"DIVPRUNE_KEEP_COUNT={DIVPRUNE_KEEP_COUNT}")
        print(f"DIVPRUNE_START={DIVPRUNE_START}")

        seed_torch(seed=SEED)
        sample_dir = os.path.join(RESULTS_DIR, SAMPLE_ID)

        adata = load_reusable_features(sample_dir) if REUSE_IMAGE_FEATURES else None
        if adata is None:
            gemcst = GeMCSTRunner(
                save_path=RESULTS_DIR,
                task="Identify_Domain",
                pre_epochs=500,
                epochs=500,
                use_gpu=True,
            )
            adata = extract_uni_features(gemcst)
            save_feature_outputs(adata, sample_dir)
        print("\nImage features ready. Continue with: python pipeline.py image ...")

    main()


def run_graph():
    """
    Part 2: build PCA features and tri-view graphs from extracted UNI features.

    Outputs:
    - {sample}_preprocessed.h5ad
    - {sample}_triview_expression_adj.npz
    - {sample}_triview_spatial_adj.npz
    - {sample}_triview_edge_adj.npz
    - {sample}_triview_config.json
    """

    import os
    import sys
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import json

    import anndata
    import numpy as np
    import scipy.sparse as sp

    from .utils import seed_torch

    SEED = int(os.environ.get("SEED", "0"))
    DATA_DIR = os.environ.get("DATA_DIR", "/root/autodl-tmp/HANYIYU/DLPFC")
    SAMPLE_ID = os.environ.get("SAMPLE_ID", "151673")
    RESULTS_DIR = os.environ.get("RESULTS_DIR", "../Results")
    DATASET_PROFILE = os.environ.get("DATASET_PROFILE", "auto")

    def _resolve_dataset_profile(data_dir, sample_id, profile):
        profile = (profile or "auto").strip().lower()
        if profile != "auto":
            return profile
        combined = f"{str(data_dir).lower()} {str(sample_id).lower()}"
        if any(token in combined for token in ("colon", "colorectal", "crc", "016um", "032um")):
            return "colon_cancer_visium_hd"
        if "her2" in combined or str(sample_id).strip().lower() == "h1":
            return "her2_breast"
        return "auto"

    _GRAPH_PROFILE_DEFAULTS = {
        "her2_breast": {
            "pca_n_comps": 100,
            "expression_graph_k": 12,
            "spatial_graph_k": 8,
            "edge_graph_k": 8,
            "image_graph_pca_n": 50,
        },
        "v1_breast_cancer_block_a_section_1": {
            "pca_n_comps": 200,
            "expression_graph_k": 12,
            "spatial_graph_k": 6,
            "edge_graph_k": 6,
            "image_graph_pca_n": 50,
        },
        "colon_cancer_visium_hd": {
            "pca_n_comps": 100,
            "expression_graph_k": 12,
            "spatial_graph_k": 8,
            "edge_graph_k": 8,
            "image_graph_pca_n": 50,
        },
    }
    DATASET_PROFILE = _resolve_dataset_profile(DATA_DIR, SAMPLE_ID, DATASET_PROFILE)
    _graph_defaults = _GRAPH_PROFILE_DEFAULTS.get(
        DATASET_PROFILE,
        {
            "pca_n_comps": 200,
            "expression_graph_k": 12,
            "spatial_graph_k": 6,
            "edge_graph_k": 6,
            "image_graph_pca_n": 50,
        },
    )

    PCA_N_COMPS = int(os.environ.get("PCA_N_COMPS", _graph_defaults["pca_n_comps"]))
    EXPRESSION_GRAPH_K = int(os.environ.get("EXPRESSION_GRAPH_K", _graph_defaults["expression_graph_k"]))
    SPATIAL_GRAPH_K = int(os.environ.get("SPATIAL_GRAPH_K", _graph_defaults["spatial_graph_k"]))
    EDGE_GRAPH_K = int(os.environ.get("EDGE_GRAPH_K", _graph_defaults["edge_graph_k"]))
    IMAGE_GRAPH_PCA_N = int(os.environ.get("IMAGE_GRAPH_PCA_N", _graph_defaults["image_graph_pca_n"]))
    IMAGE_GRAPH_MODE = os.environ.get("IMAGE_GRAPH_MODE", "similarity").strip().lower()
    IMAGE_GRAPH_FEATURE_MODE = os.environ.get("IMAGE_GRAPH_FEATURE_MODE", "divprune").strip().lower()
    LOCAL_SUBGRAPH_ENABLED = os.environ.get("LOCAL_SUBGRAPH_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
    LOCAL_EXPR_KEEP_RATIO = float(os.environ.get("LOCAL_EXPR_KEEP_RATIO", "1.0"))
    LOCAL_SPATIAL_KEEP_RATIO = float(os.environ.get("LOCAL_SPATIAL_KEEP_RATIO", "1.0"))
    LOCAL_EDGE_KEEP_RATIO = float(os.environ.get("LOCAL_EDGE_KEEP_RATIO", "1.0"))
    LATGRL_ENABLED = os.environ.get("LATGRL_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
    LATGRL_HOMO_RATIO = float(os.environ.get("LATGRL_HOMO_RATIO", "0.35"))
    LATGRL_HETERO_RATIO = float(os.environ.get("LATGRL_HETERO_RATIO", "0.20"))

    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _feature_ready_path(sample_dir):
        return os.path.join(sample_dir, f"{SAMPLE_ID}_image_features_ready.h5ad")

    def _preprocessed_path(sample_dir):
        return os.path.join(sample_dir, f"{SAMPLE_ID}_preprocessed.h5ad")

    def _triview_paths(sample_dir):
        return {
            "expression": os.path.join(sample_dir, f"{SAMPLE_ID}_triview_expression_adj.npz"),
            "spatial": os.path.join(sample_dir, f"{SAMPLE_ID}_triview_spatial_adj.npz"),
            "edge": os.path.join(sample_dir, f"{SAMPLE_ID}_triview_edge_adj.npz"),
            "expression_local": os.path.join(sample_dir, f"{SAMPLE_ID}_triview_expression_local_adj.npz"),
            "spatial_local": os.path.join(sample_dir, f"{SAMPLE_ID}_triview_spatial_local_adj.npz"),
            "edge_local": os.path.join(sample_dir, f"{SAMPLE_ID}_triview_edge_local_adj.npz"),
            "expression_homo": os.path.join(sample_dir, f"{SAMPLE_ID}_triview_expression_homo_adj.npz"),
            "spatial_homo": os.path.join(sample_dir, f"{SAMPLE_ID}_triview_spatial_homo_adj.npz"),
            "edge_homo": os.path.join(sample_dir, f"{SAMPLE_ID}_triview_edge_homo_adj.npz"),
            "expression_hetero": os.path.join(sample_dir, f"{SAMPLE_ID}_triview_expression_hetero_adj.npz"),
            "spatial_hetero": os.path.join(sample_dir, f"{SAMPLE_ID}_triview_spatial_hetero_adj.npz"),
            "edge_hetero": os.path.join(sample_dir, f"{SAMPLE_ID}_triview_edge_hetero_adj.npz"),
            "config": os.path.join(sample_dir, f"{SAMPLE_ID}_triview_config.json"),
        }

    def _resolve_image_similarity_features(adata):
        if IMAGE_GRAPH_FEATURE_MODE in {"uni_divprune_fused", "fused", "uni+divprune"}:
            if "image_feat" in adata.obsm and "image_feat_divprune" in adata.obsm:
                raw = np.asarray(adata.obsm["image_feat"], dtype=np.float32)
                pruned = np.asarray(adata.obsm["image_feat_divprune"], dtype=np.float32)
                raw = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
                pruned = pruned / np.maximum(np.linalg.norm(pruned, axis=1, keepdims=True), 1e-12)
                return np.concatenate([raw, pruned], axis=1), "image_feat+image_feat_divprune"
            raise RuntimeError("IMAGE_GRAPH_FEATURE_MODE=uni_divprune_fused requires image_feat and image_feat_divprune.")
        if IMAGE_GRAPH_FEATURE_MODE in {"divprune", "image_feat_divprune"} and "image_feat_divprune" in adata.obsm:
            return np.asarray(adata.obsm["image_feat_divprune"], dtype=np.float32), "image_feat_divprune"
        if IMAGE_GRAPH_FEATURE_MODE in {"divprune", "image_feat_divprune"}:
            raise RuntimeError("IMAGE_GRAPH_FEATURE_MODE=divprune requires adata.obsm['image_feat_divprune']. Run the image stage with Diversity Prune enabled.")
        if IMAGE_GRAPH_FEATURE_MODE in {"raw", "uni", "image_feat"} and "image_feat" in adata.obsm:
            return np.asarray(adata.obsm["image_feat"], dtype=np.float32), "image_feat"
        if IMAGE_GRAPH_FEATURE_MODE in {"pca", "image_feat_pca"} and "image_feat_pca" in adata.obsm:
            return np.asarray(adata.obsm["image_feat_pca"], dtype=np.float32), "image_feat_pca"
        raise RuntimeError(f"No UNI features found for IMAGE_GRAPH_FEATURE_MODE={IMAGE_GRAPH_FEATURE_MODE}.")

    def _runtime_signature():
        return {
            "rna_preprocess": "all_genes",
            "image_graph_mode": IMAGE_GRAPH_MODE,
            "image_graph_feature_mode": IMAGE_GRAPH_FEATURE_MODE,
            "edge_view_input": "uni_divprune_image_features",
            "local_subgraph_enabled": LOCAL_SUBGRAPH_ENABLED,
            "local_expr_keep_ratio": LOCAL_EXPR_KEEP_RATIO,
            "local_spatial_keep_ratio": LOCAL_SPATIAL_KEEP_RATIO,
            "local_edge_keep_ratio": LOCAL_EDGE_KEEP_RATIO,
            "latgrl_enabled": LATGRL_ENABLED,
            "latgrl_homo_ratio": LATGRL_HOMO_RATIO,
            "latgrl_hetero_ratio": LATGRL_HETERO_RATIO,
        }

    def _save_triview_graphs(graph_dict, sample_dir):
        paths = _triview_paths(sample_dir)
        for view_name in ("expression", "spatial", "edge"):
            view = graph_dict["views"][view_name]
            sp.save_npz(paths[view_name], view["adj_sp"].tocsr())
            print(f"Saved {view_name} adjacency: {paths[view_name]}")
            optional_specs = (
                ("local_adj_sp", f"{view_name}_local", "local"),
                ("homo_adj_sp", f"{view_name}_homo", "homophilic"),
                ("hetero_adj_sp", f"{view_name}_hetero", "heterophilic"),
            )
            for graph_key, path_key, label in optional_specs:
                if graph_key in view:
                    sp.save_npz(paths[path_key], view[graph_key].tocsr())
                    print(f"Saved {view_name} {label} adjacency: {paths[path_key]}")
        return paths

    def main():
        print("Starting image preprocessing part 2: PCA + tri-view graph generation")
        print(f"DATA_DIR={DATA_DIR}")
        print(f"SAMPLE_ID={SAMPLE_ID}")
        print(f"RESULTS_DIR={RESULTS_DIR}")
        print(f"DATASET_PROFILE={DATASET_PROFILE}")
        print(f"PCA_N_COMPS={PCA_N_COMPS}")
        print(f"IMAGE_GRAPH_MODE={IMAGE_GRAPH_MODE}")
        print(f"IMAGE_GRAPH_FEATURE_MODE={IMAGE_GRAPH_FEATURE_MODE}")
        print(
            f"LOCAL_SUBGRAPH={LOCAL_SUBGRAPH_ENABLED} "
            f"(expr={LOCAL_EXPR_KEEP_RATIO}, spatial={LOCAL_SPATIAL_KEEP_RATIO}, edge={LOCAL_EDGE_KEEP_RATIO})"
        )
        print(
            f"LATGRL={LATGRL_ENABLED} "
            f"(homo_ratio={LATGRL_HOMO_RATIO}, hetero_ratio={LATGRL_HETERO_RATIO})"
        )
        print("Tri-view graph construction only; GNN aggregation is selected during model training.")
        print("Tri-view graphs will be used for graph-only tri-view training.")

        seed_torch(seed=SEED)
        sample_dir = os.path.join(RESULTS_DIR, SAMPLE_ID)
        feature_ready_file = _feature_ready_path(sample_dir)
        if not os.path.exists(feature_ready_file):
            raise RuntimeError(f"Missing feature-ready file: {feature_ready_file}. Run the image stage first.")

        adata = anndata.read_h5ad(feature_ready_file)
        if "image_feat_pca" not in adata.obsm:
            raise RuntimeError("Feature-ready AnnData is missing adata.obsm['image_feat_pca'].")
        image_features_for_graph, image_feature_source = _resolve_image_similarity_features(adata)
        print(f"IMAGE_GRAPH_FEATURE_SOURCE={image_feature_source}")

        gemcst = GeMCSTRunner(
            save_path=RESULTS_DIR,
            task="Identify_Domain",
            pre_epochs=500,
            epochs=500,
            use_gpu=True,
        )

        print("Running PCA dimensionality reduction...")
        data = gemcst._data_process(adata, pca_n_comps=PCA_N_COMPS)
        adata.obsm["GeMCST_pca_input"] = np.asarray(data, dtype=np.float32)

        print("Building tri-view graphs...")
        graph_dict = gemcst._get_triview_graph(
            expression_data=data,
            spatial_coords=adata.obsm["spatial"],
            image_features=image_features_for_graph,
            expression_k=EXPRESSION_GRAPH_K,
            spatial_k=SPATIAL_GRAPH_K,
            edge_spatial_k=EDGE_GRAPH_K,
            image_pca_n=IMAGE_GRAPH_PCA_N,
            image_graph_mode=IMAGE_GRAPH_MODE,
            local_subgraph_enabled=LOCAL_SUBGRAPH_ENABLED,
            local_expr_keep_ratio=LOCAL_EXPR_KEEP_RATIO,
            local_spatial_keep_ratio=LOCAL_SPATIAL_KEEP_RATIO,
            local_edge_keep_ratio=LOCAL_EDGE_KEEP_RATIO,
            latgrl_enabled=LATGRL_ENABLED,
            latgrl_homo_ratio=LATGRL_HOMO_RATIO,
            latgrl_hetero_ratio=LATGRL_HETERO_RATIO,
        )
        adata.obsm["edge_view_data"] = np.asarray(graph_dict["edge_view_data"], dtype=np.float32)

        paths = _save_triview_graphs(graph_dict, sample_dir)
        config = {
            "sample_id": SAMPLE_ID,
            "dataset_profile": DATASET_PROFILE,
            "data_dir": DATA_DIR,
            "data_shape": list(adata.shape),
            "pca_n_comps": PCA_N_COMPS,
            "expression_graph_k": EXPRESSION_GRAPH_K,
            "spatial_graph_k": SPATIAL_GRAPH_K,
            "edge_graph_k": EDGE_GRAPH_K,
            "image_graph_pca_n": IMAGE_GRAPH_PCA_N,
            "image_graph_mode": IMAGE_GRAPH_MODE,
            "image_graph_feature_source": image_feature_source,
            "image_graph_feature_mode": IMAGE_GRAPH_FEATURE_MODE,
            "edge_view_input": "uni_divprune_image_features",
            "edge_view_dim": int(adata.obsm["edge_view_data"].shape[1]),
            "runtime_signature": _runtime_signature(),
            "tri_view_params": graph_dict.get("tri_view_params", {}),
            "divprune": adata.uns.get("divprune", {"enabled": False}),
            "pipeline": "run_image_part2_triview",
        }

        with open(paths["config"], "w", encoding="utf-8") as f:
            json.dump(_json_safe(config), f, indent=2, ensure_ascii=False)
        print(f"Saved tri-view config: {paths['config']}")

        config_file = os.path.join(sample_dir, f"{SAMPLE_ID}_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(_json_safe(config), f, indent=2, ensure_ascii=False)
        print(f"Saved main config: {config_file}")

        preprocessed_file = _preprocessed_path(sample_dir)
        adata.uns["tri_view_precomputed"] = {
            "expression_adj": paths["expression"],
            "spatial_adj": paths["spatial"],
            "edge_adj": paths["edge"],
            "expression_local_adj": paths["expression_local"],
            "spatial_local_adj": paths["spatial_local"],
            "edge_local_adj": paths["edge_local"],
            "expression_homo_adj": paths["expression_homo"],
            "spatial_homo_adj": paths["spatial_homo"],
            "edge_homo_adj": paths["edge_homo"],
            "expression_hetero_adj": paths["expression_hetero"],
            "spatial_hetero_adj": paths["spatial_hetero"],
            "edge_hetero_adj": paths["edge_hetero"],
            "config": paths["config"],
        }
        adata.write(preprocessed_file)
        print(f"Saved preprocessed h5ad: {preprocessed_file}")
        print("\nGraphs ready. Continue with: python pipeline.py train ...")

    main()


def run_train():
    import os
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    from matplotlib import pyplot as plt
    import scanpy as sc
    import torch
    import pandas as pd  
    import numpy as np   
    import anndata    
    import sys
    import json
    import re
    import scipy.sparse as sp
    from scipy.spatial import cKDTree

    from .utils import seed_torch

    from .graph_function import graph_dict_from_sparse
    from .utils import fit_posthoc_expression_decoder
    from .utils import palette_for_dataset

    SEED = int(os.environ.get("SEED", "0"))                                   
    DATA_DIR = os.environ.get("DATA_DIR", "/root/autodl-tmp/HANYIYU/DLPFC")
    SAMPLE_ID = os.environ.get("SAMPLE_ID", "151673")
    RESULTS_DIR = os.environ.get("RESULTS_DIR", "../Results")
    DATASET_PROFILE = os.environ.get("DATASET_PROFILE", "auto")

    def _as_bool(value, default):
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    SAVE_FULL_H5AD = _as_bool(os.environ.get("SAVE_FULL_H5AD"), True)
    SAVE_EMBEDDING_NPY = _as_bool(os.environ.get("SAVE_EMBEDDING_NPY"), True)
    USE_PRECOMPUTED_TRIVIEW = _as_bool(os.environ.get("USE_PRECOMPUTED_TRIVIEW"), True)
    EXPRESSION_DECODER_ENABLED = _as_bool(os.environ.get("EXPRESSION_DECODER_ENABLED"), True)
    SAVE_ENHANCED_EXPRESSION_CSV = _as_bool(os.environ.get("SAVE_ENHANCED_EXPRESSION_CSV"), False)
    EXPRESSION_DECODER_ALPHA = float(os.environ.get("EXPRESSION_DECODER_ALPHA", "1.0"))

    DOMAIN_PALETTE = palette_for_dataset(DATASET_PROFILE, SAMPLE_ID, DATA_DIR)

    def _get_spatial_library(adata):
        spatial = adata.uns.get("spatial", None)
        if not isinstance(spatial, dict) or len(spatial) == 0:
            return None
        key = next(iter(spatial.keys()))
        return spatial.get(key, None)

    def _get_hires_image_and_scale(adata, coords):
        library = _get_spatial_library(adata)
        if not isinstance(library, dict):
            return None, 1.0
        images = library.get("images", {})
        image = images.get("hires", None)
        if image is None:
            image = images.get("lowres", None)
        if image is None:
            return None, 1.0
        scalefactors = library.get("scalefactors", {})
        scale = float(scalefactors.get("tissue_hires_scalef", 1.0))
        image = np.asarray(image)
        if image.ndim < 2:
            return None, 1.0
        h, w = image.shape[:2]
        max_x = float(np.nanmax(coords[:, 0]))
        max_y = float(np.nanmax(coords[:, 1]))
        if max_x <= w * 1.2 and max_y <= h * 1.2:
            scale = 1.0
        return image, scale

    def _median_nearest_neighbor_spacing(coords):
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] < 2:
            raise ValueError("At least two spatial coordinates are required")
        distances, _ = cKDTree(coords[:, :2]).query(coords[:, :2], k=2)
        nearest = distances[:, 1]
        valid = nearest[np.isfinite(nearest) & (nearest > 0)]
        if valid.size == 0:
            raise ValueError("Could not estimate a positive spot center spacing")
        return float(np.median(valid))

    def _spot_diameter(coords, fraction=0.65):
        return float(fraction) * _median_nearest_neighbor_spacing(coords)

    def _spot_fraction(profile=None, sample_id=None):
        profile_key = str(profile or "").strip().lower()
        sample_key = str(sample_id or "").strip().lower()
        if profile_key == "colon_cancer_visium_hd" or "colon_cancer_p1_032um" in sample_key:
            return 0.75
        return 0.65

    def _spot_alpha(profile=None, sample_id=None, with_image=True):
        profile_key = str(profile or "").strip().lower()
        sample_key = str(sample_id or "").strip().lower()
        if profile_key == "colon_cancer_visium_hd" or "colon_cancer_p1_032um" in sample_key:
            return 0.98
        if profile_key == "human_lung_cancer" or "human_lung_cancer" in sample_key:
            return 0.75
        return 0.68 if with_image else 0.74

    def _image_alpha(profile=None, sample_id=None, default=0.55):
        profile_key = str(profile or "").strip().lower()
        sample_key = str(sample_id or "").strip().lower()
        if profile_key == "colon_cancer_visium_hd" or "colon_cancer_p1_032um" in sample_key:
            return 0.45
        return float(default)

    def _domain_sort_key(value):
        text = str(value)
        try:
            return (0, float(text), text)
        except ValueError:
            parts = tuple(
                (0, int(part)) if part.isdigit() else (1, part.lower())
                for part in re.split(r"(\d+)", text)
            )
            return (1, parts, text)

    def _domain_categories(values):
        return sorted({str(value) for value in values}, key=_domain_sort_key)

    def _marker_area_from_data_diameter(ax, diameter):
        origin = ax.transData.transform((0.0, 0.0))
        x_extent = ax.transData.transform((float(diameter), 0.0))
        y_extent = ax.transData.transform((0.0, float(diameter)))
        diameter_px = min(abs(x_extent[0] - origin[0]), abs(y_extent[1] - origin[1]))
        diameter_points = max(diameter_px * 72.0 / ax.figure.dpi, 0.5)
        return diameter_points ** 2

    def _save_figure_pair(fig, output_prefix):
        pdf_file = f"{output_prefix}.pdf"
        png_file = f"{output_prefix}.png"
        fig.savefig(pdf_file, bbox_inches="tight", dpi=300)
        fig.savefig(png_file, bbox_inches="tight", dpi=300)
        return pdf_file, png_file

    def plot_spatial_domains(adata, color_key, out_prefix, title=None, image_alpha=0.55, profile=None):
        if "spatial" not in adata.obsm:
            raise ValueError("Missing adata.obsm['spatial']")
        if color_key not in adata.obs:
            raise ValueError(f"Missing adata.obs['{color_key}']")

        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        labels = adata.obs[color_key].astype(str)
        cats = _domain_categories(labels)
        codes = pd.Categorical(labels, categories=cats, ordered=True).codes

        image, scale = _get_hires_image_and_scale(adata, coords)
        plot_coords = coords.copy()
        plot_coords[:, 0] *= scale
        plot_coords[:, 1] *= scale

        with_image = image is not None
        spot_diameter = _spot_diameter(
            plot_coords,
            fraction=_spot_fraction(profile=profile, sample_id=SAMPLE_ID),
        )
        alpha = _spot_alpha(profile=profile, sample_id=SAMPLE_ID, with_image=with_image)
        palette = palette_for_dataset(profile or DATASET_PROFILE, SAMPLE_ID, DATA_DIR)
        colors = [palette[i % len(palette)] for i in codes]

        if with_image:
            h, w = image.shape[:2]
            fig_w = 7.0
            fig_h = max(5.5, fig_w * h / max(w, 1))
        else:
            fig_w, fig_h = 7.0, 6.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)

        if with_image:
            ax.imshow(image, alpha=_image_alpha(profile, SAMPLE_ID, image_alpha))
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)
        else:
            ax.set_xlim(float(np.nanmin(plot_coords[:, 0])), float(np.nanmax(plot_coords[:, 0])))
            ax.set_ylim(float(np.nanmax(plot_coords[:, 1])), float(np.nanmin(plot_coords[:, 1])))
        ax.set_aspect("equal")
        fig.canvas.draw()
        spot_size = _marker_area_from_data_diameter(ax, spot_diameter)
        ax.scatter(
            plot_coords[:, 0],
            plot_coords[:, 1],
            c=colors,
            s=spot_size,
            linewidths=0.0,
            edgecolors="none",
            alpha=alpha,
        )
        ax.axis("off")
        ax.set_title(title or color_key, fontsize=13, pad=10)

        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=palette[i % len(palette)],
                markeredgecolor="none", markersize=7, label=str(cat),
            )
            for i, cat in enumerate(cats)
        ]
        ax.legend(
            handles=handles,
            title=color_key,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            ncol=1 if len(cats) <= 10 else 2,
        )

        pdf_file, png_file = _save_figure_pair(fig, out_prefix)
        plt.close(fig)
        return pdf_file, png_file

    def _sample_path(data_dir, sample_id):
        return os.path.join(data_dir, sample_id)

    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _precomputed_triview_paths(sample_dir, sample_id):
        return {
            "expression": os.path.join(sample_dir, f"{sample_id}_triview_expression_adj.npz"),
            "spatial": os.path.join(sample_dir, f"{sample_id}_triview_spatial_adj.npz"),
            "edge": os.path.join(sample_dir, f"{sample_id}_triview_edge_adj.npz"),
            "expression_local": os.path.join(sample_dir, f"{sample_id}_triview_expression_local_adj.npz"),
            "spatial_local": os.path.join(sample_dir, f"{sample_id}_triview_spatial_local_adj.npz"),
            "edge_local": os.path.join(sample_dir, f"{sample_id}_triview_edge_local_adj.npz"),
            "expression_homo": os.path.join(sample_dir, f"{sample_id}_triview_expression_homo_adj.npz"),
            "spatial_homo": os.path.join(sample_dir, f"{sample_id}_triview_spatial_homo_adj.npz"),
            "edge_homo": os.path.join(sample_dir, f"{sample_id}_triview_edge_homo_adj.npz"),
            "expression_hetero": os.path.join(sample_dir, f"{sample_id}_triview_expression_hetero_adj.npz"),
            "spatial_hetero": os.path.join(sample_dir, f"{sample_id}_triview_spatial_hetero_adj.npz"),
            "edge_hetero": os.path.join(sample_dir, f"{sample_id}_triview_edge_hetero_adj.npz"),
            "config": os.path.join(sample_dir, f"{sample_id}_triview_config.json"),
        }

    def _attach_optional_precomputed_graph(graph_data, paths, prefix):
        optional_names = ("local", "homo", "hetero")
        for optional_name in optional_names:
            path = paths.get(f"{prefix}_{optional_name}")
            if path and os.path.exists(path):
                loaded = graph_dict_from_sparse(sp.load_npz(path))
                graph_data[f"{optional_name}_adj_sp"] = loaded["adj_sp"]
                graph_data[f"{optional_name}_adj_norm"] = loaded["adj_norm"]
        return graph_data

    def _load_precomputed_triview(sample_dir, sample_id, adata):
        paths = _precomputed_triview_paths(sample_dir, sample_id)
        required = [paths["expression"], paths["spatial"], paths["edge"]]
        if not all(os.path.exists(path) for path in required):
            return None
        if "edge_view_data" not in adata.obsm:
            return None

        expr_graph = _attach_optional_precomputed_graph(
            graph_dict_from_sparse(sp.load_npz(paths["expression"])), paths, "expression"
        )
        spatial_graph = _attach_optional_precomputed_graph(
            graph_dict_from_sparse(sp.load_npz(paths["spatial"])), paths, "spatial"
        )
        edge_graph = _attach_optional_precomputed_graph(
            graph_dict_from_sparse(sp.load_npz(paths["edge"])), paths, "edge"
        )
        union_label = torch.clamp(
            expr_graph["adj_label"] + spatial_graph["adj_label"] + edge_graph["adj_label"],
            max=1.0,
        )
        norm = float(np.mean([
            expr_graph["norm_value"],
            spatial_graph["norm_value"],
            edge_graph["norm_value"],
        ]))
        graph_dict = {
            "views": {
                "expression": expr_graph,
                "spatial": spatial_graph,
                "edge": edge_graph,
            },
            "adj_norm": spatial_graph["adj_norm"],
            "adj_label": union_label,
            "norm_value": norm,
            "edge_view_data": np.asarray(adata.obsm["edge_view_data"], dtype=np.float32),
        }
        if os.path.exists(paths["config"]):
            with open(paths["config"], "r", encoding="utf-8") as f:
                graph_dict["tri_view_params"] = json.load(f)
        return graph_dict

    def _runtime_signature():
        return {
            "rna_preprocess": "all_genes",
            "image_graph_mode": IMAGE_GRAPH_MODE,
            "image_graph_feature_mode": IMAGE_GRAPH_FEATURE_MODE,
            "edge_view_input": "uni_divprune_image_features",
            "local_subgraph_enabled": LOCAL_SUBGRAPH_ENABLED,
            "local_expr_keep_ratio": LOCAL_EXPR_KEEP_RATIO,
            "local_spatial_keep_ratio": LOCAL_SPATIAL_KEEP_RATIO,
            "local_edge_keep_ratio": LOCAL_EDGE_KEEP_RATIO,
            "latgrl_enabled": LATGRL_ENABLED,
            "latgrl_homo_ratio": LATGRL_HOMO_RATIO,
            "latgrl_hetero_ratio": LATGRL_HETERO_RATIO,
        }

    def _precomputed_triview_matches_runtime(graph_dict):
        config = graph_dict.get("tri_view_params")
        if not isinstance(config, dict):
            return False
        signature = config.get("runtime_signature")
        expected = _runtime_signature()
        if isinstance(signature, dict):
            return all(signature.get(key) == value for key, value in expected.items())
        return all(config.get(key) == value for key, value in expected.items())

    def _find_metadata_file(data_dir, sample_id):
        env_meta_path = os.environ.get("METADATA_PATH", "").strip()
        if env_meta_path and os.path.exists(env_meta_path):
            return env_meta_path

        sample_path = _sample_path(data_dir, sample_id)
        for name in ["metadata.tsv", "metadata.csv"]:
            path = os.path.join(sample_path, name)
            if os.path.exists(path):
                return path
        return None

    def _has_10x_mtx_sample(sample_path):
        """Detect Visium-like samples without relying on one exact matrix folder name."""
        if not os.path.isdir(sample_path):
            return False

        candidate_dirs = [
            sample_path,
            os.path.join(sample_path, "filtered_feature_bc_matrix"),
            os.path.join(sample_path, "filters_count_matrices"),
            os.path.join(sample_path, "filtered_count_matrices"),
            os.path.join(sample_path, "count_matrices"),
        ]
        for root, dirs, _files in os.walk(sample_path):
            rel_depth = os.path.relpath(root, sample_path).count(os.sep)
            if rel_depth >= 2:
                dirs[:] = []
                continue
            candidate_dirs.append(root)

        for candidate in dict.fromkeys(candidate_dirs):
            has_matrix = os.path.exists(os.path.join(candidate, "matrix.mtx"))
            has_barcodes = os.path.exists(os.path.join(candidate, "barcodes.tsv"))
            has_features = (
                os.path.exists(os.path.join(candidate, "features.tsv"))
                or os.path.exists(os.path.join(candidate, "genes.tsv"))
            )
            if has_matrix and has_barcodes and has_features:
                return True
        return False

    def _has_st_like_sample(sample_path, sample_id):
        """Detect HER2/ST-like samples: count table plus selection coordinates."""
        if not os.path.isdir(sample_path):
            return False
        spatial_dir = os.path.join(sample_path, "spatial")
        count_candidates = [
            os.path.join(sample_path, f"{sample_id}.tsv.gz"),
            os.path.join(sample_path, f"{sample_id}.tsv"),
        ]
        has_count = any(os.path.exists(path) for path in count_candidates)
        if not has_count:
            has_count = any(name.endswith(".tsv.gz") for name in os.listdir(sample_path))
        selection_candidates = [
            os.path.join(spatial_dir, f"{sample_id}_selection.tsv.gz"),
            os.path.join(spatial_dir, f"{sample_id}_selection.tsv"),
        ]
        has_selection = any(os.path.exists(path) for path in selection_candidates)
        return has_count and has_selection

    def _resolve_profile(data_dir, sample_id, profile):
        profile = (profile or "auto").strip().lower()
        if profile != "auto":
            return profile
        lower_dir = str(data_dir).lower()
        lower_sample = str(sample_id).lower()
        sample_path = _sample_path(data_dir, sample_id)
        combined = f"{lower_dir} {lower_sample}"
        if any(token in combined for token in ("colon", "colorectal", "crc", "016um", "032um")):
            return "colon_cancer_visium_hd"
        if lower_sample == "v1_breast_cancer_block_a_section_1":
            return "v1_breast_cancer_block_a_section_1"
        if "dlpfc" in lower_dir:
            return "dlpfc"
        if "her2" in lower_dir or _has_st_like_sample(sample_path, sample_id):
            return "her2_breast"
        if "multi" in lower_dir or _has_10x_mtx_sample(sample_path):
            return "multi_subtype_breast"
        return "dlpfc"

    def h1_v1_breast_defaults():
        """H1 defaults that reproduce the previous V1-breast-matched setting."""
        return {
            "n_domains": 6,
            "pca_n_comps": 100,
            "graph_k": 8,
            "expression_graph_k": 12,
            "spatial_graph_k": 8,
            "edge_graph_k": 8,
            "image_pca_n": 50,
            "cbr": True,
            "cbr_spatial_k": 10,
            "cbr_max_change_ratio": 0.08,
            "cbr_boundary_quantile": 0.60,
            "cbr_low_confidence_quantile": 0.45,
            "cbr_margin": 0.03,
            "cbr_lambda_spatial": 0.68,
            "cbr_lambda_uni": 0.55,
            "cbr_lambda_anchor": 1.00,
            "cbr_boundary_loss_weight": 0.75,
            "cbr_multiscale": True,
            "cbr_compartment_aware": True,
            "cbr_lambda_compartment": 0.30,
        }

    def _is_h1_her2(profile, sample_id):
        return str(profile or "").strip().lower() == "her2_breast" and str(sample_id or "").strip().lower() == "h1"

    def _apply_sample_defaults(defaults, profile, sample_id):
        defaults = dict(defaults)
        if _is_h1_her2(profile, sample_id):
            defaults.update(h1_v1_breast_defaults())
        return defaults

    def _profile_defaults(profile):
        defaults = {
            "multi_subtype_breast": dict(n_domains=5, pca_n_comps=100, graph_k=8,
                                        expression_graph_k=12, spatial_graph_k=8, edge_graph_k=8,
                                        image_pca_n=50, cbr=True,
                                        cbr_spatial_k=8, cbr_max_change_ratio=0.08,
                                        cbr_boundary_quantile=0.65, cbr_low_confidence_quantile=0.40,
                                        cbr_margin=0.03, cbr_lambda_spatial=0.60,
                                        cbr_lambda_uni=0.50, cbr_lambda_anchor=1.00,
                                        cbr_boundary_loss_weight=0.60, cbr_multiscale=True,
                                        cbr_compartment_aware=True, cbr_lambda_compartment=0.25),
            "v1_breast_cancer_block_a_section_1": dict(n_domains=20, pca_n_comps=100, graph_k=8,
                                                    expression_graph_k=12, spatial_graph_k=8, edge_graph_k=8,
                                                    image_pca_n=50, cbr=True,
                                                    cbr_spatial_k=10, cbr_max_change_ratio=0.08,
                                                    cbr_boundary_quantile=0.60, cbr_low_confidence_quantile=0.45,
                                                    cbr_margin=0.03, cbr_lambda_spatial=0.68,
                                                    cbr_lambda_uni=0.55, cbr_lambda_anchor=1.00,
                                                    cbr_boundary_loss_weight=0.75, cbr_multiscale=True,
                                                    cbr_compartment_aware=True, cbr_lambda_compartment=0.30),
            "her2_breast": dict(n_domains=5, pca_n_comps=50, graph_k=6,
                                expression_graph_k=12, spatial_graph_k=6, edge_graph_k=6,
                                image_pca_n=50, cbr=True,
                                cbr_spatial_k=8, cbr_max_change_ratio=0.06,
                                cbr_boundary_quantile=0.62, cbr_low_confidence_quantile=0.45,
                                cbr_margin=0.035, cbr_lambda_spatial=0.65,
                                cbr_lambda_uni=0.45, cbr_lambda_anchor=1.00,
                                cbr_boundary_loss_weight=0.65, cbr_multiscale=True,
                                cbr_compartment_aware=True, cbr_lambda_compartment=0.30),
            "colon_cancer_visium_hd": dict(n_domains=10, pca_n_comps=100, graph_k=8,
                                            expression_graph_k=12, spatial_graph_k=8, edge_graph_k=8,
                                            image_pca_n=50, cbr=True,
                                            cbr_spatial_k=10, cbr_max_change_ratio=0.05,
                                            cbr_boundary_quantile=0.70, cbr_low_confidence_quantile=0.35,
                                            cbr_margin=0.03, cbr_lambda_spatial=0.68,
                                            cbr_lambda_uni=0.45, cbr_lambda_anchor=1.00,
                                            cbr_boundary_loss_weight=0.65, cbr_multiscale=True,
                                            cbr_compartment_aware=True, cbr_lambda_compartment=0.30),
        }
        if profile not in defaults:
            raise ValueError(
                f"Unsupported DATASET_PROFILE={profile!r}. Use dlpfc, multi_subtype_breast, "
                "v1_breast_cancer_block_a_section_1, her2_breast, "
                "colon_cancer_visium_hd, or auto."
            )
        return defaults[profile]

    def _infer_n_domains_from_metadata(data_dir, sample_id, profile, default_n_domains, mode="auto"):
        if profile == "dlpfc":
            return default_n_domains

        mode = str(mode).strip().lower()
        if mode in {"auto", "profile", "default", "fixed"}:
            print(
                "N_DOMAINS uses dataset profile default. "
                "Manual annotation columns are kept for evaluation only. "
                "Set N_DOMAINS_MODE=gt to force matching ground-truth label count."
            )
            return default_n_domains

        meta_path = _find_metadata_file(data_dir, sample_id)
        if meta_path is None:
            if profile == "her2_breast" and str(sample_id).strip().lower() == "h1" and mode in {"major", "auto_major"}:
                print("N_DOMAINS fallback for HER2 H1: using 6 domains because metadata was not found.")
                return 6
            return default_n_domains

        sep = "\t" if meta_path.endswith(".tsv") else ","
        try:
            metadata = pd.read_csv(meta_path, sep=sep)
        except Exception as exc:
            print(f"N_DOMAINS auto inference skipped: failed to read metadata ({exc})")
            return default_n_domains

        invalid_labels = {"undetermined", "unknown", "nan", "none", ""}
        gt_modes = {"gt", "ground_truth", "classification", "match_gt"}

        if "Classification" in metadata.columns:
            labels = metadata["Classification"].dropna().astype(str)
            labels = labels[~labels.str.lower().isin(invalid_labels)]
            if labels.empty:
                return default_n_domains
            counts = labels.value_counts()
            if mode in gt_modes:
                inferred = int(len(counts))
                print(
                    f"N_DOMAINS matched to ground truth column 'Classification': {inferred} labels "
                    f"(mode={mode}; counts={counts.to_dict()})"
                )
                return inferred
            min_count = max(20, int(round(len(labels) * 0.01)))
            major_counts = counts[counts >= min_count]
            inferred = int(np.clip(len(major_counts), 3, 8))
            print(
                f"N_DOMAINS inferred from major labels in 'Classification': {inferred} "
                f"(mode={mode}, min_count={min_count}; counts={counts.to_dict()})"
            )
            return inferred

        label_col = None
        for col in ["fine_annot_type", "annot_type", "label", "ground_truth", "annotation", "layer_guess"]:
            if col in metadata.columns:
                label_col = col
                break
        if label_col is None:
            return default_n_domains

        labels = metadata[label_col].dropna().astype(str)
        labels = labels[~labels.str.lower().isin(invalid_labels)]
        if labels.empty:
            return default_n_domains

        counts = labels.value_counts()
        if mode in gt_modes and label_col in {"fine_annot_type", "annot_type"}:
            inferred = int(len(counts))
            print(
                f"N_DOMAINS matched to annotation column '{label_col}': {inferred} labels "
                f"(mode={mode}; counts={counts.to_dict()})"
            )
            return inferred

        min_count = max(20, int(round(len(labels) * 0.01)))
        major_counts = counts[counts >= min_count]
        inferred = int(np.clip(len(major_counts), 3, 8))
        print(
            f"N_DOMAINS auto inferred from {os.path.basename(meta_path)} "
            f"column '{label_col}': {inferred} major labels "
            f"(min_count={min_count}; counts={counts.to_dict()})"
        )
        return inferred

    def _training_epoch_defaults(profile, sample_id):
        profile = str(profile or "").strip().lower()
        sample = str(sample_id or "").strip().lower()

        if profile == "her2_breast" and sample == "h1":
            return {"pre_epochs": 500, "epochs": 350}, "h1_ari_recovery"
        if profile == "v1_breast_cancer_block_a_section_1" or sample == "v1_breast_cancer_block_a_section_1":
            return {"pre_epochs": 500, "epochs": 350}, "v1_breast_recovery"
        return {"pre_epochs": 500, "epochs": 460}, "default"

    def _sri_warmup_epoch_default(profile, sample_id):
        profile = str(profile or "").strip().lower()
        sample = str(sample_id or "").strip().lower()

        if profile == "her2_breast" and sample == "h1":
            return ""
        return "150"

    def _n_domains_mode_default(profile, sample_id):
        profile = str(profile or "").strip().lower()
        sample = str(sample_id or "").strip().lower()

        if profile == "her2_breast" and sample == "h1":
            return "major"
        return "auto"

    def _apply_cbr_complexity_defaults(defaults, profile, n_domains, mode="auto", sample_id=None):
        """Set cancer boundary refinement strength by dataset complexity.

        DLPFC is not a cancer boundary task, so CBR stays disabled by default. For
        cancer datasets, larger/finer annotation systems receive stronger boundary
        evidence weighting but still keep a conservative cap on the number of
        relabeled spots.
        """
        defaults = dict(defaults)
        mode = str(mode or "auto").strip().lower()
        if profile == "dlpfc" or mode in {"off", "none", "disable", "disabled"}:
            defaults.update(
                cbr=False,
                cbr_compartment_aware=False,
                cbr_max_change_ratio=min(float(defaults.get("cbr_max_change_ratio", 0.02)), 0.02),
            )
            return defaults, "off"

        if _is_h1_her2(profile, sample_id) and mode in {"auto", "v1", "v1_breast", "v1_breast_matched", "h1_recovery"}:
            defaults.update(h1_v1_breast_defaults())
            return defaults, "v1_breast_matched"

        if mode in {"mild", "low"}:
            level = "mild"
        elif mode in {"moderate", "medium"}:
            level = "moderate"
        elif mode in {"strong", "high"}:
            level = "strong"
        elif int(n_domains) >= 12:
            level = "strong"
        elif int(n_domains) >= 6:
            level = "moderate"
        else:
            level = "mild"

        if profile == "multi_subtype_breast":
            defaults.update(
                cbr=True,
                cbr_spatial_k=8,
                cbr_max_change_ratio=0.08,
                cbr_boundary_quantile=0.65,
                cbr_low_confidence_quantile=0.40,
                cbr_margin=0.03,
                cbr_lambda_spatial=0.60,
                cbr_lambda_uni=0.50,
                cbr_lambda_anchor=1.00,
                cbr_boundary_loss_weight=0.60,
                cbr_multiscale=True,
                cbr_compartment_aware=True,
                cbr_lambda_compartment=0.25,
            )
            return defaults, "mild_empirical_best"

        if level == "mild":
            defaults.update(
                cbr=True,
                cbr_spatial_k=max(int(defaults.get("cbr_spatial_k", 8)), 8),
                cbr_max_change_ratio=max(float(defaults.get("cbr_max_change_ratio", 0.06)), 0.06),
                cbr_boundary_quantile=min(float(defaults.get("cbr_boundary_quantile", 0.62)), 0.62),
                cbr_low_confidence_quantile=max(float(defaults.get("cbr_low_confidence_quantile", 0.45)), 0.45),
                cbr_margin=min(float(defaults.get("cbr_margin", 0.035)), 0.035),
                cbr_lambda_spatial=max(float(defaults.get("cbr_lambda_spatial", 0.65)), 0.65),
                cbr_lambda_uni=max(float(defaults.get("cbr_lambda_uni", 0.45)), 0.45),
                cbr_lambda_anchor=min(float(defaults.get("cbr_lambda_anchor", 1.00)), 1.00),
                cbr_boundary_loss_weight=max(float(defaults.get("cbr_boundary_loss_weight", 0.65)), 0.65),
                cbr_multiscale=True,
                cbr_compartment_aware=True,
                cbr_lambda_compartment=max(float(defaults.get("cbr_lambda_compartment", 0.30)), 0.30),
            )
        elif level == "moderate":
            defaults.update(
                cbr=True,
                cbr_spatial_k=max(int(defaults.get("cbr_spatial_k", 10)), 10),
                cbr_max_change_ratio=max(float(defaults.get("cbr_max_change_ratio", 0.08)), 0.08),
                cbr_boundary_quantile=0.58,
                cbr_low_confidence_quantile=0.50,
                cbr_margin=0.03,
                cbr_lambda_spatial=0.70,
                cbr_lambda_uni=0.55,
                cbr_lambda_anchor=0.98,
                cbr_boundary_loss_weight=0.75,
                cbr_multiscale=True,
                cbr_compartment_aware=True,
                cbr_lambda_compartment=0.35,
            )
        else:
            defaults.update(
                cbr=True,
                cbr_spatial_k=max(int(defaults.get("cbr_spatial_k", 10)), 10),
                cbr_max_change_ratio=max(float(defaults.get("cbr_max_change_ratio", 0.10)), 0.10),
                cbr_boundary_quantile=0.55,
                cbr_low_confidence_quantile=0.55,
                cbr_margin=0.025,
                cbr_lambda_spatial=0.75,
                cbr_lambda_uni=0.60,
                cbr_lambda_anchor=0.95,
                cbr_boundary_loss_weight=0.90,
                cbr_multiscale=True,
                cbr_compartment_aware=True,
                cbr_lambda_compartment=0.40,
            )
        return defaults, level

    DATASET_PROFILE = _resolve_profile(DATA_DIR, SAMPLE_ID, DATASET_PROFILE)
    _defaults = _apply_sample_defaults(_profile_defaults(DATASET_PROFILE), DATASET_PROFILE, SAMPLE_ID)
    N_DOMAINS_MODE = os.environ.get("N_DOMAINS_MODE", _n_domains_mode_default(DATASET_PROFILE, SAMPLE_ID))
    _DLPFC_TRUE_DOMAINS = int(os.environ.get("DLPFC_TRUE_N_DOMAINS", "7"))
    if "N_DOMAINS" in os.environ:
        N_DOMAINS = int(os.environ["N_DOMAINS"])
    else:
        N_DOMAINS = _infer_n_domains_from_metadata(
            DATA_DIR,
            SAMPLE_ID,
            DATASET_PROFILE,
            _defaults["n_domains"],
            mode=N_DOMAINS_MODE,
        )
    if DATASET_PROFILE == "dlpfc" and "N_DOMAINS" not in os.environ:
        N_DOMAINS = _DLPFC_TRUE_DOMAINS
        N_DOMAINS_MODE = "fixed_dlpfc_true_layers"
    CLUSTER_METHOD = os.environ.get(
        "CLUSTER_METHOD",
        "kmeans" if DATASET_PROFILE == "colon_cancer_visium_hd" else "auto",
    )
    CBR_COMPLEXITY_MODE = os.environ.get("CBR_COMPLEXITY_MODE", "auto")
    _defaults, CBR_COMPLEXITY_LEVEL = _apply_cbr_complexity_defaults(
        _defaults,
        DATASET_PROFILE,
        N_DOMAINS,
        mode=CBR_COMPLEXITY_MODE,
        sample_id=SAMPLE_ID,
    )
    PCA_N_COMPS = int(os.environ.get("PCA_N_COMPS", _defaults["pca_n_comps"]))
    GRAPH_K = int(os.environ.get("GRAPH_K", _defaults["graph_k"]))
    USE_TRIVIEW_EDGE = _as_bool(os.environ.get("USE_TRIVIEW_EDGE"), True)
    GRAPH_EMBEDDING_DIM = int(os.environ.get("GRAPH_EMBEDDING_DIM", "32"))
    GRAPH_HIDDEN_DIM = int(os.environ.get("GRAPH_HIDDEN_DIM", "64"))
    USE_GRAPHNORMV2_GCN = os.environ.get("USE_GRAPHNORMV2_GCN", "1").strip().lower() not in {"0", "false", "no", "off"}
    GRAPHNORMV2_RES_ALPHA = float(os.environ.get("GRAPHNORMV2_RES_ALPHA", "0.10"))
    GRAPHNORMV2_CENTER_INIT = float(os.environ.get("GRAPHNORMV2_CENTER_INIT", "0.25"))
    GRAPHNORMV2_OUTPUT_NORM = os.environ.get("GRAPHNORMV2_OUTPUT_NORM", "0").strip().lower() in {"1", "true", "yes", "on"}
    EXPRESSION_GRAPH_K = int(os.environ.get("EXPRESSION_GRAPH_K", _defaults["expression_graph_k"]))
    SPATIAL_GRAPH_K = int(os.environ.get("SPATIAL_GRAPH_K", _defaults["spatial_graph_k"]))
    EDGE_GRAPH_K = int(os.environ.get("EDGE_GRAPH_K", _defaults["edge_graph_k"]))
    IMAGE_GRAPH_PCA_N = int(os.environ.get("IMAGE_GRAPH_PCA_N", _defaults["image_pca_n"]))
    IMAGE_GRAPH_MODE = os.environ.get("IMAGE_GRAPH_MODE", "similarity").strip().lower()
    IMAGE_GRAPH_FEATURE_MODE = os.environ.get("IMAGE_GRAPH_FEATURE_MODE", "divprune").strip().lower()
    LOCAL_SUBGRAPH_ENABLED = _as_bool(os.environ.get("LOCAL_SUBGRAPH_ENABLED"), True)
    LOCAL_EXPR_KEEP_RATIO = float(os.environ.get("LOCAL_EXPR_KEEP_RATIO", "1.0"))
    LOCAL_SPATIAL_KEEP_RATIO = float(os.environ.get("LOCAL_SPATIAL_KEEP_RATIO", "1.0"))
    LOCAL_EDGE_KEEP_RATIO = float(os.environ.get("LOCAL_EDGE_KEEP_RATIO", "1.0"))
    LATGRL_ENABLED = _as_bool(os.environ.get("LATGRL_ENABLED"), True)
    LATGRL_HOMO_RATIO = float(os.environ.get("LATGRL_HOMO_RATIO", "0.35"))
    LATGRL_HETERO_RATIO = float(os.environ.get("LATGRL_HETERO_RATIO", "0.20"))
    USE_CBR_REFINEMENT = _as_bool(os.environ.get("USE_CBR_REFINEMENT"), _defaults["cbr"])
    CBR_SPATIAL_K = int(os.environ.get("CBR_SPATIAL_K", _defaults["cbr_spatial_k"]))
    CBR_MAX_CHANGE_RATIO = float(os.environ.get("CBR_MAX_CHANGE_RATIO", _defaults["cbr_max_change_ratio"]))
    CBR_BOUNDARY_QUANTILE = float(os.environ.get("CBR_BOUNDARY_QUANTILE", _defaults["cbr_boundary_quantile"]))
    CBR_LOW_CONFIDENCE_QUANTILE = float(os.environ.get("CBR_LOW_CONFIDENCE_QUANTILE", _defaults["cbr_low_confidence_quantile"]))
    CBR_MARGIN = float(os.environ.get("CBR_MARGIN", _defaults["cbr_margin"]))
    CBR_LAMBDA_SPATIAL = float(os.environ.get("CBR_LAMBDA_SPATIAL", _defaults["cbr_lambda_spatial"]))
    CBR_LAMBDA_UNI = float(os.environ.get("CBR_LAMBDA_UNI", _defaults["cbr_lambda_uni"]))
    CBR_LAMBDA_ANCHOR = float(os.environ.get("CBR_LAMBDA_ANCHOR", _defaults["cbr_lambda_anchor"]))
    CBR_BOUNDARY_LOSS_WEIGHT = float(os.environ.get("CBR_BOUNDARY_LOSS_WEIGHT", _defaults["cbr_boundary_loss_weight"]))
    CBR_MULTISCALE = _as_bool(os.environ.get("CBR_MULTISCALE"), _defaults["cbr_multiscale"])
    CBR_COMPARTMENT_AWARE = _as_bool(os.environ.get("CBR_COMPARTMENT_AWARE"), _defaults["cbr_compartment_aware"])
    CBR_LAMBDA_COMPARTMENT = float(os.environ.get("CBR_LAMBDA_COMPARTMENT", _defaults["cbr_lambda_compartment"]))
    SRI_CLR_ENABLED = _as_bool(os.environ.get("SRI_CLR_ENABLED"), True)
    SRI_CLR_WEIGHT = float(os.environ.get("SRI_CLR_WEIGHT", "0.06"))
    SRI_CLR_MARGIN = float(os.environ.get("SRI_CLR_MARGIN", "0.75"))
    SRI_CLR_WARMUP_RATIO = float(os.environ.get("SRI_CLR_WARMUP_RATIO", "0.30"))
    SRI_CLR_WARMUP_EPOCH_RAW = os.environ.get(
        "SRI_CLR_WARMUP_EPOCH",
        _sri_warmup_epoch_default(DATASET_PROFILE, SAMPLE_ID),
    ).strip()
    SRI_CLR_WARMUP_EPOCH = int(SRI_CLR_WARMUP_EPOCH_RAW) if SRI_CLR_WARMUP_EPOCH_RAW else None
    if SRI_CLR_WARMUP_EPOCH is None:
        os.environ.pop("SRI_CLR_WARMUP_EPOCH", None)
    else:
        os.environ["SRI_CLR_WARMUP_EPOCH"] = str(SRI_CLR_WARMUP_EPOCH)
    SRI_CLR_MAX_PAIRS = int(os.environ.get("SRI_CLR_MAX_PAIRS", "4096"))
    SOFT_POTTS_ENABLED = _as_bool(os.environ.get("SOFT_POTTS_ENABLED"), True)
    SOFT_POTTS_ETA = float(os.environ.get("SOFT_POTTS_ETA", "0.25"))
    SOFT_POTTS_LAMBDA = float(os.environ.get("SOFT_POTTS_LAMBDA", "0.05"))
    SOFT_POTTS_ITERS = int(os.environ.get("SOFT_POTTS_ITERS", "6"))
    SOFT_POTTS_LR = float(os.environ.get("SOFT_POTTS_LR", "0.35"))
    SOFT_POTTS_MAX_EDGES = int(os.environ.get("SOFT_POTTS_MAX_EDGES", "40000"))
    SOFT_POTTS_PAIRWISE = os.environ.get("SOFT_POTTS_PAIRWISE", "pcd").strip().lower()
    if DATASET_PROFILE == "dlpfc":

        USE_CBR_REFINEMENT = False
        CBR_COMPARTMENT_AWARE = False

    print("Starting deep learning model training...")

    print(f"Dataset profile: {DATASET_PROFILE}")
    print(f"DATA_DIR={DATA_DIR}")
    print(f"SAMPLE_ID={SAMPLE_ID}")
    print(f"SEED={SEED}")
    print(f"RESULTS_DIR={RESULTS_DIR}")
    print(f"N_DOMAINS_MODE={N_DOMAINS_MODE}")
    print(f"N_DOMAINS={N_DOMAINS}, PCA_N_COMPS={PCA_N_COMPS}, GRAPH_K={GRAPH_K}")
    if DATASET_PROFILE == "dlpfc":
        print(f"DLPFC_TRUE_N_DOMAINS={_DLPFC_TRUE_DOMAINS}")
    print(f"CLUSTER_METHOD={CLUSTER_METHOD}")
    print("RNA input source=adata.X; RNA matrix augmentation is disabled")
    print(
        "Tri-view edge="
        f"{USE_TRIVIEW_EDGE}, expression_k={EXPRESSION_GRAPH_K}, spatial_k={SPATIAL_GRAPH_K}, "
        f"edge_k={EDGE_GRAPH_K}, image_pca={IMAGE_GRAPH_PCA_N}, image_graph_mode={IMAGE_GRAPH_MODE}, "
        f"image_feature_mode={IMAGE_GRAPH_FEATURE_MODE}, edge_view_input=direct_UNI_DivPrune_features"
    )
    print(
        "Tri-view structure="
        f"local_subgraph={LOCAL_SUBGRAPH_ENABLED}, "
        f"local_keep=({LOCAL_EXPR_KEEP_RATIO}, {LOCAL_SPATIAL_KEEP_RATIO}, {LOCAL_EDGE_KEEP_RATIO}), "
        f"latgrl={LATGRL_ENABLED}, homo_ratio={LATGRL_HOMO_RATIO}, hetero_ratio={LATGRL_HETERO_RATIO}"
    )
    print("Embedding mode=graph_only")
    print(f"Graph hidden dims=[{GRAPH_HIDDEN_DIM}, {GRAPH_EMBEDDING_DIM}]")
    print(
        "GraphNormv2-style residual GCN="
        f"{USE_GRAPHNORMV2_GCN}, res_alpha={GRAPHNORMV2_RES_ALPHA}, "
        f"center_init={GRAPHNORMV2_CENTER_INIT}, output_norm={GRAPHNORMV2_OUTPUT_NORM}"
    )
    print(f"CBR_COMPLEXITY_MODE={CBR_COMPLEXITY_MODE}, level={CBR_COMPLEXITY_LEVEL}")
    print(
        "CBR="
        f"{USE_CBR_REFINEMENT}, max_change_ratio={CBR_MAX_CHANGE_RATIO}, "
        f"margin={CBR_MARGIN}, lambda_uni={CBR_LAMBDA_UNI}"
    )
    print(
        "CBR details="
        f"spatial_k={CBR_SPATIAL_K}, boundary_q={CBR_BOUNDARY_QUANTILE}, "
        f"low_conf_q={CBR_LOW_CONFIDENCE_QUANTILE}, lambda_spatial={CBR_LAMBDA_SPATIAL}, "
        f"lambda_anchor={CBR_LAMBDA_ANCHOR}, boundary_loss_weight={CBR_BOUNDARY_LOSS_WEIGHT}, "
        f"multiscale={CBR_MULTISCALE}, compartment_aware={CBR_COMPARTMENT_AWARE}, "
        f"lambda_compartment={CBR_LAMBDA_COMPARTMENT}"
    )
    print(
        "SRI-CLR="
        f"{SRI_CLR_ENABLED}, weight={SRI_CLR_WEIGHT}, margin={SRI_CLR_MARGIN}, "
        f"warmup_ratio={SRI_CLR_WARMUP_RATIO}, "
        f"warmup_epoch={SRI_CLR_WARMUP_EPOCH if SRI_CLR_WARMUP_EPOCH is not None else 'auto'}, "
        f"max_pairs={SRI_CLR_MAX_PAIRS}"
    )
    print(
        "Soft-Potts="
        f"{SOFT_POTTS_ENABLED}, eta={SOFT_POTTS_ETA}, lambda={SOFT_POTTS_LAMBDA}, "
        f"iters={SOFT_POTTS_ITERS}, lr={SOFT_POTTS_LR}, max_edges={SOFT_POTTS_MAX_EDGES}, "
        f"pairwise={SOFT_POTTS_PAIRWISE}"
    )

    seed_torch(seed=SEED)

    _epoch_defaults, TRAINING_EPOCH_LEVEL = _training_epoch_defaults(DATASET_PROFILE, SAMPLE_ID)
    PRETRAIN_EPOCHS = int(os.environ.get("PRETRAIN_EPOCHS", _epoch_defaults["pre_epochs"]))
    TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", _epoch_defaults["epochs"]))
    print(
        f"Training epochs={TRAIN_EPOCHS}, pre_epochs={PRETRAIN_EPOCHS}, "
        f"level={TRAINING_EPOCH_LEVEL}"
    )
    gemcst = GeMCSTRunner(
        save_path=RESULTS_DIR,
        task="Identify_Domain",
        pre_epochs=PRETRAIN_EPOCHS,
        epochs=TRAIN_EPOCHS,
        use_gpu=True
    )

    print("\nStep 1: Loading preprocessed data...")

    sample_dir = os.path.join(RESULTS_DIR, SAMPLE_ID)

    preprocessed_file = os.path.join(sample_dir, f"{SAMPLE_ID}_preprocessed.h5ad")
    if not os.path.exists(preprocessed_file):
        print(f"[ERROR] Preprocessed file does not exist: {preprocessed_file}")
        print("Please run run_image.py first to process image features")
        sys.exit(1)

    adata = anndata.read_h5ad(preprocessed_file)
    print(f"[OK] Preprocessed data loaded: {adata.shape}")
    config_file = os.path.join(sample_dir, f"{SAMPLE_ID}_config.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    print("[OK] Configuration loaded")

    print("\nStep 2: Feature engineering...")

    print("  Building spatial neighborhood graph...")
    graph_dict = gemcst._get_graph(
        adata.obsm["spatial"],
        distType="KDTree",
        k=GRAPH_K,
    )
    print("[OK] Spatial graph construction completed")

    if USE_PRECOMPUTED_TRIVIEW and "GeMCST_pca_input" in adata.obsm:
        data = np.asarray(adata.obsm["GeMCST_pca_input"], dtype=np.float32)
        print(f"[OK] Reused precomputed RNA feature input: {data.shape}")
    else:
        print("  Running PCA dimensionality reduction...")
        data = gemcst._data_process(
            adata,
            pca_n_comps=PCA_N_COMPS,
        )
        print(f"[OK] PCA completed: {data.shape}")

    if USE_TRIVIEW_EDGE:
        precomputed_graph = _load_precomputed_triview(sample_dir, SAMPLE_ID, adata)
        if precomputed_graph is None:
            raise RuntimeError(
                "Missing precomputed tri-view graphs. Run the image stage before training."
            )
        if not _precomputed_triview_matches_runtime(precomputed_graph):
            raise RuntimeError(
                "Precomputed tri-view graph config does not match the current runtime signature. "
                "Rerun the image stage, then rerun training."
            )
        graph_dict = precomputed_graph
        print("[OK] Loaded precomputed tri-view graphs")
        print(f"Pathology image third-view input: {graph_dict['edge_view_data'].shape}")
        print(
            "GNN aggregation inside all graph branches: "
            + ("GraphNormv2-style residual GCN" if USE_GRAPHNORMV2_GCN else "Basic GCN")
        )
        print("Tri-view graph construction completed.")

    print("\nStep 3: Deep learning model training...")

    print("  Starting model training...")
    gemcst_embed, gemcst_precluster_latent = gemcst._fit(
        data=data,
        graph_dict=graph_dict,
        conv_hidden=[GRAPH_HIDDEN_DIM, GRAPH_EMBEDDING_DIM],
        tri_view=USE_TRIVIEW_EDGE,
        edge_input_dim=graph_dict["edge_view_data"].shape[1] if USE_TRIVIEW_EDGE else None,
    )
    adata.obsm["GeMCST_embed"] = gemcst_embed
    adata.obsm["GeMCST_precluster_latent"] = gemcst_precluster_latent
    adata.uns["model_type"] = (
        "gemcst_tissue_edge_triview_graph_only" if USE_TRIVIEW_EDGE else "gemcst_anchor_graph_only"
    )
    adata.uns["dataset_profile"] = DATASET_PROFILE
    adata.uns["run_model_parameters"] = {
        "n_domains": N_DOMAINS,
        "n_domains_mode": N_DOMAINS_MODE,
        "true_n_domains": _DLPFC_TRUE_DOMAINS if DATASET_PROFILE == "dlpfc" else None,
        "cluster_method": CLUSTER_METHOD,
        "pca_n_comps": PCA_N_COMPS,
        "graph_k": GRAPH_K,
        "rna_input_source": adata.uns.get("rna_input_source", "unknown"),
        "rna_matrix_augment": False,
        "use_triview_edge": USE_TRIVIEW_EDGE,
        "embedding_type": "graph_only",
        "umap_representation_key": "GeMCST_precluster_latent",
        "umap_representation_stage": "continuous_encoder_latent_before_dec_soft_potts",
        "graph_hidden_dim": GRAPH_HIDDEN_DIM,
        "graph_embedding_dim": GRAPH_EMBEDDING_DIM,
        "use_graphnormv2_gcn": USE_GRAPHNORMV2_GCN,
        "graphnormv2_res_alpha": GRAPHNORMV2_RES_ALPHA,
        "graphnormv2_center_init": GRAPHNORMV2_CENTER_INIT,
        "graphnormv2_output_norm": GRAPHNORMV2_OUTPUT_NORM,
        "expression_graph_k": EXPRESSION_GRAPH_K,
        "spatial_graph_k": SPATIAL_GRAPH_K,
        "edge_graph_k": EDGE_GRAPH_K,
        "image_graph_pca_n": IMAGE_GRAPH_PCA_N,
        "image_graph_mode": IMAGE_GRAPH_MODE,
        "image_graph_feature_mode": IMAGE_GRAPH_FEATURE_MODE,
        "local_subgraph_enabled": LOCAL_SUBGRAPH_ENABLED,
        "local_expr_keep_ratio": LOCAL_EXPR_KEEP_RATIO,
        "local_spatial_keep_ratio": LOCAL_SPATIAL_KEEP_RATIO,
        "local_edge_keep_ratio": LOCAL_EDGE_KEEP_RATIO,
        "latgrl_enabled": LATGRL_ENABLED,
        "latgrl_homo_ratio": LATGRL_HOMO_RATIO,
        "latgrl_hetero_ratio": LATGRL_HETERO_RATIO,
        "edge_view_input": "uni_divprune_image_features",
        "edge_view_dim": int(graph_dict["edge_view_data"].shape[1]) if USE_TRIVIEW_EDGE else None,
        "use_cbr_refinement": USE_CBR_REFINEMENT,
        "cbr_spatial_k": CBR_SPATIAL_K,
        "cbr_max_change_ratio": CBR_MAX_CHANGE_RATIO,
        "cbr_boundary_quantile": CBR_BOUNDARY_QUANTILE,
        "cbr_low_confidence_quantile": CBR_LOW_CONFIDENCE_QUANTILE,
        "cbr_margin": CBR_MARGIN,
        "cbr_lambda_spatial": CBR_LAMBDA_SPATIAL,
        "cbr_lambda_uni": CBR_LAMBDA_UNI,
        "cbr_lambda_anchor": CBR_LAMBDA_ANCHOR,
        "cbr_boundary_loss_weight": CBR_BOUNDARY_LOSS_WEIGHT,
        "cbr_multiscale": CBR_MULTISCALE,
        "cbr_compartment_aware": CBR_COMPARTMENT_AWARE,
        "cbr_lambda_compartment": CBR_LAMBDA_COMPARTMENT,
        "sri_clr_enabled": SRI_CLR_ENABLED,
        "sri_clr_weight": SRI_CLR_WEIGHT,
        "sri_clr_margin": SRI_CLR_MARGIN,
        "sri_clr_warmup_ratio": SRI_CLR_WARMUP_RATIO,
        "sri_clr_warmup_epoch": SRI_CLR_WARMUP_EPOCH,
        "sri_clr_max_pairs": SRI_CLR_MAX_PAIRS,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "train_epochs": TRAIN_EPOCHS,
        "training_epoch_level": TRAINING_EPOCH_LEVEL,
        "soft_potts_enabled": SOFT_POTTS_ENABLED,
        "soft_potts_eta": SOFT_POTTS_ETA,
        "soft_potts_lambda": SOFT_POTTS_LAMBDA,
        "soft_potts_iters": SOFT_POTTS_ITERS,
        "soft_potts_lr": SOFT_POTTS_LR,
        "soft_potts_max_edges": SOFT_POTTS_MAX_EDGES,
        "soft_potts_pairwise": SOFT_POTTS_PAIRWISE,
        "expression_decoder_enabled": EXPRESSION_DECODER_ENABLED,
        "expression_decoder_alpha": EXPRESSION_DECODER_ALPHA,
    }
    config["run_model_parameters"] = adata.uns["run_model_parameters"]
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(_json_safe(config), f, indent=2, ensure_ascii=False)
    print(f"[OK] Updated run configuration saved to: {config_file}")
    print(f"[OK] Model training completed; embedding shape: {gemcst_embed.shape}")

    enhanced_expression_metrics = None
    if EXPRESSION_DECODER_ENABLED:
        print("\nStep 3b: Fitting frozen post-hoc expression decoder...")
        enhanced_expression, enhanced_expression_metrics = fit_posthoc_expression_decoder(
            adata.obsm["GeMCST_embed"],
            adata.X,
            alpha=EXPRESSION_DECODER_ALPHA,
        )
        adata.layers["enhanced_expression"] = enhanced_expression
        decoder_metrics_file = os.path.join(
            sample_dir, f"{SAMPLE_ID}_reconstruction_metrics.json"
        )
        with open(decoder_metrics_file, "w", encoding="utf-8") as f:
            json.dump(_json_safe(enhanced_expression_metrics), f, indent=2, ensure_ascii=False)
        if SAVE_ENHANCED_EXPRESSION_CSV:
            enhanced_csv_file = os.path.join(
                sample_dir, f"{SAMPLE_ID}_enhanced_expression.csv"
            )
            enhanced_df = pd.DataFrame(
                enhanced_expression, index=adata.obs_names, columns=adata.var_names
            )
            enhanced_df.to_csv(enhanced_csv_file, index_label="barcode")
            print(f"[OK] Enhanced expression CSV saved: {enhanced_csv_file}")
        print(f"[OK] Enhanced expression layer saved in result h5ad: adata.layers['enhanced_expression']")
        print(f"[OK] Decoder metrics saved: {decoder_metrics_file}")
    else:
        print("Post-hoc expression decoder disabled (EXPRESSION_DECODER_ENABLED=0)")

    print("\nStep 4: Spatial domain detection...")

    adata = gemcst._get_cluster_data(
        adata,
        n_domains=N_DOMAINS,
        priori=True,
        cluster_method=CLUSTER_METHOD,
        use_cancer_boundary_refine=USE_CBR_REFINEMENT,
        cbr_spatial_k=CBR_SPATIAL_K,
        cbr_max_change_ratio=CBR_MAX_CHANGE_RATIO,
        cbr_boundary_quantile=CBR_BOUNDARY_QUANTILE,
        cbr_low_confidence_quantile=CBR_LOW_CONFIDENCE_QUANTILE,
        cbr_margin=CBR_MARGIN,
        cbr_lambda_spatial=CBR_LAMBDA_SPATIAL,
        cbr_lambda_uni=CBR_LAMBDA_UNI,
        cbr_lambda_anchor=CBR_LAMBDA_ANCHOR,
        cbr_boundary_loss_weight=CBR_BOUNDARY_LOSS_WEIGHT,
        cbr_multiscale=CBR_MULTISCALE,
        cbr_compartment_aware=CBR_COMPARTMENT_AWARE,
        cbr_lambda_compartment=CBR_LAMBDA_COMPARTMENT,
    )
    print("[OK] Spatial domain detection completed")

    print("\nStep 5: Visualization and result saving...")

    print("  Generating spatial domain visualization...")
    output_prefix = os.path.join(sample_dir, f"{SAMPLE_ID}_domains")
    pdf_file, png_file = plot_spatial_domains(
        adata,
        color_key="GeMCST_refine_domain",
        out_prefix=output_prefix,
        title=f"Spatial Domains - {SAMPLE_ID}",
        profile=DATASET_PROFILE,
    )

    print(f"[OK] Spatial domain plot saved to: {pdf_file}")
    print(f"[OK] PNG quick-look saved: {png_file}")

    print("  Saving full results...")
    result_file = os.path.join(sample_dir, f"{SAMPLE_ID}_gemcst_result.h5ad")
    if SAVE_FULL_H5AD:
        adata.write(result_file)
        print(f"[OK] Full results saved to: {result_file}")
    else:
        print(f"Skip full h5ad save by default. Set SAVE_FULL_H5AD=1 to save: {result_file}")

    print("  Saving additional files...")

    clusters_df = pd.DataFrame({
        "barcode": adata.obs_names,
        "GeMCST_refine_domain": adata.obs["GeMCST_refine_domain"]
    })
    for col in [
        "GeMCST_refine_domain_anchor",
        "cbr_changed",
        "cbr_boundary_score",
        "cbr_boundary_evidence",
        "cbr_rna_gradient",
        "cbr_morphology_gradient",
        "cbr_anchor_confidence",
        "cbr_gain",
        "cbr_reason",
        "cbr_epithelial_score",
        "cbr_immune_score",
        "cbr_stroma_score",
        "cbr_hypoxia_score",
        "cbr_invasion_score",
        "cbr_mixed_compartment_score",
        "cbr_top_compartment",
    ]:
        if col in adata.obs:
            clusters_df[col] = adata.obs[col].values
    clusters_file = os.path.join(sample_dir, f"{SAMPLE_ID}_clusters.csv")
    clusters_df.to_csv(clusters_file, index=False)
    print(f"[OK] Cluster results saved to: {clusters_file}")

    embedding_file = os.path.join(sample_dir, f"{SAMPLE_ID}_embedding.npy")
    if SAVE_EMBEDDING_NPY:
        np.save(embedding_file, adata.obsm["GeMCST_embed"])
        print(f"[OK] Embedding matrix saved to: {embedding_file}")
    else:
        print(f"Skip embedding npy save. Set SAVE_EMBEDDING_NPY=1 to save: {embedding_file}")

    precluster_latent_file = os.path.join(sample_dir, f"{SAMPLE_ID}_precluster_latent.npy")
    if SAVE_EMBEDDING_NPY:
        np.save(precluster_latent_file, adata.obsm["GeMCST_precluster_latent"])
        print(f"[OK] Pre-clustering latent matrix saved to: {precluster_latent_file}")

    metadata_df = pd.DataFrame({
        "barcode": adata.obs_names,
        "GeMCST_refine_domain": adata.obs["GeMCST_refine_domain"]
    })
    for col in [
        "GeMCST_refine_domain_anchor",
        "cbr_changed",
        "cbr_boundary_score",
        "cbr_boundary_evidence",
        "cbr_rna_gradient",
        "cbr_morphology_gradient",
        "cbr_anchor_confidence",
        "cbr_epithelial_score",
        "cbr_immune_score",
        "cbr_stroma_score",
        "cbr_hypoxia_score",
        "cbr_invasion_score",
        "cbr_mixed_compartment_score",
        "cbr_top_compartment",
    ]:
        if col in adata.obs:
            metadata_df[col] = adata.obs[col].values
    metadata_file = os.path.join(sample_dir, f"{SAMPLE_ID}_metadata_aligned.csv")
    metadata_df.to_csv(metadata_file, index=False)
    print(f"[OK] Metadata saved to: {metadata_file}")

    print("\nStep 6: Performance summary...")

    n_clusters = len(adata.obs["GeMCST_refine_domain"].unique())
    cluster_sizes = adata.obs["GeMCST_refine_domain"].value_counts()

    print("Performance statistics:")
    print(f"   Detected spatial domain count: {n_clusters}")
    print(f"   Expected spatial domain count: {N_DOMAINS}")
    print("   Spots per domain:")
    for cluster, size in cluster_sizes.items():
        print(f"     - Domain {cluster}: {size} spots")

    print("\nDeep learning model training completed.")
    print(f"All results saved to: {sample_dir}")

    n_clusters = len(adata.obs["GeMCST_refine_domain"].unique())
    files = [
        f"{SAMPLE_ID}_gemcst_result.h5ad",
        f"{SAMPLE_ID}_domains.pdf",
        f"{SAMPLE_ID}_domains.png",
        f"{SAMPLE_ID}_clusters.csv",
        f"{SAMPLE_ID}_embedding.npy",
        f"{SAMPLE_ID}_metadata_aligned.csv",
        f"{SAMPLE_ID}_reconstruction_metrics.json"
    ]
    if SAVE_ENHANCED_EXPRESSION_CSV:
        files.append(f"{SAMPLE_ID}_enhanced_expression.csv")

    for file in files:
        file_path = os.path.join(sample_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024      
            print(f"   [OK] {file} ({file_size:.1f} KB)")
        else:
            print(f"   [MISSING] {file}")

    print("\nDeep learning model training script finished")
    print("\nPipeline summary:")
    print("   1. [OK] Image feature processing")
    print("   2. [OK] Deep learning model training")
    print("   3. [OK] Spatial domain detection and visualization")
    print("   4. [OK] Result saving and performance summary")


def run_evaluate():
    import os
    import re
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    from sklearn.metrics import (
        adjusted_rand_score, 
        normalized_mutual_info_score,
        silhouette_score,
        davies_bouldin_score
    )
    from sklearn.preprocessing import LabelEncoder
    from umap import UMAP
    from .utils import palette_for_dataset
    from .utils import select_precluster_umap_input
    from .utils import (
        configure_evaluation_style,
        should_show_h1_umap_metrics,
        spatial_panel_title,
    )

    configure_evaluation_style()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(BASE_DIR, "Results"))
    DATA_DIR = os.environ.get("DATA_DIR", "")
    SAMPLE_ID = os.environ.get("SAMPLE_ID", "151673")
    METADATA_PATH = os.environ.get("METADATA_PATH", os.path.join(BASE_DIR, "DLPFC", SAMPLE_ID, "metadata.tsv"))
    EVAL_DIR = os.environ.get("EVAL_DIR", os.path.join(RESULTS_DIR, f"{SAMPLE_ID}_evaluation"))
    os.makedirs(EVAL_DIR, exist_ok=True)
    TOP_MARKERS_PER_DOMAIN = int(os.environ.get("TOP_MARKERS_PER_DOMAIN", "50"))

    HEATMAP_MARKERS_PER_DOMAIN = 3
    GROUND_TRUTH_COL = os.environ.get("GROUND_TRUTH_COL", "Classification")
    MAX_SILHOUETTE_SAMPLES = int(os.environ.get("MAX_SILHOUETTE_SAMPLES", "5000"))
    DATASET_PROFILE = os.environ.get("DATASET_PROFILE", "auto")

    DOMAIN_PALETTE = palette_for_dataset(DATASET_PROFILE, SAMPLE_ID, DATA_DIR)
    '''DOMAIN_PALETTE_OLD = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
        "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#2f4b7c",
    ]
    '''

    def _median_nearest_neighbor_spacing(coords):
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] < 2:
            raise ValueError("At least two spatial coordinates are required")
        distances, _ = cKDTree(coords[:, :2]).query(coords[:, :2], k=2)
        nearest = distances[:, 1]
        valid = nearest[np.isfinite(nearest) & (nearest > 0)]
        if valid.size == 0:
            raise ValueError("Could not estimate a positive spot center spacing")
        return float(np.median(valid))

    def _spot_diameter(coords, fraction=0.65):
        return float(fraction) * _median_nearest_neighbor_spacing(coords)

    def _spot_fraction(profile=None, sample_id=None):
        profile_key = str(profile or "").strip().lower()
        sample_key = str(sample_id or "").strip().lower()
        if profile_key == "colon_cancer_visium_hd" or "colon_cancer_p1_032um" in sample_key:
            return 0.75
        return 0.65

    def _spot_alpha(profile=None, sample_id=None, with_image=True):
        profile_key = str(profile or "").strip().lower()
        sample_key = str(sample_id or "").strip().lower()
        if profile_key == "colon_cancer_visium_hd" or "colon_cancer_p1_032um" in sample_key:
            return 0.98
        if profile_key == "human_lung_cancer" or "human_lung_cancer" in sample_key:
            return 0.75
        return 0.68 if with_image else 0.74

    def _image_alpha(profile=None, sample_id=None, default=0.55):
        profile_key = str(profile or "").strip().lower()
        sample_key = str(sample_id or "").strip().lower()
        if profile_key == "colon_cancer_visium_hd" or "colon_cancer_p1_032um" in sample_key:
            return 0.45
        return float(default)

    def _domain_sort_key(value):
        text = str(value)
        try:
            return (0, float(text), text)
        except ValueError:
            parts = tuple(
                (0, int(part)) if part.isdigit() else (1, part.lower())
                for part in re.split(r"(\d+)", text)
            )
            return (1, parts, text)

    def _domain_categories(values):
        return sorted({str(value) for value in values}, key=_domain_sort_key)

    def _apply_domain_palette(adata, color_key):
        categories = _domain_categories(adata.obs[color_key].astype(str))
        adata.obs[color_key] = pd.Categorical(
            adata.obs[color_key].astype(str), categories=categories, ordered=True
        )
        palette = palette_for_dataset(DATASET_PROFILE, SAMPLE_ID, DATA_DIR)
        colors = [palette[i % len(palette)] for i in range(len(categories))]
        adata.uns[f"{color_key}_colors"] = colors
        return colors

    def _save_figure_pair(fig, output_prefix):
        pdf_file = f"{output_prefix}.pdf"
        png_file = f"{output_prefix}.png"
        fig.savefig(pdf_file, bbox_inches="tight", dpi=300)
        fig.savefig(png_file, bbox_inches="tight", dpi=300)
        return pdf_file, png_file

    def _label_priority(columns):
        priority = [
            GROUND_TRUTH_COL,
            "fine_annot_type",
            "annot_type",
            "Classification",
            "ground_truth",
            "label",
            "layer_guess",
            "annotation",
            "domain",
            "cluster",
        ]
        ordered = []
        seen = set()
        for col in priority:
            if col and col in columns and col not in seen:
                ordered.append(col)
                seen.add(col)
        return ordered

    def _valid_label_mask(values):
        labels = values.astype(str)
        invalid = labels.str.lower().isin({"undetermined", "unknown", "nan", "none", ""})
        return values.notna() & ~invalid

    def _print_label_counts(values, column_name):
        counts = values.astype(str).value_counts()
        print(f"Ground truth column used: {column_name}")
        print(f"Ground truth class counts: {counts.to_dict()}")

    def load_metadata(meta_path):

        if not meta_path or not os.path.exists(meta_path):
            return None

        sep = '\t' if meta_path.endswith('.tsv') else ','
        metadata = pd.read_csv(meta_path, sep=sep)

        if 'Row.names' in metadata.columns:
            metadata.set_index('Row.names', inplace=True)
        elif 'barcode' in metadata.columns:
            metadata.set_index('barcode', inplace=True)
        elif metadata.columns[0] == '' or metadata.iloc[:, 0].dtype == 'object':
            metadata.set_index(metadata.columns[0], inplace=True)

        label_candidates = _label_priority(metadata.columns)
        for col in label_candidates:
            if col in metadata.columns:
                metadata['ground_truth'] = metadata[col]
                print(f"Successfully matched label column: {col}")
                break

        return metadata

    def load_data():

        possible_paths = [

            os.path.join(RESULTS_DIR, SAMPLE_ID, f"{SAMPLE_ID}_gemcst_result.h5ad"),

            os.path.join(RESULTS_DIR, "Data", SAMPLE_ID, f"{SAMPLE_ID}_raw.h5ad"),
            os.path.join(RESULTS_DIR, SAMPLE_ID, f"{SAMPLE_ID}_raw.h5ad"),
        ]

        adata = None
        for path in possible_paths:
            if os.path.exists(path):
                adata = sc.read_h5ad(path)
                print(f"Successfully loaded data file: {path}")
                print(f"Data dimensions: {adata.n_obs} spots, {adata.n_vars} genes")
                break

        if adata is None:
            raise FileNotFoundError(
                f"Data file not found. Paths tried:\n" + 
                "\n".join(f"  - {p}" for p in possible_paths)
            )

        metadata = None
        if os.path.exists(METADATA_PATH):
            metadata = load_metadata(METADATA_PATH)
            if metadata is not None:
                print(f"Successfully loaded metadata: {metadata.shape[0]} samples")
        else:
            print(f"Metadata file not found: {METADATA_PATH}")

        required_fields = ["GeMCST_refine_domain", "GeMCST_embed"]
        missing_fields = []
        for field in required_fields:
            if field not in adata.obs.columns and field not in adata.obsm.keys():
                missing_fields.append(field)

        if missing_fields:

            if "_raw.h5ad" in path:
                raise ValueError(
                    f"The loaded file contains raw data and is missing GeMCST training result fields: {missing_fields}\n"
                    f"Please run run_model.py first to generate a complete results file:\n"
                    f"  {os.path.join(RESULTS_DIR, SAMPLE_ID, f'{SAMPLE_ID}_gemcst_result.h5ad')}\n"
                    f"Alternatively, run evaluation.py with the path to a complete results file"
                )
            else:
                raise ValueError(f"Missing required fields: {missing_fields}")

        return adata, metadata

    def align_barcodes(adata, metadata):

        if metadata is None:
            print("No metadata available; skipping barcode alignment")
            return adata, None

        print("\nAligning barcodes...")

        common_barcodes = list(set(adata.obs_names) & set(metadata.index))

        if len(common_barcodes) > 0:
            print(f"Direct match succeeded: {len(common_barcodes)} shared barcodes")
            adata = adata[common_barcodes].copy()
            metadata = metadata.loc[common_barcodes].copy()
            return adata, metadata

        print("Direct matching failed; trying fuzzy matching...")

        if 'x' in metadata.columns and 'y' in metadata.columns:
            def force_clean_id(x, y):
                try:
                    return f"{int(round(float(x)))}x{int(round(float(y)))}"
                except:
                    return "invalid"

            metadata['clean_id'] = metadata.apply(
                lambda row: force_clean_id(row['x'], row['y']), 
                axis=1
            )

            label_candidates = _label_priority(metadata.columns)
            label_col = label_candidates[0] if label_candidates else metadata.columns[0]
            label_map = dict(zip(metadata['clean_id'], metadata[label_col]))

            cleaned_obs_names = []
            for i in adata.obs_names:
                try:
                    parts = str(i).split('x')
                    if len(parts) == 2:
                        cleaned_id = force_clean_id(parts[0], parts[1])
                        cleaned_obs_names.append(cleaned_id)
                    else:
                        cleaned_obs_names.append(str(i))
                except:
                    cleaned_obs_names.append(str(i))

            adata.obs_names = cleaned_obs_names
            adata.obs['label'] = adata.obs_names.map(label_map)

            matched_count = adata.obs['label'].notna().sum()
            print(f"Number of successful matches: {matched_count}")

            if matched_count > 0:
                valid_mask = adata.obs['label'].notna() & (adata.obs['label'] != 'undetermined')
                adata_eval = adata[valid_mask].copy()

                if adata_eval.shape[0] > 0:
                    metadata_eval = metadata[metadata['clean_id'].isin(adata_eval.obs_names)].copy()
                    return adata_eval, metadata_eval

        raise ValueError("No shared barcodes found; unable to align")

    def _metadata_true_labels(metadata, valid_indices, true_label_col):
        metadata_eval = metadata.loc[valid_indices].copy()
        le_true = LabelEncoder()
        return le_true.fit_transform(metadata_eval[true_label_col].astype(str)), le_true

    def encode_labels(adata, metadata):
        """Use the final CBR labels directly and encode reference labels for metrics."""
        print("\nEncoding final CBR labels...")

        prediction_col = "GeMCST_refine_domain"
        if prediction_col not in adata.obs.columns:
            raise ValueError(f"Required final CBR label column {prediction_col!r} is missing")

        embeddings = np.asarray(adata.obsm["GeMCST_embed"]) if "GeMCST_embed" in adata.obsm else None

        true_label_col = next((col for col in _label_priority(adata.obs.columns) if col in adata.obs.columns), None)
        if true_label_col:
            valid_mask = _valid_label_mask(adata.obs[true_label_col])
            if valid_mask.sum() > 0:
                adata_eval = adata[valid_mask].copy()
                le_true = LabelEncoder()
                true_labels = le_true.fit_transform(adata_eval.obs[true_label_col].astype(str))
                pred_labels = LabelEncoder().fit_transform(adata_eval.obs[prediction_col].astype(str))
                eval_embeddings = adata_eval.obsm["GeMCST_embed"] if "GeMCST_embed" in adata_eval.obsm else None
                print(f"Using final CBR labels from: {prediction_col}")
                print(f"Ground-truth label column: {true_label_col}")
                print(f"Ground-truth class count: {len(le_true.classes_)}")
                print(f"Valid sample count: {len(true_labels)}")
                print(f"Final CBR domain count: {len(np.unique(pred_labels))}")
                return pred_labels, true_labels, eval_embeddings

        if metadata is not None:
            label_columns = _label_priority(metadata.columns)
            true_label_col = next((col for col in label_columns if col in metadata.columns), None)
            pseudo_reference = False
            if true_label_col is None:
                candidate_cols = [col for col in metadata.columns if col == "GraphBased" or re.fullmatch(r"KMeans_\d+", col)]
                if candidate_cols:
                    true_label_col = candidate_cols[0]
                    pseudo_reference = True
            if true_label_col:
                print(f"Detected {'pseudo-reference' if pseudo_reference else 'ground-truth'} label column: {true_label_col}")
                common_indices = adata.obs_names.intersection(metadata.index)
                if len(common_indices) > 0:
                    metadata_filtered = metadata.loc[common_indices].copy()
                    valid_mask = _valid_label_mask(metadata_filtered[true_label_col])
                    valid_indices = common_indices[valid_mask]
                    if len(valid_indices) > 0:
                        adata_eval = adata[valid_indices].copy()
                        true_labels, le_true = _metadata_true_labels(metadata, valid_indices, true_label_col)
                        pred_labels = LabelEncoder().fit_transform(adata_eval.obs[prediction_col].astype(str))
                        eval_embeddings = adata_eval.obsm["GeMCST_embed"] if "GeMCST_embed" in adata_eval.obsm else None
                        print(f"Using final CBR labels from: {prediction_col}")
                        print(f"Ground-truth class count: {len(le_true.classes_)}")
                        print(f"Valid sample count: {len(true_labels)}")
                        print(f"Final CBR domain count: {len(np.unique(pred_labels))}")
                        return pred_labels, true_labels, eval_embeddings

        pred_labels = LabelEncoder().fit_transform(adata.obs[prediction_col].astype(str))
        print(f"Using final CBR labels from: {prediction_col}")
        print(f"Final CBR domain count: {len(np.unique(pred_labels))}")
        print("[WARN] No usable ground-truth label column was found; ARI/NMI will be skipped.")
        return pred_labels, None, embeddings

    def calculate_metrics(pred_labels, true_labels, embeddings=None):

        print("\nCalculating evaluation metrics...")

        metrics = {}

        if true_labels is not None:
            metrics["ARI"] = adjusted_rand_score(true_labels, pred_labels)
            metrics["NMI"] = normalized_mutual_info_score(true_labels, pred_labels)
            print("Supervised metrics calculated")

        if embeddings is not None:
            sample_size = min(MAX_SILHOUETTE_SAMPLES, len(pred_labels))
            metrics["Silhouette"] = silhouette_score(
                embeddings,
                pred_labels,
                sample_size=sample_size,
                random_state=0,
            )
            metrics["Davies-Bouldin"] = davies_bouldin_score(embeddings, pred_labels)
            print(f"Unsupervised metrics calculated (Silhouette sample size={sample_size}）")

        return metrics

    def print_metrics(metrics):

        print("\n" + "="*60)
        print("Clustering performance evaluation results")
        print("="*60)

        for name, value in metrics.items():
            print(f"{name:20}: {value:.4f}")

        print("="*60)

        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])
        metrics_df.to_csv(os.path.join(EVAL_DIR, f"{SAMPLE_ID}_metrics.csv"))
        print(f"Metrics saved to: {os.path.join(EVAL_DIR, f'{SAMPLE_ID}_metrics.csv')}")

    def plot_marker_heatmap_clustered(work, markers, label_key, output_prefix, seed=0):
        """Plot a compact, domain-mean marker heatmap without modifying ``work``."""
        from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
        from matplotlib.colors import ListedColormap

        if label_key not in work.obs:
            raise KeyError(f"Missing cluster column: {label_key}")

        marker_domains = {}
        if isinstance(markers, pd.DataFrame):
            if "names" not in markers:
                raise KeyError("Marker table must contain a 'names' column")
            marker_table = markers.copy()
            if "group" in marker_table:
                marker_table["_domain"] = marker_table["group"].astype(str)
                max_display_markers = min(
                    HEATMAP_MARKERS_PER_DOMAIN,
                    max(1, 30 // len(_domain_categories(work.obs[label_key].astype(str)))),
                )
                marker_table = marker_table.sort_values(
                    ["_domain", "pvals_adj", "scores"],
                    ascending=[True, True, False],
                )
                marker_table = marker_table.groupby("_domain", group_keys=False, observed=False).head(
                    max_display_markers
                )
                marker_domains = dict(zip(marker_table["names"].astype(str), marker_table["_domain"]))
            genes = marker_table["names"].astype(str).drop_duplicates().tolist()
        else:
            genes = [str(g) for g in markers]

        genes = [gene for gene in genes if gene in work.var_names]
        if not genes:
            print("No marker genes available for the heatmap; skipping")
            return
        original_genes = list(genes)

        labels = work.obs[label_key].astype(str)
        categories = _domain_categories(labels)
        base_categories = list(categories)
        expression = work[:, genes].X
        if hasattr(expression, "toarray"):
            expression = expression.toarray()
        expression = np.log1p(np.clip(np.asarray(expression, dtype=float), 0, None))

        domain_means = np.vstack([
            expression[labels.to_numpy() == domain].mean(axis=0)
            for domain in categories
        ])
        gene_sd = domain_means.std(axis=0, keepdims=True)
        z_means = (domain_means - domain_means.mean(axis=0, keepdims=True)) / np.where(
            gene_sd > 0, gene_sd, 1.0
        )

        linkage_matrix = None
        if len(categories) >= 2:
            linkage_matrix = linkage(z_means, method="average", metric="euclidean")
            domain_order = leaves_list(linkage_matrix)
            z_means = z_means[domain_order]
            categories = [categories[index] for index in domain_order]

        if marker_domains:
            genes = [
                gene for domain in categories for gene in genes
                if marker_domains.get(gene) == domain
            ] + [gene for gene in genes if gene not in marker_domains]
            selected_indices = [original_genes.index(gene) for gene in genes]
            z_means = z_means[:, selected_indices]
        else:
            selected_indices = np.argsort(np.argmax(z_means, axis=0))
            z_means = z_means[:, selected_indices]
            genes = [genes[index] for index in selected_indices]

        domain_palette = [
            "#1976B5", "#F28E2B", "#159A72", "#D7192C", "#7A2CBF", "#8C564B",
            "#D85AA6", "#B5B43C", "#159AC0", "#A9C7E6", "#F4B183", "#65B96E",
            "#EF8E8E", "#B29BC8", "#B88779", "#E7A0B9", "#C7C66B", "#8AC6D9",
            "#B84C4C", "#9A6500", "#4E79A7", "#59A14F", "#EDC948", "#76B7B2",
            "#FF9DA7", "#9C755F", "#BAB0AB", "#003F5C", "#58508D", "#BC5090",
        ]
        if len(base_categories) > len(domain_palette):
            raise ValueError("Marker heatmap supports at most 30 domains without repeating colors")

        domain_colors = [domain_palette[base_categories.index(domain)] for domain in categories]

        n_domains = len(categories)
        x_centers = np.arange(n_domains) * 10 + 5
        x_max = n_domains * 10
        fig = plt.figure(figsize=(10, 7), dpi=180)
        grid = fig.add_gridspec(5, 1, height_ratios=[1.35, 0.22, 0.32, 5.6, 0.34], hspace=0.10)
        ax_tree = fig.add_subplot(grid[0, 0])
        ax_domain_bar = fig.add_subplot(grid[1, 0], sharex=ax_tree)
        ax_domain_labels = fig.add_subplot(grid[2, 0], sharex=ax_tree)
        ax_heat = fig.add_subplot(grid[3, 0], sharex=ax_tree)
        ax_expression_bar = fig.add_subplot(grid[4, 0])

        if linkage_matrix is not None:
            dendrogram(linkage_matrix, ax=ax_tree, no_labels=True, color_threshold=0, above_threshold_color="black")
        ax_tree.set_xlim(0, x_max)
        ax_tree.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_tree.set_yticks([])
        for spine in ax_tree.spines.values():
            spine.set_visible(False)

        domain_cmap = ListedColormap(domain_colors)
        ax_domain_bar.imshow(
            np.arange(n_domains)[None, :], aspect="auto", interpolation="nearest",
            cmap=domain_cmap, vmin=0, vmax=max(n_domains - 1, 1), extent=(0, x_max, 0, 1),
        )
        ax_domain_bar.set_yticks([])
        ax_domain_bar.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for spine in ax_domain_bar.spines.values():
            spine.set_visible(False)

        ax_domain_labels.set_xlim(0, x_max)
        ax_domain_labels.set_ylim(0, 1)
        ax_domain_labels.set_yticks([])
        ax_domain_labels.set_xticks(x_centers)
        ax_domain_labels.set_xticklabels(categories, rotation=0, fontsize=8)
        ax_domain_labels.tick_params(axis="x", length=0, pad=0)
        for spine in ax_domain_labels.spines.values():
            spine.set_visible(False)

        image = ax_heat.imshow(
            z_means.T, aspect="auto", interpolation="nearest", cmap="viridis", vmin=-2.5, vmax=2.5,
            extent=(0, x_max, len(genes), 0),
        )
        ax_heat.set_xlim(0, x_max)
        ax_heat.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_heat.set_yticks(np.arange(len(genes)) + 0.5)
        ax_heat.set_yticklabels(genes, fontsize=8)
        ax_heat.tick_params(axis="y", length=0, pad=3, labelright=False, right=False)
        ax_heat.set_ylabel("Marker genes", fontsize=9)

        colorbar = fig.colorbar(image, cax=ax_expression_bar, orientation="horizontal")
        colorbar.set_label("Gene-wise z-score of log1p expression", fontsize=8, labelpad=2)
        colorbar.ax.tick_params(labelsize=7, length=2)
        fig.suptitle(f"Marker-gene expression across {n_domains} spatial domains", fontsize=12, y=0.985)
        fig.subplots_adjust(left=0.18, right=0.93, top=0.91, bottom=0.10)
        fig.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(f"{output_prefix}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Cluster marker heatmap saved to: {output_prefix}.png, {output_prefix}.pdf")

    def marker_gene_analysis(adata, layer=None, suffix=""):
        """Marker analysis on observed X or an explicit expression layer."""
        work = adata.copy()
        label = "enhanced" if layer else "observed"
        print(f"\nMarker gene analysis ({label})...")
        sc.tl.dendrogram(work, groupby="GeMCST_refine_domain", use_rep="GeMCST_embed")
        work.obsm["X_pca"] = work.obsm["GeMCST_embed"][:, :50]
        rank_layer = None
        if layer is None:
            sc.pp.log1p(work)
        else:
            if layer not in work.layers:
                raise KeyError(f"Expression layer not found: {layer}")
            rank_layer = "_marker_expression"
            values = work.layers[layer]
            if hasattr(values, "toarray"):
                values = values.toarray()
            work.layers[rank_layer] = np.log1p(np.asarray(values, dtype=np.float32))
        sc.tl.rank_genes_groups(
            work,
            groupby="GeMCST_refine_domain",
            method="wilcoxon",
            layer=rank_layer,
            use_raw=False,
        )

        marker_genes = sc.get.rank_genes_groups_df(work, group=None)
        if "group" in marker_genes.columns and TOP_MARKERS_PER_DOMAIN > 0:
            marker_genes = (
                marker_genes
                .sort_values(["group", "pvals_adj", "scores"], ascending=[True, True, False])
                .groupby("group", group_keys=False, observed=False)
                .head(TOP_MARKERS_PER_DOMAIN)
            )
        output_stem = f"{SAMPLE_ID}_marker_genes{suffix}"
        marker_genes.to_csv(os.path.join(EVAL_DIR, f"{output_stem}.csv"), index=False)

        print(f"Marker genes saved to: {os.path.join(EVAL_DIR, f'{output_stem}.csv')}")

        plot_marker_heatmap_clustered(
            work,
            marker_genes,
            label_key="GeMCST_refine_domain",
            output_prefix=os.path.join(EVAL_DIR, f"{SAMPLE_ID}_marker_heatmap{suffix}"),
        )

    def plot_umap(adata, save_dir, metrics=None):

        print("\nGenerating UMAP visualization...")

        umap_embedding = UMAP(
            n_components=2,
            n_neighbors=int(os.environ.get("UMAP_N_NEIGHBORS", "50")),
            min_dist=float(os.environ.get("UMAP_MIN_DIST", "0.20")),
            spread=float(os.environ.get("UMAP_SPREAD", "1.0")),
            metric=os.environ.get("UMAP_METRIC", "cosine"),
            random_state=0,
            init="spectral",
        ).fit_transform(select_precluster_umap_input(adata))
        adata.obsm["X_umap"] = umap_embedding
        palette = _apply_domain_palette(adata, "GeMCST_refine_domain")

        labels = adata.obs["GeMCST_refine_domain"].astype(str).to_numpy()
        categories = _domain_categories(labels)
        fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=180)
        point_size = 3.2 if adata.n_obs > 3000 else 5.0
        for i, category in enumerate(categories):
            mask = labels == category
            ax.scatter(
                umap_embedding[mask, 0], umap_embedding[mask, 1],
                s=point_size, c=palette[i % len(palette)], alpha=0.86,
                linewidths=0, edgecolors="none", rasterized=True, label=str(category),
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
        ax.set_title(
            spatial_panel_title("GeMCST", metrics, show_metrics=should_show_h1_umap_metrics(SAMPLE_ID)),
            fontsize=13,
            pad=10,
        )
        ax.margins(0.03)
        legend_cols = min(5, max(1, int(np.ceil(len(categories) / 4))))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.035),
                frameon=False, fontsize=8, ncol=legend_cols,
                columnspacing=1.0, handletextpad=0.3, markerscale=1.4)
        fig.subplots_adjust(bottom=0.17)
        pdf_file, png_file = _save_figure_pair(
            fig, os.path.join(EVAL_DIR, f"{SAMPLE_ID}_umap")
        )
        plt.close(fig)
        print(f"UMAP plot saved to: {pdf_file}, {png_file}")

        return umap_embedding

    def plot_spatial(adata, save_dir, metrics=None):

        print("\nGenerating spatial distribution plot...")

        if "spatial" not in adata.obsm:
            raise ValueError("Missing adata.obsm['spatial']")
        spot_size = _spot_diameter(
            adata.obsm["spatial"],
            fraction=_spot_fraction(profile=DATASET_PROFILE, sample_id=SAMPLE_ID),
        )
        palette = _apply_domain_palette(adata, "GeMCST_refine_domain")

        sc.pl.spatial(
            adata,
            color=["GeMCST_refine_domain"],
            palette=palette,
            frameon=False,
            spot_size=spot_size,
            alpha_img=_image_alpha(DATASET_PROFILE, SAMPLE_ID),
            alpha=_spot_alpha(DATASET_PROFILE, SAMPLE_ID, with_image=True),
            linewidth=0,
            edgecolor="none",
            legend_loc="right margin",
            legend_fontsize=8,
            title=spatial_panel_title(
                "GeMCST", metrics, show_metrics=should_show_h1_umap_metrics(SAMPLE_ID)
            ),
            show=False
        )

        fig = plt.gcf()
        pdf_file, png_file = _save_figure_pair(
            fig, os.path.join(EVAL_DIR, f"{SAMPLE_ID}_spatial")
        )
        plt.close(fig)
        print(f"Spatial distribution plot saved to: {pdf_file}, {png_file}")

    def plot_ground_truth_spatial(adata, metadata, save_dir):
        """Plot the manually annotated spatial domains when metadata is available."""
        if "spatial" not in adata.obsm:
            print("Spatial coordinates not found; skipping the manually annotated spatial domain plot")
            return None

        label_col = None
        if metadata is not None:
            candidates = _label_priority(metadata.columns)
            label_col = next((col for col in candidates if col in metadata.columns), None)
            if label_col is None:
                excluded = {"x", "y", "pxl_row_in_fullres", "pxl_col_in_fullres", "clean_id", "barcode"}
                fallback = []
                for col in metadata.columns:
                    if str(col).lower() in excluded:
                        continue
                    values = metadata[col].dropna()
                    if 1 < values.nunique() <= 50:
                        fallback.append(col)
                if fallback:
                    label_col = fallback[0]
                    print(f"No standard label column matched; using low-cardinality candidate column: {label_col}")
            if label_col is not None:
                aligned = metadata.reindex(adata.obs_names)[label_col]

                if int(_valid_label_mask(aligned).sum()) == 0 and "clean_id" in metadata.columns:
                    clean_map = metadata.drop_duplicates("clean_id").set_index("clean_id")[label_col]
                    aligned = pd.Series(adata.obs_names, index=adata.obs_names).map(clean_map)
                adata.obs["ground_truth"] = aligned.to_numpy()
        if label_col is None and "ground_truth" in adata.obs.columns:
            label_col = "ground_truth"
        if label_col is not None and int(_valid_label_mask(adata.obs.get("ground_truth", pd.Series(index=adata.obs_names))).sum()) == 0:
            if "label" in adata.obs.columns:
                adata.obs["ground_truth"] = adata.obs["label"].to_numpy()
                label_col = "label"
        if label_col is None:
            print("Manually annotated label column not found; skipping the manually annotated spatial domain plot")
            return None

        valid = _valid_label_mask(adata.obs["ground_truth"])
        if int(valid.sum()) == 0:
            print("Manually annotated labels contain no valid values; skipping the manually annotated spatial domain plot")
            return None
        work = adata[valid].copy()
        work.obs["ground_truth"] = work.obs["ground_truth"].astype(str)
        cats = _domain_categories(work.obs["ground_truth"])
        codes = pd.Categorical(work.obs["ground_truth"], categories=cats, ordered=True).codes
        coords = np.asarray(work.obsm["spatial"], dtype=float)
        spatial = work.uns.get("spatial", {})
        image, scale = None, 1.0
        if isinstance(spatial, dict) and spatial:
            library = spatial[next(iter(spatial))]
            if isinstance(library, dict):
                images = library.get("images", {})
                image = images.get("hires", images.get("lowres"))
                scale = float(library.get("scalefactors", {}).get("tissue_hires_scalef", 1.0))
        if image is not None:
            image = np.asarray(image)
            if image.ndim < 2:
                image = None
            elif np.nanmax(coords[:, 0]) <= image.shape[1] * 1.2 and np.nanmax(coords[:, 1]) <= image.shape[0] * 1.2:
                scale = 1.0
        plot_coords = coords * scale
        with_image = image is not None
        fig_w, fig_h = (7.0, max(5.5, 7.0 * image.shape[0] / max(image.shape[1], 1))) if with_image else (7.0, 6.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
        if with_image:
            ax.imshow(image, alpha=_image_alpha(DATASET_PROFILE, SAMPLE_ID))
            ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0)
        else:
            ax.set_xlim(np.nanmin(plot_coords[:, 0]), np.nanmax(plot_coords[:, 0]))
            ax.set_ylim(np.nanmax(plot_coords[:, 1]), np.nanmin(plot_coords[:, 1]))
        ax.set_aspect("equal"); fig.canvas.draw()
        diameter = _spot_diameter(plot_coords, fraction=_spot_fraction(DATASET_PROFILE, SAMPLE_ID))
        origin = ax.transData.transform((0.0, 0.0))
        x_extent = ax.transData.transform((float(diameter), 0.0))
        y_extent = ax.transData.transform((0.0, float(diameter)))
        diameter_px = min(abs(x_extent[0] - origin[0]), abs(y_extent[1] - origin[1]))
        spot_size = max(diameter_px * 72.0 / fig.dpi, 0.5) ** 2
        palette = palette_for_dataset(DATASET_PROFILE, SAMPLE_ID, DATA_DIR)
        ax.scatter(plot_coords[:, 0], plot_coords[:, 1], c=[palette[i % len(palette)] for i in codes],
                s=spot_size, linewidths=0.0, edgecolors="none",
                alpha=_spot_alpha(DATASET_PROFILE, SAMPLE_ID, with_image))
        ax.axis("off"); ax.set_title(spatial_panel_title("manual annotation"), fontsize=13, pad=10)
        handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[i % len(palette)],
                            markeredgecolor="none", markersize=7, label=str(cat)) for i, cat in enumerate(cats)]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                frameon=False, fontsize=8, ncol=1 if len(cats) <= 10 else 2)
        pdf_file, png_file = _save_figure_pair(
            fig, os.path.join(EVAL_DIR, f"{SAMPLE_ID}_ground_truth_spatial")
        )
        plt.close(fig)
        print(f"Manually annotated spatial domain plot saved to: {pdf_file}, {png_file}")
        return pdf_file, png_file

    def main():
        print("Complete evaluation and downstream analysis of spatial transcriptomics")
        print("="*60)

        try:
            adata, metadata = load_data()
            adata, metadata = align_barcodes(adata, metadata)
            pred_labels, true_labels, filtered_embeddings = encode_labels(adata, metadata)
            metrics = calculate_metrics(
                pred_labels=pred_labels,
                true_labels=true_labels,
                embeddings=filtered_embeddings
            )
            print_metrics(metrics)
            plot_umap(adata, RESULTS_DIR, metrics=metrics)
            plot_spatial(adata, RESULTS_DIR, metrics=metrics)
            plot_ground_truth_spatial(adata, metadata, RESULTS_DIR)
            marker_gene_analysis(adata, layer=None, suffix="")
            if os.environ.get("EVALUATE_ENHANCED_EXPRESSION", "1").strip().lower() in {"1", "true", "yes", "on"}:
                if "enhanced_expression" in adata.layers:
                    marker_gene_analysis(adata, layer="enhanced_expression", suffix="_enhanced")
                    print("Enhanced expression marker analysis completed; clustering metrics remain embedding-based.")
                else:
                    print("No enhanced_expression layer found; skipping enhanced marker analysis.")
            print("\nAnalysis complete!")

        except Exception as e:
            print(f"\nAnalysis failed: {e}")
            return 1

        return 0

    exit(main())
