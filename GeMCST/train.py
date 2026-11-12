import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from typing import Optional
import scanpy as sc
import pandas as pd
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import connected_components
import os
import scipy.sparse as sp

class train():
    BEST_COMMUNITY_PRESET = {
        "community_match_ratio": 0.58,
        "boundary_quantile": 0.76,
        "safe_quantile": 0.45,
        "strong_pos_per_node": 6,
        "weak_pos_per_node": 6,
        "neg_per_node": 8,
        "neg_boundary_boost": 1.55,
        "weak_pos_weight": 0.55,
    }

    def __init__(self,
                processed_data: np.ndarray,
                graph_dict: dict,
                model: nn.Module,
                pre_epochs: int = 200,
                epochs: int = 200,
                corrupt: float = 0.001,
                lr: float = 5e-4,
                weight_decay: float = 1e-4,
                domains: np.ndarray = None,
                kl_weight: float = 100,
                mse_weight: float = 10,
                bce_kld_weight: float = 0.1,
                domain_weight: float = 1,
                use_gpu: bool = True,
                coords: Optional[np.ndarray] = None,
                ):
        """
        Initialize GeMCST Trainer
        
        Parameters:
        -----------
        processed_data : np.ndarray
            Processed input data [n_spots, n_features]
        graph_dict : dict
            Dictionary containing graph information with keys:
            - 'adj_norm': Normalized adjacency matrix
            - 'adj_label': Original adjacency labels
            - 'norm_value': Normalization factor
        model : nn.Module
            GeMCST model instance
        pre_epochs : int
            Number of pretraining epochs (default: 200)
        epochs : int
            Number of training epochs (default: 200)
        corrupt : float
            Corruption rate for masking noise (default: 0.001)
        lr : float
            Learning rate (default: 5e-4)
        weight_decay : float
            Weight decay for optimizer (default: 1e-4)
        domains : np.ndarray, optional
            Domain labels for adversarial training (default: None)
        kl_weight : float
            Weight for KL divergence loss (default: 100)
        mse_weight : float
            Weight for MSE reconstruction loss (default: 10)
        bce_kld_weight : float
            Weight for BCE and KLD losses (default: 0.1)
        domain_weight : float
            Weight for domain classification loss (default: 1)
        use_gpu : bool
            Whether to use GPU if available (default: True)
        """
                    
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self.processed_data = processed_data
        self.data = torch.FloatTensor(processed_data).to(self.device)
        self.adj = self._prepare_graph_input(graph_dict)
        self.adj_label = graph_dict['adj_label'].to(self.device)
        self.norm = graph_dict['norm_value']
        self.num_spots = self.data.shape[0]
        self.graph_dict = graph_dict

        if coords is not None:
            self.coords = torch.FloatTensor(coords).to(self.device)
        else:
            self.coords = None

        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )

        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.corrupt = corrupt
        self.kl_weight = kl_weight
        self.mse_weight = mse_weight
        self.bce_kld_weight = bce_kld_weight
        self.domain_weight = domain_weight

        self.dec_tol = 0                                           
        self.q_stride = 20                                             
        self.use_sri_clr = os.environ.get("SRI_CLR_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.soft_potts_enabled = os.environ.get("SOFT_POTTS_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.soft_potts_eta = float(os.environ.get("SOFT_POTTS_ETA", "0.25"))
        self.soft_potts_lambda = float(os.environ.get("SOFT_POTTS_LAMBDA", "0.05"))
        self.soft_potts_iters = int(os.environ.get("SOFT_POTTS_ITERS", "6"))
        self.soft_potts_lr = float(os.environ.get("SOFT_POTTS_LR", "0.35"))
        self.soft_potts_max_edges = int(os.environ.get("SOFT_POTTS_MAX_EDGES", "40000"))
        self.soft_potts_pairwise = os.environ.get("SOFT_POTTS_PAIRWISE", "pcd").strip().lower()
        self.soft_potts_edges = self._prepare_soft_potts_edges(graph_dict)
        self.sri_clr_weight = float(os.environ.get("SRI_CLR_WEIGHT", "0.06"))
        self.sri_clr_margin = float(os.environ.get("SRI_CLR_MARGIN", "0.75"))
        self.sri_clr_warmup_ratio = float(os.environ.get("SRI_CLR_WARMUP_RATIO", "0.30"))
        fixed_warmup_epoch = os.environ.get("SRI_CLR_WARMUP_EPOCH", "").strip()
        self.sri_clr_warmup_epoch = int(fixed_warmup_epoch) if fixed_warmup_epoch else None
        self.sri_clr_max_pairs = int(os.environ.get("SRI_CLR_MAX_PAIRS", "4096"))
        self.sri_strong_pos_pairs = None
        self.sri_weak_pos_pairs = None
        self.sri_neg_pairs = None
        self.rna_community_labels = None
        self.rna_community_stable_mask = None
        self.rna_boundary_mask = None
        self.rna_safe_mask = None
        preset_enabled = os.environ.get("RNA_USE_BEST_COMMUNITY_PRESET", "1").strip().lower() in {"1", "true", "yes", "on"}
        preset = self.BEST_COMMUNITY_PRESET if preset_enabled else {}
        self.community_match_ratio = float(os.environ.get("RNA_COMMUNITY_MATCH_RATIO", str(preset.get("community_match_ratio", 0.56))))
        self.boundary_quantile = float(os.environ.get("RNA_BOUNDARY_QUANTILE", str(preset.get("boundary_quantile", 0.76))))
        self.safe_quantile = float(os.environ.get("RNA_SAFE_QUANTILE", str(preset.get("safe_quantile", 0.45))))
        self.strong_pos_per_node = int(os.environ.get("RNA_STRONG_POS_PER_NODE", str(preset.get("strong_pos_per_node", 6))))
        self.weak_pos_per_node = int(os.environ.get("RNA_WEAK_POS_PER_NODE", str(preset.get("weak_pos_per_node", 4))))
        self.neg_per_node = int(os.environ.get("RNA_NEG_PER_NODE", str(preset.get("neg_per_node", 8))))
        self.neg_boundary_boost = float(os.environ.get("RNA_NEG_BOUNDARY_BOOST", str(preset.get("neg_boundary_boost", 1.80))))
        self.weak_pos_weight = float(os.environ.get("RNA_WEAK_POS_WEIGHT", str(preset.get("weak_pos_weight", 0.40))))
        self._sri_debug_printed = False
        self._sri_active_printed = False
        self._sri_running_loss = 0.0
        self._sri_running_steps = 0
        self._prepare_sri_clr_prior()

        self.domains = None
        if domains is not None:
            self.domains = torch.from_numpy(domains).to(self.device)

    def _prepare_soft_potts_edges(self, graph_dict: dict):
        adj_sp = None
        if "views" in graph_dict and "expression" in graph_dict["views"]:
            expr_graph = graph_dict["views"]["expression"]
            adj_sp = expr_graph.get("local_adj_sp", expr_graph.get("adj_sp"))
        elif "adj_sp" in graph_dict:
            adj_sp = graph_dict["adj_sp"]
        if adj_sp is None:
            return None

        adj_sp = self._symmetrize_adj(adj_sp)
        rows, cols = adj_sp.nonzero()
        if rows.size == 0:
            return None
        keep = rows < cols
        rows = rows[keep]
        cols = cols[keep]
        if rows.size == 0:
            return None
        if rows.size > self.soft_potts_max_edges:
            rng = np.random.default_rng(0)
            take = rng.choice(rows.size, size=self.soft_potts_max_edges, replace=False)
            rows = rows[take]
            cols = cols[take]
        return (
            torch.as_tensor(rows.astype(np.int64), device=self.device),
            torch.as_tensor(cols.astype(np.int64), device=self.device),
        )

    @staticmethod
    def _collision_cross_entropy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -torch.log((pred * target).sum(dim=1).clamp_min(1e-8)).mean()

    def _soft_potts_pairwise_loss(self, y_prob: torch.Tensor) -> torch.Tensor:
        if self.soft_potts_edges is None:
            return torch.zeros((), device=self.device, dtype=y_prob.dtype)
        rows, cols = self.soft_potts_edges
        y_i = y_prob.index_select(0, rows)
        y_j = y_prob.index_select(0, cols)
        if self.soft_potts_pairwise == "pnq":
            y_i_n = F.normalize(y_i, p=2, dim=1)
            y_j_n = F.normalize(y_j, p=2, dim=1)
            return 0.5 * (y_i_n - y_j_n).pow(2).sum(dim=1).mean()
        sim = F.normalize(y_i, p=2, dim=1).mul(F.normalize(y_j, p=2, dim=1)).sum(dim=1)
        return -torch.log(sim.clamp_min(1e-8)).mean()

    def _update_soft_pseudolabels(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.detach().clamp_min(1e-8)
        if (not self.soft_potts_enabled) or self.soft_potts_edges is None:
            return self.model.target_distribution(sigma)

        y_logits = sigma.log().detach().clone().requires_grad_(True)
        optimizer_y = optim.Adam([y_logits], lr=self.soft_potts_lr)
        for _ in range(max(1, self.soft_potts_iters)):
            optimizer_y.zero_grad()
            y_prob = torch.softmax(y_logits, dim=1)
            loss_unary = self._collision_cross_entropy(y_prob, sigma)
            loss_pairwise = self._soft_potts_pairwise_loss(y_prob)
            loss_y = self.soft_potts_eta * loss_unary + self.soft_potts_lambda * loss_pairwise
            loss_y.backward()
            optimizer_y.step()
        return torch.softmax(y_logits.detach(), dim=1)

    def _prepare_graph_input(self, graph_dict: dict):
        """Move either a single graph or the tri-view graph bundle to the target device."""
        if "views" not in graph_dict:
            return graph_dict["adj_norm"].to(self.device)

        graph_input = {
            "views": {
                view_name: view_graph["adj_norm"].to(self.device)
                for view_name, view_graph in graph_dict["views"].items()
            }
        }
        if "edge_view_data" in graph_dict:
            graph_input["edge_view_data"] = torch.FloatTensor(
                graph_dict["edge_view_data"]
            ).to(self.device)
        return graph_input

    def _sample_pairs(self, rows: np.ndarray, cols: np.ndarray):
        if rows.size == 0:
            return None
        pair_count = rows.shape[0]
        if pair_count > self.sri_clr_max_pairs:
            rng = np.random.default_rng(0)
            keep = rng.choice(pair_count, size=self.sri_clr_max_pairs, replace=False)
            rows = rows[keep]
            cols = cols[keep]
        pairs = np.stack([rows, cols], axis=1).astype(np.int64)
        return torch.as_tensor(pairs, device=self.device)

    @staticmethod
    def _symmetrize_adj(adj: sp.spmatrix) -> sp.csr_matrix:
        adj = adj.tocsr().astype(np.int8)
        adj = adj.maximum(adj.T)
        adj.setdiag(0)
        adj.eliminate_zeros()
        return adj

    def _build_rna_communities(self, expr_adj: sp.spmatrix) -> np.ndarray:
        expr_adj = self._symmetrize_adj(expr_adj)
        n_nodes = expr_adj.shape[0]
        labels = None
        try:
            adata = sc.AnnData(np.zeros((n_nodes, 1), dtype=np.float32))
            adata.obsp["connectivities"] = expr_adj.astype(np.float32)
            adata.obsp["distances"] = expr_adj.astype(np.float32)
            adata.uns["neighbors"] = {
                "connectivities_key": "connectivities",
                "distances_key": "distances",
            }
            res = float(os.environ.get("RNA_COMMUNITY_RESOLUTION", "1.0"))
            method = os.environ.get("RNA_COMMUNITY_METHOD", "leiden").strip().lower()
            if method == "louvain":
                sc.tl.louvain(adata, resolution=res, random_state=0)
                labels = pd.Categorical(adata.obs["louvain"]).codes.astype(np.int64)
            else:
                sc.tl.leiden(adata, resolution=res, random_state=0)
                labels = pd.Categorical(adata.obs["leiden"]).codes.astype(np.int64)
        except Exception:
            _, labels = connected_components(expr_adj, directed=False, connection="weak")
            labels = labels.astype(np.int64)
        if labels is None or labels.shape[0] != n_nodes:
            labels = np.arange(n_nodes, dtype=np.int64)
        return labels

    def _community_support(self, adj: sp.spmatrix, labels: np.ndarray) -> np.ndarray:
        adj = self._symmetrize_adj(adj)
        n = adj.shape[0]
        support = np.zeros(n, dtype=np.float32)
        indptr = adj.indptr
        indices = adj.indices
        for i in range(n):
            start, end = indptr[i], indptr[i + 1]
            neigh = indices[start:end]
            if neigh.size == 0:
                support[i] = 0.0
                continue
            same = np.sum(labels[neigh] == labels[i])
            support[i] = same / float(neigh.size)
        return support

    def _build_pairs_from_masks(self, adj: sp.spmatrix, labels: np.ndarray, node_mask: np.ndarray, pair_limit: int, mode: str):
        adj = self._symmetrize_adj(adj)
        rows, cols = adj.nonzero()
        if rows.size == 0:
            return None
        if mode == "same":
            keep = labels[rows] == labels[cols]
        elif mode == "diff":
            keep = labels[rows] != labels[cols]
        else:
            keep = np.ones(rows.shape[0], dtype=bool)
        if node_mask is not None:
            keep = keep & node_mask[rows] & node_mask[cols]
        rows = rows[keep]
        cols = cols[keep]
        if rows.size == 0:
            return None
        if rows.size > pair_limit:
            rng = np.random.default_rng(0)
            keep_idx = rng.choice(rows.size, size=pair_limit, replace=False)
            rows = rows[keep_idx]
            cols = cols[keep_idx]
        return torch.as_tensor(np.stack([rows, cols], axis=1).astype(np.int64), device=self.device)

    def _build_negative_pairs(
        self,
        adj: sp.spmatrix,
        labels: np.ndarray,
        boundary_mask: np.ndarray,
        stable_mask: np.ndarray,
        pair_limit: int,
    ):
        adj = self._symmetrize_adj(adj)
        rows, cols = adj.nonzero()
        if rows.size == 0:
            return None
        diff = labels[rows] != labels[cols]
        boundary_touch = boundary_mask[rows] | boundary_mask[cols]
        stable_touch = stable_mask[rows] & stable_mask[cols]

        hard_keep = diff & boundary_touch
        fallback_keep = diff & stable_touch
        keep = hard_keep | fallback_keep
        rows = rows[keep]
        cols = cols[keep]
        if rows.size == 0:
            rows, cols = adj.nonzero()
            keep = labels[rows] != labels[cols]
            rows = rows[keep]
            cols = cols[keep]
        if rows.size == 0:
            return None
        if rows.size > pair_limit:
            rng = np.random.default_rng(1)
            hard_mask = boundary_mask[rows] | boundary_mask[cols]
            hard_idx = np.where(hard_mask)[0]
            soft_idx = np.where(~hard_mask)[0]
            hard_quota = min(hard_idx.size, int(round(pair_limit * self.neg_boundary_boost / (1.0 + self.neg_boundary_boost))))
            hard_quota = max(hard_quota, min(hard_idx.size, pair_limit // 2))
            soft_quota = max(0, pair_limit - hard_quota)
            chosen = []
            if hard_idx.size > 0 and hard_quota > 0:
                chosen.append(rng.choice(hard_idx, size=min(hard_quota, hard_idx.size), replace=False))
            if soft_idx.size > 0 and soft_quota > 0:
                chosen.append(rng.choice(soft_idx, size=min(soft_quota, soft_idx.size), replace=False))
            if chosen:
                keep_idx = np.concatenate(chosen)
                if keep_idx.size > pair_limit:
                    keep_idx = rng.choice(keep_idx, size=pair_limit, replace=False)
                rows = rows[keep_idx]
                cols = cols[keep_idx]
            else:
                keep_idx = rng.choice(rows.size, size=pair_limit, replace=False)
                rows = rows[keep_idx]
                cols = cols[keep_idx]
        return torch.as_tensor(np.stack([rows, cols], axis=1).astype(np.int64), device=self.device)

    def _prepare_sri_clr_prior(self):
        if not self.use_sri_clr or "views" not in self.graph_dict:
            return
        expr_view = self.graph_dict["views"].get("expression", {})
        spatial_view = self.graph_dict["views"].get("spatial", {})
        edge_view = self.graph_dict["views"].get("edge", {})
        expr_adj = expr_view.get("adj_sp")
        spatial_adj = spatial_view.get("adj_sp")
        if expr_adj is None or spatial_adj is None:
            return
        expr_adj = self._symmetrize_adj(expr_adj)
        spatial_adj = self._symmetrize_adj(spatial_adj)

        refined_adj = spatial_adj.copy()
        edge_adj = edge_view.get("adj_sp")
        if edge_adj is not None:
            refined_adj = refined_adj.maximum(self._symmetrize_adj(edge_adj))

        self.rna_community_labels = self._build_rna_communities(expr_adj)
        support_rna = self._community_support(expr_adj, self.rna_community_labels)
        support_refined = self._community_support(refined_adj, self.rna_community_labels)
        support = 0.5 * support_rna + 0.5 * support_refined
        support_q = float(np.quantile(support, self.boundary_quantile))
        safe_q = float(np.quantile(support, self.safe_quantile))
        self.rna_boundary_mask = support <= support_q
        self.rna_safe_mask = support >= safe_q
        self.rna_community_stable_mask = support >= self.community_match_ratio

        strong_mask = self.rna_safe_mask & self.rna_community_stable_mask
        weak_mask = self.rna_boundary_mask & self.rna_community_stable_mask

        self.sri_strong_pos_pairs = self._build_pairs_from_masks(
            expr_adj, self.rna_community_labels, strong_mask, self.strong_pos_per_node * self.num_spots, "same"
        )
        self.sri_weak_pos_pairs = self._build_pairs_from_masks(
            refined_adj, self.rna_community_labels, weak_mask, self.weak_pos_per_node * self.num_spots, "same"
        )

        self.sri_neg_pairs = self._build_negative_pairs(
            refined_adj,
            self.rna_community_labels,
            self.rna_boundary_mask,
            self.rna_community_stable_mask,
            self.neg_per_node * self.num_spots,
        )
        if not self._sri_debug_printed:
            strong_count = 0 if self.sri_strong_pos_pairs is None else int(self.sri_strong_pos_pairs.shape[0])
            weak_count = 0 if self.sri_weak_pos_pairs is None else int(self.sri_weak_pos_pairs.shape[0])
            neg_count = 0 if self.sri_neg_pairs is None else int(self.sri_neg_pairs.shape[0])
            print(
                "[SRI-CLR] prior prepared: "
                f"enabled={self.use_sri_clr}, strong_pos_pairs={strong_count}, "
                f"weak_pos_pairs={weak_count}, neg_pairs={neg_count}, "
                f"warmup_ratio={self.sri_clr_warmup_ratio}, "
                f"warmup_epoch={self._get_sri_clr_warmup_epoch()}, weight={self.sri_clr_weight}, "
                f"margin={self.sri_clr_margin}, community_match_ratio={self.community_match_ratio}, "
                f"weak_pos_weight={self.weak_pos_weight}, neg_boundary_boost={self.neg_boundary_boost}, "
                "community_mode=single"
            )
            self._sri_debug_printed = True

    def _get_sri_clr_warmup_epoch(self) -> int:
        if self.sri_clr_warmup_epoch is not None:
            return max(0, min(self.epochs, self.sri_clr_warmup_epoch))
        return int(self.epochs * self.sri_clr_warmup_ratio)

    def _sri_clr_loss(self, z: torch.Tensor, epoch: int) -> torch.Tensor:
        if not self.use_sri_clr or self.sri_neg_pairs is None:
            return z.new_tensor(0.0)
        warmup_epoch = self._get_sri_clr_warmup_epoch()
        if epoch < warmup_epoch:
            return z.new_tensor(0.0)
        if not self._sri_active_printed:
            print(f"[SRI-CLR] activated at epoch {epoch} / {self.epochs}")
            self._sri_active_printed = True
        loss_terms = []
        if self.sri_strong_pos_pairs is not None:
            pos_i = self.sri_strong_pos_pairs[:, 0]
            pos_j = self.sri_strong_pos_pairs[:, 1]
            pos_dist = (z[pos_i] - z[pos_j]).pow(2).sum(dim=1).mean()
            loss_terms.append(pos_dist)
        if self.sri_weak_pos_pairs is not None:
            pos_i = self.sri_weak_pos_pairs[:, 0]
            pos_j = self.sri_weak_pos_pairs[:, 1]
            weak_pos_dist = (z[pos_i] - z[pos_j]).pow(2).sum(dim=1).mean()
            loss_terms.append(self.weak_pos_weight * weak_pos_dist)
        neg_i = self.sri_neg_pairs[:, 0]
        neg_j = self.sri_neg_pairs[:, 1]
        neg_dist = (z[neg_i] - z[neg_j]).pow(2).sum(dim=1)
        neg_margin = self.sri_clr_margin
        if self.rna_boundary_mask is not None:
            boundary_scale = 1.0 - float(self.rna_boundary_mask.mean())
            neg_margin = max(0.28, neg_margin * (0.95 + 0.25 * boundary_scale))
        neg_loss = F.relu(neg_margin - neg_dist).mean()
        loss_terms.append(neg_loss)
        return sum(loss_terms)

    def pretrain(self, grad_clip: float = 5.0) -> None:
        """
        Pretrain the model without clustering loss
        
        Parameters:
        -----------
        grad_clip : float
            Maximum gradient norm for clipping (default: 5.0)
        """
        with tqdm(total=self.pre_epochs, 
                desc="Pretraining initial model",
                bar_format="{l_bar}{bar} [ time left: {remaining} ]", ncols=80, dynamic_ncols=True, leave=True) as pbar:
            
            for epoch in range(self.pre_epochs):
                                                
                inputs_corr = self._masking_noise(self.data, self.corrupt)

                self.model.train()
                self.optimizer.zero_grad()
                
                if self.domains is None:
                    z, mu, logvar, de_feat, _, _ = self.model(inputs_corr, self.adj)
                    preds = self.model.dc(z)
                else:
                    z, mu, logvar, de_feat, _, _, domain_pred = self.model(inputs_corr, self.adj)
                    preds = self.model.model.dc(z)

                loss = self.model.gemcst_loss(
                            decoded=de_feat,
                            x=self.data,
                            preds=preds,
                            labels=self.adj_label,
                            mu=mu,
                            logvar=logvar,
                            n_nodes=self.num_spots,
                            norm=self.norm,
                            mask=self.adj_label,
                            mse_weight=self.mse_weight,
                            bce_kld_weight=self.bce_kld_weight
                            )

                if self.domains is not None:
                    domain_loss = F.cross_entropy(domain_pred, self.domains)
                    loss += self.domain_weight * domain_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                
                pbar.update(1)

    @torch.no_grad()
    def process(self) -> tuple:
        """
        Process data through trained model
        
        Returns:
        --------
        tuple: (z, q)
            z : np.ndarray
                Latent representation [n_spots, n_features]
            q : np.ndarray
                Soft cluster assignments [n_spots, n_clusters]
        """
        self.model.eval()
        if self.domains is None:
            z, _, _, _, q, _ = self.model(self.data, self.adj)
        else:
            z, _, _, _, q, _, _ = self.model(self.data, self.adj)
            
        return z.cpu().numpy(), q.cpu().numpy()

    def save_model(self, save_path: str) -> None:
        """
        Save model state to file
        
        Parameters:
        -----------
        save_path : str
            Path to save model state
        """
        torch.save({'state_dict': self.model.state_dict()}, save_path)
        print(f'Saved model to {save_path}')

    def load_model(self, load_path: str) -> None:
        """
        Load model state from file
        
        Parameters:
        -----------
        load_path : str
            Path to load model state from
        """
        state_dict = torch.load(load_path)
        self.model.load_state_dict(state_dict['state_dict'])
        print(f'Loaded model from {load_path}')

    def fit(self, 
            cluster_n: int = 20,
            cluster_type: str = 'Louvain',
            resolution: float = 1.0,
            pretrain: bool = True) -> np.ndarray:
        """
        Train the full GeMCST model with clustering
        
        Parameters:
        -----------
        cluster_n : int
            Number of clusters (default: 20)
        cluster_type : str
            Clustering method ['Louvain', 'KMeans'] (default: 'Louvain')
        resolution : float
            Resolution parameter for Louvain clustering (default: 1.0)
        pretrain : bool
            Whether to run pretraining (default: True)
            
        Returns:
        --------
        np.ndarray
            Final cluster assignments [n_spots]
        """

        if pretrain:
            self.pretrain()
        pre_z, _ = self.process()
        self.precluster_latent = np.array(pre_z, copy=True)

        y_pred_last = self._initialize_clusters(pre_z, cluster_n, cluster_type, resolution)

        with tqdm(total=self.epochs,
                desc="Training final model",
                bar_format="{l_bar}{bar} [ time left: {remaining} ]", ncols=80, dynamic_ncols=True, leave=True) as pbar:
            
            for epoch in range(self.epochs):
                                                        
                if epoch % self.q_stride == 0:
                    _, q = self.process()
                    q = self._update_soft_pseudolabels(torch.Tensor(q).to(self.device))
                    y_pred = q.cpu().numpy().argmax(1)

                    delta_label = np.sum(y_pred != y_pred_last) / y_pred.shape[0]
                    y_pred_last = y_pred.copy()
                    
                    if epoch > 0 and delta_label < self.dec_tol:
                        print(f'delta_label {delta_label:.4f} < tol {self.dec_tol}')
                        print('Reached tolerance threshold. Stopping training.')
                        break

                self._train_step(q, epoch)
                pbar.update(1)

        _, q = self.process()
        return q.argmax(1)

    def _train_step(self, q: torch.Tensor, epoch: int) -> None:
        """Perform a single training step"""
        self.model.train()
        self.optimizer.zero_grad()

        if self.domains is None:
            z, mu, logvar, de_feat, out_q, _ = self.model(self.data, self.adj)
            preds = self.model.dc(z)
        else:
            z, mu, logvar, de_feat, out_q, _, domain_pred = self.model(self.data, self.adj)
            preds = self.model.model.dc(z)

        loss_gemcst = self.model.gemcst_loss(
            decoded = de_feat,
            x = self.data,
            preds = preds,
            labels = self.adj_label,
            mu = mu,
            logvar = logvar,
            n_nodes = self.num_spots,
            norm = self.norm,
            mask = self.adj_label,
            mse_weight = self.mse_weight,
            bce_kld_weight = self.bce_kld_weight
        )
        
        if self.soft_potts_enabled:
            loss_kl = self._collision_cross_entropy(out_q, q)
        else:
            loss_kl = F.kl_div(out_q.log(), q, reduction='batchmean')
        loss_sri = self._sri_clr_loss(z, epoch)
        has_active_pairs = (
            self.sri_strong_pos_pairs is not None
            or self.sri_weak_pos_pairs is not None
            or self.sri_neg_pairs is not None
        )
        if self.use_sri_clr and has_active_pairs and epoch >= self._get_sri_clr_warmup_epoch():
            self._sri_running_loss += float(loss_sri.detach().item())
            self._sri_running_steps += 1
            report_stride = max(1, self.q_stride)
            if (epoch + 1) % report_stride == 0:
                avg_sri = self._sri_running_loss / max(self._sri_running_steps, 1)
                print(
                    f"[SRI-CLR] epoch={epoch + 1}, current={float(loss_sri.detach().item()):.6f}, "
                    f"avg_active={avg_sri:.6f}"
                )

        if self.domains is None:
            loss = self.kl_weight * loss_kl + loss_gemcst + self.sri_clr_weight * loss_sri
        else:
            domain_loss = F.cross_entropy(domain_pred, self.domains)
            loss = self.kl_weight * loss_kl + loss_gemcst + self.domain_weight * domain_loss + self.sri_clr_weight * loss_sri

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()

    def _initialize_clusters(self, 
                        z: np.ndarray,
                        cluster_n: int,
                        cluster_type: str,
                        resolution: float) -> np.ndarray:
        """
        Initialize cluster centers using specified method
        
        Returns:
        --------
        np.ndarray
            Initial cluster assignments
        """
        if cluster_type == 'KMeans':
            kmeans = KMeans(n_clusters=cluster_n, n_init=cluster_n*2, random_state=88)
            y_pred = kmeans.fit_predict(z)
            centers = kmeans.cluster_centers_
        elif cluster_type == 'Louvain':
            cluster_data = sc.AnnData(z)
            sc.pp.neighbors(cluster_data, n_neighbors=cluster_n, random_state=0)
            sc.tl.louvain(cluster_data, resolution=resolution, random_state=0)
            y_pred = cluster_data.obs['louvain'].astype(int).to_numpy()

            features = pd.DataFrame(z, index=np.arange(z.shape[0]))
            groups = pd.Series(y_pred, index=np.arange(z.shape[0]), name="Group")
            merged = pd.concat([features, groups], axis=1)
            centers = np.asarray(merged.groupby("Group").mean())

        target_model = self.model if self.domains is None else self.model.model
        target_model.cluster_layer.data = torch.tensor(centers).to(self.device)
        
        return y_pred

    @staticmethod
    def _masking_noise(data: torch.Tensor, frac: float) -> torch.Tensor:
        """
        Apply masking noise to input data
        
        Parameters:
        -----------
        data : torch.Tensor
            Input data tensor
        frac : float
            Fraction of elements to mask
            
        Returns:
        --------
        torch.Tensor
            Noisy data tensor
        """
        data_noise = data.clone()
        mask = torch.rand(data.size()) < frac
        data_noise[mask.to(data.device)] = 0
        return data_noise
