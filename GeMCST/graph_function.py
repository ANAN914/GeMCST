import os,sys
import numpy as np
import torch
from scipy import stats
import scipy.sparse as sp
from scipy.spatial import distance
from torch_sparse import SparseTensor
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

class graph():
    def __init__(self, 
                data, 
                k,  
                rad_cutoff = 150,
                distType = 'BallTree',):
        super(graph, self).__init__()
        self.data = data
        self.distType = distType
        self.k = k
        self.rad_cutoff = rad_cutoff

    def graph_computing(self):
        """
        Input: -adata.obsm['spatial']
               -distanceType:
                    -if get more information, https://docs.scipy.org/doc/scipy/reference/generated/scipy.
                     spatial.distance.cdist.html#scipy.spatial.distance.cdist
               -k: number of neighbors
        Return: graphList
        """
        dist_list = ["euclidean","braycurtis","canberra","mahalanobis","chebyshev","cosine",
                    "jensenshannon","mahalanobis","minkowski","seuclidean","sqeuclidean","hamming",
                    "jaccard", "jensenshannon", "kulsinski", "mahalanobis","matching", "minkowski", 
                    "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
                    "sqeuclidean", "wminkowski", "yule"]

        if self.distType == 'spearmanr':
            SpearA, _= stats.spearmanr(self.data, axis=1)
            graphList = []
            for node_idx in range(self.data.shape[0]):
                tmp = SpearA[node_idx, :].reshape(1, -1)
                res = tmp.argsort()[0][-(self.k+1):]
                for j in np.arange(0, self.k):
                    graphList.append((node_idx, res[j]))
            print('%.4f neighbors per cell on average.' %(len(graphList)/self.data.shape[0]))

        elif self.distType == "BallTree":
            from sklearn.neighbors import BallTree
            tree = BallTree(self.data)
            dist, ind = tree.query(self.data, k=self.k+1)
            indices = ind[:, 1:]
            graphList=[]
            for node_idx in range(self.data.shape[0]):
                for j in np.arange(0, indices.shape[1]):
                    graphList.append((node_idx, indices[node_idx][j]))
            print('%.4f neighbors per cell on average.' %(len(graphList)/self.data.shape[0]))

        elif self.distType == "KDTree":
            from sklearn.neighbors import KDTree
            tree = KDTree(self.data)
            dist, ind = tree.query(self.data, k=self.k+1)
            indices = ind[:, 1:]
            graphList=[]
            for node_idx in range(self.data.shape[0]):
                for j in np.arange(0, indices.shape[1]):
                    graphList.append((node_idx, indices[node_idx][j]))
            print('%.4f neighbors per cell on average.' %(len(graphList)/self.data.shape[0]))

        elif self.distType == "kneighbors_graph":
            from sklearn.neighbors import kneighbors_graph
            A = kneighbors_graph(self.data, n_neighbors=self.k, mode='connectivity', include_self=False)
            A = A.toarray()
            graphList=[]
            for node_idx in range(self.data.shape[0]):
                indices = np.where(A[node_idx] == 1)[0]
                for j in np.arange(0, len(indices)):
                    graphList.append((node_idx, indices[j])) 
            print('%.4f neighbors per cell on average.' %(len(graphList)/self.data.shape[0]))

        elif self.distType == "Radius":
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(radius = self.rad_cutoff).fit(self.data)
            distances, indices = nbrs.radius_neighbors(self.data, return_distance=True)
            graphList=[]
            for node_idx in range(indices.shape[0]):
                for j in range(indices[node_idx].shape[0]):
                    if distances[node_idx][j] > 0 :
                        graphList.append((node_idx, indices[node_idx][j]))
            print('%.4f neighbors per cell on average.' %(len(graphList)/self.data.shape[0]))

        elif self.distType in dist_list:
            graphList = []
            for node_idx in range(self.data.shape[0]):
                tmp = self.data[node_idx, :].reshape(1, -1)
                distMat = distance.cdist(tmp, self.data, self.distType)
                res = distMat.argsort()[:self.k + 1]
                tmpdist = distMat[0, res[0][1:self.k + 1]]
                boundary = np.mean(tmpdist) + np.std(tmpdist)
                for j in np.arange(1, self.k+1):
                    if distMat[0, res[0][j]] <= boundary:
                        graphList.append((node_idx, res[0][j]))
                    else:
                        pass
                print('%.4f neighbors per cell on average.' %(len(graphList)/self.data.shape[0]))

        else: 
            raise ValueError(
                f"""\
                {self.distType!r} does not support. Disttype must in {dist_list} """)
        
        return graphList

    def List2Dict(self, graphList):
        """
        Return dict: eg {0: [0, 3542, 2329, 1059, 397, 2121, 485, 3099, 904, 3602],
                     1: [1, 692, 2334, 1617, 1502, 1885, 3106, 586, 3363, 101],
                     2: [2, 1849, 3024, 2280, 580, 1714, 3311, 255, 993, 2629],...}
        """
        graphdict = {}
        tdict = {}
        for graph in graphList:
            end1 = graph[0]
            end2 = graph[1]
            tdict[end1] = ""
            tdict[end2] = ""
            if end1 in graphdict:
                tmplist = graphdict[end1]
            else:
                tmplist = []
            tmplist.append(end2)
            graphdict[end1] = tmplist

        for i in range(self.data.shape[0]):
            if i not in tdict:
                graphdict[i] = []

        return graphdict

    def mx2SparseTensor(self, mx):

        """Convert a scipy sparse matrix to a torch SparseTensor."""
        mx = mx.tocoo().astype(np.float32)
        row = torch.from_numpy(mx.row).to(torch.long)
        col = torch.from_numpy(mx.col).to(torch.long)
        values = torch.from_numpy(mx.data)
        adj = SparseTensor(row=row, col=col, \
                           value=values, sparse_sizes=mx.shape)
        adj_ = adj.t()
        return adj_

    def pre_graph(self, adj):
        
        """ Graph preprocessing."""
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
             
        return self.mx2SparseTensor(adj_normalized)

    def main(self):
        adj_mtx = self.graph_computing()
        graphdict = self.List2Dict(adj_mtx)
        adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))

        """ Store original adjacency matrix (without diagonal entries) for later """
        adj_pre = adj_org
        adj_pre = adj_pre - sp.dia_matrix((adj_pre.diagonal()[np.newaxis, :], [0]), shape=adj_pre.shape)
        adj_pre.eliminate_zeros()

        """ Some preprocessing."""
        adj_norm = self.pre_graph(adj_pre)
        adj_label = adj_pre + sp.eye(adj_pre.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        norm = adj_pre.shape[0] * adj_pre.shape[0] / float((adj_pre.shape[0] * adj_pre.shape[0] - adj_pre.sum()) * 2)

        graph_dict = {
                     "adj_norm": adj_norm,
                     "adj_label": adj_label,
                     "norm_value": norm }

        return graph_dict

def combine_graph_dict(dict_1, dict_2):

    tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())
    
    graph_dict = {
        "adj_norm": SparseTensor.from_dense(tmp_adj_norm),
        "adj_label": torch.block_diag(dict_1['adj_label'], dict_2['adj_label']),
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])}
    return graph_dict

def graph_dict_from_sparse(adj):
    """Build a GeMCST graph dictionary from a scipy sparse adjacency matrix."""
    adj_pre = sp.coo_matrix(adj)
    adj_pre = adj_pre.maximum(adj_pre.T)
    adj_pre = adj_pre - sp.dia_matrix((adj_pre.diagonal()[np.newaxis, :], [0]), shape=adj_pre.shape)
    adj_pre.eliminate_zeros()

    helper = graph(np.zeros((adj_pre.shape[0], 2), dtype=np.float32), k=1)
    adj_norm = helper.pre_graph(adj_pre)
    adj_label = adj_pre.copy()
    adj_label.data = np.ones_like(adj_label.data, dtype=np.float32)
    adj_label = adj_label + sp.eye(adj_pre.shape[0], dtype=np.float32)
    adj_label = torch.FloatTensor(adj_label.toarray())
    denom = float((adj_pre.shape[0] * adj_pre.shape[0] - adj_pre.sum()) * 2)
    norm = adj_pre.shape[0] * adj_pre.shape[0] / denom if denom > 0 else 1.0

    return {
        "adj_norm": adj_norm,
        "adj_label": adj_label,
        "norm_value": norm,
        "adj_sp": adj_pre.tocsr(),
    }

def _knn_adjacency(data, k, metric="euclidean", weights=None):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    n_obs = data.shape[0]
    if n_obs < 2:
        raise ValueError("at least two spots are required to build a graph")

    k = int(max(1, min(k, n_obs - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(data)
    distances, indices = nn.kneighbors(data, return_distance=True)

    rows = np.repeat(np.arange(n_obs), k)
    cols = indices[:, 1:].reshape(-1)
    if weights is None:
        vals = np.ones(rows.shape[0], dtype=np.float32)
    else:
        vals = np.asarray(weights(distances[:, 1:]), dtype=np.float32).reshape(-1)

    adj = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
    return adj.maximum(adj.T).tocsr()

def build_expression_graph(expression, k=12):
    """Expression-similarity graph Af from X."""
    expression = np.asarray(expression, dtype=np.float64)
    expression = normalize(expression, norm="l2", axis=1)
    return graph_dict_from_sparse(_knn_adjacency(expression, k=k, metric="cosine"))

def build_spatial_graph(spatial_coords, k=6):
    """Spatial adjacency graph As from coordinates C."""
    return graph_dict_from_sparse(_knn_adjacency(spatial_coords, k=k, metric="euclidean"))

def build_image_similarity_graph(image_features, k=6, image_pca_n=50):
    """
    BMCST-style image similarity graph.

    This graph is built directly from UNI morphology similarity and is not
    restricted to local spatial neighborhoods.
    """
    image_features = np.asarray(image_features, dtype=np.float64)
    if image_features.ndim != 2:
        raise ValueError(f"image_features must be 2D, got shape {image_features.shape}")
    if image_features.shape[1] > image_pca_n:
        from sklearn.decomposition import PCA
        image_features = PCA(
            n_components=image_pca_n,
            random_state=0,
            svd_solver="randomized",
        ).fit_transform(image_features)
    image_features = normalize(image_features, norm="l2", axis=1)
    return graph_dict_from_sparse(_knn_adjacency(image_features, k=k, metric="cosine"))

def _sanitize_sparse_adj(adj):
    adj = sp.csr_matrix(adj, dtype=np.float32)
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj

def _attach_optional_views(base_graph, local_adj=None, homo_adj=None, hetero_adj=None):
    graph_data = dict(base_graph)
    if local_adj is not None:
        local_graph = graph_dict_from_sparse(_sanitize_sparse_adj(local_adj))
        graph_data["local_adj_sp"] = local_graph["adj_sp"]
        graph_data["local_adj_norm"] = local_graph["adj_norm"]
    if homo_adj is not None:
        homo_graph = graph_dict_from_sparse(_sanitize_sparse_adj(homo_adj))
        graph_data["homo_adj_sp"] = homo_graph["adj_sp"]
        graph_data["homo_adj_norm"] = homo_graph["adj_norm"]
    if hetero_adj is not None:
        hetero_graph = graph_dict_from_sparse(_sanitize_sparse_adj(hetero_adj))
        graph_data["hetero_adj_sp"] = hetero_graph["adj_sp"]
        graph_data["hetero_adj_norm"] = hetero_graph["adj_norm"]
    return graph_data

def _build_local_subgraph(base_adj, anchor_adj):
    base_adj = _sanitize_sparse_adj(base_adj)
    anchor_adj = _sanitize_sparse_adj(anchor_adj)
    local_adj = base_adj.multiply(anchor_adj)
    if local_adj.nnz == 0:
        local_adj = base_adj.copy()
    return _sanitize_sparse_adj(local_adj)

def _split_homo_hetero_edges(adj, features, homo_ratio=0.35, hetero_ratio=0.20):
    adj = _sanitize_sparse_adj(adj)
    features = np.asarray(features, dtype=np.float32)
    if adj.nnz == 0 or features.shape[0] != adj.shape[0]:
        zero = sp.csr_matrix(adj.shape, dtype=np.float32)
        return zero, zero

    features = normalize(features, norm="l2", axis=1)
    coo = adj.tocoo()
    scores = np.sum(features[coo.row] * features[coo.col], axis=1).astype(np.float32)
    if scores.size == 0:
        zero = sp.csr_matrix(adj.shape, dtype=np.float32)
        return zero, zero

    homo_ratio = float(np.clip(homo_ratio, 0.0, 1.0))
    hetero_ratio = float(np.clip(hetero_ratio, 0.0, 1.0))
    if homo_ratio + hetero_ratio > 1.0:
        scale = 1.0 / (homo_ratio + hetero_ratio)
        homo_ratio *= scale
        hetero_ratio *= scale

    homo_thr = np.quantile(scores, 1.0 - homo_ratio) if homo_ratio > 0 else np.inf
    hetero_thr = np.quantile(scores, hetero_ratio) if hetero_ratio > 0 else -np.inf

    homo_mask = scores >= homo_thr
    hetero_mask = scores <= hetero_thr
    homo_adj = sp.coo_matrix(
        (np.ones(int(homo_mask.sum()), dtype=np.float32), (coo.row[homo_mask], coo.col[homo_mask])),
        shape=adj.shape,
    ).tocsr()
    hetero_adj = sp.coo_matrix(
        (np.ones(int(hetero_mask.sum()), dtype=np.float32), (coo.row[hetero_mask], coo.col[hetero_mask])),
        shape=adj.shape,
    ).tocsr()
    return _sanitize_sparse_adj(homo_adj), _sanitize_sparse_adj(hetero_adj)

def build_triview_graph_dict(
    expression,
    spatial_coords,
    image_features,
    expression_k=12,
    spatial_k=6,
    edge_spatial_k=6,
    image_pca_n=50,
    image_graph_mode="similarity",
    local_subgraph_enabled=True,
    local_expr_keep_ratio=1.0,
    local_spatial_keep_ratio=1.0,
    local_edge_keep_ratio=1.0,
    latgrl_enabled=True,
    latgrl_homo_ratio=0.35,
    latgrl_hetero_ratio=0.20,
):
    """Build RNA, spatial and pathology-morphology views for tri-view training."""
    expression = np.asarray(expression, dtype=np.float32)
    spatial_coords = np.asarray(spatial_coords, dtype=np.float32)
    edge_view_data = np.asarray(image_features, dtype=np.float32)
    image_graph_mode = str(image_graph_mode).strip().lower()
    if image_graph_mode not in {"boundary", "ae", "edge", "similarity", "bmcst", "image_similarity"}:
        raise ValueError("image_graph_mode must be one of: boundary, similarity")
    edge_graph = build_image_similarity_graph(
        image_features=image_features,
        k=edge_spatial_k,
        image_pca_n=image_pca_n,
    )
    expr_graph = build_expression_graph(expression, k=expression_k)
    spatial_graph = build_spatial_graph(spatial_coords, k=spatial_k)

    expr_adj = expr_graph["adj_sp"]
    spatial_adj = spatial_graph["adj_sp"]
    edge_adj = edge_graph["adj_sp"]

    expr_local = _build_local_subgraph(expr_adj, spatial_adj) if local_subgraph_enabled else expr_adj
    spatial_local = spatial_adj
    edge_local = _build_local_subgraph(edge_adj, spatial_adj) if local_subgraph_enabled else edge_adj

    expr_homo, expr_hetero = _split_homo_hetero_edges(
        expr_local, expression, homo_ratio=latgrl_homo_ratio, hetero_ratio=latgrl_hetero_ratio
    ) if latgrl_enabled else (None, None)
    spatial_features = normalize(np.asarray(spatial_coords, dtype=np.float32), norm="l2", axis=1)
    spatial_homo, spatial_hetero = _split_homo_hetero_edges(
        spatial_local, spatial_features, homo_ratio=latgrl_homo_ratio, hetero_ratio=latgrl_hetero_ratio
    ) if latgrl_enabled else (None, None)
    edge_homo, edge_hetero = _split_homo_hetero_edges(
        edge_local, image_features, homo_ratio=latgrl_homo_ratio, hetero_ratio=latgrl_hetero_ratio
    ) if latgrl_enabled else (None, None)

    expr_graph = _attach_optional_views(expr_graph, expr_local, expr_homo, expr_hetero)
    spatial_graph = _attach_optional_views(spatial_graph, spatial_local, spatial_homo, spatial_hetero)
    edge_graph = _attach_optional_views(edge_graph, edge_local, edge_homo, edge_hetero)

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
        "edge_view_data": edge_view_data,
        "tri_view_params": {
            "expression_k": int(expression_k),
            "spatial_k": int(spatial_k),
            "edge_spatial_k": int(edge_spatial_k),
            "image_pca_n": int(image_pca_n),
            "edge_only_within_spatial_neighborhood": False,
            "image_graph_mode": image_graph_mode,
            "resolved_edge_graph_mode": "similarity",
            "edge_view_input": "uni_divprune_image_features",
            "tri_view_graph_only": True,
            "local_subgraph_enabled": bool(local_subgraph_enabled),
            "local_expr_keep_ratio": float(local_expr_keep_ratio),
            "local_spatial_keep_ratio": float(local_spatial_keep_ratio),
            "local_edge_keep_ratio": float(local_edge_keep_ratio),
            "latgrl_enabled": bool(latgrl_enabled),
            "latgrl_homo_ratio": float(latgrl_homo_ratio),
            "latgrl_hetero_ratio": float(latgrl_hetero_ratio),
        },
    }
    return graph_dict
