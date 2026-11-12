import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import Sequential, BatchNorm
from typing import Optional

class GraphNormV2SingleGraph(nn.Module):
    """GraphNormv2-style normalization for one full-batch spatial graph."""

    def __init__(self, in_channels: int, eps: float = 1e-5, center_init: float = 0.25):
        super().__init__()
        self.in_channels = int(in_channels)
        self.eps = float(eps)
        center_init = min(max(float(center_init), 1e-4), 1.0 - 1e-4)
        self.center_logit = nn.Parameter(
            torch.full((self.in_channels,), torch.logit(torch.tensor(center_init)).item())
        )
        self.weight = nn.Parameter(torch.ones(self.in_channels))
        self.bias = nn.Parameter(torch.zeros(self.in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center_strength = torch.sigmoid(self.center_logit).view(1, -1)
        center = x.mean(dim=0, keepdim=True)
        out = x - center_strength * center
        scale = torch.sqrt(out.pow(2).mean(dim=0, keepdim=True) + self.eps)
        return self.weight.view(1, -1) * (out / scale) + self.bias.view(1, -1)

class InitialResidualGraphConv(nn.Module):
    """Graph convolution with initial residual and optional GraphNormv2-style norm."""

    def __init__(
        self,
        conv_class,
        in_channels: int,
        out_channels: int,
        initial_channels: int,
        residual_alpha: float = 0.10,
        use_norm: bool = True,
        center_init: float = 0.25,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv = conv_class(in_channels, out_channels)
        self.residual_alpha = float(residual_alpha)
        self.initial_projection = nn.Identity()
        if int(initial_channels) != int(out_channels):
            self.initial_projection = nn.Linear(initial_channels, out_channels, bias=False)
        self.norm = GraphNormV2SingleGraph(out_channels, center_init=center_init) if use_norm else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, initial_x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x, edge_index)
        if self.residual_alpha > 0:
            initial = self.initial_projection(initial_x)
            out = (1.0 - self.residual_alpha) * out + self.residual_alpha * initial
        out = self.norm(out)
        return self.activation(out)

class GeMCSTModel(nn.Module):
    def __init__(self, 
                input_dim: int,
                Conv_type: str = 'GCNConv',
                linear_encoder_hidden: list = [32, 20],
                linear_decoder_hidden: list = [32],
                conv_hidden: list = [32, 8],
                p_drop: float = 0.01,
                dec_cluster_n: int = 15,
                alpha: float = 0.9,
                activate: str = "relu",
                tri_view: bool = False,
                edge_input_dim: int = None):
        """
        GeMCST Model Initialization
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features (number of genes)
        Conv_type : str
            Type of graph convolutional layer (default: 'GCNConv')
            Options: ['GCNConv', 'SAGEConv', 'GraphConv', 'GatedGraphConv', 
                    'ResGatedGraphConv', 'TransformerConv', 'TAGConv',
                    'ARMAConv', 'SGConv', 'MFConv', 'RGCNConv',
                    'FeaStConv', 'LEConv', 'ClusterGCNConv']
        linear_encoder_hidden : list
            List of hidden layer sizes for the encoder (default: [32, 20])
        linear_decoder_hidden : list
            List of hidden layer sizes for the decoder (default: [32])
        conv_hidden : list
            List of hidden layer sizes for the graph convolutional layers (default: [32, 8])
        p_drop : float
            Dropout probability (default: 0.01)
        dec_cluster_n : int
            Number of clusters for DEC (default: 15)
        alpha : float
            Parameter for student's t-distribution (default: 0.9)
        activate : str
            Activation function (default: 'relu')
            Options: ['relu', 'sigmoid']
        """
        super(GeMCSTModel, self).__init__()
        self.input_dim = input_dim
        self.Conv_type = Conv_type
        self.alpha = alpha
        self.conv_hidden = conv_hidden
        self.linear_encoder_hidden = linear_encoder_hidden
        self.linear_decoder_hidden = linear_decoder_hidden
        self.activate = activate
        self.p_drop = p_drop
        self.dec_cluster_n = dec_cluster_n
        self.tri_view = tri_view
        self.edge_input_dim = int(edge_input_dim) if edge_input_dim is not None else self.input_dim
        self.embedding_dim = self.conv_hidden[-1]
        self.last_view_outputs = None
        self.use_graphnormv2_gcn = os.environ.get("USE_GRAPHNORMV2_GCN", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.graphnormv2_res_alpha = float(os.environ.get("GRAPHNORMV2_RES_ALPHA", "0.10"))
        self.graphnormv2_center_init = float(os.environ.get("GRAPHNORMV2_CENTER_INIT", "0.25"))
        self.graphnormv2_output_norm = os.environ.get("GRAPHNORMV2_OUTPUT_NORM", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        self.encoder = self._build_encoder(self.input_dim, "encoder")
        self.edge_encoder = self.encoder
        if self.tri_view and self.edge_input_dim != self.input_dim:
            self.edge_encoder = self._build_encoder(self.edge_input_dim, "edge_encoder")

        current_dim = self.embedding_dim
        self.decoder = nn.Sequential()
        for i, hidden_size in enumerate(linear_decoder_hidden):
            self.decoder.add_module(
                f'decoder_L{i}',
                self._build_network(current_dim, hidden_size)
            )
            current_dim = hidden_size

        self.decoder.add_module(
            'decoder_out', 
            nn.Linear(current_dim, self.input_dim)
        )

        self._build_graph_conv_layers()
        if self.tri_view:
            self._build_triview_graph_conv_layers()

        self.dc = InnerProductDecoder(p_drop)

        self.cluster_layer = Parameter(torch.Tensor(
            self.dec_cluster_n, 
            self.embedding_dim
        ))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def _build_encoder(self, input_dim: int, prefix: str) -> nn.Sequential:
        encoder = nn.Sequential()
        current_dim = input_dim
        for i, hidden_size in enumerate(self.linear_encoder_hidden):
            encoder.add_module(
                f'{prefix}_L{i}',
                self._build_network(current_dim, hidden_size)
            )
            current_dim = hidden_size
        return encoder

    def _build_network(self, in_features: int, out_features: int) -> nn.Sequential:
        """Build a basic network block with linear, batchnorm, activation and dropout"""
        layers = [
            nn.Linear(in_features, out_features),
            BatchNorm(out_features, momentum=0.01, eps=0.001)
        ]
        
        if self.activate == "relu":
            layers.append(nn.ELU())
        elif self.activate == "sigmoid":
            layers.append(nn.Sigmoid())
            
        if self.p_drop > 0:
            layers.append(nn.Dropout(self.p_drop))
            
        return nn.Sequential(*layers)

    def _build_graph_conv_layers(self):
        """Build graph convolutional layers based on specified type"""
        conv_class = self._get_conv_class()
        if self.use_graphnormv2_gcn:
            initial_dim = self.linear_encoder_hidden[-1]
            hidden_dim = self.conv_hidden[0] * 2
            out_dim = self.conv_hidden[-1]
            self.conv = InitialResidualGraphConv(
                conv_class,
                initial_dim,
                hidden_dim,
                initial_dim,
                residual_alpha=self.graphnormv2_res_alpha,
                use_norm=True,
                center_init=self.graphnormv2_center_init,
                activation=nn.ReLU(inplace=True),
            )
            self.conv_mean = InitialResidualGraphConv(
                conv_class,
                hidden_dim,
                out_dim,
                initial_dim,
                residual_alpha=self.graphnormv2_res_alpha,
                use_norm=self.graphnormv2_output_norm,
                center_init=self.graphnormv2_center_init,
            )
            self.conv_logvar = InitialResidualGraphConv(
                conv_class,
                hidden_dim,
                out_dim,
                initial_dim,
                residual_alpha=self.graphnormv2_res_alpha,
                use_norm=self.graphnormv2_output_norm,
                center_init=self.graphnormv2_center_init,
            )
            return

        self.conv = Sequential('x, edge_index', [
            (conv_class(self.linear_encoder_hidden[-1], self.conv_hidden[0]*2), 
            'x, edge_index -> x'),
            BatchNorm(self.conv_hidden[0]*2),
            nn.ReLU(inplace=True), 
        ])

        self.conv_mean = Sequential('x, edge_index', [
            (conv_class(self.conv_hidden[0]*2, self.conv_hidden[-1]), 
            'x, edge_index -> x')
        ])
        
        self.conv_logvar = Sequential('x, edge_index', [
            (conv_class(self.conv_hidden[0]*2, self.conv_hidden[-1]), 
            'x, edge_index -> x')
        ])

    def _make_graph_branch(self):
        conv_class = self._get_conv_class()
        if self.use_graphnormv2_gcn:
            initial_dim = self.linear_encoder_hidden[-1]
            hidden_dim = self.conv_hidden[0] * 2
            out_dim = self.conv_hidden[-1]
            return nn.ModuleDict({
                "conv": InitialResidualGraphConv(
                    conv_class,
                    initial_dim,
                    hidden_dim,
                    initial_dim,
                    residual_alpha=self.graphnormv2_res_alpha,
                    use_norm=True,
                    center_init=self.graphnormv2_center_init,
                    activation=nn.ReLU(inplace=True),
                ),
                "mean": InitialResidualGraphConv(
                    conv_class,
                    hidden_dim,
                    out_dim,
                    initial_dim,
                    residual_alpha=self.graphnormv2_res_alpha,
                    use_norm=self.graphnormv2_output_norm,
                    center_init=self.graphnormv2_center_init,
                ),
                "logvar": InitialResidualGraphConv(
                    conv_class,
                    hidden_dim,
                    out_dim,
                    initial_dim,
                    residual_alpha=self.graphnormv2_res_alpha,
                    use_norm=self.graphnormv2_output_norm,
                    center_init=self.graphnormv2_center_init,
                ),
            })
        return nn.ModuleDict({
            "conv": Sequential('x, edge_index', [
                (conv_class(self.linear_encoder_hidden[-1], self.conv_hidden[0] * 2),
                'x, edge_index -> x'),
                BatchNorm(self.conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ]),
            "mean": Sequential('x, edge_index', [
                (conv_class(self.conv_hidden[0] * 2, self.conv_hidden[-1]),
                'x, edge_index -> x')
            ]),
            "logvar": Sequential('x, edge_index', [
                (conv_class(self.conv_hidden[0] * 2, self.conv_hidden[-1]),
                'x, edge_index -> x')
            ]),
        })

    def _build_triview_graph_conv_layers(self):
        self.view_names = ("expression", "spatial", "edge")
        self.view_branches = nn.ModuleDict({
            name: self._make_graph_branch()
            for name in self.view_names
        })
        self.view_attention = nn.Linear(self.conv_hidden[-1], 1, bias=False)
        self.last_view_attention = None

    def _get_conv_class(self):
        """Get the appropriate graph convolution class"""
        conv_classes = {
            "GCNConv": self._import_conv_class("GCNConv"),
            "GATConv": self._import_conv_class("GATConv"),
            "SAGEConv": self._import_conv_class("SAGEConv"),
            "GraphConv": self._import_conv_class("GraphConv"),
            "GatedGraphConv": self._import_conv_class("GatedGraphConv"),
            "ResGatedGraphConv": self._import_conv_class("ResGatedGraphConv"),
            "TransformerConv": self._import_conv_class("TransformerConv"),
            "TAGConv": self._import_conv_class("TAGConv"),
            "ARMAConv": self._import_conv_class("ARMAConv"),
            "SGConv": self._import_conv_class("SGConv"),
            "MFConv": self._import_conv_class("MFConv"),
            "RGCNConv": self._import_conv_class("RGCNConv"),
            "FeaStConv": self._import_conv_class("FeaStConv"),
            "LEConv": self._import_conv_class("LEConv"),
            "ClusterGCNConv": self._import_conv_class("ClusterGCNConv")
        }
        return conv_classes[self.Conv_type]
    
    def _import_conv_class(self, class_name: str):
        """Dynamically import the required graph convolution class"""
        from torch_geometric.nn import __dict__ as geom_nn_dict
        return geom_nn_dict[class_name]

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> tuple:
        """
        Encode input data through encoder and graph convolutional layers
        
        Parameters:
        -----------
        x : torch.Tensor
            Input feature matrix [n_nodes, input_dim]
        adj : torch.Tensor
            Adjacency matrix or edge index [2, n_edges]
            
        Returns:
        --------
        tuple: (mu, logvar, projected_x)
            mu : torch.Tensor
                Mean of latent distribution [n_nodes, conv_hidden[-1]]
            logvar : torch.Tensor
                Log variance of latent distribution [n_nodes, conv_hidden[-1]]
            projected_x : torch.Tensor
            Projected node features [n_nodes, linear_encoder_hidden[-1]]
        """
        projected_x = self.encoder(x)
        if self.use_graphnormv2_gcn:
            conv_x = self.conv(projected_x, adj, projected_x)
            return (
                self.conv_mean(conv_x, adj, projected_x),
                self.conv_logvar(conv_x, adj, projected_x),
                projected_x,
            )
        conv_x = self.conv(projected_x, adj)
        return self.conv_mean(conv_x, adj), self.conv_logvar(conv_x, adj), projected_x

    def _encode_branch(self, projected_x: torch.Tensor, adj: torch.Tensor, view_name: str) -> tuple:
        branch = self.view_branches[view_name]
        if self.use_graphnormv2_gcn:
            conv_x = branch["conv"](projected_x, adj, projected_x)
            return (
                branch["mean"](conv_x, adj, projected_x),
                branch["logvar"](conv_x, adj, projected_x),
            )
        conv_x = branch["conv"](projected_x, adj)
        return branch["mean"](conv_x, adj), branch["logvar"](conv_x, adj)

    def encode_triview(self, x: torch.Tensor, graph_bundle: dict) -> tuple:
        """
        Encode RNA, spatial and pathology-morphology views and fuse them with attention.

        The third view uses UNI/DivPrune image morphology features directly,
        together with the image-similarity graph.
        """
        views = graph_bundle["views"]
        edge_x = graph_bundle.get("edge_view_data")
        if edge_x is None:
            raise ValueError("Tri-view training requires graph_bundle['edge_view_data'] for the pathology image view.")
        projected_x = self.encoder(x)
        edge_projected_x = self.edge_encoder(edge_x)

        branch_inputs = {
            "expression": projected_x,
            "spatial": projected_x,
            "edge": edge_projected_x,
        }
        mus = []
        logvars = []
        zs = []
        view_zs = {}
        for view_name in self.view_names:
            mu, logvar = self._encode_branch(
                branch_inputs[view_name],
                views[view_name],
                view_name,
            )
            z = self.reparameterize(mu, logvar)
            mus.append(mu)
            logvars.append(logvar)
            zs.append(z)
            view_zs[view_name] = z

        z_stack = torch.stack(zs, dim=1)
        attn_logits = self.view_attention(z_stack).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1)
        fused_gnn_z = torch.sum(z_stack * attn.unsqueeze(-1), dim=1)

        mu_stack = torch.stack(mus, dim=1)
        logvar_stack = torch.stack(logvars, dim=1)
        fused_mu = torch.sum(mu_stack * attn.unsqueeze(-1), dim=1)
        fused_logvar = torch.sum(logvar_stack * attn.unsqueeze(-1), dim=1)
        self.last_view_attention = attn.detach()
        self.last_view_outputs = {
            "names": self.view_names,
            "embeddings": view_zs,
            "attention": attn.detach(),
        }

        return fused_mu, fused_logvar, fused_gnn_z

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution
        
        Parameters:
        -----------
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution
            
        Returns:
        --------
        torch.Tensor
            Sampled latent variables
        """
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        return mu

    def target_distribution(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution for DEC clustering
        
        Parameters:
        -----------
        target : torch.Tensor
            Current soft cluster assignments
            
        Returns:
        --------
        torch.Tensor
            Target distribution
        """
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def gemcst_loss(self, 
                decoded: torch.Tensor, 
                x: torch.Tensor, 
                preds: torch.Tensor, 
                labels: torch.Tensor, 
                mu: torch.Tensor, 
                logvar: torch.Tensor, 
                n_nodes: int, 
                norm: float, 
                mask: Optional[torch.Tensor] = None, 
                mse_weight: float = 10, 
                bce_kld_weight: float = 0.1,
                negative_weight: float = 0.0) -> torch.Tensor:
        """
        Compute GeMCST loss function
        
        Parameters:
        -----------
        decoded : torch.Tensor
            Decoded/reconstructed features
        x : torch.Tensor
            Original input features
        preds : torch.Tensor
            Predicted adjacency matrix
        labels : torch.Tensor
            True adjacency matrix
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution
        n_nodes : int
            Number of nodes in graph
        norm : float
            Normalization factor
        mask : torch.Tensor, optional
            Mask for adjacency matrix
        mse_weight : float
            Weight for reconstruction loss
        bce_kld_weight : float
            Weight for BCE and KLD losses
        negative_weight : float
            Additional non-edge BCE weight (default: 0, disabled)
            
        Returns:
        --------
        torch.Tensor
            Combined loss value
        """
        mse_loss = F.mse_loss(decoded, x)
        
        pred_prob = preds  # Decoder already applies sigmoid.
        if mask is not None:
            valid_mask = mask.bool()
            bce_loss = F.binary_cross_entropy(
                pred_prob[valid_mask], labels[valid_mask].float()
            )
        else:
            bce_loss = F.binary_cross_entropy(pred_prob, labels.float())

        bce_loss = norm * bce_loss
        if mask is not None and negative_weight > 0:
            negative_mask = ~labels.bool()
            if negative_mask.any():
                negative_loss = F.binary_cross_entropy(
                    pred_prob[negative_mask], labels[negative_mask].float()
                )
                bce_loss = bce_loss + norm * negative_weight * negative_loss

        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        
        return mse_weight * mse_loss + bce_kld_weight * (bce_loss + KLD)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple:
        """
        Forward pass of GeMCST model
        
        Parameters:
        -----------
        x : torch.Tensor
            Input feature matrix [n_nodes, input_dim]
        adj : torch.Tensor
            Adjacency matrix or edge index [2, n_edges]
        Returns:
        --------
        tuple: (z, mu, logvar, de_feat, q, gnn_z)
            z : torch.Tensor
                Graph-dominant latent features [n_nodes, conv_hidden[-1]]
            mu : torch.Tensor
                Mean of latent distribution
            logvar : torch.Tensor
                Log variance of latent distribution
            de_feat : torch.Tensor
                Decoded features [n_nodes, input_dim]
            q : torch.Tensor
                Soft cluster assignments [n_nodes, dec_cluster_n]
            gnn_z : torch.Tensor
                Graph latent features [n_nodes, conv_hidden[-1]]
        """
        if self.tri_view and isinstance(adj, dict) and "views" in adj:
            mu, logvar, gnn_z = self.encode_triview(x, adj)
        else:
            self.last_view_outputs = None
            mu, logvar, _ = self.encode(x, adj)
            gnn_z = self.reparameterize(mu, logvar)
        z = gnn_z
        de_feat = self.decoder(z)

        q = 1.0 / ((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, mu, logvar, de_feat, q, gnn_z

class InnerProductDecoder(nn.Module):
    """
    Inner Product Decoder for graph reconstruction
    
    Parameters:
    -----------
    dropout : float
        Dropout probability
    act : callable
        Activation function (default: torch.sigmoid)
    """
    def __init__(self, dropout: float, act: callable = torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct adjacency matrix from latent features
        
        Parameters:
        -----------
        z : torch.Tensor
            Latent features [n_nodes, n_features]
            
        Returns:
        --------
        torch.Tensor
            Reconstructed adjacency matrix [n_nodes, n_nodes]
        """
        z = F.dropout(z, self.dropout, training=self.training)
        return self.act(torch.mm(z, z.t()))

class GradientReverseLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training
    
    This layer reverses and scales gradients during backpropagation
    while performing a no-op during forward pass.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: float) -> torch.Tensor:
        """
        Forward pass (no operation)
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        weight : float
            Weight for gradient scaling
            
        Returns:
        --------
        torch.Tensor
            Same as input tensor
        """
        ctx.weight = weight
        return x.view_as(x) * 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass with gradient reversal
        
        Parameters:
        -----------
        grad_output : torch.Tensor
            Gradient from subsequent layers
            
        Returns:
        --------
        tuple: (rev_grad, None)
            rev_grad : torch.Tensor
                Reversed and scaled gradients
            None : 
                Placeholder for weight gradient
        """
        return (grad_output * -1 * ctx.weight), None

class AdversarialNetwork(nn.Module):
    """
    Adversarial Network for Domain Adaptation
    
    Parameters:
    -----------
    model : GeMCSTModel
        The base GeMCST model
    n_domains : int
        Number of domains to adapt between (default: 2)
    weight : float
        Weight for gradient reversal (default: 1)
    n_layers : int
        Number of hidden layers (default: 2)
    """
    def __init__(self,
                model: GeMCSTModel,
                n_domains: int = 2,
                weight: float = 1,
                n_layers: int = 2):
        super(AdversarialNetwork, self).__init__()
        self.model = model
        self.n_domains = n_domains
        self.weight = weight
        self.n_layers = n_layers

        input_dim = self.model.embedding_dim
        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.extend([
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
            ])

        self.domain_clf = nn.Sequential(
            *hidden_layers,
            nn.Linear(input_dim, self.n_domains),
        )

    def set_rev_grad_weight(self, weight: float) -> None:
        """Set the gradient reversal weight"""
        self.weight = weight

    def target_distribution(
        self, 
        target
        ):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def gemcst_loss(self, 
                decoded: torch.Tensor, 
                x: torch.Tensor, 
                preds: torch.Tensor, 
                labels: torch.Tensor, 
                mu: torch.Tensor, 
                logvar: torch.Tensor, 
                n_nodes: int, 
                norm: float, 
                mask: Optional[torch.Tensor] = None, 
                mse_weight: float = 10, 
                bce_kld_weight: float = 0.1,
                negative_weight: float = 0.0) -> torch.Tensor:
        """
        Compute GeMCST loss function
        
        Parameters:
        -----------
        decoded : torch.Tensor
            Decoded/reconstructed features
        x : torch.Tensor
            Original input features
        preds : torch.Tensor
            Predicted adjacency matrix
        labels : torch.Tensor
            True adjacency matrix
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution
        n_nodes : int
            Number of nodes in graph
        norm : float
            Normalization factor
        mask : torch.Tensor, optional
            Mask for adjacency matrix
        mse_weight : float
            Weight for reconstruction loss
        bce_kld_weight : float
            Weight for BCE and KLD losses
        negative_weight : float
            Additional non-edge BCE weight (default: 0, disabled)
            
        Returns:
        --------
        torch.Tensor
            Combined loss value
        """
        mse_loss = F.mse_loss(decoded, x)
        
        pred_prob = preds  # Decoder already applies sigmoid.
        if mask is not None:
            valid_mask = mask.bool()
            bce_loss = F.binary_cross_entropy(
                pred_prob[valid_mask], labels[valid_mask].float()
            )
        else:
            bce_loss = F.binary_cross_entropy(pred_prob, labels.float())

        bce_loss = norm * bce_loss
        if mask is not None and negative_weight > 0:
            negative_mask = ~labels.bool()
            if negative_mask.any():
                negative_loss = F.binary_cross_entropy(
                    pred_prob[negative_mask], labels[negative_mask].float()
                )
                bce_loss = bce_loss + norm * negative_weight * negative_loss

        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        
        return mse_weight * mse_loss + bce_kld_weight * (bce_loss + KLD)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        """
        Forward pass with domain classification
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features [n_nodes, input_dim]
        edge_index : torch.Tensor
            Graph edge indices [2, n_edges]
            
        Returns:
        --------
        tuple: (z, mu, logvar, de_feat, q, gnn_z, domain_pred)
            All outputs from base model plus:
            domain_pred : torch.Tensor
                Domain classification logits [n_nodes, n_domains]
        """
                                
        z, mu, logvar, de_feat, q, gnn_z = self.model(x, edge_index)

        x_rev = GradientReverseLayer.apply(z, self.weight)

        domain_pred = self.domain_clf(x_rev)
        
        return z, mu, logvar, de_feat, q, gnn_z, domain_pred
