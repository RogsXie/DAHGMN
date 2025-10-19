import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from thop import profile


class ChannelAttention(nn.Module):
    """Channel Attention Module"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()[:2]
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return x * (avg_out + max_out).view(b, c, 1, 1)


class ChannelPool(nn.Module):
    """Channel Pooling Layer"""

    def forward(self, x):
        return torch.cat((
            torch.max(x, 1)[0].unsqueeze(1),
            torch.mean(x, 1, keepdim=True)
        ), dim=1)


class SpatialGate(nn.Module):
    """Spatial Attention Gate"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class DualAttention(nn.Module):
    """Dual Attention Module (Channel + Spatial)"""

    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialGate()

    def forward(self, x):
        x1 = self.channel_att(x)
        x2 = self.spatial_att(x)
        return x1 + x2


class HSI_Processor(nn.Module):
    """Hyperspectral Image Processor with PCA preprocessing"""

    def __init__(self, band=144, pca_dim=64, embed_dim=64):
        super().__init__()
        # PCA dimensionality reduction
        self.pca = nn.Conv2d(band, pca_dim, kernel_size=1)

        # Feature extraction
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 4, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(4),
            nn.GELU(),
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(4 * 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

    def forward(self, x):
        # PCA reduction [B,144,H,W] -> [B,30,H,W]
        x = self.pca(x)

        # 3D convolution processing
        b, c, h, w = x.shape
        x = x.view(b, 1, c, h, w)
        x = self.conv3d(x)
        x = x.view(b, -1, h, w)
        return self.conv2d(x)


class LiDAR_Processor(nn.Module):
    """LiDAR Feature Extractor with spatial attention"""

    def __init__(self, band=1, embed_dim=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(band, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.feature_extractor(x)


class CrossModalMamba(nn.Module):
    """Optimized Mamba Fusion Module with dimension fixes"""

    def __init__(self, embed_dim=64):
        super().__init__()
        # Dual Mamba modules
        self.mamba_forward = Mamba(
            d_model=embed_dim, d_state=32,
            d_conv=4, expand=2
        )
        self.mamba_reverse = Mamba(
            d_model=embed_dim, d_state=32,
            d_conv=4, expand=2
        )

        # Enhanced normalization
        self.norm_pre = nn.LayerNorm(embed_dim)
        self.norm_post = nn.LayerNorm(embed_dim)

        # Gated FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

    def _reshape_to_sequence(self, x):
        B, C, H, W = x.size()
        return x.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]

    def _bidirectional_scan(self, x):
        # Forward pass
        forward_out = self.mamba_forward(x)
        # Reverse pass
        reverse_out = self.mamba_reverse(torch.flip(x, dims=[1]))
        return forward_out + torch.flip(reverse_out, dims=[1])

    def forward(self, fused):
        # Sequence processing
        seq = self._reshape_to_sequence(fused)

        # Bidirectional scanning
        seq = self.norm_pre(seq)
        mamba_out = self._bidirectional_scan(seq)

        # Residual connection
        mamba_out = self.norm_post(mamba_out + seq)

        # Gated FFN
        ffn_out = self.ffn(mamba_out)
        final_out = mamba_out + ffn_out

        # Restore spatial dimensions [B,N,C] -> [B,C,H,W]
        B, N, C = final_out.shape
        H = W = int(N ** 0.5)
        return final_out.permute(0, 2, 1).view(B, C, H, W)

class GCN(nn.Module):
    """Dual-channel GCN compatible with analysis tools"""

    def __init__(self, embed_dim, layers_count):
        super().__init__()
        self.layers = nn.ModuleList([
            GCNLayer(embed_dim, embed_dim) for _ in range(layers_count)
        ])
        self.input_adapter = nn.Identity()  # Dummy input adapter

    def forward(self, H, A):
        H = self.input_adapter(H)  # Adapter maintains input tracking
        for layer in self.layers:
            H, A = layer(H, A)
        return H


class GCNLayer(nn.Module):
    """Dynamic Dimension GCN Layer"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, H, A):
        B, N, _ = H.shape

        # Dynamic identity matrix (fixes dimension mismatch)
        I = torch.eye(N, device=H.device).unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
        A_hat = A + I

        # Degree matrix calculation
        D_inv_sqrt = torch.pow(A_hat.sum(dim=2).clamp(min=1e-6), -0.5)
        D_hat = torch.einsum('bi,bij,bj->bij', D_inv_sqrt, A_hat, D_inv_sqrt)

        # Feature transformation
        H_transformed = self.feature_transform(H.reshape(-1, H.size(-1))).view(B, N, -1)
        return torch.bmm(D_hat, H_transformed), A


class CrossModalGraph(nn.Module):
    """Dynamic Adjacency Graph Convolution Module"""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.gcn = GCN(embed_dim, layers_count=2)

        # Enhanced gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Sigmoid()
        )

    def _build_adjacency(self, H, W):


        coords = torch.cartesian_prod(torch.arange(H), torch.arange(W))  # (H*W, 2)
        coords = coords.float()

        dist_matrix = torch.cdist(coords, coords, p=2)  # (H*W, H*W)

        sigma = 1.0
        adjacency = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

        adjacency = adjacency / adjacency.sum(dim=1, keepdim=True)

        return adjacency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # Dynamic adjacency matrix (critical fix)
        A = self._build_adjacency(H, W).to(x.device)
        A_batch = A.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        # Node feature processing
        nodes = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # Graph convolution
        graph_features = self.gcn(nodes, A_batch)

        # Gated fusion
        gate = self.gate(torch.cat([nodes, graph_features], dim=-1))
        fused = gate * graph_features + nodes

        # Restore original shape
        return fused.permute(0, 2, 1).view(B, C, H, W)


class HGM(nn.Module):
    """Hybid GCNMamba output"""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.mamba_path = CrossModalMamba(embed_dim)
        self.graph_path = CrossModalGraph(embed_dim)
        self.DA = DualAttention(embed_dim)

        # Learnable fusion coefficients
        self.coeff1 = torch.nn.Parameter(torch.Tensor([0.5]))

        # self.cross_attn_gate = CrossAttentionGate(embed_dim)
    def forward(self, hsi_feat, lidar_feat):
        fused1 = self.coeff1 * hsi_feat + (1-self.coeff1) * lidar_feat
        fused = self.DA(fused1)

        mamba_out = self.mamba_path(fused)
        gcn_out = self.graph_path(fused)
        # gcn_out = self.cross_attn_gate(gcn_out, mamba_out)
        # mamba_out = self.cross_attn_gate(mamba_out, gcn_out)
        return mamba_out, gcn_out, mamba_out+gcn_out


class CNN_Classifier(nn.Module):
    """CNN-based Classifier"""

    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.FC = nn.Sequential(
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.pool(x).view(x.size(0), -1)
        x = self.FC(x)
        return F.softmax(x, dim=1)


class DAHGM(nn.Module):
    """Complete Network Architecture"""

    def __init__(self, hsi_bands, lidar_bands, num_classes=15):
        super().__init__()
        self.hsi_net = HSI_Processor(hsi_bands)
        self.lidar_net = LiDAR_Processor(lidar_bands)
        self.fusion = HGM()

        # Learnable fusion coefficients
        self.coeff1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.coeff2 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.coeff3 = torch.nn.Parameter(torch.Tensor([0.5]))

        self.refine = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        self.cnn_classifier = CNN_Classifier(num_classes)

        self._init_weights()

    def forward(self, hsi, lidar):
        # Feature extraction
        hsi_feat = self.hsi_net(hsi)
        lidar_feat = self.lidar_net(lidar)

        # Multi-modal fusion
        fused1, fused2, fused3= self.fusion(hsi_feat, lidar_feat)

        # Classification
        fused1 = self.refine(fused1).flatten(1)
        fused2 = self.refine(fused2).flatten(1)
        out1 = F.softmax(self.classifier(fused1), dim=1)
        out2 = F.softmax(self.classifier(fused2), dim=1)
        out3 = self.cnn_classifier(fused3)

        # Weighted fusion of predictions
        return out1*self.coeff1+out2*self.coeff2+out3*self.coeff3

    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAHGM(144, 1, 15).to(device)

    hsi = torch.randn(1, 144, 13, 13).to(device)
    lidar = torch.randn(1, 1, 13, 13).to(device)

    flops, params = profile(model, inputs=(hsi, lidar))
    print(f"FLOPs: {flops / 1e6:.2f}M")

    print(f"Parameters: {params}")

