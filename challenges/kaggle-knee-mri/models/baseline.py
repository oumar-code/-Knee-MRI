import timm
import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, 1)

    def forward(self, x):  # x: (B, S, D)
        w = torch.softmax(self.q(x).squeeze(-1), dim=1)  # (B, S)
        return (x * w.unsqueeze(-1)).sum(1)


class SliceModel(nn.Module):
    def __init__(self, backbone_name='tf_efficientnet_b3', pretrained=True, n_outputs=12, in_channels=1):
        super().__init__()
        # create backbone with no classifier head
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='')
        # if the backbone expects 3 channels but we have 1, use a simple conv to expand
        self.in_channels = in_channels
        feat_dim = self.backbone.num_features
        self.expand = None
        # detect backbone default in-channels
        try:
            backbone_in_ch = self.backbone.conv_stem.in_channels
        except Exception:
            backbone_in_ch = 3
        if in_channels != backbone_in_ch:
            # small conv to map in_channels -> backbone_in_ch
            self.expand = nn.Conv2d(in_channels, backbone_in_ch, kernel_size=1)
        self.pool = AttentionPool(feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_outputs)
        )

    def forward(self, x):
        # x: (B, S, C, H, W) -> treat as batch of (B*S, C, H, W)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        if self.expand is not None:
            x = self.expand(x)
        f = self.backbone.forward_features(x)  # (B*S, D, ...) depending on backbone
        if f.ndim > 2:
            f = f.view(f.size(0), f.size(1))
        f = f.view(B, S, -1)
        s = self.pool(f)
        out = self.head(s)
        return out
