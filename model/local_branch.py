import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLocalFusion(nn.Module):
    def __init__(self, in_channels, num_parts=3, reduce_dim=256):
        super(GlobalLocalFusion, self).__init__()
        self.num_parts = num_parts
        self.reduce_dim = reduce_dim
        self.output_dim = reduce_dim * (num_parts + 1)

        self.global_reduce = nn.Sequential(
            nn.Linear(in_channels, reduce_dim),
            nn.BatchNorm1d(reduce_dim),
            nn.ReLU(inplace=True)
        )

        self.local_reduce = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, reduce_dim),
                nn.BatchNorm1d(reduce_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_parts)
        ])

    def forward(self, feat_map):
        """
        feat_map: [B, C, H, W]
        """
        if feat_map.dim() != 4:
            raise ValueError('GlobalLocalFusion expects a 4D feature map.')

        batch_size, channels, height, _ = feat_map.shape

        global_feat = F.adaptive_avg_pool2d(feat_map, (1, 1)).view(batch_size, channels)
        global_feat = self.global_reduce(global_feat)

        local_feats = []
        part_h = max(1, height // self.num_parts)
        for idx in range(self.num_parts):
            start = idx * part_h
            end = height if idx == self.num_parts - 1 else min(height, (idx + 1) * part_h)
            if start >= height:
                part = feat_map[:, :, height - 1:height, :]
            else:
                part = feat_map[:, :, start:end, :]
            part_feat = F.adaptive_avg_pool2d(part, (1, 1)).view(batch_size, channels)
            part_feat = self.local_reduce[idx](part_feat)
            local_feats.append(part_feat)

        final_feat = torch.cat([global_feat] + local_feats, dim=1)
        return final_feat, global_feat, local_feats
