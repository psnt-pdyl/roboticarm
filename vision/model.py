"""Part-finding network: a fully convolutional heat map, read out with a soft-argmax.

Six 3x3 convolutions (two 2x2 max-pools) turn the 160x40 camera image into a 40x10 map
of scores. A softmax makes that a probability map over the image, and its expected
position is the part's centre. Plain convolutions only, so the forward pass is easy to
port to JavaScript.
"""
import torch
import torch.nn.functional as F
from torch import nn

from scene import X0, X1, Z0, Z1

POOL_AFTER = (1, 3)          # max-pool after these conv layers


class VisionNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        c = width
        self.convs = nn.ModuleList(nn.Conv2d(i, o, 3, padding=1) for i, o in
                                   [(3, c), (c, c), (c, 2 * c), (2 * c, 2 * c), (2 * c, 2 * c), (2 * c, 1)])

    def forward(self, img):
        """img [B,3,H,W] in [0,1] -> part centre [B,2] as (x, z) in mm."""
        x = img - 0.5
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
            if i in POOL_AFTER:
                x = F.max_pool2d(x, 2)
        B, _, h, w = x.shape
        p = torch.softmax(x.view(B, -1), 1).view(B, h, w)
        u = (p.sum(1) * (torch.arange(w, device=x.device) + 0.5) / w).sum(1)
        v = (p.sum(2) * (torch.arange(h, device=x.device) + 0.5) / h).sum(1)
        return torch.stack([X0 + u * (X1 - X0), Z1 - v * (Z1 - Z0)], -1)
