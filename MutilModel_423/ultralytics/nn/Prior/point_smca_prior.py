"""
Point-cloud inspired depth prior with SMCA-style RGB fusion.

Pipeline:
  1) reliability-aware depth filtering
  2) depth + intrinsics backprojection to 3D
  3) regular voxelization in camera coordinates
  4) voxel scoring and candidate extraction
  5) 2D Gaussian prior projection (SMCA-style spatial modulation)
  6) multi-scale prior feature generation and fusion with RGB
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PointSMCAPriorBranch", "PointSMCAPriorFusion"]


def _norm01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    b = x.shape[0]
    mn = x.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
    mx = x.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
    return (x - mn) / (mx - mn + eps)


class _DepthReliabilityFilter(nn.Module):
    def __init__(self, depth_range: Tuple[float, float] = (0.3, 20.0), tau: float = 0.6):
        super().__init__()
        self.depth_min = float(depth_range[0])
        self.depth_max = float(depth_range[1])
        self.tau = float(tau)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.register_buffer("lap", lap, persistent=False)

    def forward(self, depth: torch.Tensor):
        if depth.shape[1] != 1:
            depth = depth.mean(dim=1, keepdim=True)

        valid = ((depth > self.depth_min) & (depth < self.depth_max)).float()
        depth = depth.clamp(min=self.depth_min, max=self.depth_max)

        smooth = F.avg_pool2d(depth, kernel_size=5, stride=1, padding=2)
        consistency = torch.exp(-torch.abs(depth - smooth) / self.tau) * valid
        filtered = valid * (consistency * depth + (1.0 - consistency) * smooth)

        sx = self.sobel_x.to(device=depth.device, dtype=depth.dtype)
        sy = self.sobel_y.to(device=depth.device, dtype=depth.dtype)
        lap = self.lap.to(device=depth.device, dtype=depth.dtype)
        edge = _norm01(torch.sqrt(F.conv2d(filtered, sx, padding=1).pow(2) + F.conv2d(filtered, sy, padding=1).pow(2) + 1e-6))
        curvature = _norm01(torch.abs(F.conv2d(filtered, lap, padding=1)))

        reliability = (0.7 * consistency + 0.3 * valid).clamp(0.0, 1.0)
        return filtered, reliability, edge, curvature


class _PointBackProjector(nn.Module):
    def __init__(self, intrinsics: Sequence[float], image_size: Tuple[int, int] = (640, 640), eps: float = 1e-6):
        super().__init__()
        if len(intrinsics) != 4:
            raise ValueError(f"intrinsics must be [fx, fy, cx, cy], got {intrinsics}")
        self.fx, self.fy, self.cx, self.cy = [float(v) for v in intrinsics]
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.img_h, self.img_w = int(image_size[0]), int(image_size[1])
        self.eps = float(eps)

    def _scaled_intrinsics(self, h: int, w: int):
        sx = float(w) / float(max(self.img_w, 1))
        sy = float(h) / float(max(self.img_h, 1))
        return self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy

    def forward(self, depth: torch.Tensor):
        b, _, h, w = depth.shape
        fx, fy, cx, cy = self._scaled_intrinsics(h, w)
        yy = torch.arange(h, device=depth.device, dtype=depth.dtype).view(h, 1).expand(h, w)
        xx = torch.arange(w, device=depth.device, dtype=depth.dtype).view(1, w).expand(h, w)
        xx = xx.view(1, 1, h, w).expand(b, 1, h, w)
        yy = yy.view(1, 1, h, w).expand(b, 1, h, w)
        x3 = ((xx - cx) * depth) / (fx + self.eps)
        y3 = ((yy - cy) * depth) / (fy + self.eps)
        return torch.cat([x3, y3, depth], dim=1)


class _RegularVoxelizer(nn.Module):
    def __init__(
        self,
        voxel_size: Sequence[float] = (0.10, 0.10, 0.25),
        xyz_range: Sequence[Sequence[float]] = ((-10.0, 10.0), (-6.0, 6.0), (0.3, 20.0)),
    ):
        super().__init__()
        self.vx, self.vy, self.vz = [float(v) for v in voxel_size]
        self.x_range = (float(xyz_range[0][0]), float(xyz_range[0][1]))
        self.y_range = (float(xyz_range[1][0]), float(xyz_range[1][1]))
        self.z_range = (float(xyz_range[2][0]), float(xyz_range[2][1]))

    def forward(self, xyz: torch.Tensor, reliability: torch.Tensor, geom_break: torch.Tensor):
        b, _, h, w = xyz.shape
        voxel_score_map = xyz.new_zeros((b, 1, h, w))

        for bi in range(b):
            xyz_i = xyz[bi].permute(1, 2, 0).reshape(-1, 3)
            rel_i = reliability[bi, 0].reshape(-1)
            geom_i = geom_break[bi, 0].reshape(-1)

            keep = (
                (xyz_i[:, 0] >= self.x_range[0]) & (xyz_i[:, 0] <= self.x_range[1]) &
                (xyz_i[:, 1] >= self.y_range[0]) & (xyz_i[:, 1] <= self.y_range[1]) &
                (xyz_i[:, 2] >= self.z_range[0]) & (xyz_i[:, 2] <= self.z_range[1]) &
                (rel_i > 0)
            )
            if keep.sum() == 0:
                continue

            xyz_keep = xyz_i[keep]
            rel_keep = rel_i[keep]
            geom_keep = geom_i[keep]

            ix = torch.floor((xyz_keep[:, 0] - self.x_range[0]) / self.vx).long()
            iy = torch.floor((xyz_keep[:, 1] - self.y_range[0]) / self.vy).long()
            iz = torch.floor((xyz_keep[:, 2] - self.z_range[0]) / self.vz).long()
            keys = torch.stack([ix, iy, iz], dim=1)
            uniq, inverse = torch.unique(keys, dim=0, return_inverse=True)

            n_vox = uniq.shape[0]
            count = xyz_keep.new_zeros((n_vox,))
            z_sum = xyz_keep.new_zeros((n_vox,))
            rel_sum = xyz_keep.new_zeros((n_vox,))
            geom_sum = xyz_keep.new_zeros((n_vox,))

            ones = torch.ones_like(rel_keep)
            count.scatter_add_(0, inverse, ones)
            z_sum.scatter_add_(0, inverse, xyz_keep[:, 2])
            rel_sum.scatter_add_(0, inverse, rel_keep)
            geom_sum.scatter_add_(0, inverse, geom_keep)

            z_mean = z_sum / count.clamp_min(1.0)
            rel_mean = rel_sum / count.clamp_min(1.0)
            geom_mean = geom_sum / count.clamp_min(1.0)
            density = torch.log1p(count) / math.log(32.0)
            range_score = torch.exp(-0.08 * z_mean)
            voxel_score = (0.55 * geom_mean + 0.25 * density + 0.20 * range_score) * rel_mean
            voxel_score = voxel_score.clamp(min=0.0)

            pix_idx = keep.nonzero(as_tuple=False).squeeze(1)
            score_per_point = voxel_score[inverse]
            voxel_score_map[bi, 0].view(-1)[pix_idx] = score_per_point

        return _norm01(voxel_score_map)


class _CandidateProjector(nn.Module):
    def __init__(self, topk: int = 80, radius_gain: float = 14.0, radius_min: float = 1.5, radius_max: float = 18.0):
        super().__init__()
        self.topk = int(topk)
        self.radius_gain = float(radius_gain)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)

    def _compact_connected_prior(self, score_map: torch.Tensor):
        occ = (score_map > 0.20).float()
        local_occ = F.avg_pool2d(occ, kernel_size=9, stride=1, padding=4)
        local_mass = F.avg_pool2d(score_map, kernel_size=9, stride=1, padding=4)
        compact = _norm01(local_occ * local_mass)
        return score_map * compact

    def _gaussian_map(self, score_map: torch.Tensor, depth: torch.Tensor):
        b, _, h, w = score_map.shape
        device, dtype = score_map.device, score_map.dtype
        yy = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)
        xx = torch.arange(w, device=device, dtype=dtype).view(1, 1, w)
        gauss = score_map.new_zeros((b, 1, h, w))

        pooled = F.max_pool2d(score_map, kernel_size=5, stride=1, padding=2)
        maxima = score_map * (score_map == pooled).float()

        for bi in range(b):
            flat = maxima[bi, 0].view(-1)
            k = min(self.topk, flat.numel())
            vals, inds = torch.topk(flat, k=k)
            vals_keep = vals > 0
            vals = vals[vals_keep]
            inds = inds[vals_keep]
            if vals.numel() == 0:
                continue

            ys = torch.div(inds, w, rounding_mode="floor").to(dtype)
            xs = (inds % w).to(dtype)
            z = depth[bi, 0, ys.long(), xs.long()].clamp_min(1e-3)
            radius = (self.radius_gain / z).clamp(self.radius_min, self.radius_max)

            dx = xx - xs.view(-1, 1, 1)
            dy = yy - ys.view(-1, 1, 1)
            sigma2 = (0.6 * radius).view(-1, 1, 1).pow(2)
            maps = vals.view(-1, 1, 1) * torch.exp(-(dx.pow(2) + dy.pow(2)) / (2.0 * sigma2 + 1e-6))
            gauss[bi, 0] = maps.max(dim=0)[0]

        return _norm01(gauss)

    def forward(self, voxel_score_map: torch.Tensor, depth: torch.Tensor):
        compact_score = self._compact_connected_prior(voxel_score_map)
        return self._gaussian_map(compact_score, depth)


class PointSMCAPriorBranch(nn.Module):
    """
    Depth-to-point prior branch with regular voxelization and 2D Gaussian projection.

    YAML args:
      [c_prior, intrinsics, image_size, in_depth_channels, topk, voxel_size, xyz_range]
    """

    _prior_cache = {}
    _mask_cache = {}
    _runtime_depth = None
    _runtime_intrinsics = None

    @classmethod
    def reset_cache(cls):
        cls._prior_cache = {}
        cls._mask_cache = {}

    @classmethod
    def set_runtime_inputs(cls, depth: Optional[torch.Tensor], intrinsics: Optional[torch.Tensor] = None):
        cls._runtime_depth = depth
        cls._runtime_intrinsics = intrinsics

    def __init__(
        self,
        c_prior: int = 64,
        intrinsics: Optional[Sequence[float]] = None,
        image_size: int = 640,
        in_depth_channels: int = 1,
        topk: int = 80,
        voxel_size: Sequence[float] = (0.10, 0.10, 0.25),
        xyz_range: Sequence[Sequence[float]] = ((-10.0, 10.0), (-6.0, 6.0), (0.3, 20.0)),
        c1: int = 0,
        c2: int = 0,
    ):
        super().__init__()
        intrinsics = intrinsics or [318.948951, 318.948951, 320.0, 240.0]
        img_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.in_depth_channels = int(in_depth_channels)

        self.depth_filter = _DepthReliabilityFilter(depth_range=(xyz_range[2][0], xyz_range[2][1]))
        self.backproject = _PointBackProjector(intrinsics=intrinsics, image_size=img_size)
        self.voxelizer = _RegularVoxelizer(voxel_size=voxel_size, xyz_range=xyz_range)
        self.candidate_projector = _CandidateProjector(topk=topk)

        self.stem = nn.Sequential(
            nn.Conv2d(7, c_prior, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_prior),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_prior, c_prior, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_prior),
            nn.SiLU(inplace=True),
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(c_prior, c_prior, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c_prior),
            nn.SiLU(inplace=True),
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(c_prior, c_prior, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c_prior),
            nn.SiLU(inplace=True),
        )
        self.p5 = nn.Sequential(
            nn.Conv2d(c_prior, c_prior, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c_prior),
            nn.SiLU(inplace=True),
        )
        self.mask_gain = nn.Parameter(torch.tensor(1.0))

    def _extract_depth(self, x_ext):
        if isinstance(x_ext, torch.Tensor):
            if self.in_depth_channels == 1:
                if x_ext.shape[1] >= 4:
                    return x_ext[:, 3:4, :, :]
                return x_ext[:, :1, :, :]
            if x_ext.shape[1] >= self.in_depth_channels:
                return x_ext[:, -self.in_depth_channels:, :, :]
            return x_ext

        if isinstance(x_ext, (list, tuple)):
            d = x_ext[-1]
            if d.shape[1] > self.in_depth_channels:
                d = d[:, : self.in_depth_channels, :, :]
            return d

        raise TypeError(f"Unsupported x_ext type: {type(x_ext)}")

    def forward(self, x_ext):
        PointSMCAPriorBranch.reset_cache()
        depth = PointSMCAPriorBranch._runtime_depth
        if depth is None:
            depth = self._extract_depth(x_ext)
        elif depth.shape[1] > self.in_depth_channels:
            depth = depth[:, : self.in_depth_channels, :, :]

        filtered_depth, reliability, edge, curvature = self.depth_filter(depth)
        xyz = self.backproject(filtered_depth)
        dx = F.pad(xyz[:, :, :, 1:] - xyz[:, :, :, :-1], (0, 1, 0, 0), mode="replicate")
        dy = F.pad(xyz[:, :, 1:, :] - xyz[:, :, :-1, :], (0, 0, 0, 1), mode="replicate")
        normal = torch.cross(dx, dy, dim=1)
        normal = F.normalize(normal, dim=1, eps=1e-6)
        normal_var = _norm01(torch.norm(normal - F.avg_pool2d(normal, 3, 1, 1), p=2, dim=1, keepdim=True))

        mean_xyz = F.avg_pool2d(xyz, kernel_size=9, stride=1, padding=4)
        mean_normal = F.normalize(F.avg_pool2d(normal, kernel_size=9, stride=1, padding=4), dim=1, eps=1e-6)
        plane_residual = torch.abs(((xyz - mean_xyz) * mean_normal).sum(dim=1, keepdim=True))
        plane_residual = _norm01(plane_residual)

        geom_break = _norm01(0.50 * plane_residual + 0.30 * normal_var + 0.20 * curvature)
        voxel_score = self.voxelizer(xyz, reliability, geom_break)
        gaussian_full = self.candidate_projector(voxel_score, filtered_depth)

        prior_input = torch.cat(
            [
                _norm01(filtered_depth),
                reliability,
                geom_break,
                curvature,
                normal_var,
                voxel_score,
                gaussian_full,
            ],
            dim=1,
        )
        base = self.stem(prior_input)
        p3 = self.p3(base)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        g3 = F.interpolate(gaussian_full, size=p3.shape[2:], mode="bilinear", align_corners=False)
        g4 = F.interpolate(gaussian_full, size=p4.shape[2:], mode="bilinear", align_corners=False)
        g5 = F.interpolate(gaussian_full, size=p5.shape[2:], mode="bilinear", align_corners=False)

        p3 = p3 * (1.0 + torch.tanh(self.mask_gain) * g3)
        p4 = p4 * (1.0 + torch.tanh(self.mask_gain) * g4)
        p5 = p5 * (1.0 + torch.tanh(self.mask_gain) * g5)

        PointSMCAPriorBranch._prior_cache[1] = p3
        PointSMCAPriorBranch._prior_cache[2] = p4
        PointSMCAPriorBranch._prior_cache[3] = p5
        PointSMCAPriorBranch._mask_cache[1] = g3
        PointSMCAPriorBranch._mask_cache[2] = g4
        PointSMCAPriorBranch._mask_cache[3] = g5
        return [p3, p4, p5]


class PointSMCAPriorFusion(nn.Module):
    """
    Fuse RGB features with prior features using SMCA-style Gaussian spatial modulation.
    """

    def __init__(
        self,
        c_out: int = 256,
        stage_idx: int = 1,
        mode: str = "gated",
        c1: int = 0,
        c2: int = 0,
    ):
        super().__init__()
        self.c_out = int(c_out)
        self.stage_idx = int(stage_idx)
        self.mode = str(mode)
        self._proj = None
        self._mix = None
        self._out = None
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        if c1 > 0 and c2 > 0:
            self._init_layers(int(c1), int(c2))

    def _init_layers(self, c_rgb: int, c_prior: int):
        out_c = self.c_out if self.c_out > 0 else c_rgb
        self._proj = nn.Conv2d(c_prior, c_rgb, 1, bias=False)
        self._mix = nn.Sequential(
            nn.Conv2d(2 * c_rgb, c_rgb, 1, bias=False),
            nn.BatchNorm2d(c_rgb),
            nn.SiLU(inplace=True),
        )
        self._out = nn.Sequential(
            nn.Conv2d(2 * c_rgb, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

    def _lazy_init(self, c_rgb: int, c_prior: int, device):
        if self._proj is None or self._proj.in_channels != c_prior or self._proj.out_channels != c_rgb:
            self._proj = nn.Conv2d(c_prior, c_rgb, 1, bias=False).to(device)
        mix_ok = (
            self._mix is not None
            and isinstance(self._mix, nn.Sequential)
            and isinstance(self._mix[0], nn.Conv2d)
            and self._mix[0].in_channels == (2 * c_rgb)
            and self._mix[0].out_channels == c_rgb
        )
        if not mix_ok:
            self._mix = nn.Sequential(
                nn.Conv2d(2 * c_rgb, c_rgb, 1, bias=False),
                nn.BatchNorm2d(c_rgb),
                nn.SiLU(inplace=True),
            ).to(device)
        out_c = self.c_out if self.c_out > 0 else c_rgb
        out_ok = (
            self._out is not None
            and isinstance(self._out, nn.Sequential)
            and isinstance(self._out[0], nn.Conv2d)
            and self._out[0].in_channels == (2 * c_rgb)
            and self._out[0].out_channels == out_c
        )
        if not out_ok:
            self._out = nn.Sequential(
                nn.Conv2d(2 * c_rgb, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU(inplace=True),
            ).to(device)

    def forward(self, x):
        rgb_feat, prior_out = x[0], x[1]
        prior_feat = prior_out[self.stage_idx - 1] if isinstance(prior_out, (list, tuple)) else prior_out
        prior_mask = PointSMCAPriorBranch._mask_cache.get(self.stage_idx, None)

        if prior_feat.shape[2:] != rgb_feat.shape[2:]:
            prior_feat = F.interpolate(prior_feat, size=rgb_feat.shape[2:], mode="bilinear", align_corners=False)
        if prior_mask is None:
            prior_mask = _norm01(prior_feat.mean(dim=1, keepdim=True))
        elif prior_mask.shape[2:] != rgb_feat.shape[2:]:
            prior_mask = F.interpolate(prior_mask, size=rgb_feat.shape[2:], mode="bilinear", align_corners=False)

        self._lazy_init(rgb_feat.shape[1], prior_feat.shape[1], rgb_feat.device)
        prior_proj = self._proj(prior_feat)
        spatial_bias = prior_mask.clamp_min(1e-4).log()
        spatial_gate = torch.sigmoid(prior_proj + torch.tanh(self.beta) * spatial_bias)

        if self.mode == "concat":
            fused = self._mix(torch.cat([rgb_feat, prior_proj * spatial_gate], dim=1))
        else:
            fused = rgb_feat * (1.0 + torch.tanh(self.gamma) * spatial_gate)

        return self._out(torch.cat([fused, prior_proj], dim=1))
