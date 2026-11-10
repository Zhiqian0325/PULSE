"""DENSE_DISK_V1: a shared soft/hard-mask disk integral, in pixel cubed."""

import math
import torch
from torch.nn import functional as F


def volume(mask, fixed_axis=None):
    m = mask.float() if mask.dtype != torch.float64 else mask
    n, h, w = m.shape
    if (h, w) != (128, 128):
        raise ValueError("Expected 128x128 geometry")
    coord = torch.arange(128, dtype=m.dtype, device=m.device)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    mass = m.sum((1, 2))
    norm = mass.clamp_min(1e-8)
    with torch.no_grad():
        cx = (m * xx).sum((1, 2)) / norm
        cy = (m * yy).sum((1, 2)) / norm
        dx, dy = xx - cx[:, None, None], yy - cy[:, None, None]
        a = (m * dx.square()).sum((1, 2)) / norm
        b = (m * dx * dy).sum((1, 2)) / norm
        d = (m * dy.square()).sum((1, 2)) / norm
        gap = torch.sqrt((a - d).square() + 4 * b.square())
        top = (a + d + gap) / 2
        ambiguous = (top <= 1e-6) | (gap / top.clamp_min(1e-6) < 0.05)
        angle = 0.5 * torch.atan2(2 * b, a - d)
        angle = torch.where(ambiguous, torch.full_like(angle, math.pi / 2), angle)
        axis = torch.stack((angle.cos(), angle.sin()), 1)
        axis = torch.where(axis[:, 1:2] < 0, -axis, axis)
        if fixed_axis is not None:
            axis = fixed_axis.detach().to(m)
        v = torch.stack((-axis[:, 1], axis[:, 0]), 1)
        t = torch.arange(192, device=m.device, dtype=m.dtype) - 95.5
        u_grid, v_grid = torch.meshgrid(t, t, indexing="ij")
        grid = (
            u_grid[None, :, :, None] * axis[:, None, None, :]
            + v_grid[None, :, :, None] * v[:, None, None, :]
            + 63.5
        )
        grid = grid * (2.0 / 127.0) - 1.0
    rotated = F.grid_sample(
        m[:, None], grid, align_corners=True, mode="bilinear", padding_mode="zeros"
    )[:, 0]
    diameters = rotated.sum(2)
    values = math.pi / 4 * diameters.square().sum(1)
    return {
        "volume": values,
        "mass": mass,
        "axis": axis,
        "ambiguous": ambiguous,
        "diameters": diameters,
    }


def pair_geometry(masks):
    b = len(masks)
    g = volume(masks.reshape(-1, 128, 128))
    v = g["volume"].reshape(b, 2)
    mass = g["mass"].reshape(b, 2)
    z = (v[:, 1] + 1).log() - (v[:, 0] + 1).log()
    soft_ef = 100 * (1 - z.clamp(-7.0, math.log(2.0)).exp())
    raw = 100 * (1 - v[:, 1] / v[:, 0].clamp_min(1e-8))
    valid = (
        (mass >= 8).all(1)
        & (v > 0).all(1)
        & torch.isfinite(v).all(1)
        & torch.isfinite(raw)
    )
    return {
        "volume": v,
        "mass": mass,
        "soft_ef": soft_ef,
        "raw_ef": raw,
        "valid": valid,
        "upper_clamped": z > math.log(2.0),
        "ambiguous": g["ambiguous"].reshape(b, 2),
    }
