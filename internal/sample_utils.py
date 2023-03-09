import torch
import numpy as np
from internal import utils

def euler_angles_to_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)

        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")

    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip("XYZ", torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

@torch.no_grad()
def sample_noisy_rays(rays: utils.Rays, rendering: dict,
                      sample_angle_range: float = 0., 
                      sample_noise_size: int = 128,
                      warmup_ratio: float = 1.) -> utils.Rays:
    xyz_angle = torch.zeros(3, device=rendering['distance'].device).uniform_(
        0, sample_angle_range/180 * np.pi * warmup_ratio)
    T = euler_angles_to_matrix(xyz_angle)

    distance = rendering['distance'][:sample_noise_size]
    if len(distance.shape) == len(rays.origins.shape)-1:
        distance = distance[..., None][:sample_noise_size]
    elif len(distance.shape) != len(rays.origins.shape):
        raise ValueError('The dimension of distance is wrong.')
    
    viewdirs_ = rays.viewdirs[:sample_noise_size]@T.T
    directions_ = rays.directions[:sample_noise_size]@T.T
    origins_ = rays.origins[:sample_noise_size] + \
               distance * rays.directions[:sample_noise_size] - \
               distance * directions_

    return utils.Rays(
        origins=origins_,
        directions=directions_,
        viewdirs=viewdirs_,
        radii=rays.radii[:sample_noise_size],
        imageplane=rays.imageplane[:sample_noise_size],
        lossmult=rays.lossmult[:sample_noise_size],
        near=rays.near[:sample_noise_size],
        far=rays.far[:sample_noise_size],
        cam_idx=rays.cam_idx[:sample_noise_size],
    )