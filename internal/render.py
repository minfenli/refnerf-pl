# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for shooting and rendering rays."""

import torch
from internal import stepfun
from internal import image


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d[..., None, :] * t_mean[..., None]

    eps = torch.tensor(1e-10, device=d.device)
    d_mag_sq = torch.maximum(eps, torch.sum(d**2, dim=-1, keepdims=True))

    if diag:
        d_outer_diag = d**2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1], device=d.device)
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `d` is normalized.

    Args:
      d: torch.float32 3-vector, the axis of the cone
      t0: float, the starting distance of the frustum.
      t1: float, the ending distance of the frustum.
      base_radius: float, the scale of the radius as a function of distance.
      diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
      stable: boolean, whether or not to use the stable computation described in
        the paper (setting this to False will cause catastrophic failure).

    Returns:
      a Gaussian (mean and covariance).
    """
    if stable:
        # Equation 7 in the paper (https://arxiv.org/abs/2103.13415).
        mu = (t0 + t1) / 2  # The average of the two `t` values.
        hw = (t1 - t0) / 2  # The half-width of the two `t` values.
        eps = torch.tensor(torch.finfo(torch.float32).eps)
        t_mean = mu + (2 * mu * hw**2) / torch.maximum(eps, 3 * mu**2 + hw**2)
        denom = torch.maximum(eps, 3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * hw**4 * (12 * mu**2 - hw**2) / denom**2
        r_var = (mu**2) / 4 + (5 / 12) * hw**2 - (4 / 15) * (hw**4) / denom
    else:
        # Equations 37-39 in the paper.
        t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
        r_var = 3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3)
        t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
        t_var = t_mosq - t_mean**2
    r_var *= base_radius**2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """Approximate a cylinder as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and radius is the
    radius. Does not renormalize `d`.

    Args:
      d: torch.float32 3-vector, the axis of the cylinder
      t0: float, the starting distance of the cylinder.
      t1: float, the ending distance of the cylinder.
      radius: float, the radius of the cylinder
      diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

    Returns:
      a Gaussian (mean and covariance).
    """
    t_mean = (t0 + t1) / 2
    r_var = radius**2 / 4
    t_var = (t1 - t0)**2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(tdist, origins, directions, radii, ray_shape, diag=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

    Args:
      tdist: float array, the "fencepost" distances along the ray.
      origins: float array, the ray origin coordinates.
      directions: float array, the ray direction vectors.
      radii: float array, the radii (base radii for cones) of the rays.
      ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
      diag: boolean, whether or not the covariance matrices should be diagonal.

    Returns:
      a tuple of arrays of means and covariances.
    """
    t0 = tdist[..., :-1]
    t1 = tdist[..., 1:]
    if ray_shape == 'cone':
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == 'cylinder':
        gaussian_fn = cylinder_to_gaussian
    else:
        raise ValueError('ray_shape must be \'cone\' or \'cylinder\'')
    means, covs = gaussian_fn(directions, t0, t1, radii, diag)
    means = means + origins[..., None, :]
    return means, covs


def compute_alpha_weights(density, tdist, dirs, opaque_background=False):
    """Helper function for computing alpha compositing weights."""
    t_delta = tdist[..., 1:] - tdist[..., :-1]
    delta = t_delta * torch.linalg.norm(dirs[..., None, :], dim=-1)
    density_delta = density * delta

    if opaque_background:
        # Equivalent to making the final t-interval infinitely wide.
        density_delta = torch.cat([
            density_delta[..., :-1],
            torch.full_like(density_delta[..., -1:], torch.inf)], dim=-1)

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)], dim=-1))
    weights = alpha * trans
    return weights, alpha, trans


def volumetric_rendering(rgbs,
                         weights,
                         tdist,
                         bg_rgbs,
                         t_far,
                         compute_extras,
                         extras=None,
                         srgb_mapping=False,
                         specular_rgbs=None,
                         specular_weights=None):
    """Volumetric Rendering Function.

    Args:
      rgbs: torch.ndarray(float32), color, [batch_size, num_samples, 3]
      weights: torch.ndarray(float32), weights, [batch_size, num_samples].
      tdist: torch.ndarray(float32), [batch_size, num_samples].
      bg_rgbs: torch.ndarray(float32), the color(s) to use for the background.
      t_far: torch.ndarray(float32), [batch_size, 1], the distance of the far plane.
      compute_extras: bool, if True, compute extra quantities besides color.
      extras: dict, a set of values along rays to render by alpha compositing.

    Returns:
      rendering: a dict containing an rgb image of size [batch_size, 3], and other
        visualizations if compute_extras=True.
    """
    eps = torch.tensor(torch.finfo(torch.float32).eps)
    rendering = {}

    if not specular_rgbs is None and not specular_weights is None:
        acc = weights.sum(dim=-1) + specular_weights.sum(dim=-1)
        # The weight of the background.
        bg_w = torch.maximum(torch.tensor(0), 1 - acc[..., None])
        diffuse_rgb = (weights[..., None] * rgbs).sum(dim=-2)
        specular_rgb = (specular_weights[..., None] * specular_rgbs).sum(dim=-2)
        rgb = diffuse_rgb + specular_rgb + bg_w * bg_rgbs
        rendering['diffuse'] = diffuse_rgb
        rendering['specular'] = specular_rgb
    else:
        acc = weights.sum(dim=-1)
        # The weight of the background.
        bg_w = torch.maximum(torch.tensor(0), 1 - acc[..., None])
        rgb = (weights[..., None] * rgbs).sum(dim=-2) + bg_w * bg_rgbs


    if srgb_mapping:
        torch.clip(image.linear_to_srgb(rgb), 0.0, 1.0)
    rendering['rgb'] = rgb

    t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
    rendering['distance'] = (weights[..., None] * t_mids[..., None]).sum(dim=-2)
    rendering['acc'] = acc

    if compute_extras:
        if extras is not None:
            for k, v in extras.items():
                if v is not None:
                    rendering[k] = (weights[..., None] * v).sum(dim=-2)       

        def expectation(x):
            return (weights * x).sum(dim=-1) / torch.max(eps, acc)

        # For numerical stability this expectation is computing using log-distance.
        rendering['distance_mean'] = torch.clip(
            torch.nan_to_num(
                torch.exp(expectation(torch.log(t_mids))), torch.inf),
            tdist[..., 0], tdist[..., -1])

        # Add an extra fencepost with the far distance at the end of each ray, with
        # whatever weight is needed to make the new weight vector sum to exactly 1
        # (`weights` is only guaranteed to sum to <= 1, not == 1).
        t_aug = torch.cat([tdist, t_far], dim=-1)
        weights_aug = torch.cat([weights, bg_w], dim=-1)

        ps = [5, 50, 95]
        distance_percentiles = stepfun.weighted_percentile(
            t_aug, weights_aug, ps)

        for i, p in enumerate(ps):
            s = 'median' if p == 50 else 'percentile_' + str(p)
            rendering['distance_' + s] = distance_percentiles[..., i]

    return rendering
