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

"""NeRF and its MLPs, with helper functions for construction and rendering."""

import functools
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Text, Tuple
from itertools import chain
import math as python_math
import logging
import gin.torch
import torch
from torch import nn
from internal import configs
from internal import coord
from internal import geopoly
from internal import image
from internal import math
from internal import ref_utils
from internal import render
from internal import stepfun
from internal import utils
from internal import sample_utils


# apply same default initialization as the Jax version
def reset_parameters(self) -> None:
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    nn.init.kaiming_uniform_(self.weight, a=python_math.sqrt(5))
    # torch.nn.init.constant_(self.weight, val=1e-3)
    if self.bias is not None:
        nn.init.constant_(self.bias, val=0)

nn.Linear.reset_parameters = reset_parameters


@gin.configurable
class Model(nn.Module):
    """ A mip-Nerf360 model containing all MLPs. """

    def __init__(
            self,
            config: Any = None,
            num_prop_samples: int = 64,
            num_nerf_samples: int = 32,
            num_levels: int = 3,
            bg_intensity_range: Tuple[float] = (1., 1.),
            anneal_slope: float = 10,
            use_viewdirs: bool = True,
            raydist_fn: Callable[..., Any] = None,
            ray_shape: str = 'cone',
            disable_integration: bool = False,
            single_jitter: bool = True,
            dilation_bias: float = 0.0025,
            dilation_multiplier: float = 0.5,
            single_mlp: bool = False,
            resample_padding: float = 0.0,
            opaque_background: bool = False,
            init_s_near: float = 0.,
            init_s_far: float = 1.,
            ):
        """
        Initializes the mip-Nerf360 model

        Args:
            config (Any): A Config class, must be set upon construction. Defaults to None.
            num_prop_samples (int, optional): The number of samples for each proposal level. Defaults to 64.
            num_nerf_samples (int, optional): The number of samples the final nerf level. Defaults to 32.
            num_levels (int, optional): The number of sampling levels (3==2 proposals, 1 nerf). Defaults to 3.
            bg_intensity_range (Tuple[float], optional): The range of background colors. Defaults to (1., 1.).
            anneal_slope (float): Higher results in more rapid annealing. Defaults to 10.
            use_viewdirs (bool, optional): If True, use view directions as input. Defaults to True.
            raydist_fn (Callable[..., Any], optional): The curve used for ray dists. Defaults to None.
            ray_shape (str, optional): The shape of cast rays ('cone' or 'cylinder'). Defaults to 'cone'
            disable_integration (bool, optional): If True, use PE instead of IDE. Defaults to False.
            single_jitter (bool, optional): If True, jitter whole rays instead of samples. Defaults to True.
            dilation_bias (float, optional): How much to dilate intervals absolutely.
            dilation_multiplier (float, optional): How much to dilate intervals relatively.
            single_mlp (bool, optional): Use the NerfMLP for all rounds of sampling. Defaults to False.
            resample_padding (bool, optional): Dirichlet/alpha "padding" on the histogram. Defaults to 0.
            opaque_background (bool, optional): If true, make the background opaque. Defaults to False.
            init_s_near (float, optional): Initial values for near bound of the rays. Defaults to 0.
            init_s_far (float, optional): Initial values for far bound of the rays. Defaults to 1.
        """
        super().__init__()
        self.config = config
        self.single_mlp = single_mlp
        self.num_prop_samples = num_prop_samples
        self.num_nerf_samples = num_nerf_samples
        self.num_levels = num_levels
        self.dilation_bias = dilation_bias
        self.dilation_multiplier = dilation_multiplier
        self.bg_intensity_range = bg_intensity_range
        self.anneal_slope = anneal_slope
        self.use_viewdirs = use_viewdirs
        self.raydist_fn = raydist_fn
        self.ray_shape = ray_shape
        self.disable_integration = disable_integration
        self.single_jitter = single_jitter
        self.single_jitter = single_jitter
        self.single_mlp = single_mlp
        self.resample_padding = resample_padding
        self.opaque_background = opaque_background
        self.init_s_near = init_s_near
        self.init_s_far = init_s_far

        # Construct MLPs. WARNING: Construction order may matter, if MLP weights are
        # being regularized.
        self.nerf_mlp = NerfMLP()
        self.prop_mlp = self.nerf_mlp if self.single_mlp else PropMLP()

    @property
    def device(self):
        return next(self.parameters()).device

    def __call__(
        self,
        rays,
        train_frac,
        compute_extras,
    ):
        """The Ref-NeRF Model.

        Args:
          rays: util.Rays -> ray origins, directions, and viewdirs.
          train_frac: float in [0, 1], what fraction of training is complete.
          compute_extras: bool, if True, compute extra quantities besides color.

        Returns:
          ret: list, [*(rgb, distance, acc)]
        """

        # Define the mapping from normalized to metric ray distance.
        _, s_to_t = coord.construct_ray_warps(
            self.raydist_fn, rays.near, rays.far)

        # Initialize the range of (normalized) distances for each ray to [0, 1],
        # and assign that single interval a weight of 1. These distances and weights
        # will be repeatedly updated as we proceed through sampling levels.
        sdist = torch.cat([
            torch.full_like(rays.near, self.init_s_near),
            torch.full_like(rays.far, self.init_s_far)
        ], dim=-1)
        weights = torch.ones_like(rays.near)
        prod_num_samples = 1

        ray_history = []
        renderings = []
        for i_level in range(self.num_levels):
            is_prop = i_level < (self.num_levels - 1)
            num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples

            # Dilate by some multiple of the expected span of each current interval,
            # with some bias added in.
            dilation = self.dilation_bias + self.dilation_multiplier * (
                self.init_s_far - self.init_s_near) / prod_num_samples

            # Record the product of the number of samples seen so far.
            prod_num_samples *= num_samples

            # After the first level (where dilation would be a no-op) optionally
            # dilate the interval weights along each ray slightly so that they're
            # overestimates, which can reduce aliasing.
            use_dilation = self.dilation_bias > 0 or self.dilation_multiplier > 0
            if i_level > 0 and use_dilation:
                sdist, weights = stepfun.max_dilate_weights(
                    sdist,
                    weights,
                    dilation,
                    domain=(torch.tensor(self.init_s_near),
                            torch.tensor(self.init_s_far)),
                    renormalize=True)
                sdist = sdist[..., 1:-1]
                weights = weights[..., 1:-1]

            # Optionally anneal the weights as a function of training iteration.
            if self.anneal_slope > 0:
                # Schlick's bias function, see https://arxiv.org/abs/2010.09714
                bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
                anneal = bias(train_frac, self.anneal_slope)
            else:
                anneal = 1.


            # A slightly more stable way to compute weights**anneal. If the distance
            # between adjacent intervals is zero then its weight is fixed to 0.
            logits_resample = torch.where(
                sdist[..., 1:] > sdist[..., :-1],
                anneal * torch.log(weights + self.resample_padding),
                -float('inf'))

            # Draw sampled intervals from each ray's current weights.
            # Optimization will usually go nonlinear if you propagate gradients
            # through sampling so detach sdist
            sdist = stepfun.sample_intervals(
                sdist,
                logits_resample,
                num_samples,
                single_jitter=self.single_jitter,
                domain=(self.init_s_near, self.init_s_far),
                use_gpu_resampling=False,
                ).detach()

            # Convert normalized distances to metric distances.
            tdist = s_to_t(sdist)

            # Cast our rays, by turning our distance intervals into Gaussians.
            gaussians = render.cast_rays(
                tdist,
                rays.origins,
                rays.directions,
                rays.radii,
                self.ray_shape,
                diag=False)

            if self.disable_integration:
                # Setting the covariance of our Gaussian samples to 0 disables the
                # "integrated" part of integrated positional encoding.
                gaussians = (gaussians[0], torch.zeros_like(gaussians[1]))


            # Push our Gaussians through one of our two MLPs.
            mlp = self.prop_mlp if is_prop else self.nerf_mlp
            ray_results = mlp(
                gaussians,
                viewdirs=rays.viewdirs if self.use_viewdirs else None,
                imageplane=rays.imageplane,
            )

            # Get the weights used by volumetric rendering (and our other losses).
            weights = render.compute_alpha_weights(
                ray_results['density'],
                tdist,
                rays.directions,
                opaque_background=self.opaque_background,
            )[0]
            if self.config.render_with_specular_density:
                if not 'specular_density' in ray_results.keys():
                    ValueError('Specular density prediction from mlps should be enabled.')
                specular_weights = render.compute_alpha_weights(
                    ray_results['specular_density'],
                    tdist,
                    rays.directions,
                    opaque_background=self.opaque_background,
                )[0]

            # Define or sample the background color for each ray.
            if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
                # If the min and max of the range are equal, just take it.
                bg_rgbs = self.bg_intensity_range[0]
            else:
                # If rendering is deterministic, use the midpoint of the range.
                bg_rgbs = (
                    self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2

            # Render each ray.
            if not self.config.render_with_specular_density:
                rendering = render.volumetric_rendering(
                    ray_results['rgb'],
                    weights,
                    tdist,
                    bg_rgbs,
                    rays.far,
                    compute_extras,
                    extras={
                        k: v
                        for k, v in ray_results.items()
                        if k.startswith('normals') or k in [
                            'roughness', 'diffuse', 'specular', 'tint']
                    },
                    srgb_mapping=self.config.srgb_mapping_type \
                        if self.config.srgb_mapping_when_rendering
                        else 'none'
                    )
            else:
                if not self.config.srgb_mapping_when_rendering:
                    ValueError('Srgb mapping should be done during rendering when using specular density.')
                rendering = render.volumetric_rendering(
                    ray_results['diffuse'],
                    weights,
                    tdist,
                    bg_rgbs,
                    rays.far,
                    compute_extras,
                    extras={
                        k: v
                        for k, v in ray_results.items()
                        if k.startswith('normals') or k in [
                            'roughness', 'tint']
                    },
                    srgb_mapping=self.config.srgb_mapping_type,
                    specular_rgbs=ray_results['specular'],
                    specular_weights=specular_weights
                    )

            if compute_extras:
                # Collect some rays to visualize directly. By naming these quantities
                # with `ray_` they get treated differently downstream --- they're
                # treated as bags of rays, rather than image chunks.
                n = self.config.vis_num_rays
                rendering['ray_sdist'] = sdist.reshape(
                    [-1, sdist.shape[-1]])[:n, :]
                rendering['ray_weights'] = (
                    weights.reshape([-1, weights.shape[-1]])[:n, :])
                rgb = ray_results['rgb']
                rendering['ray_rgbs'] = (rgb.reshape(
                    (-1,) + rgb.shape[-2:]))[:n, :, :]

            renderings.append(rendering)
            ray_results['sdist'] = sdist.clone()
            ray_results['weights'] = weights.clone()
            ray_history.append(ray_results)

        if compute_extras:
            # Because the proposal network doesn't produce meaningful colors, for
            # easier visualization we replace their colors with the final average
            # color.
            weights = [r['ray_weights'] for r in renderings]
            rgbs = [r['ray_rgbs'] for r in renderings]
            final_rgb = torch.sum(rgbs[-1] * weights[-1][..., None], dim=-2)
            avg_rgbs = [
                torch.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]
            ]
            for i in range(len(avg_rgbs)):
                renderings[i]['ray_rgbs'] = avg_rgbs[i]

        return renderings, ray_history


def construct_model(rays, config):
    """Construct a mip-NeRF 360 model.

    Args:
      rays: an example of input Rays.
      config: A Config class.

    Returns:
      model: initialized nn.Module, a NeRF model with parameters.
    """
    model = Model(config=config)
    # call model once to initialize lazy layers
    _ = model(
        rays=rays,
        train_frac=1.,
        compute_extras=False)
    return model


class MLP(nn.Module):
    """ A PosEnc MLP. """

    def __init__(
            self,
            net_depth: int = 8,
            net_width: int = 256,
            bottleneck_width: int = 256,
            net_depth_viewdirs: int = 1,
            net_width_viewdirs: int = 128,
            net_activation: Callable[..., Any] = torch.nn.functional.relu,
            min_deg_point: int = 0,
            max_deg_point: int = 12,
            weight_init: str = 'he_uniform',
            skip_layer: int = 4,
            skip_layer_dir: int = 4,
            num_rgb_channels: int = 3,
            deg_view: int = 4,
            use_reflections: bool = False,
            use_directional_enc: bool = False,
            enable_pred_roughness: bool = False,
            roughness_activation: Callable[..., Any] = torch.nn.functional.softplus,
            roughness_bias: float = -1.,
            use_diffuse_color: bool = False,
            use_specular_tint: bool = False,
            use_n_dot_v: bool = False,
            enable_pred_specular_density: bool = False,
            bottleneck_noise: float = 0.0,
            density_activation: Callable[..., Any] = torch.nn.functional.softplus,
            density_bias: float = -1.,
            density_noise: float = 0.,
            rgb_premultiplier: float = 1.,
            rgb_activation: Callable[..., Any] = torch.sigmoid,
            rgb_bias: float = 0.,
            rgb_padding: float = 0.001,
            enable_pred_normals: bool = False,
            disable_density_normals: bool = False,
            disable_rgb: bool = False,
            srgb_mapping: bool = True,
            warp_fn: Callable[..., Any] = None,
            basis_shape: str = 'icosahedron',
            basis_subdivisions: int = 2,
        ):
        """
        Initializes the PosEnc MLP

        Args:
            net_depth (int, optional): The depth of the first part of MLP. Defaults to 8.
            net_width (int, optional): The width of the first part of MLP. Defaults to 256.
            bottleneck_width (int, optional): The width of the bottleneck vector. Defaults to 256.
            net_depth_viewdirs (int, optional): The depth of the second part of MLP. Defaults to 1.
            net_width_viewdirs (int, optional): The width of the second part of MLP. Defaults to 128.
            net_activation (Callable[..., Any], optional): The activation function. Defaults to nn.ReLU.
            min_deg_point (int, optional): Min degree of positional encoding for 3D points. Defaults to 0.
            max_deg_point (int, optional): Max degree of positional encoding for 3D points. Defaults to 12.
            weight_init (str, optional): Initializer for the weights of the MLP. Defaults to 'he_uniform'.
            skip_layer (int, optional): Add a skip connection to the output of every N layers. Defaults to 4.
            skip_layer_dir (int, optional): Add a skip connection to 2nd MLP every N layers. Defaults to 4.
            num_rgb_channels (int, optional): The number of RGB channels. Defaults to 3.
            deg_view (int, optional): Degree of encoding for viewdirs or refdirs. Defaults to 4.
            use_reflections (bool, optional): If True, use refdirs instead of viewdirs. Defaults to False.
            use_directional_enc (bool, optional): If True, use IDE to encode directions. Defaults to False.
            enable_pred_roughness (bool, optional): If False and if use_directional_enc is True, use zero roughness in IDE. Defaults to False.
            roughness_activation (Callable[..., Any], optional): Roughness activation function. Defaults to nn.Softplus.
            roughness_bias (float, optional): Shift added to raw roughness pre-activation. Defaults to -1..
            use_diffuse_color (bool, optional): If True, predict diffuse & specular colors. Defaults to False.
            use_specular_tint (bool, optional): If True, predict tint. Defaults to False.
            use_n_dot_v (bool, optional): If True, feed dot(n * viewdir) to 2nd MLP. Defaults to False.
            bottleneck_noise (float, optional): Std. deviation of noise added to bottleneck. Defaults to 0.0.
            density_activation (Callable[..., Any], optional): Density activation. Defaults to nn.Softplus.
            density_bias (float, optional): Shift added to raw densities pre-activation. Defaults to -1..
            density_noise (float, optional): Standard deviation of noise added to raw density. Defaults to 0..
            rgb_premultiplier (float, optional): Premultiplier on RGB before activation. Defaults to 1..
            rgb_activation (Callable[..., Any], optional): The RGB activation. Defaults to nn.Sigmoid.
            rgb_bias (float, optional): The shift added to raw colors pre-activation. Defaults to 0..
            rgb_padding (float, optional): Padding added to the RGB outputs. Defaults to 0.001.
            enable_pred_normals (bool, optional): If True compute predicted normals. Defaults to False.
            disable_density_normals (bool, optional): If True don't compute normals. Defaults to False.
            disable_rgb (bool, optional): If True don't output RGB. Defaults to False.
            warp_fn (Callable[..., Any], optional): The ray warp function. Defaults to None.
            basis_shape (str, optional):  `octahedron` or `icosahedron`. Defaults to 'icosahedron'.
            basis_subdivisions (int, optional): Tesselation count. 'octahedron' + 1 == eye(3). Defaults to 2.

        Raises:
            ValueError: If use_reflections is set normals estimation is disabled
        """
        super().__init__()

        self.net_depth = net_depth
        self.net_width = net_width
        self.bottleneck_width = bottleneck_width
        self.net_depth_viewdirs = net_depth_viewdirs
        self.net_width_viewdirs = net_width_viewdirs
        self.net_activation = net_activation
        self.min_deg_point = min_deg_point
        self.max_deg_point = max_deg_point
        self.weight_init = weight_init
        self.skip_layer = skip_layer
        self.skip_layer_dir = skip_layer_dir
        self.num_rgb_channels = num_rgb_channels
        self.deg_view = deg_view
        self.use_reflections = use_reflections
        self.use_directional_enc = use_directional_enc
        self.enable_pred_roughness = enable_pred_roughness
        self.roughness_activation = roughness_activation
        self.roughness_bias = roughness_bias
        self.use_diffuse_color = use_diffuse_color
        self.use_specular_tint = use_specular_tint
        self.use_n_dot_v = use_n_dot_v
        self.enable_pred_specular_density = enable_pred_specular_density
        self.bottleneck_noise = bottleneck_noise
        self.density_activation = density_activation
        self.density_bias = density_bias
        self.density_noise = density_noise
        self.rgb_premultiplier = rgb_premultiplier
        self.rgb_activation = rgb_activation
        self.rgb_bias = rgb_bias
        self.rgb_padding = rgb_padding
        self.enable_pred_normals = enable_pred_normals
        self.disable_density_normals = disable_density_normals
        self.disable_rgb = disable_rgb
        self.srgb_mapping = srgb_mapping
        self.warp_fn = warp_fn
        self.basis_shape = basis_shape
        self.basis_subdivisions = basis_subdivisions

        # Make sure that normals are computed if reflection direction is used.
        if self.use_reflections and not (self.enable_pred_normals or
                                         not self.disable_density_normals):
            raise ValueError(
                'Normals must be computed for reflection directions.')

        # Make sure that...
        if self.enable_pred_specular_density and not self.use_diffuse_color:
            raise ValueError(
                'Specular density is useless if not using diffuse color.')

        # Precompute and store (the transpose of) the basis being used.
        self.pos_basis_t = torch.tensor(
            geopoly.generate_basis(self.basis_shape, self.basis_subdivisions)).T

        # Precompute and define viewdir or refdir encoding function.
        if self.use_directional_enc:
            self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
        else:
            def dir_enc_fn(direction, _):
                return coord.pos_enc(
                    direction, min_deg=0, max_deg=self.deg_view,
                    append_identity=True)
            self.dir_enc_fn = dir_enc_fn

        # spatial MLP
        self.spatial_net = nn.ModuleList(
            [nn.LazyLinear(self.net_width) for i in range(self.net_depth)])

        # raw density layer
        self.raw_density = nn.Linear(self.net_width, 1)
        if self.enable_pred_specular_density:
            self.raw_specular_density = nn.Linear(self.net_width, 1)

        # predicted normals
        if self.enable_pred_normals:
            self.grad_pred = nn.Linear(self.net_width, 3)

        # roughness layer
        if self.enable_pred_roughness:
            self.raw_roughness = nn.Linear(self.net_width, 1)

        # diffuse layer
        if self.use_diffuse_color:
            self.raw_rgb_diffuse = nn.Linear(self.net_width, self.num_rgb_channels)

        # tint layer
        if self.use_specular_tint:
            self.raw_tint = nn.Linear(self.net_width, 3)

        # bottleneck layer
        if self.bottleneck_width > 0:
            self.bottleneck = nn.Linear(self.net_width, self.bottleneck_width)

        # directional MLP
        self.viewdir_mlp = nn.ModuleList(
            [nn.LazyLinear(self.net_width_viewdirs)
             for i in range(self.net_depth_viewdirs)])

        # rgb layer
        self.rgb = nn.LazyLinear(self.num_rgb_channels)

    def __call__(self,
                 gaussians,
                 viewdirs=None,
                 imageplane=None,
        ):
        """Evaluate the MLP.

        Args:
          gaussians: a tuple containing:                                           /
            - mean: [..., n, 3], coordinate means, and                             /
            - cov: [..., n, 3{, 3}], coordinate covariance matrices.
          viewdirs: torch.tensor(float32), [..., 3], if not None, this variable will
            be part of the input to the second part of the MLP concatenated with the
            output vector of the first part of the MLP. If None, only the first part
            of the MLP will be used with input x. In the original paper, this
            variable is the view direction.
          imageplane: torch.tensor(float32), [batch, 2], xy image plane coordinates
            for each ray in the batch. Useful for image plane operations such as a
            learned vignette mapping.

        Returns:
          rgb: torch.tensor(float32), with a shape of [..., num_rgb_channels].
          density: torch.tensor(float32), with a shape of [...].
          normals_pred: torch.tensor(float32), with a shape of [..., 3], or None.
          roughness: torch.tensor(float32), with a shape of [..., 1], or None.
        """
        # get inputs in the form of means and variances representation the ray segments
        means, covs = gaussians
        
        if self.training:
            means.requires_grad_()

        # lift means and vars of position input
        lifted_means, lifted_vars = (
            coord.lift_and_diagonalize(means, covs, self.pos_basis_t.to(means.device)))
        
        # apply integrated position encoding to position input
        x = coord.integrated_pos_enc(lifted_means, lifted_vars,
                                     self.min_deg_point, self.max_deg_point)
        inputs = x

        # Evaluate network to produce the output density.
        inputs = x
        for i, layer in enumerate(self.spatial_net):
            x = layer(x)
            x = self.net_activation(x)
            if i % self.skip_layer == 0 and i > 0:
                x = torch.concatenate([x, inputs], dim=-1)

        raw_density = self.raw_density(x)[..., 0]
        if self.enable_pred_specular_density:
            raw_specular_density = self.raw_specular_density(x)[..., 0]

        # Add noise to regularize the density predictions if needed.
        if self.density_noise > 0:
            raw_density += self.density_noise * torch.normal(0, 1, raw_density.shape)
            if self.enable_pred_specular_density:
                raw_specular_density += self.density_noise * torch.normal(0, 1, raw_specular_density.shape)

        # calculate normals through density gradients
        normals = None
        # elif self.training:
        #     # https://github.com/Enigmatisms/NeRF/blob/1c535492f89dccb483aa8810106733d2d6a9a52b/py/ref_model.py#L120
        #     grad, = torch.autograd.grad(
        #         raw_density, means, torch.ones_like(raw_density),
        #         retain_graph=True)
        #     grad_norm = grad.norm(dim=-1, keepdim=True)
        #     normals = -ref_utils.l2_normalize(grad_norm)
        # else:
        #     normals = None
        if not self.disable_density_normals and self.training:
            # https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/fields/base_field.py
            # raw_density.backward(
            #     gradient=torch.ones_like(raw_density),
            #     inputs=means, retain_graph = True)
            # normals = -ref_utils.l2_normalize(means.grad)
            normals = -ref_utils.l2_normalize(torch.autograd.grad(raw_density.sum(), means, retain_graph=True)[0])

        if self.enable_pred_normals:
            # predict normals
            grad_pred = self.grad_pred(x)
            # normalize negative predicted gradients to get predicted normal vectors.
            normals_pred = -ref_utils.l2_normalize(grad_pred)
            normals_to_use = normals_pred
        else:
            grad_pred = None
            normals_pred = None
            normals_to_use = normals

        # Apply bias and activation to raw density
        density = self.density_activation(raw_density + self.density_bias)
        if self.enable_pred_specular_density:
            specular_density = self.density_activation(raw_specular_density + self.density_bias)

        roughness = 0
        if self.disable_rgb:
            rgb = torch.zeros_like(means)
        else:
            if viewdirs is not None:
                # Predict diffuse color.
                if self.use_diffuse_color:
                    raw_rgb_diffuse = self.raw_rgb_diffuse(x)

                if self.use_specular_tint:
                    tint = torch.sigmoid(self.raw_tint(x))

                if self.enable_pred_roughness:
                    roughness = self.roughness_activation(
                        self.raw_roughness(x) + self.roughness_bias)

                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    bottleneck = self.bottleneck(x)

                    # Add bottleneck noise.
                    if self.bottleneck_noise > 0:
                        bottleneck += self.bottleneck_noise * torch.normal(
                            0, 1, bottleneck.shape)

                    x = [bottleneck]
                else:
                    x = []

                # Encode view (or reflection) directions.
                if self.use_reflections:
                    # Compute reflection directions. Note that we flip viewdirs before
                    # reflecting, because they point from the camera to the point,
                    # whereas ref_utils.reflect() assumes they point toward the camera.
                    # Returned refdirs then point from the point to the environment.
                    refdirs = ref_utils.reflect(
                        -viewdirs[..., None, :], normals_to_use)
                    # Encode reflection directions.
                    dir_enc = self.dir_enc_fn(refdirs, roughness)
                else:
                    # Encode view directions.
                    dir_enc = self.dir_enc_fn(viewdirs, roughness)

                    # broadcast directional encoding to bottleneck's dimensions
                    dir_enc = torch.broadcast_to(
                        dir_enc[..., None, :],
                        bottleneck.shape[:-1] + (dir_enc.shape[-1],))

                # Append view (or reflection) direction encoding to bottleneck vector.
                x.append(dir_enc)

                # Append dot product between normal vectors and view directions.
                if self.use_n_dot_v:
                    dotprod = torch.sum(
                        normals_to_use * viewdirs[..., None, :],
                        dim=-1, keepdims=True)
                    x.append(dotprod)

                # Concatenate bottleneck, directional encoding, and nv product
                x = torch.cat(x, dim=-1)

                # Output of the second part of MLP.
                inputs = x
                for i, layer in enumerate(self.viewdir_mlp):
                    x = layer(x)
                    x = self.net_activation(x)
                    if i % self.skip_layer == 0 and i > 0:
                        x = torch.concatenate([x, inputs], dim=-1)


            # If using diffuse/specular colors, then `rgb` is treated as linear
            # specular color. Otherwise it's treated as the color itself.
            rgb = self.rgb_activation(
                self.rgb_premultiplier * self.rgb(x) + self.rgb_bias)

            if self.use_diffuse_color:
                # Initialize linear diffuse color around 0.25, so that the combined
                # linear color is initialized around 0.5.
                three = torch.tensor(3.0, dtype=torch.float32)
                diffuse_linear = torch.sigmoid(raw_rgb_diffuse - torch.log(three))
                if self.use_specular_tint:
                    specular_linear = tint * rgb
                else:
                    specular_linear = 0.5 * rgb

                if self.srgb_mapping:
                    # Combine specular and diffuse components and tone map to sRGB.
                    rgb = torch.clip(
                        image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)
                else:
                    rgb = specular_linear + diffuse_linear
            # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        ray_results = dict(
            density=density,
            rgb=rgb,
        )
        if not self.disable_density_normals:
            ray_results['normals'] = normals
        if self.enable_pred_normals:
            ray_results['normals_pred'] = normals_pred
            ray_results['grad_pred'] = grad_pred
        if self.use_specular_tint:
            ray_results['tint'] = tint
        if self.use_diffuse_color:
            ray_results['diffuse'] = diffuse_linear
            ray_results['specular'] = specular_linear
            if self.enable_pred_specular_density:
                ray_results['specular_density'] = specular_density
        if self.enable_pred_roughness:
            ray_results['roughness'] = roughness

        return ray_results


@gin.configurable
class NerfMLP(MLP):
    pass


@gin.configurable
class PropMLP(MLP):
    pass


def render_image(render_fn: Callable[[torch.tensor, utils.Rays],
                                     Tuple[List[Mapping[Text, torch.tensor]],
                                           List[Tuple[torch.tensor, ...]]]],
                 rays: utils.Rays,
                 config: configs.Config,
                 verbose: bool = True,
                 device=torch.device('cuda')) -> MutableMapping[Text, Any]:
    """Render all the pixels of an image (in test mode).

    Args:
      render_fn: function, jit-ed render function mapping (rays) -> pytree.
      rays: a `Rays` pytree, the rays to be rendered.
      config: A Config class.
      verbose: print progress indicators.

    Returns:
      rgb: torch.tensor, rendered color image.
      disp: torch.tensor, rendered disparity image.
      acc: torch.tensor, rendered accumulated weights per pixel.
    """
    torch.cuda.synchronize()
    height, width = rays.origins.shape[:2]
    num_rays = height * width
    rays = rays.reshape(num_rays, -1)
    chunks = []
    idx0s = range(0, num_rays, config.render_chunk_size)

    for idx0 in idx0s:
        chunk_rays = rays[idx0:idx0 + config.render_chunk_size]
        chunk_rays.to(device)
        chunk_renderings, _ = render_fn(chunk_rays)

        # Gather the final pass for 2D buffers and all passes for ray bundles.
        chunk_rendering = chunk_renderings[-1]
        for k in chunk_renderings[0]:
            if k.startswith('ray_'):
                chunk_rendering[k] = [r[k] for r in chunk_renderings]

        chunk_rendering = {k: utils.recursive_detach(v)
                           for k, v in chunk_rendering.items()}
        chunks.append(chunk_rendering)

    # Concatenate all chunks
    rendering = utils.merge_chunks(chunks)

    # reshape renderings 2D images
    for k, z in rendering.items():
        if not k.startswith('ray_'):
            # Reshape 2D buffers into original image shape.
            rendering[k] = z.reshape((height, width) + z.shape[1:])

    # After all of the ray bundles have been concatenated together, extract a
    # new random bundle (deterministically) from the concatenation that is the
    # same size as one of the individual bundles.
    keys = [k for k in rendering if k.startswith('ray_')]
    if keys:
        temp_num_rays = rendering[keys[0]][0].shape[0]
        ray_idx = torch.randperm(temp_num_rays)
        ray_idx = ray_idx[:config.vis_num_rays]
        for k in keys:
            rendering[k] = [r[ray_idx] for r in rendering[k]]

    return rendering
