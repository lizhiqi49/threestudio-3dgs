import numpy as np
import torch
from pytorch3d.ops import knn_points
from pytorch3d.transforms import quaternion_apply, matrix_to_quaternion, quaternion_to_matrix
from ..geometry.gaussian_base import GaussianBaseModel

scale_activation = torch.exp
scale_inverse_activation = torch.log


def _initialize_radiuses_gauss_rasterizer(sugar):
    """Function to initialize the  of a SuGaR model.

    Args:
        sugar (SuGaR): SuGaR model.

    Returns:
        Tensor: Tensor with shape (n_points, 4+3) containing
            the initial quaternions and scaling factors.
    """
    # Initialize learnable radiuses
    # sugar.image_height = int(sugar.nerfmodel.training_cameras.height[0].item())
    # sugar.image_width = int(sugar.nerfmodel.training_cameras.width[0].item())
    #
    # all_camera_centers = sugar.nerfmodel.training_cameras.camera_to_worlds[..., 3]
    # all_camera_dists = torch.cdist(sugar.points, all_camera_centers)[None]
    # d_charac = all_camera_dists.mean(-1, keepdim=True)
    #
    # ndc_factor = 1.
    # sugar.min_ndc_radius = ndc_factor * 2. / min(sugar.image_height, sugar.image_width)
    # sugar.max_ndc_radius = ndc_factor * 2. * 0.05  # 2. * 0.01
    # sugar.min_radius = sugar.min_ndc_radius / sugar.focal_factor * d_charac
    # sugar.max_radius = sugar.max_ndc_radius / sugar.focal_factor * d_charac

    knn = knn_points(sugar.points[None], sugar.points[None], K=4)
    use_sqrt = True
    use_mean = False
    initial_radius_normalization = 1.  # 1., 0.1
    if use_sqrt:
        knn_dists = torch.sqrt(knn.dists[..., 1:])
    else:
        knn_dists = knn.dists[..., 1:]
    if use_mean:
        print("Use mean to initialize scales.")
        radiuses = knn_dists.mean(-1, keepdim=True).clamp_min(0.0000001) * initial_radius_normalization
    else:
        print("Use min to initialize scales.")
        radiuses = knn_dists.min(-1, keepdim=True)[0].clamp_min(0.0000001) * initial_radius_normalization

    res = inverse_radius_fn(radiuses=radiuses)
    sugar.radius_dim = res.shape[-1]

    return res


def inverse_radius_fn(radiuses: torch.Tensor):
    scales = scale_inverse_activation(radiuses.expand(-1, -1, 3).clone())
    quaternions = matrix_to_quaternion(
        torch.eye(3)[None, None].repeat(1, radiuses.shape[1], 1, 1).to(radiuses.device)
    )
    return torch.cat([quaternions, scales], dim=-1)


class SuGaR():
    def __init__(
        self,
        gaussians: GaussianBaseModel,
        initialize: bool = True,
        keep_track_of_knn: bool = False,
        knn_to_track: int = 16,
        surface_mesh_to_bind=None,  # Open3D mesh
        beta_mode='average',  # 'learnable', 'average', 'weighted_average'
    ):
        """
        Args:
            gaussians (GaussianSplattingWrapper): A vanilla Gaussian Splatting model trained for 7k iterations.
            initialize (bool, optional): Whether to initialize the radiuses. Defaults to True.

            keep_track_of_knn (bool, optional): Whether to keep track of the KNN information for training regularization. Defaults to False.
            knn_to_track (int, optional): Number of KNN to track. Defaults to 16.
            surface_mesh_to_bind (None, optional): Surface mesh to bind the Gaussians to. Defaults to None.
            beta_mode (str, optional): Whether to use a learnable beta, or to average the beta values. Defaults to 'average'.

        """

        self.gaussians = gaussians
        # initialize points
        if surface_mesh_to_bind is not None:
            ## wait to finish
            self._n_points = len(self.gaussians.get_xyz)
            pass
        else:
            self.binded_to_surface_mesh = False
            n_points = len(self.gaussians.get_xyz)

        # initialize radiues
        # self.scale_activation = scale_activation
        # self.scale_inverse_activation = scale_inverse_activation
        # if not self.binded_to_surface_mesh:
        #     if initialize:
        #         radiuses = _initialize_radiuses_gauss_rasterizer(self, )
        #         print("Initialized radiuses for 3D Gauss Rasterizer")
        #     else:
        #         radiuses = torch.rand(1, n_points, self.radius_dim, device=gaussians.device)
        #         self.min_radius = self.min_ndc_radius / self.focal_factor * 0.005  # 0.005
        #         self.max_radius = self.max_ndc_radius / self.focal_factor * 2.  # 2.
        #
        #     # reset the scaling and rotation for regularization
        #     gaussians.set_scaling(radiuses[0, ..., 4:])
        #     gaussians.set_rotation(radiuses[0, ..., :4])

        self.keep_track_of_knn = keep_track_of_knn
        if keep_track_of_knn:
            self.knn_to_track = knn_to_track
            knns = knn_points(gaussians.get_xyz[None], gaussians.get_xyz[None], K=knn_to_track)
            self.knn_dists = knns.dists[0]
            self.knn_idx = knns.idx[0]

        # Beta mode
        self.beta_mode = beta_mode
        if beta_mode == 'learnable':
            with torch.no_grad():
                log_beta = self.gaussians.get_scaling.mean().log().view(1, )
            self._log_beta = torch.nn.Parameter(
                log_beta.to(self.gaussians.device),
            ).to(self.gaussians.device)


    @property
    def points(self):
        return self.gaussians.get_xyz

    @property
    def scaling(self):
        if not self.binded_to_surface_mesh:
            scales = self.gaussians.get_scaling
        else:
            scales = None
            # scales = torch.cat([
            #     self.surface_mesh_thickness * torch.ones(len(self.gaussians.get_scaling), 1, device=self.device),
            #     self.scale_activation(self._scales)
            # ], dim=-1)
        return scales

    @property
    def strengths(self):
        return self.gaussians.get_opacity

    @property
    def n_points(self):
        if not self.binded_to_surface_mesh:
            return len(self.gaussians.get_xyz)
        else:
            return self._n_points

    @property
    def device(self):
        return self.gaussians.device

    @property
    def quaternions(self):
        if not self.binded_to_surface_mesh:
            quaternions = self.gaussians.get_rotation
        else:
            quaternions = None
        return quaternions

    def sample_points_in_gaussians(self, num_samples, sampling_scale_factor=1., mask=None,
                                   probabilities_proportional_to_opacity=False,
                                   probabilities_proportional_to_volume=True, ):
        """Sample points in the Gaussians.

        Args:
            num_samples (_type_): _description_
            sampling_scale_factor (_type_, optional): _description_. Defaults to 1..
            mask (_type_, optional): _description_. Defaults to None.
            probabilities_proportional_to_opacity (bool, optional): _description_. Defaults to False.
            probabilities_proportional_to_volume (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if mask is None:
            scaling = self.scaling
        else:
            scaling = self.scaling[mask]

        if probabilities_proportional_to_volume:
            areas = scaling[..., 0] * scaling[..., 1] * scaling[..., 2]
        else:
            areas = torch.ones_like(scaling[..., 0])

        if probabilities_proportional_to_opacity:
            if mask is None:
                areas = areas * self.strengths.view(-1)
            else:
                areas = areas * self.strengths[mask].view(-1)
        areas = areas.abs()
        cum_probs = areas.cumsum(dim=-1) / areas.sum(dim=-1, keepdim=True)

        # sample from multinomial distribution with probability cum_probs, return indices
        # if probabilities_proportional_to_volume and probabilities_proportional_to_opacity are None, then sample uniformly from 0~unmasked points num
        random_indices = torch.multinomial(cum_probs, num_samples=num_samples, replacement=True)

        if mask is not None:
            # get the valid indices for 3d gaussians
            valid_indices = torch.arange(self.n_points, device=self.device)[mask]
            random_indices = valid_indices[random_indices]

        # add noise to point
        random_points = self.points[random_indices] + quaternion_apply(
            self.quaternions[random_indices],
            sampling_scale_factor * self.scaling[random_indices] * torch.randn_like(self.points[random_indices]))

        return random_points, random_indices


    def reset_neighbors(self, knn_to_track: int = None):
        if self.binded_to_surface_mesh:
            print("WARNING! You should not reset the neighbors of a surface mesh.")
            print("Then, neighbors reset will be ignored.")
        else:
            if not hasattr(self, 'knn_to_track'):
                if knn_to_track is None:
                    knn_to_track = 16
                self.knn_to_track = knn_to_track
            else:
                if knn_to_track is None:
                    knn_to_track = self.knn_to_track
                    # Compute KNN
            with torch.no_grad():
                self.knn_to_track = knn_to_track
                knns = knn_points(self.points[None], self.points[None], K=knn_to_track)
                self.knn_dists = knns.dists[0]
                self.knn_idx = knns.idx[0]

    def get_covariance(self, return_full_matrix=False, return_sqrt=False, inverse_scales=False):
        scaling = self.scaling
        if inverse_scales:
            scaling = 1. / scaling.clamp(min=1e-8)
        scaled_rotation = quaternion_to_matrix(self.quaternions) * scaling[:, None]
        if return_sqrt:
            return scaled_rotation

        cov3Dmatrix = scaled_rotation @ scaled_rotation.transpose(-1, -2)
        if return_full_matrix:
            return cov3Dmatrix

        cov3D = torch.zeros((cov3Dmatrix.shape[0], 6), dtype=torch.float, device=self.device)
        cov3D[:, 0] = cov3Dmatrix[:, 0, 0]
        cov3D[:, 1] = cov3Dmatrix[:, 0, 1]
        cov3D[:, 2] = cov3Dmatrix[:, 0, 2]
        cov3D[:, 3] = cov3Dmatrix[:, 1, 1]
        cov3D[:, 4] = cov3Dmatrix[:, 1, 2]
        cov3D[:, 5] = cov3Dmatrix[:, 2, 2]

        return cov3D

    def get_field_values(self, x, gaussian_idx=None,
                         closest_gaussians_idx=None,
                         gaussian_strengths=None,
                         gaussian_centers=None,
                         gaussian_inv_scaled_rotation=None,
                         return_sdf=True, density_threshold=1., density_factor=1.,
                         return_sdf_grad=False, sdf_grad_max_value=10.,
                         opacity_min_clamp=1e-16,
                         return_closest_gaussian_opacities=False,
                         return_beta=False, ):
        if gaussian_strengths is None:
            gaussian_strengths = self.strengths
        if gaussian_centers is None:
            gaussian_centers = self.points
        if gaussian_inv_scaled_rotation is None:
            gaussian_inv_scaled_rotation = self.get_covariance(return_full_matrix=True, return_sqrt=True,
                                                               inverse_scales=True)
        if closest_gaussians_idx is None:
            closest_gaussians_idx = self.knn_idx[gaussian_idx]
        closest_gaussian_centers = gaussian_centers[closest_gaussians_idx]
        closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[closest_gaussians_idx]
        closest_gaussian_strengths = gaussian_strengths[closest_gaussians_idx]

        fields = {}

        # Compute the density field as a sum of local gaussian opacities
        # TODO: Change the normalization of the density (maybe learn the scaling parameter?)
        shift = (x[:, None] - closest_gaussian_centers)
        warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
        neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
        neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(
            -1. / 2 * neighbor_opacities)
        densities = neighbor_opacities.sum(dim=-1)
        fields['density'] = densities.clone()
        density_mask = densities >= 1.
        # normalize density bigger or equal than 1
        densities[density_mask] = densities[density_mask] / (densities[density_mask].detach() + 1e-12)

        if return_closest_gaussian_opacities:
            fields['closest_gaussian_opacities'] = neighbor_opacities

        if return_sdf or return_sdf_grad or return_beta:
            # --- Old way
            # beta = self.scaling.min(dim=-1)[0][closest_gaussians_idx].mean(dim=1)
            # --- New way
            beta = self.get_beta(x,
                                 closest_gaussians_idx=closest_gaussians_idx,
                                 closest_gaussians_opacities=neighbor_opacities,
                                 densities=densities,
                                 opacity_min_clamp=opacity_min_clamp,
                                 )
            clamped_densities = densities.clamp(min=opacity_min_clamp)

        if return_beta:
            fields['beta'] = beta

        # Compute the signed distance field
        if return_sdf:
            sdf_values = beta * (
                torch.sqrt(-2. * torch.log(clamped_densities))  # TODO: Change the max=1. to something else?
                - np.sqrt(-2. * np.log(min(density_threshold, 1.)))
            )
            fields['sdf'] = sdf_values

        # Compute the gradient of the signed distance field
        if return_sdf_grad:
            sdf_grad = neighbor_opacities[..., None] * (closest_gaussian_inv_scaled_rotation @ warped_shift)[..., 0]
            sdf_grad = sdf_grad.sum(dim=-2)
            sdf_grad = \
                (beta / (clamped_densities * torch.sqrt(-2. * torch.log(clamped_densities))).clamp(
                    min=opacity_min_clamp))[
                    ..., None] * sdf_grad
            fields['sdf_grad'] = sdf_grad.clamp(min=-sdf_grad_max_value, max=sdf_grad_max_value)

        return fields

    def get_beta(self, x,
                 closest_gaussians_idx=None,
                 closest_gaussians_opacities=None,
                 densities=None,
                 opacity_min_clamp=1e-32, ):
        """_summary_

        Args:
            x (_type_): Should have shape (n_points, 3)
            closest_gaussians_idx (_type_, optional): Should have shape (n_points, n_neighbors).
                Defaults to None.
            closest_gaussians_opacities (_type_, optional): Should have shape (n_points, n_neighbors).
            densities (_type_, optional): Should have shape (n_points, ).

        Returns:
            _type_: _description_
        """
        if self.beta_mode == 'learnable':
            return torch.exp(self._log_beta).expand(len(x))

        elif self.beta_mode == 'average':
            if closest_gaussians_idx is None:
                raise ValueError("closest_gaussians_idx must be provided when using beta_mode='average'.")
            return self.scaling.min(dim=-1)[0][closest_gaussians_idx].mean(dim=1)

        elif self.beta_mode == 'weighted_average':
            if closest_gaussians_idx is None:
                raise ValueError("closest_gaussians_idx must be provided when using beta_mode='weighted_average'.")
            if closest_gaussians_opacities is None:
                raise ValueError(
                    "closest_gaussians_opacities must be provided when using beta_mode='weighted_average'.")

            min_scaling = self.scaling.min(dim=-1)[0][closest_gaussians_idx]

            # if densities is None:
            if True:
                opacities_sum = closest_gaussians_opacities.sum(dim=-1, keepdim=True)
            else:
                opacities_sum = densities.view(-1, 1)
            # weights = neighbor_opacities.clamp(min=opacity_min_clamp) / opacities_sum.clamp(min=opacity_min_clamp)
            weights = closest_gaussians_opacities / opacities_sum.clamp(min=opacity_min_clamp)

            # Three methods to handle the case where all opacities are 0.
            # Important because we need to avoid beta == 0 at all cost for these points!
            # Indeed, beta == 0. gives sdf == 0.
            # However these points are far from gaussians, so they should have a sdf != 0.

            # Method 1: Give 1-weight to closest gaussian (Not good)
            if False:
                one_at_closest_gaussian = torch.zeros(1, neighbor_opacities.shape[1], device=rc.device)
                one_at_closest_gaussian[0, 0] = 1.
                weights[opacities_sum[..., 0] == 0.] = one_at_closest_gaussian
                beta = (rc.scaling.min(dim=-1)[0][closest_gaussians_idx] * weights).sum(dim=1)

            # Method 2: Give the maximum scaling value in neighbors as beta (Not good if neighbors have low scaling)
            if False:
                beta = (min_scaling * weights).sum(dim=-1)
                mask = opacities_sum[..., 0] == 0.
                beta[mask] = min_scaling.max(dim=-1)[0][mask]

            # Method 3: Give a constant, large beta value (better control)
            if True:
                beta = (min_scaling * weights).sum(dim=-1)
                with torch.no_grad():
                    if False:
                        # Option 1: beta = camera_spatial_extent
                        beta[opacities_sum[..., 0] == 0.] = rc.get_cameras_spatial_extent()
                    else:
                        # Option 2: beta = largest min_scale in the scene
                        beta[opacities_sum[..., 0] == 0.] = min_scaling.max().detach()

            return beta

        else:
            raise ValueError("Unknown beta_mode.")
