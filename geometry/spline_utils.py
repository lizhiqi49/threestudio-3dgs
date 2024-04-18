"""
SE(3) B-spline trajectory library
"""

from dataclasses import dataclass, field
from typing import Tuple, Type

import pypose as pp
import torch
from jaxtyping import Float, Int
from pypose import LieTensor
from torch import nn, Tensor
from typing_extensions import assert_never
from einops import rearrange

_EPS = 1e-6


def linear_interpolation_mid(
    ctrl_knots: Float[LieTensor, "*batch_size 2 7"],
) -> Float[LieTensor, "*batch_size 7"]:
    """Get the midpoint between batches of two SE(3) poses by linear interpolation.

    Args:
        ctrl_knots: The control knots.

    Returns:
        The midpoint poses.
    """
    start_pose, end_pose = ctrl_knots[..., 0, :], ctrl_knots[..., 1, :]
    t_start, q_start = start_pose.translation(), start_pose.rotation()
    t_end, q_end = end_pose.translation(), end_pose.rotation()

    t = (t_start + t_end) * 0.5

    q_tau_0 = q_start.Inv() @ q_end
    q_t_0 = pp.Exp(pp.so3(q_tau_0.Log() * 0.5))
    q = q_start @ q_t_0

    ret = pp.SE3(torch.cat([t, q], dim=-1))
    return ret


def linear_interpolation(
    ctrl_knots: Float[LieTensor, "*batch_size 2 7"],
    u: Float[Tensor, "interpolations"] | Float[Tensor, "*batch_size interpolations"],
    enable_eps: bool = False,
) -> Float[LieTensor, "*batch_size interpolations 7"]:
    """Linear interpolation between batches of two SE(3) poses.

    Args:
        ctrl_knots: The control knots.
        u: Normalized positions between two SE(3) poses. Range: [0, 1].
        enable_eps: Whether to clip the normalized position with a small epsilon to avoid possible numerical issues.

    Returns:
        The interpolated poses.
    """
    start_pose, end_pose = ctrl_knots[..., 0, :], ctrl_knots[..., 1, :]
    batch_size = start_pose.shape[:-1]
    interpolations = u.shape[-1]

    t_start, q_start = start_pose.translation(), start_pose.rotation()
    t_end, q_end = end_pose.translation(), end_pose.rotation()

    # If u only has one dim, broadcast it to all batches. This means same interpolations for all batches.
    # Otherwise, u should have the same batch size as the control knots (*batch_size, interpolations).
    if u.dim() == 1:
        u = u.tile((*batch_size, 1))  # (*batch_size, interpolations)
    if enable_eps:
        u = torch.clip(u, _EPS, 1.0 - _EPS)

    t = pp.bvv(1 - u, t_start) + pp.bvv(u, t_end)

    q_tau_0 = q_start.Inv() @ q_end
    r_tau_0 = q_tau_0.Log()
    q_t_0 = pp.Exp(pp.so3(pp.bvv(u, r_tau_0)))
    q = q_start.unsqueeze(-2).tile((interpolations, 1)) @ q_t_0

    ret = pp.SE3(torch.cat([t, q], dim=-1))
    return ret


def cubic_bspline_interpolation(
    ctrl_knots: Float[LieTensor, "*batch_size 4 7"],
    u: Float[Tensor, "interpolations"] | Float[Tensor, "*batch_size interpolations"],
    enable_eps: bool = True,
) -> Float[LieTensor, "*batch_size interpolations 7"]:
    """Cubic B-spline interpolation with batches of four SE(3) control knots.

    Args:
        ctrl_knots: The control knots.
        u: Normalized positions on the trajectory segments. Range: [0, 1].
        enable_eps: Whether to clip the normalized position with a small epsilon to avoid possible numerical issues.

    Returns:
        The interpolated poses.
    """
    batch_size = ctrl_knots.shape[:-2]  # (N_pts,)
    interpolations = u.shape[-1]

    # If u only has one dim, broadcast it to all batches. This means same interpolations for all batches.
    # Otherwise, u should have the same batch size as the control knots (*batch_size, interpolations).
    if u.dim() == 1:
        u = u.tile((*batch_size, 1))  # (*batch_size, interpolations)
    if enable_eps:
        u = torch.clip(u, _EPS, 1.0 - _EPS)

    uu = u * u
    uuu = uu * u
    oos = 1.0 / 6.0  # one over six

    # t coefficients
    coeffs_t = torch.stack([
        oos - 0.5 * u + 0.5 * uu - oos * uuu,
        4.0 * oos - uu + 0.5 * uuu,
        oos + 0.5 * u + 0.5 * uu - 0.5 * uuu,
        oos * uuu
    ], dim=-2)

    # spline t
    t_t = torch.sum(pp.bvv(coeffs_t, ctrl_knots.translation()), dim=-3)

    # q coefficients
    coeffs_r = torch.stack([
        5.0 * oos + 0.5 * u - 0.5 * uu + oos * uuu,
        oos + 0.5 * u + 0.5 * uu - 2 * oos * uuu,
        oos * uuu
    ], dim=-2)

    # spline q
    q_adjacent = ctrl_knots[..., :-1, :].rotation().Inv() @ ctrl_knots[..., 1:, :].rotation()
    r_adjacent = q_adjacent.Log()
    q_ts = pp.Exp(pp.so3(pp.bvv(coeffs_r, r_adjacent)))
    q0 = ctrl_knots[..., 0, :].rotation()  # (*batch_size, 4)
    q_ts = torch.cat([
        q0.unsqueeze(-2).tile((interpolations, 1)).unsqueeze(-3),
        q_ts
    ], dim=-3)  # (*batch_size, num_ctrl_knots=4, interpolations, 4)
    q_t = pp.cumprod(q_ts, dim=-3, left=False)[..., -1, :, :]

    ret = pp.SE3(torch.cat([t_t, q_t.tensor()], dim=-1))
    return ret


"""
SE(3) B-spline trajectory
"""


@dataclass
class SplineConfig:
    """Configuration for spline instantiation."""

    _target: Type = field(default_factory=lambda: Spline)
    """Target class to instantiate."""
    degree: int = 1
    """Degree of the spline. 1 for linear spline, 3 for cubic spline."""
    sampling_interval: float = 0.1
    """Sampling interval of the control knots."""
    start_time: float = 0.0
    """Starting timestamp of the spline."""
    n_knots: int = 0
    """Number of control knots."""


class Spline(nn.Module):
    """
    Args:
        config: the SplineConfig used to instantiate class
    """

    config: SplineConfig
    data: Float[Tensor, "n_knots n_feature"]
    start_time: float
    end_time: float
    t_lower_bound: float
    t_upper_bound: float

    def __init__(self, config: SplineConfig):
        super().__init__()
        self.config = config
        self.data = None
        self.interp_param_names = []
        self.order = self.config.degree + 1
        self.n_knots = self.config.n_knots
        """Order of the spline, i.e. control knots per segment, 2 for linear, 4 for cubic"""

        self.set_start_time(config.start_time)
        self.update_end_time()

    def __len__(self):
        return self.n_knots

    def forward(
        self, timestamps: Float[Tensor, "batch_size"], keys: list[str] = None
    ) -> Float[Tensor, "batch_size n_pts n_feature"]:
        """Interpolate the spline at the given timestamps.

        Args:
            timestamps: Timestamps to interpolate the spline at. Range: [t_lower_bound, t_upper_bound].

        Returns:
            poses: The interpolated pose.
        """
        ts = torch.clamp(timestamps, self.t_lower_bound + _EPS, self.t_upper_bound - _EPS)
        batch_size = ts.shape[0]
        relative_time = ts - self.start_time
        normalized_time = relative_time / self.config.sampling_interval

        start_index: Int[Tensor, "B"]
        start_index = torch.floor(normalized_time).int()
        u = normalized_time - start_index
        if self.config.degree == 3:
            start_index -= 1

        if len(self.interp_param_names) == 0:
            raise ValueError("You must set control knots before interpolation!")

        outs = {}
        names = self.interp_param_names if keys is None else keys
        for name in names:
            knots: Float[Tensor, "N_pts N_knots N_feature"] = getattr(self, name)
            indices = start_index[..., None] + torch.arange(self.order, device=start_index.device)[None]
            indices = indices.flatten()

            segment: Float[Tensor, "B N_pts N_order N_feature"]
            segment = knots[:, indices, :]
            segment = rearrange(segment, "N (B K) F -> B N K F", B=batch_size, K=self.order)

            interp: Float[Tensor, "B N_pts N_feature"]
            interp = self.interpolation(
                segment.reshape(-1, *segment.shape[2:]),
                u.repeat_interleave(knots.shape[0])[..., None],
                name,  # (N, interpolation=1)
            ).reshape(batch_size, knots.shape[0], knots.shape[-1])
            outs[name] = interp

        return outs

    def interpolation(self, segment, u, name):
        if self.config.degree == 1:
            return self.linear_interpolation(segment, u, name)
        elif self.config.degree == 3:
            return self.cubic_bspline_interpolation(segment, u, name)
        else:
            assert_never(self.cfg.degree)

    def cubic_bspline_interpolation(
        self,
        ctrl_knots: Float[Tensor, "N 4 n_feature"],
        u: Float[Tensor, "N 1"],
        name: str,
        enable_eps: bool = False
    ):
        N = ctrl_knots.shape[0]  # (N_pts,)
        interpolations = u.shape[-1]
        # If u only has one dim, broadcast it to all batches. This means same interpolations for all batches.
        # Otherwise, u should have the same batch size as the control knots (*batch_size, interpolations).
        if u.dim() == 1:
            u = u.tile((N, 1))  # (*batch_size, interpolations)
        if enable_eps:
            u = torch.clip(u, _EPS, 1.0 - _EPS)

        uu = u * u
        uuu = uu * u
        oos = 1.0 / 6.0  # one over six

        if name in ["xyz", "scale"]:
            assert ctrl_knots.shape[-1] == 3
            # t coefficients
            coeffs_t = torch.stack([
                oos - 0.5 * u + 0.5 * uu - oos * uuu,
                4.0 * oos - uu + 0.5 * uuu,
                oos + 0.5 * u + 0.5 * uu - 0.5 * uuu,
                oos * uuu
            ], dim=-2)

            # spline t
            ret = torch.sum(pp.bvv(coeffs_t, ctrl_knots), dim=-3).squeeze(-2)
        elif name == "rotation":
            assert ctrl_knots.shape[-1] == 4
            # q coefficients
            coeffs_r = torch.stack([
                5.0 * oos + 0.5 * u - 0.5 * uu + oos * uuu,
                oos + 0.5 * u + 0.5 * uu - 2 * oos * uuu,
                oos * uuu
            ], dim=-2)

            # spline q
            q_adjacent = ctrl_knots[..., :-1, :].Inv() @ ctrl_knots[..., 1:, :]
            r_adjacent = q_adjacent.Log()
            q_ts = pp.Exp(pp.so3(pp.bvv(coeffs_r, r_adjacent)))
            q0 = ctrl_knots[..., 0, :]  # (*batch_size, 4)
            q_ts = torch.cat([
                q0.unsqueeze(-2).tile((interpolations, 1)).unsqueeze(-3),
                q_ts
            ], dim=-3)  # (*batch_size, num_ctrl_knots=4, interpolations, 4)
            q_t = pp.cumprod(q_ts, dim=-3, left=False)[..., -1, :, :]
            ret = q_t.squeeze(-2)
            # ret = ret / ret.norm(dim=-1, keepdim=True)

        return ret

    def insert(self, name, new_knot: Float[Tensor, "*N 1 n_feature"]):
        """Insert a control knot"""
        # data = self.data[name]
        data = getattr(self, name)
        data = torch.cat([data, new_knot], dim=-2)
        # self.data[name] = data
        self.__setattr__(name, data)
        self.n_knots = data.shape[-2]
        self.update_end_time()

    def set_data(self, name, data: Float[Tensor, "num_knots n_feature"]):
        """Set the spline data."""
        # self.data[name] = data
        self.__setattr__(name, data)
        self.n_knots = data.shape[-2]
        if name not in self.interp_param_names:
            self.interp_param_names.append(name)
        self.update_end_time()

    def set_start_time(self, start_time: float):
        """Set the starting timestamp of the spline."""
        self.start_time = start_time
        if self.config.degree == 1:
            self.t_lower_bound = self.start_time
        elif self.config.degree == 3:
            self.t_lower_bound = self.start_time + self.config.sampling_interval
        else:
            assert_never(self.config.degree)

    def update_end_time(self):
        self.end_time = self.start_time + self.config.sampling_interval * (self.n_knots - 1)
        if self.config.degree == 1:
            self.t_upper_bound = self.end_time
        elif self.config.degree == 3:
            self.t_upper_bound = self.end_time - self.config.sampling_interval
        else:
            assert_never(self.config.degree)

# class Spline(nn.Module):
#     """SE(3) spline trajectory.

#     Args:
#         config: the SplineConfig used to instantiate class
#     """

#     config: SplineConfig
#     data: Float[LieTensor, "num_knots 7"]
#     start_time: float
#     end_time: float
#     t_lower_bound: float
#     t_upper_bound: float

#     def __init__(self, config: SplineConfig):
#         super().__init__()
#         self.config = config
#         self.data = pp.identity_SE3(0)
#         self.order = self.config.degree + 1
#         """Order of the spline, i.e. control knots per segment, 2 for linear, 4 for cubic"""

#         self.set_start_time(config.start_time)
#         self.update_end_time()

#     def __len__(self):
#         return self.data.shape[0]

#     def forward(self, timestamps: Float[Tensor, "*batch_size"]) -> Float[LieTensor, "*batch_size 7"]:
#         """Interpolate the spline at the given timestamps.

#         Args:
#             timestamps: Timestamps to interpolate the spline at. Range: [t_lower_bound, t_upper_bound].

#         Returns:
#             poses: The interpolated pose.
#         """
#         segment, u = self.get_segment(timestamps)
#         u = u[..., None]  # (*batch_size) to (*batch_size, interpolations=1)
#         if self.config.degree == 1:
#             poses = linear_interpolation(segment, u)
#         elif self.config.degree == 3:
#             poses = cubic_bspline_interpolation(segment, u)
#         else:
#             assert_never(self.config.degree)
#         return poses.squeeze()

#     def get_segment(
#             self,
#             timestamps: Float[Tensor, "*batch_size"]
#     ) -> Tuple[
#         Float[LieTensor, "*batch_size self.order 7"],
#         Float[Tensor, "*batch_size"]
#     ]:
#         """Get the spline segment and normalized position on segment at the given timestamp.

#         Args:
#             timestamps: Timestamps to get the spline segment and normalized position at.

#         Returns:
#             segment: The spline segment.
#             u: The normalized position on the segment.
#         """
#         # assert torch.all(timestamps >= self.t_lower_bound)
#         # assert torch.all(timestamps <= self.t_upper_bound)
#         timestamps = torch.clamp(timestamps, self.t_lower_bound + _EPS, self.t_upper_bound - _EPS)
#         batch_size = timestamps.shape
#         relative_time = timestamps - self.start_time
#         normalized_time = relative_time / self.config.sampling_interval
#         start_index = torch.floor(normalized_time).int()
#         u = normalized_time - start_index
#         if self.config.degree == 3:
#             start_index -= 1

#         indices = (start_index.tile((self.order, 1)).T +
#                    torch.arange(self.order).tile((*batch_size, 1)).to(start_index.device))
#         indices = indices[..., None].tile(7)
#         # segment = pp.SE3(torch.gather(self.data.expand(*batch_size, -1, -1), 1, indices))
#         segment = pp.SE3(torch.gather(self.data, -2, indices)) # (N_points, 4, 7)

#         return segment, u

#     def insert(self, pose: Float[LieTensor, "1 7"]):
#         """Insert a control knot"""
#         self.data = pp.SE3(torch.cat([self.data, pose]))
#         self.update_end_time()

#     def set_data(self, data: Float[LieTensor, "num_knots 7"] | pp.Parameter):
#         """Set the spline data."""
#         self.data = data
#         self.update_end_time()

#     def set_start_time(self, start_time: float):
#         """Set the starting timestamp of the spline."""
#         self.start_time = start_time
#         if self.config.degree == 1:
#             self.t_lower_bound = self.start_time
#         elif self.config.degree == 3:
#             self.t_lower_bound = self.start_time + self.config.sampling_interval
#         else:
#             assert_never(self.config.degree)

#     def update_end_time(self):
#         self.end_time = self.start_time + self.config.sampling_interval * (self.data.shape[-2] - 1)
#         if self.config.degree == 1:
#             self.t_upper_bound = self.end_time
#         elif self.config.degree == 3:
#             self.t_upper_bound = self.end_time - self.config.sampling_interval
#         else:
#             assert_never(self.config.degree)
