import torch
from pytorch3d.ops import knn_points


class SuGaR():
    def __init__(
        self,
        points: torch.Tensor,
        keep_track_of_knn: bool = False,
        knn_to_track: int = 16,
        surface_mesh_to_bind=None,  # Open3D mesh
    ):
        """
        Args:
            points (torch.Tensor): Initial positions of the Gaussians (not used when wrapping).

            keep_track_of_knn (bool, optional): Whether to keep track of the KNN information for training regularization. Defaults to False.
            knn_to_track (int, optional): Number of KNN to track. Defaults to 16.
            surface_mesh_to_bind (None, optional): Surface mesh to bind the Gaussians to. Defaults to None.

        """

        if surface_mesh_to_bind is not None:
            ## wait to finish
            pass
        else:
            self.binded_to_surface_mesh = False
            self._points = points
            n_points = len(self._points)

        self.keep_track_of_knn = keep_track_of_knn
        if keep_track_of_knn:
            self.knn_to_track = knn_to_track
            knns = knn_points(points[None], points[None], K=knn_to_track)
            self.knn_dists = knns.dists[0]
            self.knn_idx = knns.idx[0]

    @property
    def points(self):
        return self._points

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
