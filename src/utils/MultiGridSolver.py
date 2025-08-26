import numpy as np

from src.utils.HarmonicSolver import HarmonicSolver


class MultigridSolver:
    def __init__(self, field_map, max_levels=4):
        self.map = field_map
        self.max_levels = max_levels
        self.original_shape = field_map.grid.shape

    def solve(self, max_iter=1000):
        phi = self._v_cycle(self.map.grid.copy(), self.map.mask, self.map.goal, level=0, max_level=self.max_levels, max_iter=max_iter)
        self.map.grid = phi

    def _v_cycle(self, phi, mask, goal, level, max_level, max_iter):
        ny, nx = phi.shape

        # Stop recursion if grid is too small
        if level == max_level or nx < 5 or ny < 5:
            return self._relax(phi, mask, goal, iterations=max_iter)

        # Pre-smoothing
        phi = self._relax(phi, mask, goal, iterations=max_iter)

        # Compute residual
        residual = self._compute_residual(phi, mask)

        # Restrict to coarse grid
        residual_coarse = self._restrict(residual)
        mask_coarse = self._restrict(mask.astype(float)) > 0.5
        phi_coarse = np.zeros_like(residual_coarse)

        # Recursive solve for error on coarse grid
        error_coarse = self._v_cycle(phi_coarse, mask_coarse, None, level + 1, max_level, max_iter)

        # Interpolate error to fine grid and correct
        error_fine = self._interpolate(error_coarse, phi.shape)
        phi += error_fine

        # Post-smoothing
        phi = self._relax(phi, mask, goal, iterations=max_iter)

        return phi

    def _relax(self, phi, mask, goal, iterations, omega=1.7):
        return HarmonicSolver.relax_SOR(phi, mask, goal, iterations=iterations, omega=omega)

    def _compute_residual(self, phi, mask):
        ny, nx = phi.shape
        residual = np.zeros_like(phi)
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if mask[i, j]:
                    continue
                laplacian = phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] - 4 * phi[i, j]
                residual[i, j] = laplacian
        return residual

    def _restrict(self, fine):
        ny, nx = fine.shape
        coarse = np.zeros((ny // 2, nx // 2))
        for i in range(1, coarse.shape[0] - 1):
            for j in range(1, coarse.shape[1] - 1):
                i2, j2 = 2 * i, 2 * j
                coarse[i, j] = 0.25 * (fine[i2, j2] + fine[i2-1, j2] + fine[i2+1, j2] + fine[i2, j2-1] + fine[i2, j2+1])
        return coarse

    def _interpolate(self, coarse, shape):
        ny, nx = shape
        fine = np.zeros((ny, nx))
        for i in range(1, coarse.shape[0] - 1):
            for j in range(1, coarse.shape[1] - 1):
                i2, j2 = 2 * i, 2 * j
                fine[i2, j2] = coarse[i, j]
                if i2+1 < ny:
                    fine[i2+1, j2] = 0.5 * (coarse[i, j] + coarse[i+1, j])
                if j2+1 < nx:
                    fine[i2, j2+1] = 0.5 * (coarse[i, j] + coarse[i, j+1])
                if i2+1 < ny and j2+1 < nx:
                    fine[i2+1, j2+1] = 0.25 * (coarse[i, j] + coarse[i+1, j] + coarse[i, j+1] + coarse[i+1, j+1])
        return fine