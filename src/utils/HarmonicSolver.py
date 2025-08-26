import numpy as np
class HarmonicSolver:
    def __init__(self, field_map):
        self.map = field_map

    def solve_SOR(self, omega=1.7, max_iter=1000, tol=1e-10):
        phi = self.map.grid.copy()
        mask = self.map.mask
        goal = self.map.goal
        for it in range(max_iter):
            phi_old = phi.copy()

            # Use the static relax function
            phi = HarmonicSolver.relax_SOR(phi, mask, goal, omega=omega, iterations=1)

            # Convergence check
            max_delta = np.max(np.abs(phi - phi_old))
            if max_delta < tol:
                print(f"SOR converged in {it + 1} iterations.")
                break

        self.map.grid = phi
    @staticmethod
    def relax_SOR(phi, mask, goal, omega=1.7, iterations=3):
        ny, nx = phi.shape
        for _ in range(iterations):
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    if mask[i, j] or (goal is not None and (i, j) == goal):
                        continue
                    new_val = 0.25 * (phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1])
                    phi[i, j] = (1 - omega) * phi[i, j] + omega * new_val
            if goal is not None:
                phi[goal[0], goal[1]] = 0
        return phi
