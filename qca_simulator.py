# qca_simulator.py
import numpy as np

class QCASimulator:
    def __init__(self, N_branches=5, alpha=0.7, gamma=1/12e-3, beta_eff=1.2, dt=0.001, T_total=1.0):
        self.N = N_branches
        self.alpha = alpha
        self.gamma = gamma
        self.beta_eff = beta_eff
        self.dt = dt
        self.T_total = T_total
        self.timesteps = int(T_total / dt)
        self.A_k = np.zeros((self.timesteps, self.N))
        self.w_k = np.zeros((self.timesteps, self.N))
        self.Gamma_k = np.zeros((self.timesteps, self.N))
        self._run_simulation()

    def fidelity(self, k, t):
        return 0.5 + 0.4 * np.cos(2 * np.pi * k / self.N + 0.1 * t)

    def phi_k(self, k, t):
        raw_phi = 1 + 0.5 * np.sin(0.2 * t + k)
        return raw_phi / (1 + raw_phi)

    def affinity(self, k, t):
        return self.alpha * self.fidelity(k, t) + (1 - self.alpha) * self.phi_k(k, t)

    def decoherence_rate(self, A_k_t):
        return np.exp(-self.beta_eff * A_k_t)

    def _run_simulation(self):
        for t_idx in range(self.timesteps):
            t = t_idx * self.dt
            total_w = 0
            for k in range(self.N):
                A_k_t = self.affinity(k, t)
                self.A_k[t_idx, k] = A_k_t
                Gamma_k_t = self.decoherence_rate(A_k_t)
                self.Gamma_k[t_idx, k] = Gamma_k_t
                self.w_k[t_idx, k] = np.maximum(0, A_k_t - 0.65)
                total_w += self.w_k[t_idx, k]
            if total_w > 0:
                self.w_k[t_idx] /= total_w
