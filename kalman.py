import numpy as np

# Filtro de Kalman escalar para suavizar la secuencia de probabilidades de anomalía producida por el modelo.
class KalmanSmoother:

    def __init__(self, sigma2_w, sigma2_v,
                 initial_state=0.5, initial_covariance=1.0):
        self.sigma2_w = float(sigma2_w)
        self.sigma2_v = float(sigma2_v)
        self.s = float(initial_state)
        self.P = float(initial_covariance)

    def reset(self, initial_state=0.5, initial_covariance=1.0):
        self.s = float(initial_state)
        self.P = float(initial_covariance)

    def update(self, p_t):
        # Predicción
        # Bajo random walk, el estado predicho es igual al posterior anterior.
        s_pred = self.s
        P_pred = self.P + self.sigma2_w

        # Ganancia de Kalman
        # K_t = P_{t|t-1} / (P_{t|t-1} + σ_v²)
        K = P_pred / (P_pred + self.sigma2_v)

        # Actualización con la observación p_t
        self.s = s_pred + K * (p_t - s_pred)
        self.P = (1.0 - K) * P_pred

        return self.s

    def smooth_sequence(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        smoothed = np.empty_like(probs)
        for t, p in enumerate(probs):
            smoothed[t] = self.update(p)
        return smoothed