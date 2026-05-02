import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Proyecciones lineales de la entrada para generar Delta, B, C
        self.proj_delta = nn.Linear(d_model, d_model)
        self.proj_B = nn.Linear(d_model, d_state)
        self.proj_C = nn.Linear(d_model, d_state)

        # A inicializada negativa para garantizar estabilidad bajo discretización
        self.A = nn.Parameter(-torch.rand(d_model, d_state))

    def forward(self, u):
        batch_size, seq_len, _ = u.shape

        # Softplus para asegurar que el paso de tiempo discreto (Delta) sea positivo
        delta = F.softplus(self.proj_delta(u))
        B = self.proj_B(u)
        C = self.proj_C(u)

        h = torch.zeros(batch_size, self.d_model, self.d_state, device=u.device)
        y = []

        for t in range(seq_len):
            u_t = u[:, t, :]
            delta_t = delta[:, t, :].unsqueeze(-1)
            B_t = B[:, t, :].unsqueeze(1)
            C_t = C[:, t, :].unsqueeze(1)

            # Discretización tipo Euler: h_t = (1 + dt*A) h_{t-1} + dt*B*u
            h = (1 + delta_t * self.A) * h + delta_t * B_t * u_t.unsqueeze(-1)

            # Salida: y_t = sum_n C_t[n] * h_t[:, :, n]
            y_t = torch.sum(C_t * h, dim=-1)
            y.append(y_t)

        return torch.stack(y, dim=1)

class TemporalBlock(nn.Module):
    def __init__(self, d_model, d_state, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model
        )
        self.silu = nn.SiLU()
        self.ssm = SelectiveSSM(d_model, d_state)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x

        x_n = self.layer_norm(x)

        # Convolución local depthwise + SiLU
        x_conv = self.conv(x_n.transpose(1, 2)).transpose(1, 2)
        x_act = self.silu(x_conv)
        x_ssm = self.ssm(x_act)

        g = F.silu(self.gate_proj(x_n))

        # Conexión residual
        return res + (x_ssm * g)

class SpectralBlock(nn.Module):
    def __init__(self, d_model, top_k):
        super().__init__()
        self.top_k = top_k
        self.d_model = d_model
        # Filtro complejo aprendible POR CANAL: un peso por canal, componente_frec
        self.complex_filter = nn.Parameter(
            torch.randn(top_k, d_model, dtype=torch.cfloat) * 0.02
        )

    def forward(self, x):
        # x: (B, L, d_model)
        seq_len = x.size(1)

        # FFT a lo largo del eje temporal
        X_f = torch.fft.rfft(x, dim=1)

        # Top-k componentes por magnitud (por canal)
        magnitudes = torch.abs(X_f)
        k = min(self.top_k, magnitudes.size(1))
        _, topk_indices = torch.topk(magnitudes, k, dim=1)

        selected = torch.gather(X_f, 1, topk_indices)

        # Aplicar filtro complejo aprendible (broadcast en batch)
        filtered = selected * self.complex_filter.unsqueeze(0)

        # Reinsertar en el espectro completo, ceros en el resto
        X_filtered = torch.zeros_like(X_f)
        X_filtered.scatter_(1, topk_indices, filtered)

        # IFFT al dominio temporal
        x_out = torch.fft.irfft(X_filtered, n=seq_len, dim=1)
        return x_out

class Fusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.ones(d_model))

    def forward(self, x_t, x_f):
        return (self.alpha * x_t) + (self.beta * x_f)

class AnomalyDetector(nn.Module):

    def __init__(self, input_dim, d_model, d_state, top_k, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        self.temporal = TemporalBlock(d_model, d_state)
        self.spectral = SpectralBlock(d_model, top_k)
        self.fusion = Fusion(d_model)

        # Cabeza de clasificación: salida = 2 logits (clase 0 y clase 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, L, input_dim)
        x_proj = self.input_proj(x)

        # Ramas en paralelo
        x_t = self.temporal(x_proj)
        x_f = self.spectral(x_proj)

        z = self.fusion(x_t, x_f)

        z_pooled = z.mean(dim=1)

        # Logits de 2 clases
        return self.classifier(z_pooled)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            return probs[:, 1]  # solo prob de anomalía

    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_proj.in_features,
                'd_model': self.input_proj.out_features,
                'd_state': self.temporal.ssm.d_state,
                'top_k': self.spectral.top_k,
            }
        }, path)

    @classmethod
    def load(cls, path, map_location=None):
        # Se carga un modelo previamente guardado con .save().
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model