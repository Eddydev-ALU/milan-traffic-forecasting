"""Models 2 and 3: LSTM and Temporal Convolutional Network.

Both consume the same ``WindowedData`` produced by ``src/data/windows.py``:
a length-``L`` window of scaled past traffic plus (optionally) a small vector
of calendar features describing the *target* timestamp. Only the encoder
differs, which is what makes the comparison a comparison of architectures
rather than of preprocessing pipelines.

* **LSTM** -- gated recurrence, processes the window sequentially, well suited
  to the nonlinear, regime-switching behaviour (weekday/weekend, day/night)
  visible in the exploratory analysis. Cost: sequential, so training does not
  parallelise along time.
* **TCN** -- stacked dilated causal convolutions (Bai et al., 2018). The
  receptive field grows as ``1 + 2(k-1)(2^n - 1)``, so four levels with k=3
  already cover 31 steps and six levels cover 127; the whole window is
  processed in parallel. Cost: a fixed receptive field, no explicit memory.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except Exception:  # pragma: no cover - also catches a broken CUDA install
    HAS_TORCH = False
    torch = None
    nn = object


def _require_torch() -> None:
    if not HAS_TORCH:  # pragma: no cover
        raise ImportError("PyTorch is required: pip install torch")


def get_device(prefer: str = "auto"):
    _require_torch()
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class LSTMForecaster(nn.Module):
        """LSTM encoder + optional static-feature fusion + linear head."""

        def __init__(self, hidden_size: int = 64, num_layers: int = 2,
                     dropout: float = 0.2, n_static: int = 0):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size + n_static, 1)

        def forward(self, x, static=None):
            # x: (B, L, 1)
            out, _ = self.lstm(x)
            h = self.dropout(out[:, -1, :])            # last hidden state
            if static is not None and static.shape[1] > 0:
                h = torch.cat([h, static], dim=1)
            return self.head(h).squeeze(-1)

    class Chomp1d(nn.Module):
        """Trim the right-hand padding so the convolution stays causal."""

        def __init__(self, chomp_size: int):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x):
            return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x

    class TemporalBlock(nn.Module):
        """Two dilated causal convolutions with a residual connection."""

        def __init__(self, n_in: int, n_out: int, kernel_size: int,
                     dilation: int, dropout: float):
            super().__init__()
            pad = (kernel_size - 1) * dilation
            self.net = nn.Sequential(
                nn.Conv1d(n_in, n_out, kernel_size, padding=pad, dilation=dilation),
                Chomp1d(pad),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(n_out, n_out, kernel_size, padding=pad, dilation=dilation),
                Chomp1d(pad),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    class TCNForecaster(nn.Module):
        """Stack of temporal blocks with exponentially increasing dilation."""

        def __init__(self, channels=(32, 32, 32, 32), kernel_size: int = 3,
                     dropout: float = 0.15, n_static: int = 0):
            super().__init__()
            layers = []
            n_in = 1
            for i, n_out in enumerate(channels):
                layers.append(TemporalBlock(n_in, n_out, kernel_size, 2 ** i, dropout))
                n_in = n_out
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(channels[-1] + n_static, 1)
            self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** len(channels) - 1)

        def forward(self, x, static=None):
            # x: (B, L, 1) -> (B, 1, L) for Conv1d
            h = self.tcn(x.transpose(1, 2))[:, :, -1]   # last causal position
            if static is not None and static.shape[1] > 0:
                h = torch.cat([h, static], dim=1)
            return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    train_loss: list = field(default_factory=list)
    val_mae_raw: list = field(default_factory=list)
    best_epoch: int = -1
    best_val_mae: float = float("inf")
    epochs_run: int = 0
    stopped_early: bool = False


class NeuralForecaster:
    """Framework-level wrapper: batching, early stopping, inverse scaling.

    Early stopping monitors validation MAE **in original activity units**, not
    the scaled loss, so model selection optimises the quantity actually
    reported in the results tables.
    """

    def __init__(self, arch: str = "lstm", scaler=None, device: str = "auto",
                 lr: float = 1e-3, batch_size: int = 128, max_epochs: int = 60,
                 patience: int = 8, grad_clip: float = 1.0, loss: str = "huber",
                 seed: int = 42, **arch_kwargs):
        _require_torch()
        self.arch = arch.lower()
        self.scaler = scaler
        self.device = get_device(device)
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.loss_name = loss
        self.seed = seed
        self.arch_kwargs = arch_kwargs
        self.model = None
        self.history = TrainingHistory()
        self.name = "LSTM" if self.arch == "lstm" else "TCN"

    # -- helpers ------------------------------------------------------------
    def _build(self, n_static: int):
        torch.manual_seed(self.seed)
        if self.arch == "lstm":
            return LSTMForecaster(n_static=n_static, **self.arch_kwargs).to(self.device)
        if self.arch == "tcn":
            return TCNForecaster(n_static=n_static, **self.arch_kwargs).to(self.device)
        raise ValueError(f"Unknown architecture: {self.arch}")

    def _tensors(self, data):
        x = torch.tensor(np.asarray(data.X, dtype=np.float32)).unsqueeze(-1)  # (N, L, 1)
        y = torch.tensor(np.asarray(data.y, dtype=np.float32))
        s = (torch.tensor(np.asarray(data.static, dtype=np.float32))
             if data.static is not None else torch.zeros(len(y), 0))
        return x, s, y

    def _criterion(self):
        return nn.HuberLoss(delta=1.0) if self.loss_name == "huber" else nn.MSELoss()

    def _to_raw(self, z: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(z) if self.scaler is not None else z

    # -- API ----------------------------------------------------------------
    def fit(self, train_data, val_data=None, verbose: bool = False) -> "NeuralForecaster":
        xt, st, yt = self._tensors(train_data)
        n_static = st.shape[1]
        self.model = self._build(n_static)

        loader = DataLoader(
            TensorDataset(xt, st, yt),
            batch_size=self.batch_size,
            shuffle=True,          # shuffling *windows* is fine; the chronological
            drop_last=False,       # split already happened upstream.
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        crit = self._criterion()

        best_state, bad_epochs = None, 0
        for epoch in range(self.max_epochs):
            self.model.train()
            total, seen = 0.0, 0
            for xb, sb, yb in loader:
                xb, sb, yb = xb.to(self.device), sb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(xb, sb), yb)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt.step()
                total += loss.item() * len(yb)
                seen += len(yb)
            self.history.train_loss.append(total / max(seen, 1))
            self.history.epochs_run = epoch + 1

            if val_data is not None and len(val_data) > 0:
                val_pred = self.predict(val_data)
                val_mae = float(np.mean(np.abs(val_pred - val_data.y_raw)))
                self.history.val_mae_raw.append(val_mae)
                sched.step(val_mae)
                if val_mae < self.history.best_val_mae - 1e-9:
                    self.history.best_val_mae = val_mae
                    self.history.best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.patience:
                        self.history.stopped_early = True
                        if verbose:
                            print(f"  early stop at epoch {epoch + 1}")
                        break
                if verbose:
                    print(f"  epoch {epoch + 1:>3}  train_loss={self.history.train_loss[-1]:.5f}"
                          f"  val_MAE={val_mae:,.1f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, data) -> np.ndarray:
        """Batched inference returning predictions in original activity units."""
        self.model.eval()
        x, s, _ = self._tensors(data)
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), 1024):
                xb = x[i: i + 1024].to(self.device)
                sb = s[i: i + 1024].to(self.device)
                preds.append(self.model(xb, sb).detach().cpu().numpy())
        return self._to_raw(np.concatenate(preds) if preds else np.zeros(0))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters()) if self.model else 0

    def receptive_field(self) -> int | None:
        return getattr(self.model, "receptive_field", None)
