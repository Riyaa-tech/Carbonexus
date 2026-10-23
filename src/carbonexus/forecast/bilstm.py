"""The paper's forecaster: 2-layer bidirectional LSTM (Fig. 3), T_in=24 -> T_out=12,
13 features, MSE + AdamW(lr 5e-4, wd 1e-5), dropout 0.15, MC-dropout uncertainty.

Requires the optional `torch` extra. Differences from the paper (all forced by data size
or by gaps in the manuscript) are listed in docs/methodology.md:
  * trained on region-aggregated series pooled over regions (paper: "city-level");
  * walk-forward split by months: train targets <= T-24, validation targets <= T-12,
    test = last 12 months;
  * future exogenous features (meteorology, land use) use the training-period monthly
    climatology during autoregressive roll-out (the paper does not say).
Nothing here reproduces the paper's reported RMSE/R2; metrics are computed from the data.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def require_torch():
    if torch is None:
        raise RuntimeError("PyTorch not installed: pip install 'carbonexus[torch]'")


if nn is not None:
    class StackedBiLSTMRegressor(nn.Module):
        def __init__(self, input_dim=13, hidden=128, out_len=12, dropout=0.15):
            super().__init__()
            self.lstm1 = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
            self.norm1 = nn.LayerNorm(2 * hidden)
            self.drop1 = nn.Dropout(dropout)
            self.lstm2 = nn.LSTM(2 * hidden, hidden, batch_first=True, bidirectional=True)
            self.norm2 = nn.LayerNorm(2 * hidden)
            self.drop2 = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, out_len))

        def forward(self, x):  # [B,T,C] -> [B,out_len]
            x, _ = self.lstm1(x)
            x = self.drop1(self.norm1(x))
            x, _ = self.lstm2(x)
            x = self.drop2(self.norm2(x))
            return self.head(x[:, -1, :])


def make_windows(feat: np.ndarray, tin: int, tout: int, target_col: int, max_end: int | None = None, min_end: int = 0):
    """feat [T,C]. Windows whose target block ends (exclusive) at index e satisfy min_end < e <= max_end."""
    T = feat.shape[0]
    X, Y, S = [], [], []
    for s in range(0, T - tin - tout + 1):
        e = s + tin + tout
        if e > (max_end if max_end is not None else T) or e <= min_end:
            continue
        w = feat[s:s + tin]
        y = feat[s + tin:e, target_col]
        if np.isfinite(w).all() and np.isfinite(y).all():
            X.append(w); Y.append(y); S.append(s)
    return (np.array(X, np.float32), np.array(Y, np.float32), S)


def train_bilstm(feats: dict[str, np.ndarray], target_col: int, tin=24, tout=12, epochs=200,
                 hidden=128, seed=0, patience=30, lr=5e-4, wd=1e-5, verbose=False):
    """feats: region -> [T,C] features (already standardised). Returns (model, report)."""
    require_torch()
    torch.manual_seed(seed); np.random.seed(seed)
    T = next(iter(feats.values())).shape[0]
    C = next(iter(feats.values())).shape[1]
    tr = [make_windows(f, tin, tout, target_col, max_end=T - 24) for f in feats.values()]
    va = [make_windows(f, tin, tout, target_col, max_end=T - 12, min_end=T - 24) for f in feats.values()]
    Xtr = np.concatenate([a[0] for a in tr if len(a[0])]); Ytr = np.concatenate([a[1] for a in tr if len(a[1])])
    Xva = np.concatenate([a[0] for a in va if len(a[0])]); Yva = np.concatenate([a[1] for a in va if len(a[1])])
    model = StackedBiLSTMRegressor(C, hidden, tout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xv, Yv = torch.from_numpy(Xva), torch.from_numpy(Yva)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 16):
            idx = perm[i:i + 16]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), Yt[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xv), Yv).item()
        hist.append(v)
        if v < best - 1e-6:
            best, best_state, bad = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 20 == 0:
            print(f"epoch {ep} val_mse {v:.4f}")
    model.load_state_dict(best_state)
    return model, {"train_windows": int(len(Xtr)), "val_windows": int(len(Xva)), "epochs_run": len(hist),
                   "best_val_mse": best}


def predict_mc(model, window: np.ndarray, n_samples=50):
    """MC-dropout: mean and std of forecast for a [Tin,C] window."""
    require_torch()
    model.train()  # dropout on
    x = torch.from_numpy(window[None].astype(np.float32)).repeat(n_samples, 1, 1)
    with torch.no_grad():
        p = model(x).numpy()
    model.eval()
    return p.mean(0), p.std(0)


def test_metrics(model, feats: dict[str, np.ndarray], target_col: int, tin=24, tout=12, n_samples=50):
    """Score the final `tout` months of every region, with MC-dropout mean as the point forecast."""
    from .baseline import metrics
    ys, ps, sds = [], [], []
    for f in feats.values():
        T = f.shape[0]
        w = f[T - tout - tin:T - tout]
        y = f[T - tout:, target_col]
        if np.isfinite(w).all() and np.isfinite(y).all():
            m, s = predict_mc(model, w, n_samples)
            ys.append(y); ps.append(m); sds.append(s)
    if not ys:
        return {"error": "no complete test windows"}
    out = metrics(np.concatenate(ys), np.concatenate(ps))
    out["mean_mc_std"] = float(np.mean(np.concatenate(sds)))
    return out


def roll_forward(model, window: np.ndarray, target_col: int, exog_climatology: np.ndarray,
                 next_month: int, rounds: int, n_samples=50):
    """Autoregressive multi-round forecast (paper: each 12-month output is appended to the
    input). exog_climatology [12,C] gives the calendar-month climatology used for the
    non-target features of appended months; next_month (0=Jan) is the calendar month of the
    first forecast step. Returns (mean, std), each [rounds*tout]. The std is the MC-dropout
    spread of each round only: it does NOT propagate earlier rounds' uncertainty, so it
    understates long-horizon uncertainty."""
    cur = window.copy()
    means, stds = [], []
    month = next_month
    for _ in range(rounds):
        m, s = predict_mc(model, cur, n_samples)
        means.append(m); stds.append(s)
        new = np.stack([exog_climatology[(month + k) % 12] for k in range(len(m))]).copy()
        new[:, target_col] = m
        cur = np.concatenate([cur, new])[-window.shape[0]:]
        month = (month + len(m)) % 12
    return np.concatenate(means), np.concatenate(stds)
