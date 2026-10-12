"""Decision curve analysis (Vickers & Elkin 2006): net benefit vs threshold probability."""

from __future__ import annotations

import numpy as np

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def net_benefit(y, p, thresholds=THRESHOLDS) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    n, prev = len(y), float(np.mean(y))
    model, treat_all = [], []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1)) / n
        fp = np.sum(pred & (y == 0)) / n
        w = t / (1 - t)
        model.append(float(tp - fp * w))
        treat_all.append(float(prev - (1 - prev) * w))
    return {"thresholds": [float(t) for t in thresholds], "model": model, "treat_all": treat_all,
            "treat_none": [0.0] * len(thresholds), "prevalence": prev}


def plot_dca(curves: dict[str, dict], path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (name, c) in zip(axes, curves.items(), strict=True):
        ax.plot(c["thresholds"], c["model"], label="model", color="#2a78d6", lw=2)
        ax.plot(c["thresholds"], c["treat_all"], label="treat all", color="#8a8986", ls="--")
        ax.axhline(0, color="black", lw=0.8, label="treat none")
        lo = min(min(c["model"]), 0) - 0.02
        ax.set_ylim(max(lo, -0.1), max(c["prevalence"], max(c["model"])) + 0.05)
        ax.set_title(name)
        ax.set_xlabel("threshold probability")
    axes[0].set_ylabel("net benefit")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
