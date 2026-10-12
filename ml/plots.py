"""Static report figures (PNG). Palette: validated categorical order from the dataviz reference palette."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
REF = "#8a8986"
INK = "#0b0b0b"
INK2 = "#52514e"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(REF)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def reliability(curves: dict[str, list[dict]], path, title: str):
    fig, axes = plt.subplots(1, len(curves), figsize=(3.2 * len(curves), 3.2))
    for ax, (t, bins) in zip(np.atleast_1d(axes), curves.items(), strict=True):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=REF, ls="--", lw=1)
        x = [b["mean_pred"] for b in bins]
        y = [b["frac_pos"] for b in bins]
        s = [8 + 2 * b["n"] for b in bins]
        ax.plot(x, y, color=SERIES[0], lw=2)
        ax.scatter(x, y, s=s, color=SERIES[0], edgecolor="white", linewidth=1.5, zorder=3)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=t, xlabel="mean predicted")
    np.atleast_1d(axes)[0].set_ylabel("observed fraction")
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cv_comparison(summary: dict[str, dict], variants: list[str], targets: list[str], path, metric="roc_auc"):
    fig, ax = plt.subplots(figsize=(1.3 * len(targets) + 4, 3.6))
    _style(ax)
    w = 0.8 / len(variants)
    x = np.arange(len(targets))
    for i, v in enumerate(variants):
        m = [summary[v][t][metric]["mean"] for t in targets]
        sd = [summary[v][t][metric]["sd"] for t in targets]
        ax.bar(x + i * w - 0.4 + w / 2, m, w * 0.92, yerr=sd, color=SERIES[i % 8], label=v,
               error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 2})
    ax.set_xticks(x, targets)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel(f"{metric} (mean ± SD, 5×5 nested CV)")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def lines(series: dict[str, tuple[list, list]], path, xlabel, ylabel, title, ref_line=None):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    _style(ax)
    for i, (name, (x, y)) in enumerate(series.items()):
        ax.plot(x, y, color=SERIES[i % 8], lw=2, marker="o", ms=4, label=name)
    if ref_line is not None:
        ax.axhline(ref_line, color=REF, ls="--", lw=1)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def hbar(values: dict[str, float], path, xlabel, title, top=15):
    items = sorted(values.items(), key=lambda kv: kv[1])[-top:]
    fig, ax = plt.subplots(figsize=(5.2, 0.28 * len(items) + 1.2))
    _style(ax)
    ax.barh([k for k, _ in items], [v for _, v in items], color=SERIES[0], height=0.7)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
