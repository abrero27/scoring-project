# src/radar_plot.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _radar_angles(n: int):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    return angles


def plot_radar_from_df(df: pd.DataFrame, title: str, outfile: str):
    """
    df must have: axis, score_0100
    """
    df = df.copy()
    df["score_0100"] = pd.to_numeric(df["score_0100"], errors="coerce")
    df = df.dropna(subset=["score_0100"])

    labels = df["axis"].tolist()
    values = df["score_0100"].tolist()

    if len(labels) < 3:
        raise ValueError("Need at least 3 axes for a radar chart.")

    values += values[:1]
    angles = _radar_angles(len(labels))

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 100)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.15)

    ax.set_title(title, y=1.10)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)
