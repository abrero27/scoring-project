from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FWPositionPack:
    df_fw: pd.DataFrame
    feature_cols: list[str]
    cluster_map: dict[int, str]
    thresholds: dict[str, float]


def _parse_minutes_series(s: pd.Series) -> pd.Series:
    """
    Your Excel is cleaned:
    - thousands are plain integers (e.g. 2578)
    - decimals are with dot (e.g. 2.5)
    """
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric (robust to object dtype)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _available_numeric(df: pd.DataFrame, candidates: list[str], min_non_nan_ratio: float = 0.60) -> list[str]:
    """Keep columns that exist and have enough non-NaN coverage after coercion."""
    keep = []
    n = len(df)
    for c in candidates:
        if c in df.columns:
            non_nan = df[c].notna().sum()
            if n > 0 and (non_nan / n) >= min_non_nan_ratio:
                keep.append(c)
    return keep


def _fit_gmm_2class(Xz_train: np.ndarray, random_state: int) -> GaussianMixture:
    """
    Conservative 2-component GMM (like DF stage):
    - diag covariance => less overfit
    - stronger regularization
    """
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="diag",
        random_state=random_state,
        n_init=30,
        max_iter=1500,
        reg_covar=1e-3,
    )
    gmm.fit(Xz_train)
    return gmm


def assign_fw_positions(
    df_players: pd.DataFrame,
    min_train_minutes: int = 900,
    prob_threshold: float = 0.60,
    hybrid_margin: float = 0.20,
    proba_temperature: float = 1.30,
    random_state: int = 42,
    export_filename: str = "attackers_fw_posts.csv",
) -> FWPositionPack:
    """
    Stage: Among PURE forwards (Pos == 'FW'), assign:
      - ST
      - NON_ST

    We DO NOT create a HYBRID class.
    Instead we create:
      - hybrid_flag (bool) if player sits between both profiles:
          |p_ST - p_NON_ST| < hybrid_margin  OR  max(p) < prob_threshold

    Training:
      - fit on PURE FW with Min_num >= min_train_minutes
    Inference:
      - predict probabilities for ALL PURE FW
      - always output fw_post in {ST, NON_ST}
    """
    df_fw = df_players[df_players["Pos"].astype(str).str.upper().eq("FW")].copy().reset_index(drop=True)
    if df_fw.empty:
        raise ValueError("No PURE FW found: Pos == 'FW'.")

    if "Min" not in df_fw.columns:
        raise ValueError("Expected column 'Min' in df_players.")

    # Minutes
    df_fw["Min_num"] = _parse_minutes_series(df_fw["Min"]).fillna(0.0)
    df_fw["low_min_flag"] = df_fw["Min_num"] < float(min_train_minutes)

    # ---------------------------------------------------------
    # Feature design: specific, low-noise (ST vs NON-ST)
    # ---------------------------------------------------------
    candidates = [
        # Finishing / box presence
        "Touches_AttPen", "npxG", "xG", "Gls", "Sh/90", "SoT/90", "SoT",
        # Creation / wide
        "xA", "xAG", "KP", "PPA", "SCA90", "Crs", "Pass_Cross", "TakeOn_Att", "TakeOn_Succ",
        # spatial
        "Touches_Att3rd",
    ]

    df_fw = _coerce_numeric(df_fw, candidates)
    usable = _available_numeric(df_fw, candidates, min_non_nan_ratio=0.60)

    # Keep it compact & stable order (avoid overfitting)
    preferred_order = [
        "Touches_AttPen", "npxG", "xG", "Sh/90", "SoT/90",
        "xA", "KP", "SCA90", "Crs", "Pass_Cross", "TakeOn_Att", "Touches_Att3rd",
    ]
    feature_cols = [c for c in preferred_order if c in usable]

    if len(feature_cols) < 6:
        raise ValueError(f"Too few usable FW features. Found: {feature_cols}")

    # Train subset (>= min_train_minutes)
    train = df_fw[~df_fw["low_min_flag"]].copy()
    if train.shape[0] < 10:
        raise ValueError(
            f"Not enough PURE FW with Min >= {min_train_minutes} to train. "
            f"Have {train.shape[0]}. Lower min_train_minutes or widen data."
        )

    # Impute with TRAIN medians (more stable)
    train_medians = train[feature_cols].median(numeric_only=True)

    X_train = train[feature_cols].copy().fillna(train_medians)

    scaler = StandardScaler()
    Xz_train = scaler.fit_transform(X_train.values)

    gmm = _fit_gmm_2class(Xz_train, random_state=random_state)

    # Determine which component is ST vs NON_ST using centers (z-space)
    centers = pd.DataFrame(gmm.means_, columns=feature_cols)

    def get(col: str) -> pd.Series:
        return centers[col] if col in centers.columns else 0.0

    # ST score: box touches + xG + shooting
    st_score = (
        1.4 * get("Touches_AttPen")
        + 1.1 * get("npxG")
        + 0.9 * get("xG")
        + 0.9 * get("Sh/90")
        + 0.9 * get("SoT/90")
        - 0.6 * get("Crs")
        - 0.4 * get("Pass_Cross")
        - 0.4 * get("TakeOn_Att")
        - 0.3 * get("xA")
        - 0.2 * get("KP")
    )

    st_component = int(pd.Series(st_score).idxmax())
    nonst_component = int([i for i in [0, 1] if i != st_component][0])

    cluster_map = {st_component: "ST", nonst_component: "NON_ST"}

    # Predict for ALL PURE FW (use TRAIN medians for impute)
    X_all = df_fw[feature_cols].copy().fillna(train_medians)
    Xz_all = scaler.transform(X_all.values)

    proba = gmm.predict_proba(Xz_all)  # (n, 2)

    # Optional: soften extreme probabilities for nicer reporting
    if proba_temperature and proba_temperature > 1.0:
        proba = np.power(proba, 1.0 / float(proba_temperature))
        proba = proba / proba.sum(axis=1, keepdims=True)

    # Map to p_ST / p_NON_ST (sum = 1)
    df_fw["p_ST"] = proba[:, st_component]
    df_fw["p_NON_ST"] = proba[:, nonst_component]

    # Best label (ONLY ST / NON_ST)
    best_component = proba.argmax(axis=1)
    best_p = proba.max(axis=1)

    df_fw["fw_cluster"] = best_component
    df_fw["fw_post_model"] = pd.Series(best_component).map(cluster_map).values
    df_fw["fw_post_proba"] = best_p
    df_fw["fw_post"] = df_fw["fw_post_model"]

    # HYBRID flag (not a class)
    gap = np.abs(df_fw["p_ST"] - df_fw["p_NON_ST"])
    df_fw["prob_gap"] = gap  # diagnostic (optional)

    df_fw["hybrid_flag"] = (gap < float(hybrid_margin)) | (df_fw["fw_post_proba"] < float(prob_threshold))

    # Export (short, not all raw stats)
    out_cols = [
        "Player", "Squad", "Pos", "Min", "Min_num", "low_min_flag",
        "fw_post", "fw_post_proba", "p_ST", "p_NON_ST",
        "hybrid_flag", "prob_gap",
    ]
    out_cols = [c for c in out_cols if c in df_fw.columns]
    out_path = PROCESSED_DIR / export_filename
    df_fw[out_cols].sort_values(["Squad", "Player"]).to_csv(out_path, index=False)

    return FWPositionPack(
        df_fw=df_fw,
        feature_cols=feature_cols,
        cluster_map=cluster_map,
        thresholds={
            "min_train_minutes": float(min_train_minutes),
            "prob_threshold": float(prob_threshold),
            "hybrid_margin": float(hybrid_margin),
            "proba_temperature": float(proba_temperature),
        },
    )
