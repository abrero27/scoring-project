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
class DFPositionPack:
    df_df: pd.DataFrame
    feature_cols: list[str]
    cluster_map: dict[int, str]
    thresholds: dict[str, float]


# -------------------------
# Utils
# -------------------------
def _parse_minutes_series(s: pd.Series) -> pd.Series:
    # Excel cleaned: thousands are plain ints, decimals use dot
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce").fillna(0.0)


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _available_numeric(df: pd.DataFrame, candidates: list[str], min_non_nan_ratio: float = 0.60) -> list[str]:
    keep = []
    n = len(df)
    if n == 0:
        return keep
    for c in candidates:
        if c in df.columns:
            non_nan = df[c].notna().sum()
            if (non_nan / n) >= float(min_non_nan_ratio):
                keep.append(c)
    return keep


def _soften_proba(proba: np.ndarray, temperature: float | None) -> np.ndarray:
    if temperature is None or temperature <= 1.0:
        return proba
    p = np.power(proba, 1.0 / float(temperature))
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


def _fit_gmm_2class(Xz_train: np.ndarray, random_state: int) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="diag",
        random_state=random_state,
        n_init=40,
        max_iter=2000,
        reg_covar=2e-3,
    )
    gmm.fit(Xz_train)
    return gmm


# -------------------------
# Main API
# -------------------------
def assign_df_positions(
    df_players: pd.DataFrame,
    min_train_minutes: int = 900,
    proba_temperature: float | None = 1.20,
    random_state: int = 42,
    export_filename: str = "defenders_df_posts.csv",
) -> DFPositionPack:
    """
    PURE DF only (Pos == 'DF').

    2-class DF post assignment:
      - CB
      - FB_WB

    Training:
      - fit on PURE DF with Min_num >= min_train_minutes

    Inference:
      - predict probabilities for ALL PURE DF
      - always output df_post in {CB, FB_WB}
      - optional: hybrid_flag if "between" both profiles:
          |p_CB - p_FB_WB| < hybrid_margin OR max(p) < prob_threshold
    """
    df_df = df_players[df_players["Pos"].astype(str).str.upper().eq("DF")].copy().reset_index(drop=True)
    if df_df.empty:
        raise ValueError("No PURE DF found: Pos == 'DF'.")

    if "Min" not in df_df.columns:
        raise ValueError("Expected column 'Min' in df_players.")

    # Minutes
    df_df["Min_num"] = _parse_minutes_series(df_df["Min"])
    df_df["low_min_flag"] = df_df["Min_num"] < float(min_train_minutes)

    # ---- Feature design: compact + discriminant (CB vs FB/WB) ----
    # CB: aerials/clear/defpen/interceptions
    # FB/WB: crosses + advanced touches + carries into final third
    candidates = [
        "Aerials_Won%",
        "Clearances",
        "Interceptions",
        "Touches_DefPen",
        "Crs",
        "Touches_Att3rd",
        "Carry_FinalThird",
        # optional build-up discriminator (helps wide CB vs true FB)
        "Pass_Switch",
    ]
    df_df = _coerce_numeric(df_df, candidates)
    usable = _available_numeric(df_df, candidates, min_non_nan_ratio=0.60)

    preferred = [
        "Aerials_Won%",
        "Clearances",
        "Touches_DefPen",
        "Interceptions",
        "Crs",
        "Touches_Att3rd",
        "Carry_FinalThird",
        "Pass_Switch",
    ]
    feature_cols = [c for c in preferred if c in usable]

    # Keep it compact/stable
    feature_cols = feature_cols[:7]  # <= 7 features max (avoid noise)
    if len(feature_cols) < 5:
        raise ValueError(f"Too few usable DF features. Found: {feature_cols}")

    # Train subset (>= min_train_minutes)
    train = df_df[~df_df["low_min_flag"]].copy()
    if train.shape[0] < 12:
        raise ValueError(
            f"Not enough PURE DF with Min >= {min_train_minutes} to train. "
            f"Have {train.shape[0]}. Lower min_train_minutes or widen data."
        )

    # Impute with TRAIN medians (stable)
    train_medians = train[feature_cols].median(numeric_only=True)

    X_train = train[feature_cols].copy().fillna(train_medians)

    scaler = StandardScaler()
    Xz_train = scaler.fit_transform(X_train.values)

    gmm = _fit_gmm_2class(Xz_train, random_state=random_state)

    # Determine which component is CB vs FB_WB using centers in z-space
    centers = pd.DataFrame(gmm.means_, columns=feature_cols)

    def g(col: str) -> pd.Series:
        return centers[col] if col in centers.columns else 0.0

    cb_score = (
        1.35 * g("Aerials_Won%")
        + 1.10 * g("Clearances")
        + 1.05 * g("Touches_DefPen")
        + 0.85 * g("Interceptions")
        - 1.05 * g("Crs")
        - 0.95 * g("Touches_Att3rd")
        - 0.90 * g("Carry_FinalThird")
    )

    cb_component = int(pd.Series(cb_score).idxmax())
    fb_component = int([i for i in [0, 1] if i != cb_component][0])
    cluster_map = {cb_component: "CB", fb_component: "FB_WB"}

    # Predict for ALL DF (use TRAIN medians)
    X_all = df_df[feature_cols].copy().fillna(train_medians)
    Xz_all = scaler.transform(X_all.values)

    proba = gmm.predict_proba(Xz_all)

    # soften (reporting)
    proba = _soften_proba(proba, proba_temperature)

    df_df["p_CB"] = proba[:, cb_component]
    df_df["p_FB_WB"] = proba[:, fb_component]

    best_component = proba.argmax(axis=1)
    best_p = proba.max(axis=1)

    df_df["df_cluster"] = best_component
    df_df["df_post_model"] = pd.Series(best_component).map(cluster_map).values
    df_df["df_post_proba"] = best_p
    df_df["df_post"] = df_df["df_post_model"]  # ALWAYS CB / FB_WB

    # Export (clean)
    out_cols = [
        "Player", "Squad", "Pos", "Min", "Min_num", "low_min_flag",
        "df_post", "df_post_proba", "p_CB", "p_FB_WB",
    ]
    out_cols = [c for c in out_cols if c in df_df.columns]
    out_path = PROCESSED_DIR / export_filename
    df_df[out_cols].sort_values(["Squad", "Player"]).to_csv(out_path, index=False)

    return DFPositionPack(
        df_df=df_df,
        feature_cols=feature_cols,
        cluster_map=cluster_map,
        thresholds={
            "min_train_minutes": float(min_train_minutes),
            "proba_temperature": float(proba_temperature or 1.0),
        },
    )
