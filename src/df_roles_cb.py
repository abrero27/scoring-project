from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Output container
# -------------------------
@dataclass
class CBRolePack:
    df_cb_roles: pd.DataFrame
    feature_cols: list[str]
    chosen_k: int
    bic: dict[int, float]
    role_map: dict[int, str]      # cluster -> template role
    thresholds: dict[str, float]


# -------------------------
# Utils (same style as FW/MF)
# -------------------------
def _parse_minutes_series(s: pd.Series) -> pd.Series:
    """
    Robust minutes parsing:
    - handles "2,598" or "2.598" as 2598 (thousands separator)
    - trims spaces
    """
    x = s.astype(str).str.strip()
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace(".", "", regex=False)
    return pd.to_numeric(x, errors="coerce").fillna(0.0)


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


def _fit_best_gmm_by_bic(
    Xz: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    covariance_type: str = "diag",
) -> tuple[GaussianMixture, int, dict[int, float]]:
    bic: dict[int, float] = {}
    best_model: GaussianMixture | None = None
    best_k: int | None = None
    best_bic = np.inf

    n = Xz.shape[0]
    for k in range(k_min, k_max + 1):
        if n < k:
            continue

        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=40,
            max_iter=3000,
            reg_covar=2e-3,
        )
        gmm.fit(Xz)
        b = float(gmm.bic(Xz))
        bic[k] = b

        if b < best_bic:
            best_bic = b
            best_model = gmm
            best_k = k

    if best_model is None or best_k is None:
        raise ValueError(f"Not enough samples to fit GMM for k={k_min}..{k_max}. n={n}")

    return best_model, int(best_k), bic


def _soften_proba(proba: np.ndarray, temperature: float | None) -> np.ndarray:
    """temperature > 1 => less extreme probs, rows still sum to 1."""
    if temperature is None or temperature <= 1.0:
        return proba
    p = np.power(proba, 1.0 / float(temperature))
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


# ------------------------------
# Role templates (football logic)
# ------------------------------
def _cb_role_scores(centers: pd.DataFrame, elite_build_threshold: float = 3.0) -> dict[str, pd.Series]:
    """
    Goal (based on your cluster summary):
      - Cluster 0 => CB_DUEL
      - Cluster 1 => CB_HYBRID (positive build but not elite, weak duel)
      - Cluster 2 => CB_BUILDER (elite progression / relance)
    centers are in z-space (StandardScaler).
    """
    def g(col: str) -> pd.Series:
        return centers[col] if col in centers.columns else 0.0

    # --- Duel index (dominance in defensive actions + aerial) ---
    duel = (
        1.15 * g("Clearances")
        + 1.00 * g("Blocks_Total")
        + 1.05 * g("Interceptions")
        + 1.00 * g("Tackles_Interceptions")
        + 1.10 * g("Aerials_Won")
        + 0.85 * g("Aerials_Won%")
    )

    # --- Elite build index (THIS is what separates your cluster #2) ---
    elite_build = (
        1.00 * g("PrgDist")
        + 1.00 * g("FinalThird_Pass")
        + 0.60 * g("PrgP")
        + 0.60 * g("Carry_PrgDist")
        + 0.30 * g("Pass_Switch")
        + 0.30 * g("Pass_Cmp%")
    )

    # Gate: builder only if elite_build is REALLY high (z-space)
    gate = (elite_build >= float(elite_build_threshold)).astype(float)  # 0 or 1
    gap = (duel - elite_build).abs()

    # --- Scores ---
    # Builder is "very mean":
    # - if gate=0 => strong penalty (-5)
    # - if gate=1 => strong bonus (+5)
    # So cluster1 cannot become builder anymore, cluster2 will.
    base_builder = (elite_build - 0.25 * duel) - 0.10 * gap
    score_builder = base_builder + (10.0 * gate - 5.0)

    # Duel easier to get: duel dominates and build penalizes a bit
    score_duel = duel - 0.55 * elite_build

    # Hybrid: ONLY when not elite-builder (gate=0)
    # Wants: positive build, not too extreme, and not a duel monster.
    score_hybrid = (1.0 - gate) * (
        0.95 * elite_build
        - 0.15 * duel.abs()
        - 0.10 * gap
    )

    return {
        "CB_DUEL": score_duel,
        "CB_HYBRID": score_hybrid,
        "CB_BUILDER": score_builder,
    }


def _cluster_to_role_by_argmax(centers: pd.DataFrame, elite_build_threshold: float) -> dict[int, str]:
    """
    Map each cluster to the role where it scores highest.
    Then we aggregate component probabilities by role (FW/MF style).
    """
    scores = _cb_role_scores(centers, elite_build_threshold=elite_build_threshold)
    role_names = list(scores.keys())  # ["CB_DUEL","CB_HYBRID","CB_BUILDER"]
    mat = np.vstack([scores[r].values for r in role_names]).T  # (k, 3)
    best_role_idx = mat.argmax(axis=1)
    return {c: role_names[int(best_role_idx[c])] for c in range(mat.shape[0])}


# ------------------------------
# Main API
# ------------------------------
def assign_cb_roles(
    df_players: pd.DataFrame,
    df_df_posts: pd.DataFrame,             # needs Player, Squad, df_post
    min_train_minutes: int = 700,
    k_min: int = 2,
    k_max: int = 6,
    proba_temperature: float | None = 1.20,
    low_min_shrink: float = 0.0,          # keep 0.0 by default (you said you don't care)
    random_state: int = 42,
    export_filename: str = "defenders_cb_roles.csv",
    summary_filename: str = "defenders_cb_roles_cluster_summary.csv",
    elite_build_threshold: float = 3.0,   # <-- the knob (higher = builder rarer)
) -> CBRolePack:
    df = df_players.copy()

    # PURE DF
    df_df = df[df["Pos"].astype(str).str.upper().eq("DF")].copy().reset_index(drop=True)
    if df_df.empty:
        raise ValueError("No PURE DF found: Pos == 'DF'.")

    # merge df_post
    join_cols = ["Player", "Squad"]
    tmp = df_df_posts.copy()
    for c in join_cols:
        if c not in tmp.columns:
            raise ValueError(f"df_df_posts must contain column '{c}'.")
        tmp[c] = tmp[c].astype(str)
        df_df[c] = df_df[c].astype(str)

    if "df_post" not in tmp.columns:
        raise ValueError("df_df_posts must contain column 'df_post'.")

    df_df = df_df.merge(tmp[join_cols + ["df_post"]], on=join_cols, how="left")
    if df_df["df_post"].isna().any():
        missing = int(df_df["df_post"].isna().sum())
        raise ValueError(f"{missing} DF rows have no df_post after merge. Check join keys (Player/Squad).")

    # CB only
    df_cb = df_df[df_df["df_post"].astype(str).eq("CB")].copy().reset_index(drop=True)
    if df_cb.empty:
        raise ValueError("No CB found: df_post == 'CB'.")

    # minutes
    if "Min" not in df_cb.columns:
        raise ValueError("Expected column 'Min'.")
    df_cb["Min_num"] = _parse_minutes_series(df_cb["Min"])
    df_cb["low_min_flag"] = df_cb["Min_num"] < float(min_train_minutes)

    # -------------------------
    # Feature set (short + discriminant)
    # -------------------------
    candidates = [
        # duel/defense
        "Clearances", "Blocks_Total", "Interceptions", "Tackles_Interceptions",
        "Aerials_Won", "Aerials_Won%",
        # builder/progression (quality, not volume)
        "PrgP", "PrgDist", "Pass_Switch", "FinalThird_Pass", "Carry_PrgDist", "Pass_Cmp%",
    ]
    df_cb = _coerce_numeric(df_cb, candidates)
    usable = _available_numeric(df_cb, candidates, min_non_nan_ratio=0.60)

    preferred = [
        "Clearances", "Blocks_Total", "Interceptions", "Tackles_Interceptions",
        "Aerials_Won", "Aerials_Won%",
        "PrgP", "PrgDist", "Pass_Switch", "FinalThird_Pass", "Carry_PrgDist", "Pass_Cmp%",
    ]
    feature_cols = [c for c in preferred if c in usable]
    if len(feature_cols) < 8:
        raise ValueError(f"Too few usable CB role features. Found: {feature_cols}")

    # -------------------------
    # TRAIN STRICT (Min >= min_train_minutes)
    # -------------------------
    train = df_cb[~df_cb["low_min_flag"]].copy()
    if train.shape[0] < (k_min + 6):
        raise ValueError(f"Not enough CB with Min >= {min_train_minutes} to train. Have {train.shape[0]}.")

    train_medians = train[feature_cols].median(numeric_only=True)
    X_train = train[feature_cols].copy().fillna(train_medians)

    scaler = StandardScaler()
    Xz_train = scaler.fit_transform(X_train.values)

    gmm, chosen_k, bic = _fit_best_gmm_by_bic(Xz_train, k_min, k_max, random_state, "diag")
    centers = pd.DataFrame(gmm.means_, columns=feature_cols)

    # Map clusters -> roles (3 roles)
    role_map = _cluster_to_role_by_argmax(centers, elite_build_threshold=elite_build_threshold)
    roles = ["CB_DUEL", "CB_HYBRID", "CB_BUILDER"]

    # -------------------------
    # PREDICT ALL CB (including low-min)
    # -------------------------
    X_all = df_cb[feature_cols].copy().fillna(train_medians)
    Xz_all = scaler.transform(X_all.values)

    proba_comp = _soften_proba(gmm.predict_proba(Xz_all), proba_temperature)

    if low_min_shrink and low_min_shrink > 0:
        u = np.ones_like(proba_comp) / proba_comp.shape[1]
        mask = df_cb["low_min_flag"].values
        proba_comp[mask] = (1.0 - float(low_min_shrink)) * proba_comp[mask] + float(low_min_shrink) * u[mask]

    # aggregate component probs -> role probs
    role_probs = {r: np.zeros(len(df_cb), dtype=float) for r in roles}
    for j in range(proba_comp.shape[1]):
        r = role_map.get(j)
        if r in role_probs:
            role_probs[r] += proba_comp[:, j]

    mat = np.vstack([role_probs[r] for r in roles]).T
    mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-12)

    best_idx = mat.argmax(axis=1)
    best_role = [roles[i] for i in best_idx]
    best_p = mat.max(axis=1)

    df_out = df_cb.copy()
    df_out["cb_role_cluster"] = proba_comp.argmax(axis=1)
    df_out["cb_role"] = best_role
    df_out["cb_role_proba"] = best_p
    df_out["p_cb_role_CB_DUEL"] = mat[:, 0]
    df_out["p_cb_role_CB_HYBRID"] = mat[:, 1]
    df_out["p_cb_role_CB_BUILDER"] = mat[:, 2]

    # -------------------------
    # Export
    # -------------------------
    base_cols = [
        "Player","Squad","Pos","Min","Min_num","low_min_flag",
        "df_post","cb_role","cb_role_proba","cb_role_cluster",
        "p_cb_role_CB_DUEL","p_cb_role_CB_HYBRID","p_cb_role_CB_BUILDER",
    ]
    out_cols = [c for c in base_cols if c in df_out.columns]
    df_out[out_cols].sort_values(["Squad","Player"]).to_csv(PROCESSED_DIR / export_filename, index=False)

    # cluster summary (centers in z-space)
    centers_out = centers.copy()
    centers_out["cluster"] = range(len(centers_out))
    centers_out["mapped_role"] = centers_out["cluster"].map(role_map).astype(str)
    centers_out["avg_abs_z"] = centers_out[feature_cols].abs().mean(axis=1)
    cols_sum = ["cluster","mapped_role","avg_abs_z"] + feature_cols
    centers_out[cols_sum].to_csv(PROCESSED_DIR / summary_filename, index=False)

    return CBRolePack(
        df_cb_roles=df_out,
        feature_cols=feature_cols,
        chosen_k=chosen_k,
        bic=bic,
        role_map=role_map,
        thresholds={
            "min_train_minutes": float(min_train_minutes),
            "k_min": float(k_min),
            "k_max": float(k_max),
            "proba_temperature": float(proba_temperature or 1.0),
            "low_min_shrink": float(low_min_shrink),
            "elite_build_threshold": float(elite_build_threshold),
        },
    )
