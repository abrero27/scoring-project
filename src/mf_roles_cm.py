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
class CMRolePack:
    df_cm_roles: pd.DataFrame
    feature_cols: list[str]
    chosen_k: int
    bic: dict[int, float]
    role_map: dict[int, str]   # cluster -> role template
    thresholds: dict[str, float]


# -------------------------
# Utils
# -------------------------
def _parse_minutes_series(s: pd.Series) -> pd.Series:
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
        if c in df.columns and (df[c].notna().sum() / n) >= float(min_non_nan_ratio):
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
    best_model = None
    best_k = None
    best_bic = np.inf

    n = int(Xz.shape[0])
    for k in range(int(k_min), int(k_max) + 1):
        if n < k:
            continue

        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=40,
            max_iter=2500,
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
        raise ValueError(f"Not enough samples to fit GMM in range k={k_min}..{k_max}. n={n}")

    return best_model, int(best_k), bic


def _soften_proba(proba: np.ndarray, temperature: float | None) -> np.ndarray:
    """temperature > 1 => less extreme probs, rows still sum to 1."""
    if temperature is None or temperature <= 1.0:
        return proba
    p = np.power(proba, 1.0 / float(temperature))
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


# ------------------------------
# CM role templates
# ------------------------------
def _role_scores_cm(centers: pd.DataFrame) -> dict[str, pd.Series]:
    """
    CM roles (ONLY among CM players):
      - CM_PROGRESSOR: progresses mainly by PASSING (PrgP, PrgDist, switches/longs, volume)
      - CM_BOX_TO_BOX: progresses mainly by CARRY / movement (PrgC, carries, Att3rd involvement)

    NOTE (rapport-friendly):
      - Ici "BOX_TO_BOX" signifie "CM très mobile / carry-driven" (profil 'carrier/runner').
      - Les CM très contrôleurs (pass-heavy) basculent plutôt en PROGRESSOR.
    centers are in z-space.
    """
    def g(col: str) -> pd.Series:
        return centers[col] if col in centers.columns else 0.0

    # PASS progression (progressor signature)
    pass_prog = (
        1.35 * g("PrgP")
        + 1.05 * g("PrgDist")
        + 0.95 * g("Pass_Att")
        + 0.80 * g("FinalThird_Pass")
        + 0.85 * g("Pass_Switch")
        + 0.55 * g("Pass_Long_Att")
        + 0.45 * g("PPA")
    )

    # CARRY progression (box-to-box signature)
    carry_prog = (
        1.30 * g("PrgC")
        + 1.15 * g("Carry_PrgDist")
        + 0.90 * g("Carry_FinalThird")
        + 0.55 * g("PrgR")
        + 0.55 * g("Touches_Att3rd")
        + 0.35 * g("SCA90")
    )

    # very light defense baseline (should not drive the split)
    defense_light = 0.15 * g("Tackles_Interceptions") + 0.10 * g("Interceptions")

    scores: dict[str, pd.Series] = {}

    # IMPORTANT: make BOX_TO_BOX stricter by penalizing pass_prog more strongly
    scores["CM_PROGRESSOR"] = (1.20 * pass_prog + 0.10 * defense_light) - (1.00 * carry_prog)
    scores["CM_BOX_TO_BOX"] = (1.35 * carry_prog + 0.10 * defense_light) - (1.25 * pass_prog)

    return scores


def _cluster_to_role_by_argmax(centers: pd.DataFrame) -> dict[int, str]:
    scores = _role_scores_cm(centers)  # role -> series over clusters
    role_names = list(scores.keys())   # ["CM_PROGRESSOR","CM_BOX_TO_BOX"]
    mat = np.vstack([scores[r].values for r in role_names]).T  # (k, 2)
    best_role_idx = mat.argmax(axis=1)
    return {c: role_names[int(best_role_idx[c])] for c in range(mat.shape[0])}


# ------------------------------
# Main API
# ------------------------------
def assign_cm_roles(
    df_players: pd.DataFrame,
    df_mf_posts: pd.DataFrame,
    min_train_minutes: int = 700,     # train on >= 700, but assign to all
    k_min: int = 2,
    k_max: int = 6,
    proba_temperature: float | None = 1.20,
    random_state: int = 42,
    export_filename: str = "midfielders_cm_roles.csv",
    summary_filename: str = "midfielders_cm_roles_cluster_summary.csv",
) -> CMRolePack:
    """
    PURE MF only, CM only:
      - filter Pos == 'MF'
      - merge mf_post from mf_positions output
      - keep mf_post == 'CM'
      - train on Min >= min_train_minutes (but assign a role to everyone)

    Output:
      - cm_role, cm_role_proba
      - cm_role_cluster (GMM component id)
      - p_cm_role_CM_PROGRESSOR / p_cm_role_CM_BOX_TO_BOX
      - cluster summary with centers (z-space)
    """
    df = df_players.copy()

    # PURE MF
    df_mf = df[df["Pos"].astype(str).str.upper().eq("MF")].copy().reset_index(drop=True)
    if df_mf.empty:
        raise ValueError("No PURE MF found: Pos == 'MF'.")

    # merge mf_post
    join_cols = ["Player", "Squad"]
    tmp = df_mf_posts.copy()
    for c in join_cols:
        if c not in tmp.columns:
            raise ValueError(f"df_mf_posts must contain column '{c}'.")
        tmp[c] = tmp[c].astype(str)
        df_mf[c] = df_mf[c].astype(str)

    if "mf_post" not in tmp.columns:
        raise ValueError("df_mf_posts must contain column 'mf_post'.")

    df_mf = df_mf.merge(tmp[join_cols + ["mf_post"]], on=join_cols, how="left")
    if df_mf["mf_post"].isna().any():
        missing = int(df_mf["mf_post"].isna().sum())
        raise ValueError(f"{missing} MF rows have no mf_post after merge. Check join keys (Player/Squad).")

    # keep CM only
    df_cm = df_mf[df_mf["mf_post"].astype(str).eq("CM")].copy().reset_index(drop=True)
    if df_cm.empty:
        raise ValueError("No CM found: mf_post == 'CM' (check your mf_positions output).")

    # minutes
    if "Min" not in df_cm.columns:
        raise ValueError("Expected column 'Min'.")
    df_cm["Min_num"] = _parse_minutes_series(df_cm["Min"])
    df_cm["low_min_flag"] = df_cm["Min_num"] < float(min_train_minutes)

    # ---------------------------------------------------------
    # Feature design (PASS vs CARRY)
    # ---------------------------------------------------------
    candidates = [
        # PASS progression
        "PrgP", "PrgDist", "Pass_Att", "FinalThird_Pass", "PPA", "Pass_Switch", "Pass_Long_Att",
        # CARRY progression
        "PrgC", "Carry_PrgDist", "Carry_FinalThird", "PrgR",
        # context / involvement
        "Touches_Att3rd", "Touches_Mid3rd",
        # light creation
        "SCA90",
        # very light defense
        "Tackles_Interceptions", "Interceptions",
    ]
    df_cm = _coerce_numeric(df_cm, candidates)
    usable = _available_numeric(df_cm, candidates, min_non_nan_ratio=0.60)

    preferred_order = [
        "PrgP", "PrgDist", "Pass_Att", "FinalThird_Pass", "PPA", "Pass_Switch", "Pass_Long_Att",
        "PrgC", "Carry_PrgDist", "Carry_FinalThird", "PrgR",
        "Touches_Att3rd", "Touches_Mid3rd",
        "SCA90",
        "Tackles_Interceptions", "Interceptions",
    ]
    feature_cols = [c for c in preferred_order if c in usable]
    if len(feature_cols) < 8:
        raise ValueError(f"Too few usable CM role features. Found: {feature_cols}")

    # Train subset (>= min_train_minutes). If too few, fallback: all CM.
    train = df_cm[~df_cm["low_min_flag"]].copy()
    if train.shape[0] < (k_min + 6):
        train = df_cm.copy()

    # Impute with TRAIN medians
    train_medians = train[feature_cols].median(numeric_only=True)
    X_train = train[feature_cols].copy().fillna(train_medians)

    scaler = StandardScaler()
    Xz_train = scaler.fit_transform(X_train.values)

    gmm, chosen_k, bic = _fit_best_gmm_by_bic(
        Xz_train, k_min=k_min, k_max=k_max, random_state=random_state, covariance_type="diag"
    )

    centers = pd.DataFrame(gmm.means_, columns=feature_cols)  # z-space
    role_map = _cluster_to_role_by_argmax(centers)
    templates = ["CM_PROGRESSOR", "CM_BOX_TO_BOX"]

    # Predict ALL CM (including low-min) using train medians + scaler
    X_all = df_cm[feature_cols].copy().fillna(train_medians)
    Xz_all = scaler.transform(X_all.values)

    proba_comp = _soften_proba(gmm.predict_proba(Xz_all), proba_temperature)  # (n, k)

    # Aggregate component probs -> role probs
    role_probs = {r: np.zeros(len(df_cm), dtype=float) for r in templates}
    for j in range(proba_comp.shape[1]):
        r = role_map.get(j)
        if r in role_probs:
            role_probs[r] += proba_comp[:, j]

    mat = np.vstack([role_probs[r] for r in templates]).T
    mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-12)

    best_idx = mat.argmax(axis=1)
    best_role = [templates[i] for i in best_idx]
    best_p = mat.max(axis=1)

    df_out = df_cm.copy()
    df_out["cm_role_cluster"] = proba_comp.argmax(axis=1)  # ✅ important for debug / report
    df_out["cm_role"] = best_role
    df_out["cm_role_proba"] = best_p
    df_out["p_cm_role_CM_PROGRESSOR"] = mat[:, 0]
    df_out["p_cm_role_CM_BOX_TO_BOX"] = mat[:, 1]

    # Export
    out_cols = [
        "Player", "Squad", "Pos", "Min", "Min_num", "low_min_flag", "mf_post",
        "cm_role", "cm_role_proba", "cm_role_cluster",
        "p_cm_role_CM_PROGRESSOR", "p_cm_role_CM_BOX_TO_BOX",
    ]
    out_cols = [c for c in out_cols if c in df_out.columns]
    (PROCESSED_DIR / export_filename).write_text(
        df_out[out_cols].sort_values(["Squad", "Player"]).to_csv(index=False),
        encoding="utf-8",
    )

    # Summary export (centers in z-space)
    centers_out = centers.copy()
    centers_out["cluster"] = range(len(centers_out))
    centers_out["mapped_role"] = centers_out["cluster"].map(role_map).astype(str)
    centers_out["avg_abs_z"] = centers_out[feature_cols].abs().mean(axis=1)
    cols_sum = ["cluster", "mapped_role", "avg_abs_z"] + feature_cols
    (PROCESSED_DIR / summary_filename).write_text(centers_out[cols_sum].to_csv(index=False), encoding="utf-8")

    return CMRolePack(
        df_cm_roles=df_out,
        feature_cols=feature_cols,
        chosen_k=chosen_k,
        bic=bic,
        role_map=role_map,
        thresholds={
            "min_train_minutes": float(min_train_minutes),
            "k_min": float(k_min),
            "k_max": float(k_max),
            "proba_temperature": float(proba_temperature or 1.0),
        },
    )
