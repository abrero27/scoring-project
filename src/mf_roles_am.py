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
class AMRolePack:
    df_am_roles: pd.DataFrame
    feature_cols: list[str]
    chosen_k: int
    bic: dict[int, float]
    role_map: dict[int, str]
    thresholds: dict[str, float]

# -------------------------
# Utils
# -------------------------
def _parse_minutes(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce").fillna(0.0)


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _available_numeric(df: pd.DataFrame, candidates: list[str], min_ratio: float = 0.60) -> list[str]:
    keep = []
    n = len(df)
    if n == 0:
        return keep
    for c in candidates:
        if c in df.columns and (df[c].notna().sum() / n) >= float(min_ratio):
            keep.append(c)
    return keep


def _fit_best_gmm_by_bic(
    Xz: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    covariance_type: str = "diag",
) -> tuple[GaussianMixture, int, dict[int, float]]:
    bic = {}
    best_model = None
    best_k = None
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

    if best_model is None:
        raise ValueError(f"Not enough samples to fit GMM for k={k_min}..{k_max}. n={n}")

    return best_model, int(best_k), bic


def _soften_proba(proba: np.ndarray, temperature: float | None) -> np.ndarray:
    if temperature is None or temperature <= 1.0:
        return proba
    p = np.power(proba, 1.0 / float(temperature))
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


# -------------------------
# Cluster -> role mapping (2 roles)
# -------------------------
def _am_role_scores(centers: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Only 2 AM roles:
      - AM_ORGANIZER: plays lower, high pass volume/control, switches, progression from deeper zones
      - AM_CLASSIC_10: final-third creator, high KP/xA/SCA + penetrative passes + advanced touches
    centers are z-space.
    """
    def g(col: str) -> pd.Series:
        return centers[col] if col in centers.columns else 0.0

    # Organizer: deeper involvement + volume/control + progression
    score_org = (
        1.25 * g("Touches_Mid3rd")
        + 0.85 * g("Touches_Def3rd")
        + 1.35 * g("Pass_Att")
        + 0.95 * g("Pass_Switch")
        + 1.05 * g("PrgP")
        + 0.90 * g("PrgDist")
        + 0.55 * g("FinalThird_Pass")
        # organizer should not be a pure final-third killer
        - 0.90 * g("Touches_AttPen")
        - 0.45 * g("npxG")
        - 0.45 * g("Sh/90")
        - 0.55 * g("KP")
        - 0.55 * g("xA")
        - 0.45 * g("SCA90")
    )

    # Classic 10: decisive creation in last third
    score_c10 = (
        1.25 * g("KP")
        + 1.15 * g("xA")
        + 1.00 * g("SCA90")
        + 0.95 * g("PPA")
        + 0.95 * g("Touches_Att3rd")
        + 0.85 * g("Touches_AttPen")
        + 0.70 * g("FinalThird_Pass")
        # penalize being too "low / controller"
        - 0.95 * g("Touches_Def3rd")
        - 0.85 * g("Touches_Mid3rd")
        - 0.80 * g("Pass_Att")
        - 0.65 * g("Pass_Switch")
        - 0.55 * g("PrgDist")
    )

    return {"AM_ORGANIZER": score_org, "AM_CLASSIC_10": score_c10}


def _cluster_to_role_by_argmax(centers: pd.DataFrame) -> dict[int, str]:
    scores = _am_role_scores(centers)
    role_names = list(scores.keys())
    mat = np.vstack([scores[r].values for r in role_names]).T  # (k,2)
    best_idx = mat.argmax(axis=1)
    return {c: role_names[int(best_idx[c])] for c in range(mat.shape[0])}


# -------------------------
# Main API
# -------------------------
def assign_am_roles(
    df_players: pd.DataFrame,
    df_mf_posts: pd.DataFrame,
    min_train_minutes: int = 700,
    k_min: int = 2,
    k_max: int = 5,
    proba_temperature: float | None = 1.20,
    low_min_shrink: float = 0.15,   # <= IMPORTANT: makes low-min proba less extreme (no flag)
    random_state: int = 42,
    export_filename: str = "midfielders_am_roles.csv",
    summary_filename: str = "midfielders_am_roles_cluster_summary.csv",
) -> AMRolePack:
    df = df_players.copy()

    # merge mf_post
    join_cols = ["Player", "Squad"]
    tmp = df_mf_posts.copy()
    for c in join_cols:
        if c not in tmp.columns:
            raise ValueError(f"df_mf_posts must contain column '{c}'.")
        tmp[c] = tmp[c].astype(str)
        df[c] = df[c].astype(str)

    if "mf_post" not in tmp.columns:
        raise ValueError("df_mf_posts must contain column 'mf_post'.")

    df = df.merge(tmp[join_cols + ["mf_post"]], on=join_cols, how="left")
    df_am = df[df["mf_post"].astype(str).eq("AM")].copy().reset_index(drop=True)
    if df_am.empty:
        raise ValueError("No AM found: mf_post == 'AM'.")

    # minutes
    if "Min" not in df_am.columns:
        raise ValueError("Expected column 'Min'.")
    df_am["Min_num"] = _parse_minutes(df_am["Min"])
    df_am["low_min_flag"] = df_am["Min_num"] < float(min_train_minutes)

    # -------------------------
    # Features: fewer + more discriminant
    # (avoid redundant triplets like TotDist + PrgDist + Pass_Long_Att all together)
    # -------------------------
    candidates = [
        # creation (decisive)
        "KP", "xA", "SCA90", "PPA",
        # zones (where he plays)
        "Touches_Def3rd", "Touches_Mid3rd", "Touches_Att3rd", "Touches_AttPen",
        # control / distribution
        "Pass_Att", "Pass_Switch", "PrgP",
        # movement/progression (1 only)
        "PrgDist",
        # mild goal threat
        "npxG", "Sh/90",
        # link into final third
        "FinalThird_Pass",
    ]
    df_am = _coerce_numeric(df_am, candidates)
    usable = _available_numeric(df_am, candidates, min_ratio=0.60)

    preferred = [
        "KP", "xA", "SCA90", "PPA",
        "FinalThird_Pass",
        "Touches_Def3rd", "Touches_Mid3rd", "Touches_Att3rd", "Touches_AttPen",
        "Pass_Att", "Pass_Switch", "PrgP",
        "PrgDist",
        "npxG", "Sh/90",
    ]
    feature_cols = [c for c in preferred if c in usable]

    if len(feature_cols) < 8:
        raise ValueError(f"Too few usable AM features. Found: {feature_cols}")

    # -------------------------
    # TRAIN STRICTLY on Min >= min_train_minutes
    # -------------------------
    train = df_am[~df_am["low_min_flag"]].copy()
    if train.shape[0] < (k_min + 6):
        raise ValueError(f"Not enough AM with Min >= {min_train_minutes} to train. Have {train.shape[0]}.")

    train_medians = train[feature_cols].median(numeric_only=True)
    X_train = train[feature_cols].copy().fillna(train_medians)

    scaler = StandardScaler()
    Xz_train = scaler.fit_transform(X_train.values)

    gmm, chosen_k, bic = _fit_best_gmm_by_bic(Xz_train, k_min, k_max, random_state, "diag")
    centers = pd.DataFrame(gmm.means_, columns=feature_cols)

    # Map each cluster -> role (2 roles only)
    role_map = _cluster_to_role_by_argmax(centers)
    roles = ["AM_ORGANIZER", "AM_CLASSIC_10"]

    # -------------------------
    # PREDICT FOR ALL AM (including low-min)
    # -------------------------
    X_all = df_am[feature_cols].copy().fillna(train_medians)
    Xz_all = scaler.transform(X_all.values)

    proba_comp = _soften_proba(gmm.predict_proba(Xz_all), proba_temperature)

    # shrink low-min toward uniform (no flag, just less extreme proba)
    if low_min_shrink and low_min_shrink > 0:
        u = np.ones_like(proba_comp) / proba_comp.shape[1]
        mask = df_am["low_min_flag"].values
        proba_comp[mask] = (1.0 - float(low_min_shrink)) * proba_comp[mask] + float(low_min_shrink) * u[mask]

    # aggregate component probs -> role probs
    role_probs = {r: np.zeros(len(df_am), dtype=float) for r in roles}
    for j in range(proba_comp.shape[1]):
        r = role_map.get(j)
        if r in role_probs:
            role_probs[r] += proba_comp[:, j]

    mat = np.vstack([role_probs[r] for r in roles]).T
    mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-12)

    best_idx = mat.argmax(axis=1)
    best_role = [roles[i] for i in best_idx]
    best_p = mat.max(axis=1)

    df_out = df_am.copy()
    df_out["mf_role_cluster"] = proba_comp.argmax(axis=1)
    df_out["mf_role"] = best_role
    df_out["mf_role_proba"] = best_p
    df_out["p_mf_role_AM_ORGANIZER"] = mat[:, 0]
    df_out["p_mf_role_AM_CLASSIC_10"] = mat[:, 1]

    # export
    base_cols = [
        "Player", "Squad", "Pos", "Min", "Min_num", "low_min_flag",
        "mf_post", "mf_role", "mf_role_proba", "mf_role_cluster",
        "p_mf_role_AM_ORGANIZER", "p_mf_role_AM_CLASSIC_10",
    ]
    out_cols = [c for c in base_cols if c in df_out.columns]
    out_path = PROCESSED_DIR / export_filename
    df_out[out_cols].sort_values(["Squad", "Player"]).to_csv(out_path, index=False)

    # cluster summary (TRAIN centers only)
    centers_out = centers.copy()
    centers_out["cluster"] = range(len(centers_out))
    centers_out["mapped_role"] = centers_out["cluster"].map(role_map).astype(str)
    centers_out["avg_abs_z"] = centers_out[feature_cols].abs().mean(axis=1)
    cols_sum = ["cluster", "mapped_role", "avg_abs_z"] + feature_cols
    (PROCESSED_DIR / summary_filename).write_text(centers_out[cols_sum].to_csv(index=False), encoding="utf-8")

    return AMRolePack(
        df_am_roles=df_out,
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
        },
    )
