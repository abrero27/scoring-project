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
class DMRolePack:
    df_dm_roles: pd.DataFrame
    feature_cols: list[str]
    chosen_k: int
    bic: dict[int, float]
    role_map: dict[int, str]   # cluster -> template role
    thresholds: dict[str, float]


# -------------------------
# Utils (same spirit as FW)
# -------------------------
def _parse_minutes_series(s: pd.Series) -> pd.Series:
    # Robust minutes parsing (handles commas/dots if they ever appear)
    x = s.astype(str).str.strip()
    x = x.str.replace(",", "", regex=False)
    # If your Min is always integer, removing dots is OK; if not, remove the line below.
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
            max_iter=2500,
            reg_covar=2e-3,
        )
        gmm.fit(Xz)
        b = gmm.bic(Xz)
        bic[k] = float(b)

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


# ------------------------------
# DM role templates (z-space)
# ------------------------------
def _dm_role_scores(centers: pd.DataFrame) -> dict[str, pd.Series]:
    """
    3 DM roles (simple & interpretable):
      - DM_CONTROLLER: volume pass/build-up + deep touches, less "destroyer" & less AM actions
      - DM_DESTROYER: interceptions/tackles/blocks/clearances/recoveries
      - DM_DEEP_CM: progression/carries/final third passes but not much KP/xA/SCA and not much pure defending
    centers are in z-space (StandardScaler).
    """
    def g(col: str) -> pd.Series:
        return centers[col] if col in centers.columns else 0.0

    # --- Controller ---
    build = (
        1.10 * g("Pass_Att")
        + 0.80 * g("Pass_Cmp")
        + 0.85 * g("PrgP")
        + 0.70 * g("Pass_Switch")
        + 0.55 * g("Pass_Long_Att")
        + 0.70 * g("Touches_Def3rd")
        + 0.55 * g("Touches_Mid3rd")
        + 0.35 * g("Touches")
    )
    defend_penalty = (
        0.65 * g("Tackles_Interceptions")
        + 0.55 * g("Interceptions")
        + 0.40 * g("Blocks_Total")
        + 0.35 * g("Clearances")
        + 0.35 * g("Recoveries")
    )
    attack_penalty = (
        0.80 * g("KP")
        + 0.75 * g("xA")
        + 0.70 * g("SCA90")
        + 0.55 * g("Touches_AttPen")
        + 0.45 * g("Sh/90")
        + 0.40 * g("xG")
    )
    score_controller = build - (0.75 * defend_penalty + 0.85 * attack_penalty)

    # --- Destroyer ---
    destroy = (
        1.10 * g("Interceptions")
        + 1.00 * g("Tackles_Interceptions")
        + 0.85 * g("Tackles_Won")
        + 0.70 * g("Blocks_Total")
        + 0.65 * g("Clearances")
        + 0.75 * g("Recoveries")
        + 0.55 * g("Touches_Def3rd")
        + 0.35 * g("Touches_DefPen")
    )
    build_penalty = (
        0.55 * g("Pass_Att")
        + 0.45 * g("PrgP")
        + 0.35 * g("Pass_Switch")
        + 0.25 * g("Pass_Long_Att")
    )
    attack_penalty2 = (
        0.70 * g("KP")
        + 0.65 * g("xA")
        + 0.60 * g("SCA90")
        + 0.45 * g("Touches_AttPen")
    )
    score_destroyer = destroy - (0.65 * build_penalty + 0.85 * attack_penalty2)

    # --- Deep CM (CM bas "caché") ---
    prog = (
        1.05 * g("PrgP")
        + 0.95 * g("PrgC")
        + 0.80 * g("PrgR")
        + 0.90 * g("Carry_PrgDist")
        + 0.75 * g("Carry_FinalThird")
        + 0.70 * g("FinalThird_Pass")
        + 0.60 * g("Touches_Mid3rd")
        + 0.35 * g("Touches_Att3rd")
        + 0.35 * g("Pass_Att")
    )
    defend_penalty3 = (
        0.85 * g("Interceptions")
        + 0.85 * g("Tackles_Interceptions")
        + 0.55 * g("Blocks_Total")
        + 0.55 * g("Clearances")
        + 0.45 * g("Touches_DefPen")
    )
    creator_penalty = (
        0.95 * g("KP")
        + 0.90 * g("xA")
        + 0.85 * g("SCA90")
        + 0.55 * g("Touches_AttPen")
        + 0.45 * g("Sh/90")
        + 0.45 * g("xG")
    )
    score_deep_cm = prog - (0.70 * defend_penalty3 + 0.80 * creator_penalty)

    return {
        "DM_CONTROLLER": score_controller,
        "DM_DESTROYER": score_destroyer,
        "DM_DEEP_CM": score_deep_cm,
    }


def _cluster_to_role_by_argmax(centers: pd.DataFrame) -> dict[int, str]:
    scores = _dm_role_scores(centers)  # role -> series (k,)
    role_names = list(scores.keys())
    mat = np.vstack([scores[r].values for r in role_names]).T  # (k, n_roles)
    best_role_idx = mat.argmax(axis=1)
    return {c: role_names[int(best_role_idx[c])] for c in range(mat.shape[0])}


# ------------------------------
# Main API
# ------------------------------
def assign_dm_roles(
    df_players: pd.DataFrame,
    df_mf_posts: pd.DataFrame,             # needs Player, Squad, mf_post
    min_train_minutes: int = 700,
    k_min: int = 2,
    k_max: int = 5,
    proba_temperature: float | None = 1.20,
    random_state: int = 42,
    export_filename: str = "midfielders_dm_roles.csv",
    summary_filename: str = "midfielders_dm_roles_cluster_summary.csv",
) -> DMRolePack:
    """
    DM roles ONLY, trained ONLY on DM group.
    Consistent with your FW roles pipeline:
      - filter PURE MF
      - merge mf_post
      - train on Min >= min_train_minutes
      - choose K by BIC
      - map clusters -> role templates via argmax
      - aggregate component probabilities by role
      - export short table + cluster summary
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

    # DM only
    df_dm = df_mf[df_mf["mf_post"].astype(str).eq("DM")].copy().reset_index(drop=True)
    if df_dm.empty:
        raise ValueError("No DM found: mf_post == 'DM'.")

    # minutes
    if "Min" not in df_dm.columns:
        raise ValueError("Expected column 'Min'.")
    df_dm["Min_num"] = _parse_minutes_series(df_dm["Min"])
    df_dm["low_min_flag"] = df_dm["Min_num"] < float(min_train_minutes)

    # ---- feature set for DM roles ----
    candidates = [
        # build/controller
        "Pass_Att", "Pass_Cmp", "PrgP", "Pass_Switch", "Pass_Long_Att",
        "Touches", "Touches_DefPen", "Touches_Def3rd", "Touches_Mid3rd", "Touches_Att3rd",
        # destroyer
        "Interceptions", "Tackles_Interceptions", "Tackles_Won", "Blocks_Total",
        "Clearances", "Recoveries",
        # deep CM progression
        "PrgC", "PrgR", "Carry_PrgDist", "Carry_FinalThird", "FinalThird_Pass",
        # creator penalties (keep them so the model can separate)
        "xA", "KP", "SCA90", "Touches_AttPen", "Sh/90", "xG", "npxG",
    ]
    df_dm = _coerce_numeric(df_dm, candidates)
    usable = _available_numeric(df_dm, candidates, min_non_nan_ratio=0.60)

    preferred = [
        "Pass_Att", "Pass_Cmp", "PrgP", "Pass_Switch", "Pass_Long_Att",
        "Touches_Def3rd", "Touches_Mid3rd", "Touches", "Touches_DefPen", "Touches_Att3rd",
        "Interceptions", "Tackles_Interceptions", "Tackles_Won", "Blocks_Total",
        "Clearances", "Recoveries",
        "PrgC", "PrgR", "Carry_PrgDist", "Carry_FinalThird", "FinalThird_Pass",
        "xA", "KP", "SCA90", "Touches_AttPen", "Sh/90", "xG", "npxG",
    ]
    feature_cols = [c for c in preferred if c in usable]
    if len(feature_cols) < 10:
        raise ValueError(f"Too few usable DM role features. Found: {feature_cols}")

    # train subset
    train = df_dm[~df_dm["low_min_flag"]].copy()
    if train.shape[0] < (k_min + 4):
        train = df_dm.copy()

    X_train = train[feature_cols].copy().apply(lambda s: s.fillna(s.median()), axis=0)
    scaler = StandardScaler()
    Xz_train = scaler.fit_transform(X_train.values)

    gmm, chosen_k, bic = _fit_best_gmm_by_bic(Xz_train, k_min, k_max, random_state, "diag")
    centers = pd.DataFrame(gmm.means_, columns=feature_cols)

    role_map = _cluster_to_role_by_argmax(centers)
    templates = list(_dm_role_scores(centers).keys())  # ["DM_CONTROLLER","DM_DESTROYER","DM_DEEP_CM"]

    # predict all DM
    X_all = df_dm[feature_cols].copy().apply(lambda s: s.fillna(s.median()), axis=0)
    Xz_all = scaler.transform(X_all.values)

    proba_comp = gmm.predict_proba(Xz_all)
    proba_comp = _soften_proba(proba_comp, proba_temperature)

    # aggregate component probs -> role probs
    role_probs = {r: np.zeros(len(df_dm), dtype=float) for r in templates}
    for j in range(proba_comp.shape[1]):
        r = role_map.get(j)
        if r in role_probs:
            role_probs[r] += proba_comp[:, j]

    mat = np.vstack([role_probs[r] for r in templates]).T
    mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-12)

    best_idx = mat.argmax(axis=1)
    best_role = [templates[i] for i in best_idx]
    best_p = mat.max(axis=1)

    df_out = df_dm.copy()
    df_out["mf_role"] = best_role
    df_out["mf_role_proba"] = best_p

    for i, r in enumerate(templates):
        df_out[f"p_mf_role_{r}"] = mat[:, i]

    # export short
    base_cols = ["Player", "Squad", "Pos", "Min", "Min_num", "low_min_flag", "mf_post", "mf_role", "mf_role_proba"]
    prob_cols = sorted([c for c in df_out.columns if c.startswith("p_mf_role_")])
    out_cols = [c for c in base_cols if c in df_out.columns] + prob_cols

    out_path = PROCESSED_DIR / export_filename
    df_out[out_cols].sort_values(["Squad", "Player"]).to_csv(out_path, index=False)

    # cluster summary (centers in z-space)
    def centers_summary() -> pd.DataFrame:
        tmp = centers.copy()
        tmp["cluster"] = range(len(tmp))
        tmp["mapped_role"] = tmp["cluster"].map(role_map).astype(str)
        tmp["avg_abs_z"] = tmp[feature_cols].abs().mean(axis=1)
        cols = ["cluster", "mapped_role", "avg_abs_z"] + feature_cols
        return tmp[cols].sort_values(["cluster"])

    summary = centers_summary()
    (PROCESSED_DIR / summary_filename).write_text(summary.to_csv(index=False), encoding="utf-8")

    return DMRolePack(
        df_dm_roles=df_out,
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
