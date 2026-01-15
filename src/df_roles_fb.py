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
class FBRolePack:
    df_fb_roles: pd.DataFrame
    feature_cols: list[str]
    chosen_k: int
    bic: dict[int, float]
    role_map: dict[int, str]   # cluster -> role
    thresholds: dict[str, float]


# -------------------------
# Utils (same spirit as your other modules)
# -------------------------
def _parse_minutes_series(s: pd.Series) -> pd.Series:
    """
    Robust minutes parsing:
    - handles "2,598" or "2.598" as 2598 (thousands separator)
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
    """
    IMPORTANT:
    We need 3 roles => at least 3 clusters.
    So even if user passes k_min=2, we enforce k_min>=3 internally.
    """
    k_min_eff = max(int(k_min), 3)
    k_max_eff = int(k_max)

    bic: dict[int, float] = {}
    best_model: GaussianMixture | None = None
    best_k: int | None = None
    best_bic = np.inf

    n = Xz.shape[0]
    for k in range(k_min_eff, k_max_eff + 1):
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
        raise ValueError(f"Not enough samples to fit GMM for k={k_min_eff}..{k_max_eff}. n={n}")

    return best_model, int(best_k), bic


def _soften_proba(proba: np.ndarray, temperature: float | None) -> np.ndarray:
    """temperature > 1 => less extreme probs, rows still sum to 1."""
    if temperature is None or temperature <= 1.0:
        return proba
    p = np.power(proba, 1.0 / float(temperature))
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


# -------------------------
# Role templates (StatsBomb/Opta logic)
# -------------------------
def _fb_role_scores(centers: pd.DataFrame) -> dict[str, np.ndarray]:
    def g(col: str) -> np.ndarray:
        return centers[col].values if col in centers.columns else np.zeros(len(centers), dtype=float)

    wide = (
        1.35 * g("Crs")
        + 1.10 * g("CrsPA")
        + 1.05 * g("Touches_Att3rd")
        + 0.80 * g("Carry_FinalThird")
        + 0.70 * g("TakeOn_Att")
        + 0.55 * g("SCA90")
        - 0.55 * g("Touches_Mid3rd")
        - 0.35 * g("Pass_Switch")
    )

    inverted = (
        1.35 * g("Touches_Mid3rd")
        + 1.15 * g("PrgP")
        + 1.15 * g("Pass_Switch")
        + 0.95 * g("FinalThird_Pass")
        + 0.85 * g("PPA")
        - 1.15 * g("Crs")
        - 0.85 * g("CrsPA")
        - 0.55 * g("Carry_FinalThird")
        - 0.45 * g("TakeOn_Att")
    )

    # STAY_HOME: must be VERY CB-like AND VERY LOW attacking
    stay = (
        1.55 * g("Touches_DefPen")
        + 1.20 * g("Clearances")
        + 1.10 * g("Blocks_Total")
        + 1.05 * g("Interceptions")
        - 1.10 * g("Touches_Att3rd")
        - 1.05 * g("Crs")
        - 0.95 * g("Carry_FinalThird")
        - 0.85 * g("SCA90")
        - 0.70 * g("PrgP")
        - 0.60 * g("Pass_Switch")
    )

    return {"FB_WIDE": wide, "FB_INVERTED": inverted, "FB_STAY_HOME": stay}


def _cluster_to_role_unique(centers: pd.DataFrame) -> dict[int, str]:
    """
    Academic & clean: force a UNIQUE mapping:
      1) pick the most CB-like cluster => STAY_HOME
      2) among remaining, pick most inverted => INVERTED
      3) remaining => WIDE
    This prevents label swapping.
    """
    scores = _fb_role_scores(centers)
    stay = scores["FB_STAY_HOME"]
    inv = scores["FB_INVERTED"]
    wide = scores["FB_WIDE"]

    k = centers.shape[0]
    clusters = list(range(k))

    stay_c = int(np.argmax(stay))
    remaining = [c for c in clusters if c != stay_c]

    inv_c = int(remaining[int(np.argmax(inv[remaining]))])
    remaining2 = [c for c in remaining if c != inv_c]

    # last one(s): if k>3, assign the best wide among remaining2, rest also wide
    # (but you still aggregate to 3 roles)
    role_map: dict[int, str] = {stay_c: "FB_STAY_HOME", inv_c: "FB_INVERTED"}
    if len(remaining2) == 1:
        role_map[int(remaining2[0])] = "FB_WIDE"
    else:
        best_w = int(remaining2[int(np.argmax(wide[remaining2]))])
        role_map[best_w] = "FB_WIDE"
        for c in remaining2:
            if c != best_w:
                # extra clusters also mapped to the closest role (usually wide or inverted)
                # choose by max of the 3 scores
                r = max(
                    [("FB_WIDE", wide[c]), ("FB_INVERTED", inv[c]), ("FB_STAY_HOME", stay[c])],
                    key=lambda x: x[1],
                )[0]
                role_map[int(c)] = r

    return role_map


# -------------------------
# Main API
# -------------------------
def assign_fb_roles(
    df_players: pd.DataFrame,
    df_df_posts: pd.DataFrame,             # needs Player, Squad, df_post
    min_train_minutes: int = 700,
    k_min: int = 2,                        # user can pass 2, but we enforce >=3 internally
    k_max: int = 6,
    proba_temperature: float | None = 1.20,
    random_state: int = 42,
    export_filename: str = "defenders_fb_roles.csv",
    summary_filename: str = "defenders_fb_roles_cluster_summary.csv",
) -> FBRolePack:
    df = df_players.copy()

    # attach df_post
    join_cols = ["Player", "Squad"]
    tmp = df_df_posts.copy()
    for c in join_cols:
        if c not in tmp.columns:
            raise ValueError(f"df_df_posts must contain column '{c}'.")
        tmp[c] = tmp[c].astype(str)
        df[c] = df[c].astype(str)

    if "df_post" not in tmp.columns:
        raise ValueError("df_df_posts must contain column 'df_post'.")

    df = df.merge(tmp[join_cols + ["df_post"]], on=join_cols, how="left")
    df_fb = df[df["df_post"].astype(str).eq("FB_WB")].copy().reset_index(drop=True)
    if df_fb.empty:
        raise ValueError("No FB/WB found: df_post == 'FB_WB'.")

    # minutes
    if "Min" not in df_fb.columns:
        raise ValueError("Expected column 'Min'.")
    df_fb["Min_num"] = _parse_minutes_series(df_fb["Min"])
    df_fb["low_min_flag"] = df_fb["Min_num"] < float(min_train_minutes)

    # -------------------------
    # Compact + discriminant features (12 total)
    # No duplicate "Crs" vs "Pass_Cross" => we keep ONLY Crs (+ CrsPA)
    # -------------------------
    candidates = [
     # WIDE
        "Crs", "CrsPA", "Touches_Att3rd", "Carry_FinalThird", "TakeOn_Att", "SCA90",
        # INVERTED
        "Touches_Mid3rd", "PrgP", "Pass_Switch", "FinalThird_Pass", "PPA",
        # STAY HOME (CB-like)
        "Touches_DefPen", "Clearances", "Blocks_Total", "Interceptions",
    ]
    preferred = [
        "Crs", "CrsPA", "Touches_Att3rd", "Carry_FinalThird", "TakeOn_Att", "SCA90",
        "Touches_Mid3rd", "PrgP", "Pass_Switch", "FinalThird_Pass", "PPA",
        "Touches_DefPen", "Clearances", "Blocks_Total", "Interceptions",
    ]

    df_fb = _coerce_numeric(df_fb, candidates)
    usable = _available_numeric(df_fb, candidates, min_ratio=0.60)
    feature_cols = [c for c in preferred if c in usable]

    # hard cap to keep it compact (still discriminant)
    # (order matters: keep top items)
    MAX_FEATS = 12
    if len(feature_cols) > MAX_FEATS:
        feature_cols = feature_cols[:MAX_FEATS]

    if len(feature_cols) < 8:
        raise ValueError(f"Too few usable FB features. Found: {feature_cols}")

    # -------------------------
    # TRAIN STRICTLY on Min >= min_train_minutes
    # -------------------------
    train = df_fb[~df_fb["low_min_flag"]].copy()
    if train.shape[0] < 12:
        raise ValueError(f"Not enough FB with Min >= {min_train_minutes} to train. Have {train.shape[0]}.")

    med = train[feature_cols].median(numeric_only=True)
    X_train = train[feature_cols].copy().fillna(med)

    scaler = StandardScaler()
    Xz_train = scaler.fit_transform(X_train.values)

    gmm, chosen_k, bic = _fit_best_gmm_by_bic(Xz_train, k_min, k_max, random_state, "diag")
    centers = pd.DataFrame(gmm.means_, columns=feature_cols)

    # UNIQUE mapping cluster -> role (academic)
    role_map = _cluster_to_role_unique(centers)
    roles = ["FB_WIDE", "FB_INVERTED", "FB_STAY_HOME"]

    # -------------------------
    # Predict ALL FB/WB
    # -------------------------
    X_all = df_fb[feature_cols].copy().fillna(med)
    Xz_all = scaler.transform(X_all.values)
    proba_comp = _soften_proba(gmm.predict_proba(Xz_all), proba_temperature)

    # aggregate component probs -> role probs (FW/MF style)
    role_probs = {r: np.zeros(len(df_fb), dtype=float) for r in roles}
    for j in range(proba_comp.shape[1]):
        r = role_map.get(j)
        if r in role_probs:
            role_probs[r] += proba_comp[:, j]

    mat = np.vstack([role_probs[r] for r in roles]).T
    mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-12)

    best_idx = mat.argmax(axis=1)
    best_role = [roles[i] for i in best_idx]
    best_p = mat.max(axis=1)

    df_out = df_fb.copy()
    df_out["fb_role_cluster"] = proba_comp.argmax(axis=1)
    df_out["fb_role"] = best_role
    df_out["fb_role_proba"] = best_p
    for i, r in enumerate(roles):
        df_out[f"p_fb_role_{r}"] = mat[:, i]

    # export
    out_cols = [
        "Player","Squad","Pos","Min","Min_num","low_min_flag","df_post",
        "fb_role","fb_role_proba","fb_role_cluster",
        "p_fb_role_FB_WIDE","p_fb_role_FB_INVERTED","p_fb_role_FB_STAY_HOME",
    ]
    out_cols = [c for c in out_cols if c in df_out.columns]
    df_out[out_cols].sort_values(["Squad","Player"]).to_csv(PROCESSED_DIR / export_filename, index=False)

    # cluster summary (centers in z-space)
    centers_out = centers.copy()
    centers_out["cluster"] = range(len(centers_out))
    centers_out["mapped_role"] = centers_out["cluster"].map(role_map).astype(str)
    centers_out["avg_abs_z"] = centers_out[feature_cols].abs().mean(axis=1)
    cols_sum = ["cluster","mapped_role","avg_abs_z"] + feature_cols
    centers_out[cols_sum].to_csv(PROCESSED_DIR / summary_filename, index=False)

    return FBRolePack(
        df_fb_roles=df_out,
        feature_cols=feature_cols,
        chosen_k=chosen_k,
        bic=bic,
        role_map=role_map,
        thresholds={
            "min_train_minutes": float(min_train_minutes),
            "k_min_user": float(k_min),
            "k_min_effective": float(max(int(k_min), 3)),
            "k_max": float(k_max),
            "proba_temperature": float(proba_temperature or 1.0),
        },
    )
