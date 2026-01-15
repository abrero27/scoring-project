# src/fw_scoring.py
# ============================================================
# FW SCORING 
# ------------------------------------------------------------
# Outputs (one row per FW player):
#   - fw_post_rating_0100          : OFFICIAL (within ST / NON_ST)
#   - fw_role_rating_0100          : within role (e.g., ST_FINISHER)
#   - fw_role_rating_shrunk_0100   : role rating shrunk toward post rating
#
# Core principles (same style as MF/DF):
#   - Weights learned per POST only (ST roles compared within ST; NON_ST within NON_ST)
#   - Feature importance = omega-squared (ω²) from 1-way ANOVA across roles
#   - Robust rating = percentile-rank 0-100 with winsorization
#   - Shrinkage column is EXTRA (does NOT replace role rating)
#
# IMPORTANT PATCH:
#   - FBref exports often contain duplicate column names (Gls, Ast, xG, npxG, etc.)
#   - We drop duplicated columns (keep first) to avoid incoherent scoring.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Role labels 
# -----------------------------
ST_ROLES = ["ST_FINISHER", "ST_LINK", "ST_RUNNER"]
NON_ST_ROLES = ["WIDE_WINGER", "INSIDE_FORWARD"]


# -----------------------------
# Feature groups (rich but structured)
# All feature names below exist in your header (as given).
# -----------------------------

ST_GROUPS: Dict[str, List[str]] = {
    # Finishing / shot quality
    "finishing": [
        "Gls", "xG", "npxG",
        "Sh", "SoT", "SoT%",
        "Sh/90", "SoT/90",
        "npxG/Sh", "G-xG", "np:G-xG",
        "G/Sh", "G/SoT",
        "PK", "PKatt",
    ],
    # Box presence / receiving threat
    "box_presence": [
        "Touches_AttPen", "Touches_Att3rd",
        "Rec", "PrgR",
        "Aerials_Won", "Aerials_Won%",
        "Offsides",  # often informative for pure 9s
    ],
    # Link + creation
    "link_creation": [
        "Touches", "Touches_Live",
        "Pass_Att", "Pass_Cmp", "Pass_Cmp%",
        "KP", "xA", "A-xAG", "xAG",
        "SCA", "SCA90", "GCA", "GCA90",
        "PrgP", "FinalThird_Pass", "PPA",
    ],
    # Runner profile (carry / depth)
    "carry_dribble": [
        "PrgC", "Carry_PrgDist", "Carry_FinalThird", "Carry_PenArea",
        "TakeOn_Att", "TakeOn_Succ", "TakeOn_Succ%",
        "TakeOn_Tackled", "TakeOn_Tackled%",
    ],
    # Light defensive / pressing (kept small)
    "defense_work": [
        "Tackles_Won", "Interceptions", "Recoveries",
        "Fls", "Fld",
    ],
}

NON_ST_GROUPS: Dict[str, List[str]] = {
    # Threat (shooting + xG contribution)
    "threat": [
        "Gls", "xG", "npxG",
        "Sh", "SoT", "Sh/90", "SoT/90",
        "xA", "KP", "A-xAG", "xAG",
        "npxG+xAG",
    ],
    # 1v1 + carries (wing progression)
    "carry_dribble": [
        "PrgC", "Carry_PrgDist", "Carry_FinalThird", "Carry_PenArea",
        "TakeOn_Att", "TakeOn_Succ", "TakeOn_Succ%",
        "TakeOn_Tackled", "TakeOn_Tackled%",
        "PrgR",
    ],
    # Wide creation (cross / chance creation)
    "wide_creation": [
        "Pass_Cross", "CrsPA", "Crs",
        "SCA", "SCA90", "GCA", "GCA90",
        "PPA", "FinalThird_Pass",
    ],
    # Involvement / link
    "involvement": [
        "Touches", "Touches_Live",
        "Touches_Att3rd", "Touches_AttPen",
        "Pass_Att", "Pass_Cmp", "Pass_Cmp%",
        "PrgP",
        "Miscontrols", "Dispossessed",
    ],
    # Light defense
    "defense_work": [
        "Tackles_Won", "Interceptions", "Recoveries",
        "Fls", "Fld",
    ],
}


# -----------------------------
# Group caps (prevents domination by "touches/passes")
# Defendable in a Master report.
# -----------------------------
ST_GROUP_CAPS = {
    "finishing": 0.35,
    "box_presence": 0.15,
    "link_creation": 0.25,
    "carry_dribble": 0.15,
    "defense_work": 0.10,
}
NON_ST_GROUP_CAPS = {
    "threat": 0.25,
    "carry_dribble": 0.30,
    "wide_creation": 0.20,
    "involvement": 0.15,
    "defense_work": 0.10,
}


@dataclass(frozen=True)
class PostCfg:
    name: str
    fw_post_value: str
    role_col: str
    valid_roles: List[str]
    groups: Dict[str, List[str]]
    group_caps: Dict[str, float]


POSTS = [
    PostCfg("ST", "ST", "fw_role", ST_ROLES, ST_GROUPS, ST_GROUP_CAPS),
    PostCfg("NON_ST", "NON_ST", "fw_role", NON_ST_ROLES, NON_ST_GROUPS, NON_ST_GROUP_CAPS),
]


# ============================================================
# Utilities
# ============================================================

def _dedupe(seq: List[str]) -> List[str]:
    out, seen = [], set()
    for x in seq:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _safe_fill_median(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            med = df[c].median(skipna=True)
            df[c] = df[c].fillna(med)


def _zscore(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    x = df[cols].astype(float)
    mu = x.mean()
    sd = x.std(ddof=0).replace(0, np.nan)
    return (x - mu) / sd


def _rating_0100_from_rank(s: pd.Series, clip: float = 0.02) -> pd.Series:
    """
    Percentile-rank rating (robust).
    Note: top can be 100. That's normal.
    """
    s = s.astype(float)
    if s.notna().sum() < 3:
        return pd.Series(np.nan, index=s.index)

    lo = s.quantile(clip)
    hi = s.quantile(1 - clip)
    s2 = s.clip(lower=lo, upper=hi)

    r = s2.rank(pct=True, method="average")
    return 100.0 * r


def _omega_squared(x: pd.Series, g: pd.Series) -> float:
    mask = x.notna() & g.notna()
    x = x[mask].astype(float)
    g = g[mask].astype(str)

    levels = g.unique().tolist()
    k = len(levels)
    n = int(len(x))
    if k < 2 or n < (k + 1):
        return np.nan

    grand_mean = float(x.mean())
    ss_between = 0.0
    ss_within = 0.0

    for lev in levels:
        vals = x[g == lev]
        ni = int(len(vals))
        if ni == 0:
            continue
        mi = float(vals.mean())
        ss_between += ni * (mi - grand_mean) ** 2
        ss_within += float(((vals - mi) ** 2).sum())

    ss_total = ss_between + ss_within
    df_between = k - 1
    df_within = n - k

    if df_within <= 0 or ss_total <= 0:
        return np.nan

    ms_within = ss_within / df_within
    w2 = (ss_between - df_between * ms_within) / (ss_total + ms_within)
    return float(max(0.0, w2))


def _shrink_rating(role_rating: pd.Series, post_rating: pd.Series, n_role: pd.Series, k: int = 10) -> pd.Series:
    n = n_role.astype(float).fillna(0.0)
    alpha = n / (n + float(k))
    return alpha * role_rating + (1 - alpha) * post_rating


# ============================================================
# Merge table FW
# ============================================================

def build_fw_merged(
    df_outfield: pd.DataFrame,
    df_fw: pd.DataFrame,
    df_fw_roles: pd.DataFrame,
) -> pd.DataFrame:
    base = df_fw[["Player", "Squad", "fw_post"]].merge(
        df_fw_roles[["Player", "Squad", "fw_role"]],
        on=["Player", "Squad"],
        how="left",
    )

    df_stats = df_outfield.copy()

    # 🔥 CRITICAL PATCH: drop duplicate columns (FBref often duplicates Gls/xG/etc.)
    df_stats = df_stats.loc[:, ~df_stats.columns.duplicated(keep="first")].copy()

    # normalize squad for merge safety
    if "Squad" in df_stats.columns:
        df_stats["Squad"] = df_stats["Squad"].astype(str).str.strip().str.lower()
    if "Squad" in base.columns:
        base["Squad"] = base["Squad"].astype(str).str.strip().str.lower()

    df_stats = df_stats.drop_duplicates(subset=["Player", "Squad"], keep="first")
    out = base.merge(df_stats, on=["Player", "Squad"], how="left", suffixes=("", "_stats"))

    if "Min_num" not in out.columns and "Min" in out.columns:
        out["Min_num"] = pd.to_numeric(out["Min"], errors="coerce")

    return out


# ============================================================
# Weights per POST (with group caps)
# ============================================================

def _available_features_for_post(df: pd.DataFrame, post: PostCfg) -> Dict[str, List[str]]:
    gmap: Dict[str, List[str]] = {}
    for g, feats in post.groups.items():
        feats2 = [f for f in _dedupe(feats) if f in df.columns]
        if feats2:
            gmap[g] = feats2
    return gmap


def compute_post_weights(
    df: pd.DataFrame,
    post: PostCfg,
    min_non_nan_ratio: float = 0.60,
) -> pd.DataFrame:
    sub = df[df["fw_post"] == post.fw_post_value].copy()

    sub = sub[sub[post.role_col].isin(post.valid_roles)].copy()
    group_feats = _available_features_for_post(sub, post)
    if not group_feats:
        raise ValueError(f"[{post.name}] No features available for any group.")

    rows = []
    for gname, feats in group_feats.items():
        _coerce_numeric(sub, feats)
        usable = [f for f in feats if sub[f].notna().mean() >= min_non_nan_ratio]
        if not usable:
            continue

        _safe_fill_median(sub, usable)
        for f in usable:
            w2 = _omega_squared(sub[f], sub[post.role_col])
            rows.append({"fw_post": post.name, "group": gname, "feature": f, "omega2": w2})

    wdf = pd.DataFrame(rows).dropna(subset=["omega2"]).copy()
    if wdf.empty:
        raise ValueError(f"[{post.name}] No usable features after NaN/ANOVA filtering.")

    wdf["omega2"] = wdf["omega2"].clip(lower=0.0)

    # normalize within group
    wdf["w_in_group"] = 0.0
    for g in wdf["group"].unique():
        mask = wdf["group"] == g
        denom = wdf.loc[mask, "omega2"].sum()
        if not np.isfinite(denom) or denom <= 0:
            wdf.loc[mask, "w_in_group"] = 1.0 / mask.sum()
        else:
            wdf.loc[mask, "w_in_group"] = wdf.loc[mask, "omega2"] / denom

    # apply caps + global renorm
    wdf["group_cap"] = wdf["group"].map(post.group_caps).fillna(0.0)
    wdf["weight"] = wdf["group_cap"] * wdf["w_in_group"]

    denom2 = wdf["weight"].sum()
    if not np.isfinite(denom2) or denom2 <= 0:
        wdf["weight"] = 1.0 / len(wdf)
    else:
        wdf["weight"] = wdf["weight"] / denom2

    return wdf.sort_values("weight", ascending=False).reset_index(drop=True)


# ============================================================
# Scoring core
# ============================================================

def score_weighted_z(df: pd.DataFrame, weights: pd.DataFrame, features: List[str]) -> pd.Series:
    feats = [f for f in features if f in df.columns]
    if not feats:
        return pd.Series(np.nan, index=df.index)

    _coerce_numeric(df, feats)
    _safe_fill_median(df, feats)

    z = _zscore(df, feats)
    wmap = weights.set_index("feature")["weight"].to_dict()
    w = np.array([wmap.get(f, 0.0) for f in z.columns], dtype=float)

    return pd.Series(z.values @ w, index=df.index)


# ============================================================
# Public API
# ============================================================

def score_forwards(
    df_outfield: pd.DataFrame,
    df_fw: pd.DataFrame,
    df_fw_roles: pd.DataFrame,
    min_non_nan_ratio: float = 0.60,
    min_train_minutes: int = 700,
    min_role_n_for_flag: int = 5,
    shrink_k: int = 10,
    export_filename: Optional[str] = None,
    export_weights_filename: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df_all = build_fw_merged(df_outfield, df_fw, df_fw_roles)

    # minutes flag (display/ranking only; we still score everyone)
    if "Min_num" in df_all.columns:
        df_all["minutes_flag"] = np.where(df_all["Min_num"] >= float(min_train_minutes), "OK", "LOW_MIN")
    else:
        df_all["minutes_flag"] = "NA"

    # weights per post (ST / NON_ST)
    weights_df = pd.concat(
        [compute_post_weights(df_all, post, min_non_nan_ratio=min_non_nan_ratio) for post in POSTS],
        ignore_index=True,
    )

    # init outputs
    df_scored = df_all.copy()
    df_scored["fw_post_score_raw"] = np.nan
    df_scored["fw_post_rating_0100"] = np.nan
    df_scored["fw_role_score_raw"] = np.nan
    df_scored["fw_role_rating_0100"] = np.nan
    df_scored["fw_role_rating_shrunk_0100"] = np.nan
    df_scored["n_post"] = np.nan
    df_scored["n_role"] = np.nan
    df_scored["role_sample_flag"] = pd.Series(index=df_scored.index, dtype="string")

    for post in POSTS:
        post_mask = (df_scored["fw_post"] == post.fw_post_value) & (df_scored[post.role_col].isin(post.valid_roles))
        post_df = df_scored.loc[post_mask].copy()

        w_post = weights_df[weights_df["fw_post"] == post.name].copy()
        feats = w_post["feature"].tolist()

        # POST rating (within ST or within NON_ST)
        post_raw = score_weighted_z(post_df, w_post, feats)
        df_scored.loc[post_df.index, "fw_post_score_raw"] = post_raw
        df_scored.loc[post_df.index, "fw_post_rating_0100"] = _rating_0100_from_rank(post_raw)
        df_scored.loc[post_df.index, "n_post"] = float(len(post_df))

        # ROLE rating (within each role)
        for r in post.valid_roles:
            ridx = post_df.index[post_df[post.role_col] == r].tolist()
            if not ridx:
                continue
            role_block = df_scored.loc[ridx].copy()
            role_raw = score_weighted_z(role_block, w_post, feats)

            df_scored.loc[ridx, "fw_role_score_raw"] = role_raw
            df_scored.loc[ridx, "fw_role_rating_0100"] = _rating_0100_from_rank(role_raw)
            df_scored.loc[ridx, "n_role"] = float(len(ridx))

        df_scored.loc[post_df.index, "role_sample_flag"] = np.where(
            df_scored.loc[post_df.index, "n_role"].astype(float) < float(min_role_n_for_flag),
            "LOW_SAMPLE",
            "OK",
        )

    # Shrunk role rating (extra column)
    df_scored["fw_role_rating_shrunk_0100"] = _shrink_rating(
        df_scored["fw_role_rating_0100"],
        df_scored["fw_post_rating_0100"],
        df_scored["n_role"],
        k=shrink_k,
    )

    # Export
    if export_filename:
        df_scored.to_csv(f"data/processed/{export_filename}", index=False)
    if export_weights_filename:
        weights_df.to_csv(f"data/processed/{export_weights_filename}", index=False)

    return weights_df, df_scored
