# src/df_scoring.py
# ============================================================
# DF SCORING 
# ------------------------------------------------------------
# Outputs (one row per DF player):
#   - df_post_rating_0100          : OFFICIAL (within CB vs within FB_WB)
#   - df_role_rating_0100          : within role (e.g., CB_BUILDER)
#   - df_role_rating_shrunk_0100   : role rating shrunk toward post rating
#
# Weights:
#   - Within each POST (CB / FB_WB), run ANOVA across ROLES for each feature
#   - Use omega-squared (ω²) effect size as "importance"
#   - Normalize ω² into weights
#
# Robust rating:
#   - Use percentile-rank 0-100 (winsorized) instead of min-max
#
# Assumptions about your pipeline:
#   - df_df_posts (from assign_df_positions): Player, Squad, df_post
#   - df_cb_roles: Player, Squad, cb_role (CB roles)
#   - df_fb_roles: Player, Squad, fb_role (FB/WB roles)
#   - df_outfield: Player, Squad + all numeric stats + Min (minutes)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Role labels (your setup)
# -----------------------------
CB_ROLES = ["CB_DUEL", "CB_HYBRID", "CB_BUILDER"]
FB_ROLES = ["FB_WIDE", "FB_INVERTED", "FB_STAY_HOME"]


# -----------------------------
# Feature space (rich but sane)
# IMPORTANT:
# - We compute weights per POST (CB vs FB_WB), not across posts.
# - Non-discriminant stats get ~0 ω² automatically.
# -----------------------------
COMMON_DF_FEATURES = [
    # Minutes context / involvement
    "Touches", "Touches_Live",
    "Touches_DefPen", "Touches_Def3rd", "Touches_Mid3rd", "Touches_Att3rd", "Touches_AttPen",

    # Defensive actions
    "Interceptions", "Tackles_Interceptions", "Tackles_Won",
    "Blocks_Total", "Clearances", "Recoveries",
    "Dribblers_Tackled", "Dribblers_Challenged", "Dribblers_Tackled_%",

    # Aerial / duels
    "Aerials_Won", "Aerials_Lost", "Aerials_Won%",

    # Passing volume / quality
    "Pass_Att", "Pass_Cmp", "Pass_Cmp%",
    "Pass_Short_Att", "Pass_Short_Cmp", "Pass_Short_Cmp%",
    "Pass_Med_Att", "Pass_Med_Cmp", "Pass_Med_Cmp%",
    "Pass_Long_Att", "Pass_Long_Cmp", "Pass_Long_Cmp%",
    "Pass_Switch", "Pass_TB",

    # Progression (pass/carry)
    "PrgP", "PrgDist",
    "PrgC", "Carry_PrgDist", "Carry_FinalThird",

    # Final third connection / chance creation (helps FB and ball-playing CB)
    "FinalThird_Pass", "PPA", "KP", "xA",
    "SCA90", "GCA90",

    # Wide actions (FB) — ok to include in common, ω² will filter for CB
    "Crs", "Pass_Cross", "CrsPA",
    "TakeOn_Att", "TakeOn_Succ", "TakeOn_Succ%",

    # Discipline / misc
    "Fls", "Fld", "CrdY", "CrdR",
]

# Post-specific extras
CB_EXTRA = [
    "Errors_LeadingShot", "Shots_Blocked", "Passes_Blocked",
]
FB_EXTRA = [
    "Touches_Att3rd", "Touches_AttPen",
    "Carry_PenArea",
]


@dataclass(frozen=True)
class PostCfg:
    name: str              # "CB" or "FB_WB"
    df_post_value: str     # "CB" / "FB_WB"
    role_col: str          # "cb_role" or "fb_role"
    valid_roles: List[str]
    extra_features: List[str]


POSTS = [
    PostCfg("CB", "CB", "cb_role", CB_ROLES, CB_EXTRA),
    PostCfg("FB_WB", "FB_WB", "fb_role", FB_ROLES, FB_EXTRA),
]


# ============================================================
# Utilities (same philosophy as MF)
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
    Stable 0-100 rating based on percentile ranks (robust to outliers & small samples).
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
    """
    One-way ANOVA effect size omega^2:
      ω² = (SSB - (k-1)*MSW) / (SST + MSW)
    Clamped at 0.
    """
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
    """
    role_shrunk = alpha*role + (1-alpha)*post, alpha=n/(n+k)
    """
    n = n_role.astype(float).fillna(0.0)
    alpha = n / (n + float(k))
    return alpha * role_rating + (1 - alpha) * post_rating


# ============================================================
# Merge table DF (posts + roles + stats)
# ============================================================

def build_df_merged(
    df_outfield: pd.DataFrame,
    df_df_posts: pd.DataFrame,
    df_cb_roles: pd.DataFrame,
    df_fb_roles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one DF table with:
      Player, Squad, df_post, cb_role, fb_role,
      Min_num + all stats from df_outfield.
    """
    base = (
        df_df_posts[["Player", "Squad", "df_post"]]
        .merge(df_cb_roles[["Player", "Squad", "cb_role"]], on=["Player", "Squad"], how="left")
        .merge(df_fb_roles[["Player", "Squad", "fb_role"]], on=["Player", "Squad"], how="left")
    )

    df_stats = df_outfield.copy()
    if "Squad" in df_stats.columns:
        df_stats["Squad"] = df_stats["Squad"].astype(str).str.strip().str.lower()
    if "Squad" in base.columns:
        base["Squad"] = base["Squad"].astype(str).str.strip().str.lower()

    df_stats = df_stats.drop_duplicates(subset=["Player", "Squad"], keep="first")
    out = base.merge(df_stats, on=["Player", "Squad"], how="left", suffixes=("", "_stats"))

    # minutes numeric
    if "Min_num" not in out.columns and "Min" in out.columns:
        out["Min_num"] = pd.to_numeric(out["Min"], errors="coerce")

    return out


# ============================================================
# Weights per POST (CB, FB_WB) based on ANOVA across roles
# ============================================================

def compute_post_weights(
    df: pd.DataFrame,
    post: PostCfg,
    min_non_nan_ratio: float = 0.60,
) -> pd.DataFrame:
    sub = df[df["df_post"] == post.df_post_value].copy()

    if post.role_col not in sub.columns:
        raise ValueError(f"[{post.name}] Missing role column: {post.role_col}")

    sub = sub[sub[post.role_col].isin(post.valid_roles)].copy()

    feats = _dedupe(COMMON_DF_FEATURES + post.extra_features)
    feats = [f for f in feats if f in sub.columns]

    if not feats:
        raise ValueError(f"[{post.name}] No features available in df.")

    _coerce_numeric(sub, feats)

    usable = [f for f in feats if sub[f].notna().mean() >= min_non_nan_ratio]
    if not usable:
        raise ValueError(f"[{post.name}] No usable features after NaN filter.")

    _safe_fill_median(sub, usable)

    rows = []
    for f in usable:
        w2 = _omega_squared(sub[f], sub[post.role_col])
        rows.append({"df_post": post.name, "feature": f, "omega2": w2})

    wdf = pd.DataFrame(rows).dropna(subset=["omega2"]).copy()
    wdf["omega2"] = wdf["omega2"].clip(lower=0.0)

    denom = wdf["omega2"].sum()
    if not np.isfinite(denom) or denom <= 0:
        wdf["weight"] = 1.0 / len(wdf)
    else:
        wdf["weight"] = wdf["omega2"] / denom

    return wdf.sort_values("weight", ascending=False).reset_index(drop=True)


# ============================================================
# Scoring core
# ============================================================

def score_weighted_z(
    df: pd.DataFrame,
    weights: pd.DataFrame,
    features: List[str],
) -> pd.Series:
    """
    Weighted sum of z-scored features.
    """
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

def score_defenders(
    df_outfield: pd.DataFrame,
    df_df_posts: pd.DataFrame,
    df_cb_roles: pd.DataFrame,
    df_fb_roles: pd.DataFrame,
    min_non_nan_ratio: float = 0.60,
    min_train_minutes: int = 700,
    min_role_n_for_flag: int = 5,
    shrink_k: int = 10,
    export_filename: Optional[str] = None,
    export_weights_filename: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      weights_df   : weights per post (CB/FB_WB)
      df_scored    : DF table with post + role ratings (plus shrunk role)
    """

    df_all = build_df_merged(df_outfield, df_df_posts, df_cb_roles, df_fb_roles)

    # minutes flag (for display + ranking; we still score everyone)
    if "Min_num" in df_all.columns:
        df_all["minutes_flag"] = np.where(df_all["Min_num"] >= float(min_train_minutes), "OK", "LOW_MIN")
    else:
        df_all["minutes_flag"] = "NA"

    # compute weights per post
    weights_list = []
    for post in POSTS:
        w = compute_post_weights(df_all, post, min_non_nan_ratio=min_non_nan_ratio)
        weights_list.append(w)
    weights_df = pd.concat(weights_list, ignore_index=True)

    # init outputs
    df_scored = df_all.copy()

    df_scored["df_post_score_raw"] = np.nan
    df_scored["df_post_rating_0100"] = np.nan

    df_scored["df_role_score_raw"] = np.nan
    df_scored["df_role_rating_0100"] = np.nan
    df_scored["df_role_rating_shrunk_0100"] = np.nan

    df_scored["n_post"] = np.nan
    df_scored["n_role"] = np.nan
    df_scored["role_sample_flag"] = pd.Series(index=df_scored.index, dtype="string")

    # score per post (CB / FB_WB)
    for post in POSTS:
        post_mask = (df_scored["df_post"] == post.df_post_value) & (df_scored[post.role_col].isin(post.valid_roles))
        post_df = df_scored.loc[post_mask].copy()

        w_post = weights_df[weights_df["df_post"] == post.name].copy()
        feats = w_post["feature"].tolist()

        # POST scoring (within post)
        post_raw = score_weighted_z(post_df, w_post, feats)
        df_scored.loc[post_df.index, "df_post_score_raw"] = post_raw
        df_scored.loc[post_df.index, "df_post_rating_0100"] = _rating_0100_from_rank(post_raw)

        df_scored.loc[post_df.index, "n_post"] = float(len(post_df))

        # ROLE scoring (within each role, same weights)
        role_col = post.role_col
        for r in post.valid_roles:
            ridx = post_df.index[post_df[role_col] == r].tolist()
            if not ridx:
                continue

            role_block = df_scored.loc[ridx].copy()
            role_raw = score_weighted_z(role_block, w_post, feats)

            df_scored.loc[ridx, "df_role_score_raw"] = role_raw
            df_scored.loc[ridx, "df_role_rating_0100"] = _rating_0100_from_rank(role_raw)
            df_scored.loc[ridx, "n_role"] = float(len(ridx))

        # sample flag (informational only)
        df_scored.loc[post_df.index, "role_sample_flag"] = np.where(
            df_scored.loc[post_df.index, "n_role"].astype(float) < float(min_role_n_for_flag),
            "LOW_SAMPLE",
            "OK",
        )

    # Shrunk role rating (extra column, does NOT replace role rating)
    df_scored["df_role_rating_shrunk_0100"] = _shrink_rating(
        df_scored["df_role_rating_0100"],
        df_scored["df_post_rating_0100"],
        df_scored["n_role"],
        k=shrink_k,
    )

    # Export
    if export_filename:
        df_scored.to_csv(f"data/processed/{export_filename}", index=False)

    if export_weights_filename:
        weights_df.to_csv(f"data/processed/{export_weights_filename}", index=False)

    return weights_df, df_scored


if __name__ == "__main__":
    print("This module is meant to be imported and called from main.py.")
