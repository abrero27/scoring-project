# src/role_projection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
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


def _z_params(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    """
    Returns mean and std (ddof=0) for cols.
    std is floored to avoid division by 0.
    """
    x = df[cols].astype(float)
    mu = x.mean(skipna=True)
    sd = x.std(ddof=0, skipna=True)
    sd = sd.replace(0, np.nan)
    return mu, sd


def _z_apply(df: pd.DataFrame, cols: List[str], mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    x = df[cols].astype(float)
    z = (x - mu) / sd
    return z


def _weighted_l1_distance(z_player: pd.Series, z_centroid: pd.Series, w: pd.Series) -> float:
    """
    Weighted L1 distance: sum_j w_j * |z_pj - z_cj|
    """
    # Align indices
    common = z_player.index.intersection(z_centroid.index).intersection(w.index)
    if len(common) == 0:
        return np.nan
    d = (w.loc[common] * (z_player.loc[common] - z_centroid.loc[common]).abs()).sum()
    return float(d)


def _distance_to_score_0100(distances: Dict[str, float]) -> Dict[str, float]:
    """
    Convert distances to 0-100 where 100 is closest (min distance).
    Normalized only across the keys given (e.g., Bruno across 7 posts).
    """
    items = {k: v for k, v in distances.items() if np.isfinite(v)}
    if not items:
        return {k: np.nan for k in distances.keys()}

    vals = np.array(list(items.values()), dtype=float)
    dmin, dmax = float(vals.min()), float(vals.max())
    if dmax == dmin:
        # All equal -> 50
        return {k: (50.0 if np.isfinite(v) else np.nan) for k, v in distances.items()}

    out = {}
    for k, d in distances.items():
        if not np.isfinite(d):
            out[k] = np.nan
        else:
            out[k] = 100.0 * (1.0 - (float(d) - dmin) / (dmax - dmin))
    return out


# -----------------------------
# Core: project one player to many targets
# -----------------------------
@dataclass(frozen=True)
class TargetSpec:
    label: str           # e.g. "DM", "CB", "ST_FINISHER"
    family: str          # "MF"/"DF"/"FW"
    post: str            # e.g. "DM", "CB", "ST", "NON_ST"
    role: Optional[str]  # role label or None for post-centroid projection


def _prepare_reference(
    df_scored: pd.DataFrame,
    post_col: str,
    post_value: str,
    role_col: Optional[str],
    role_value: Optional[str],
    feature_list: List[str],
    min_train_minutes: int = 700,
    minutes_col: str = "Min_num",
) -> pd.DataFrame:
    """
    Reference population for z-scaling and centroids.
    We use minutes >= min_train_minutes if available (stable master-level).
    """
    ref = df_scored.copy()

    # Filter by post
    ref = ref[ref[post_col] == post_value].copy()

    # Optional filter by role
    if role_col and role_value is not None:
        ref = ref[ref[role_col] == role_value].copy()

    # Stable ref: only high minutes if possible
    if minutes_col in ref.columns:
        ref[minutes_col] = pd.to_numeric(ref[minutes_col], errors="coerce")
        ref = ref[ref[minutes_col] >= float(min_train_minutes)].copy()

    # Ensure numeric
    _coerce_numeric(ref, feature_list)

    # Drop rows with all NaN on feature_list
    if feature_list:
        ref = ref[ref[feature_list].notna().any(axis=1)].copy()

    return ref

def project_one_player(
    player_name: str,
    squad_name: str,
    df_outfield: pd.DataFrame,          #Bruno features from outfield
    df_mf_scored: pd.DataFrame,
    df_df_scored: pd.DataFrame,
    df_fw_scored: pd.DataFrame,
    weights_mf: pd.DataFrame,
    weights_df: pd.DataFrame,
    weights_fw: pd.DataFrame,
    min_train_minutes: int = 700,
    clip_rank: float = 0.02,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    pn = str(player_name).strip()
    sq = str(squad_name).strip().lower()

    def _norm_team(x: str) -> str:
        return " ".join(str(x).strip().lower().split())

    def _find_player_outfield(df: pd.DataFrame) -> pd.Series:
        tmp = df.copy()
        tmp["Player"] = tmp["Player"].astype(str).str.strip()
        if "Squad" in tmp.columns:
            tmp["Squad"] = tmp["Squad"].astype(str).map(_norm_team)

        # 1) exact player + exact squad
        hit = tmp[(tmp["Player"] == pn) & (tmp["Squad"] == _norm_team(sq))]
        if len(hit) > 0:
            return hit.iloc[0]

        # 2) exact player, ignore squad (useful if your team string differs)
        hit = tmp[tmp["Player"] == pn]
        if len(hit) > 0:
            return hit.iloc[0]

        # 3) fallback: contains player (rare naming issues)
        hit = tmp[tmp["Player"].str.contains(pn, case=False, na=False)]
        if len(hit) > 0:
            return hit.iloc[0]

        raise ValueError(f"Player not found in df_outfield: {pn} / {sq}")

    # ✅ Bruno features for ALL projections come from df_outfield
    br = _find_player_outfield(df_outfield)

    # (the rest stays identical except:
    #  - br_row is always 'br'
    #  - df_scored chooses reference pops (MF/DF/FW)
    #  - weights determine feature set
    #  - we do NOT need br_mf/br_df/br_fw anymore)

    def _get_weights(weights: pd.DataFrame, post_key_col: str, post_value: str) -> pd.Series:
        w = weights[weights[post_key_col] == post_value].copy()
        if len(w) == 0:
            raise ValueError(f"No weights for {post_key_col}={post_value}")
        w = w.dropna(subset=["feature", "weight"])
        s = w.set_index("feature")["weight"].astype(float)
        s = s[s > 0]
        if s.sum() > 0:
            s = s / s.sum()
        return s

    # -----------------------------
    # Radar A: POSTS
    # -----------------------------
    post_targets: List[TargetSpec] = [
        TargetSpec("CB", "DF", "CB", None),
        TargetSpec("FB", "DF", "FB_WB", None),
        TargetSpec("DM", "MF", "DM", None),
        TargetSpec("CM", "MF", "CM", None),
        TargetSpec("AM", "MF", "AM", None),
        TargetSpec("ST", "FW", "ST", None),
        TargetSpec("NON_ST", "FW", "NON_ST", None),
    ]

    post_dist: Dict[str, float] = {}

    for t in post_targets:
        if t.family == "MF":
            df_scored = df_mf_scored
            weights = _get_weights(weights_mf, "mf_post", t.post)
            post_col = "mf_post"
        elif t.family == "DF":
            df_scored = df_df_scored
            weights = _get_weights(weights_df, "df_post", t.post)
            post_col = "df_post"
        else:
            df_scored = df_fw_scored
            weights = _get_weights(weights_fw, "fw_post", t.post)
            post_col = "fw_post"

        feats = _dedupe(list(weights.index))
        feats = [f for f in feats if f in df_scored.columns and f in df_outfield.columns]
        if not feats:
            post_dist[t.label] = np.nan
            continue

        ref_post = _prepare_reference(
            df_scored=df_scored,
            post_col=post_col,
            post_value=t.post,
            role_col=None,
            role_value=None,
            feature_list=feats,
            min_train_minutes=min_train_minutes,
            minutes_col="Min_num",
        )
        if len(ref_post) < 5:
            post_dist[t.label] = np.nan
            continue

        _coerce_numeric(ref_post, feats)
        mu, sd = _z_params(ref_post, feats)

        z_ref = _z_apply(ref_post, feats, mu, sd)
        z_centroid = z_ref.mean(skipna=True)

        br_df1 = pd.DataFrame([br.to_dict()])
        _coerce_numeric(br_df1, feats)
        z_player = _z_apply(br_df1, feats, mu, sd).iloc[0]

        d = _weighted_l1_distance(z_player, z_centroid, weights.loc[feats])
        post_dist[t.label] = d

    post_scores = _distance_to_score_0100(post_dist)
    radar_posts_df = pd.DataFrame([{"axis": k, "score_0100": v} for k, v in post_scores.items()])

    # -----------------------------
    # Radar B: ROLES (MF + FW only)
    # -----------------------------
    mf_roles = [
        ("DM", "mf_role", "DM_CONTROLLER"),
        ("DM", "mf_role", "DM_DESTROYER"),
        ("DM", "mf_role", "DM_DEEP_CM"),
        ("CM", "cm_role", "CM_PROGRESSOR"),
        ("CM", "cm_role", "CM_BOX_TO_BOX"),
        ("AM", "mf_role", "AM_ORGANIZER"),
        ("AM", "mf_role", "AM_CLASSIC_10"),
    ]
    fw_roles = [
        ("ST", "fw_role", "ST_FINISHER"),
        ("ST", "fw_role", "ST_LINK"),
        ("ST", "fw_role", "ST_RUNNER"),
        ("NON_ST", "fw_role", "WIDE_WINGER"),
        ("NON_ST", "fw_role", "INSIDE_FORWARD"),
    ]

    role_targets: List[TargetSpec] = []
    for post, _, r in mf_roles:
        role_targets.append(TargetSpec(r, "MF", post, r))
    for post, _, r in fw_roles:
        role_targets.append(TargetSpec(r, "FW", post, r))

    role_dist: Dict[str, float] = {}

    for t in role_targets:
        if t.family == "MF":
            df_scored = df_mf_scored
            weights = _get_weights(weights_mf, "mf_post", t.post)
            post_col = "mf_post"
            role_col = "mf_role" if t.post in ["DM", "AM"] else "cm_role"
        else:
            df_scored = df_fw_scored
            weights = _get_weights(weights_fw, "fw_post", t.post)
            post_col = "fw_post"
            role_col = "fw_role"

        feats = _dedupe(list(weights.index))
        feats = [f for f in feats if f in df_scored.columns and f in df_outfield.columns]
        if not feats:
            role_dist[t.label] = np.nan
            continue

        ref_post = _prepare_reference(
            df_scored=df_scored,
            post_col=post_col,
            post_value=t.post,
            role_col=None,
            role_value=None,
            feature_list=feats,
            min_train_minutes=min_train_minutes,
            minutes_col="Min_num",
        )
        if len(ref_post) < 5:
            role_dist[t.label] = np.nan
            continue

        ref_role = _prepare_reference(
            df_scored=df_scored,
            post_col=post_col,
            post_value=t.post,
            role_col=role_col,
            role_value=t.role,
            feature_list=feats,
            min_train_minutes=min_train_minutes,
            minutes_col="Min_num",
        )
        if len(ref_role) < 3:
            role_dist[t.label] = np.nan
            continue

        _coerce_numeric(ref_post, feats)
        mu, sd = _z_params(ref_post, feats)

        z_role = _z_apply(ref_role, feats, mu, sd)
        z_centroid = z_role.mean(skipna=True)

        br_df1 = pd.DataFrame([br.to_dict()])
        _coerce_numeric(br_df1, feats)
        z_player = _z_apply(br_df1, feats, mu, sd).iloc[0]

        d = _weighted_l1_distance(z_player, z_centroid, weights.loc[feats])
        role_dist[t.label] = d

    role_scores = _distance_to_score_0100(role_dist)
    radar_roles_df = pd.DataFrame([{"axis": k, "score_0100": v} for k, v in role_scores.items()]).sort_values("axis")

    return radar_posts_df, radar_roles_df
