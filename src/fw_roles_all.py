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
class FWRolePack:
    df_fw_roles: pd.DataFrame
    feature_cols: list[str]
    chosen_k_st: int
    chosen_k_wide: int
    bic_st: dict[int, float]
    bic_wide: dict[int, float]
    role_map_st: dict[int, str]      # cluster -> template role
    role_map_wide: dict[int, str]    # cluster -> template role
    thresholds: dict[str, float]


# -------------------------
# Utils
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
            if (non_nan / n) >= min_non_nan_ratio:
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
            reg_covar=2e-3,  # strong regularization to avoid weird micro-clusters
        )
        gmm.fit(Xz)
        b = gmm.bic(Xz)
        bic[k] = float(b)

        if b < best_bic:
            best_bic = b
            best_model = gmm
            best_k = k

    if best_model is None:
        raise ValueError(f"Not enough samples to fit GMM in range k={k_min}..{k_max}. n={n}")

    return best_model, int(best_k), bic


def _soften_proba(proba: np.ndarray, temperature: float | None) -> np.ndarray:
    """temperature > 1 => less extreme probs, rows still sum to 1."""
    if temperature is None or temperature <= 1.0:
        return proba
    p = np.power(proba, 1.0 / float(temperature))
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


def _zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sd = series.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return series * 0.0
    return (series - mu) / sd


# ------------------------------
# Role templates (football logic)
# ------------------------------
def _role_scores_st(centers: pd.DataFrame) -> dict[str, pd.Series]:
    """
    ST roles:
      - ST_FINISHER: box + xG + shooting, low creation/running
      - ST_LINK: creation + involvement, less pure shooting
      - ST_RUNNER: depth/progression proxies + some xG/shots (so it doesn't get crushed)
    centers are in z-space (StandardScaler).
    """
    def g(col: str) -> pd.Series:
        return centers[col] if col in centers.columns else 0.0

    # Finisher signals
    box = 1.2 * g("Touches_AttPen") + 1.0 * g("xG") + 1.0 * g("npxG")
    shooting = 1.0 * g("Sh/90") + 1.0 * g("SoT/90") + 0.8 * g("npxG/Sh")

    # Link signals
    creation = 1.0 * g("xA") + 0.9 * g("KP") + 0.8 * g("PPA") + 0.9 * g("SCA90") + 0.7 * g("FinalThird_Pass")
    involvement = 0.8 * g("Touches_Att3rd")

    # Runner signals (IMPORTANT: include xG/shots so runner isn't dominated by finisher)
    running = (
        1.0 * g("PrgR")
        + 0.9 * g("Carry_PrgDist")
        + 0.8 * g("Carry_FinalThird")
        + 0.6 * g("Carry_PenArea")
        + 0.7 * g("Touches_Att3rd")
        + 0.8 * g("Offsides")
    )
    runner_attack = 0.6 * g("xG") + 0.6 * g("npxG") + 0.5 * g("Sh/90")

    scores = {}
    scores["ST_FINISHER"] = (1.25 * box + 1.05 * shooting) - (0.85 * creation + 0.55 * running)
    scores["ST_LINK"] = (1.25 * creation + 0.65 * involvement + 0.25 * box) - (0.55 * shooting)
    scores["ST_RUNNER"] = (1.15 * running + 0.90 * runner_attack) - (0.55 * creation + 0.25 * box)
    return scores


def _role_scores_wide(centers: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Wide roles:
      - WIDE_WINGER: crosses + chance creation + take-ons, less box/shooting
      - INSIDE_FORWARD: more box/xG/shooting, less crossing
    """
    def g(col: str) -> pd.Series:
        return centers[col] if col in centers.columns else 0.0

    crosses = 1.2 * g("Crs") + 1.0 * g("Pass_Cross") + 0.9 * g("CrsPA")
    takeon = 0.9 * g("TakeOn_Att") + 0.7 * g("TakeOn_Succ")
    creation = 1.0 * g("xA") + 0.9 * g("KP") + 0.8 * g("PPA") + 0.9 * g("SCA90")

    box = 1.1 * g("Touches_AttPen") + 1.0 * g("xG") + 1.0 * g("npxG")
    shooting = 0.9 * g("Sh/90") + 0.9 * g("SoT/90") + 0.6 * g("npxG/Sh")

    scores = {}
    scores["WIDE_WINGER"] = (1.25 * crosses + 1.05 * creation + 0.75 * takeon) - (0.95 * box + 0.55 * shooting)
    scores["INSIDE_FORWARD"] = (1.20 * box + 1.00 * shooting + 0.45 * takeon) - (1.05 * crosses) + (0.15 * creation)
    return scores


def _cluster_to_role_by_argmax(centers: pd.DataFrame, score_fn) -> dict[int, str]:
    """
    IMPORTANT: NOT a unique assignment.
    Each cluster maps to the template role where it scores highest.
    Then we aggregate component probabilities by role.
    """
    scores = score_fn(centers)  # role -> series over clusters
    role_names = list(scores.keys())
    mat = np.vstack([scores[r].values for r in role_names]).T  # (k, n_roles)
    best_role_idx = mat.argmax(axis=1)
    return {c: role_names[int(best_role_idx[c])] for c in range(mat.shape[0])}


# ------------------------------
# Runner index (player-level)
# ------------------------------
def _runner_index(df: pd.DataFrame) -> pd.Series:
    """
    Player-level runner proxy:
    - Offsides: classic depth / last-line threat
    - PrgR: progressive receives
    - carries progression
    - touches in Att3rd
    All stats are already per90 in your Excel (except Min).
    We standardize within the ST family (z-scores) later.
    """
    def s(col: str) -> pd.Series:
        return df[col] if col in df.columns else 0.0

    raw = (
        1.00 * s("Offsides")
        + 0.90 * s("PrgR")
        + 0.80 * s("Carry_PrgDist")
        + 0.70 * s("Carry_FinalThird")
        + 0.50 * s("Carry_PenArea")
        + 0.60 * s("Touches_Att3rd")
        + 0.40 * s("Sh/90")        # keep some attacking volume
        + 0.35 * s("xG")           # so “runner but scoring threat” is possible
        + 0.35 * s("npxG")
    )
    return raw.fillna(0.0)


# ------------------------------
# Main API
# ------------------------------
def assign_fw_roles_all(
    df_players: pd.DataFrame,
    df_fw_posts: pd.DataFrame,
    min_train_minutes: int = 700,
    runner_min_minutes: int = 600,          # <-- what you asked
    k_min: int = 2,
    k_max_st: int = 5,
    k_max_wide: int = 4,
    proba_temperature: float | None = 1.20,
    random_state: int = 42,
    export_filename: str = "attackers_fw_roles.csv",
    summary_filename: str = "attackers_fw_roles_cluster_summary.csv",
    runner_override_z: float = 0.70,        # mean + 0.70*std (robust, “academic”)
) -> FWRolePack:
    """
    PURE FW roles learned by family:
      - ST family (fw_post == 'ST'): FINISHER / LINK / RUNNER
      - WIDE family (fw_post == 'NON_ST'): WIDE_WINGER / INSIDE_FORWARD

    Key design choices (why it's "pro"):
    - Train per family to avoid wide players dominating runner detection.
    - Map clusters -> template role by argmax (not unique), then aggregate probabilities by role.
    - Add a player-level runner_index with a controlled override for Min >= runner_min_minutes.
    """
    df = df_players.copy()

    # PURE FW
    df_fw = df[df["Pos"].astype(str).str.upper().eq("FW")].copy().reset_index(drop=True)
    if df_fw.empty:
        raise ValueError("No PURE FW found: Pos == 'FW'.")

    # attach fw_post from your existing fw_positions output
    join_cols = ["Player", "Squad"]
    tmp = df_fw_posts.copy()
    for c in join_cols:
        if c not in tmp.columns:
            raise ValueError(f"df_fw_posts must contain column '{c}'.")
        tmp[c] = tmp[c].astype(str)
        df_fw[c] = df_fw[c].astype(str)

    if "fw_post" not in tmp.columns:
        raise ValueError("df_fw_posts must contain column 'fw_post'.")

    df_fw = df_fw.merge(tmp[join_cols + ["fw_post"]], on=join_cols, how="left")
    if df_fw["fw_post"].isna().any():
        missing = int(df_fw["fw_post"].isna().sum())
        raise ValueError(f"{missing} FW rows have no fw_post after merge. Check join keys (Player/Squad).")

    # minutes
    if "Min" not in df_fw.columns:
        raise ValueError("Expected column 'Min'.")
    df_fw["Min_num"] = _parse_minutes_series(df_fw["Min"])
    df_fw["low_min_flag"] = df_fw["Min_num"] < float(min_train_minutes)

    # features (shared but compact)
    candidates = [
        # box / shooting
        "Touches_AttPen", "Touches_Att3rd",
        "npxG", "xG", "npxG/Sh",
        "Sh/90", "SoT/90",
        # creation / link
        "xA", "KP", "PPA", "SCA90", "FinalThird_Pass",
        # wide
        "Crs", "Pass_Cross", "CrsPA",
        "TakeOn_Att", "TakeOn_Succ",
        # running proxies
        "Offsides", "PrgR", "PrgC",
        "Carry_PrgDist", "Carry_FinalThird", "Carry_PenArea",
    ]
    df_fw = _coerce_numeric(df_fw, candidates)
    feature_cols = _available_numeric(df_fw, candidates, min_non_nan_ratio=0.60)

    preferred_order = [
        "Touches_AttPen", "Touches_Att3rd",
        "npxG", "xG", "npxG/Sh",
        "Sh/90", "SoT/90",
        "xA", "KP", "PPA", "SCA90", "FinalThird_Pass",
        "Crs", "Pass_Cross", "CrsPA",
        "TakeOn_Att", "TakeOn_Succ",
        "Offsides", "PrgR", "PrgC",
        "Carry_PrgDist", "Carry_FinalThird", "Carry_PenArea",
    ]
    feature_cols = [c for c in preferred_order if c in feature_cols]
    if len(feature_cols) < 10:
        raise ValueError(f"Too few usable FW role features. Found: {feature_cols}")

    # split families (IMPORTANT for your runner issue)
    df_st = df_fw[df_fw["fw_post"].astype(str).eq("ST")].copy()
    df_wide = df_fw[df_fw["fw_post"].astype(str).eq("NON_ST")].copy()
    if df_st.empty or df_wide.empty:
        raise ValueError("Need both ST and NON_ST groups present from fw_post.")

    def fit_family(df_family: pd.DataFrame, family_name: str):
        train = df_family[~df_family["low_min_flag"]].copy()
        if train.shape[0] < (k_min + 2):
            train = df_family.copy()

        X_train = train[feature_cols].copy().apply(lambda s: s.fillna(s.median()), axis=0)
        scaler = StandardScaler()
        Xz_train = scaler.fit_transform(X_train.values)

        if family_name == "ST":
            gmm, chosen_k, bic = _fit_best_gmm_by_bic(Xz_train, k_min, k_max_st, random_state, "diag")
            centers = pd.DataFrame(gmm.means_, columns=feature_cols)
            role_map = _cluster_to_role_by_argmax(centers, _role_scores_st)
            templates = list(_role_scores_st(centers).keys())
        else:
            gmm, chosen_k, bic = _fit_best_gmm_by_bic(Xz_train, k_min, k_max_wide, random_state, "diag")
            centers = pd.DataFrame(gmm.means_, columns=feature_cols)
            role_map = _cluster_to_role_by_argmax(centers, _role_scores_wide)
            templates = list(_role_scores_wide(centers).keys())

        return gmm, scaler, chosen_k, bic, role_map, centers, templates

    gmm_st, scaler_st, k_st, bic_st, role_map_st, centers_st, templates_st = fit_family(df_st, "ST")
    gmm_w, scaler_w, k_w, bic_w, role_map_w, centers_w, templates_w = fit_family(df_wide, "WIDE")

    def predict_family(df_family: pd.DataFrame, gmm, scaler, role_map: dict[int, str], templates: list[str], prefix: str):
        X_all = df_family[feature_cols].copy().apply(lambda s: s.fillna(s.median()), axis=0)
        Xz_all = scaler.transform(X_all.values)

        proba_comp = gmm.predict_proba(Xz_all)
        proba_comp = _soften_proba(proba_comp, proba_temperature)

        # aggregate component probs -> role probs
        role_probs = {r: np.zeros(len(df_family), dtype=float) for r in templates}
        for j in range(proba_comp.shape[1]):
            r = role_map.get(j)
            if r in role_probs:
                role_probs[r] += proba_comp[:, j]

        mat = np.vstack([role_probs[r] for r in templates]).T
        mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-12)

        best_idx = mat.argmax(axis=1)
        best_role = [templates[i] for i in best_idx]
        best_p = mat.max(axis=1)

        out = df_family.copy()
        out[f"{prefix}"] = best_role
        out[f"{prefix}_proba"] = best_p

        for i, r in enumerate(templates):
            out[f"p_{prefix}_{r}"] = mat[:, i]

        return out

    st_pred = predict_family(df_st, gmm_st, scaler_st, role_map_st, templates_st, "fw_role")
    w_pred = predict_family(df_wide, gmm_w, scaler_w, role_map_w, templates_w, "fw_role")

    df_out = pd.concat([st_pred, w_pred], axis=0, ignore_index=True)

    # -----------------------------
    # Runner override (ST only)
    # -----------------------------
    # runner_index computed for everyone, but calibrated inside ST group (>= runner_min_minutes)
    df_out["runner_index_raw"] = _runner_index(df_out)

    st_mask = df_out["fw_post"].astype(str).eq("ST")
    st_calib = df_out[st_mask & (df_out["Min_num"] >= float(runner_min_minutes))].copy()

    if len(st_calib) >= 8:
        z = _zscore(st_calib["runner_index_raw"])
        # threshold = mean + runner_override_z * std  (z >= runner_override_z)
        cutoff = float(runner_override_z)

        # compute z for all ST using the same mean/std (approx via re-using _zscore on concat trick)
        mu = st_calib["runner_index_raw"].mean()
        sd = st_calib["runner_index_raw"].std(ddof=0)
        if sd and not np.isnan(sd) and sd > 0:
            df_out["runner_index_z_st"] = 0.0
            df_out.loc[st_mask, "runner_index_z_st"] = (df_out.loc[st_mask, "runner_index_raw"] - mu) / sd

            override_mask = (
                st_mask
                & (df_out["Min_num"] >= float(runner_min_minutes))
                & (df_out["runner_index_z_st"] >= cutoff)
                & (df_out["fw_role"].astype(str) == "ST_FINISHER")   # ✅ ONLY finishers can flip to runner
            )
            df_out.loc[override_mask, "fw_role"] = "ST_RUNNER"

            # keep proba as-is (you said proba doesn't matter); still we expose it.
    else:
        # too few to calibrate; keep role as model output
        df_out["runner_index_z_st"] = np.nan

    # =========================================================
    # Alternative role among ALL FW roles (no "other position")
    # =========================================================
    ALL_ROLES = ["ST_FINISHER", "ST_LINK", "ST_RUNNER", "WIDE_WINGER", "INSIDE_FORWARD"]

    # 1) For EVERYONE, compute ST-role probabilities using ST model
    st_all = predict_family(df_fw, gmm_st, scaler_st, role_map_st, templates_st, "tmp_st_role")
    # st_all has p_tmp_st_role_ST_FINISHER, p_tmp_st_role_ST_LINK, p_tmp_st_role_ST_RUNNER

    # 2) For EVERYONE, compute WIDE-role probabilities using WIDE model
    w_all = predict_family(df_fw, gmm_w, scaler_w, role_map_w, templates_w, "tmp_wide_role")
    # w_all has p_tmp_wide_role_WIDE_WINGER, p_tmp_wide_role_INSIDE_FORWARD

    # Merge these probs into df_out
    keep_st = ["Player", "Squad"] + [c for c in st_all.columns if c.startswith("p_tmp_st_role_")]
    keep_w  = ["Player", "Squad"] + [c for c in w_all.columns if c.startswith("p_tmp_wide_role_")]

    df_out = df_out.merge(st_all[keep_st], on=["Player", "Squad"], how="left")
    df_out = df_out.merge(w_all[keep_w],  on=["Player", "Squad"], how="left")

    # 3) Build a single "p_any_ROLE" over the 5 roles
    df_out["p_any_ST_FINISHER"] = df_out.get("p_tmp_st_role_ST_FINISHER", 0.0).fillna(0.0)
    df_out["p_any_ST_LINK"]     = df_out.get("p_tmp_st_role_ST_LINK", 0.0).fillna(0.0)
    df_out["p_any_ST_RUNNER"]   = df_out.get("p_tmp_st_role_ST_RUNNER", 0.0).fillna(0.0)

    df_out["p_any_WIDE_WINGER"]    = df_out.get("p_tmp_wide_role_WIDE_WINGER", 0.0).fillna(0.0)
    df_out["p_any_INSIDE_FORWARD"] = df_out.get("p_tmp_wide_role_INSIDE_FORWARD", 0.0).fillna(0.0)

    denom = (
        df_out["p_any_ST_FINISHER"]
        + df_out["p_any_ST_LINK"]
        + df_out["p_any_ST_RUNNER"]
        + df_out["p_any_WIDE_WINGER"]
        + df_out["p_any_INSIDE_FORWARD"]
        + 1e-12
    )
    for r in ALL_ROLES:
        df_out[f"p_any_{r}"] = df_out[f"p_any_{r}"] / denom

    # 4) Choose alt role = best among ALL roles excluding current fw_role
    def pick_alt(row):
        main = str(row["fw_role"])
        items = [(r, float(row[f"p_any_{r}"])) for r in ALL_ROLES if r != main]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[0][0], items[0][1]

    alt = df_out.apply(pick_alt, axis=1, result_type="expand")
    df_out["alt_role_any"] = alt[0]
    df_out["alt_role_any_proba"] = alt[1]

    # optional: drop temp columns (clean)
    drop_tmp = [c for c in df_out.columns if c.startswith("p_tmp_") or c.startswith("tmp_")]
    df_out = df_out.drop(columns=drop_tmp, errors="ignore")

    # -----------------------------
    # Exports
    # -----------------------------
    base_cols = [
        "Player","Squad","Pos","Min","Min_num","low_min_flag",
        "fw_post","fw_role","fw_role_proba",
        "runner_index_raw","runner_index_z_st",
        "alt_role_any","alt_role_any_proba",
    ]

    prob_cols = sorted([c for c in df_out.columns if c.startswith("p_fw_role_")])
    any_cols  = sorted([c for c in df_out.columns if c.startswith("p_any_")])

    out_cols = [c for c in base_cols if c in df_out.columns] + prob_cols + any_cols
   
    out_path = PROCESSED_DIR / export_filename
    df_out[out_cols].sort_values(["Squad", "Player"]).to_csv(out_path, index=False)

    # Cluster summary (centers in z-space)
    def centers_summary(centers: pd.DataFrame, role_map: dict[int, str], family: str) -> pd.DataFrame:
        tmp = centers.copy()
        tmp["family"] = family
        tmp["cluster"] = range(len(tmp))
        tmp["mapped_role"] = tmp["cluster"].map(role_map)
        tmp["avg_abs_z"] = tmp[feature_cols].abs().mean(axis=1)
        cols = ["family", "cluster", "mapped_role", "avg_abs_z"] + feature_cols
        return tmp[cols].sort_values(["family", "cluster"])

    summary = pd.concat(
        [centers_summary(centers_st, role_map_st, "ST"), centers_summary(centers_w, role_map_w, "NON_ST")],
        axis=0,
        ignore_index=True,
    )
    summary_path = PROCESSED_DIR / summary_filename
    summary.to_csv(summary_path, index=False)

    return FWRolePack(
        df_fw_roles=df_out,
        feature_cols=feature_cols,
        chosen_k_st=k_st,
        chosen_k_wide=k_w,
        bic_st=bic_st,
        bic_wide=bic_w,
        role_map_st=role_map_st,
        role_map_wide=role_map_w,
        thresholds={
            "min_train_minutes": float(min_train_minutes),
            "runner_min_minutes": float(runner_min_minutes),
            "runner_override_z": float(runner_override_z),
            "k_min": float(k_min),
            "k_max_st": float(k_max_st),
            "k_max_wide": float(k_max_wide),
            "proba_temperature": float(proba_temperature or 1.0),
        },
    )
