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
class MFPositionPack:
    df_mf: pd.DataFrame
    feature_cols_stage1: list[str]
    feature_cols_stage2: list[str]
    cluster_map_stage1: dict[int, str]   # 0/1 -> {"DM","NON_DM"}
    cluster_map_stage2: dict[int, str]   # 0/1 -> {"AM","CM"}
    thresholds: dict[str, float]


# -------------------------
# Utils
# -------------------------
def _parse_minutes_series(s: pd.Series) -> pd.Series:
    # Your Excel is cleaned: thousands are plain ints, decimals use dot.
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
                if (df[c].notna().sum() / n) >= float(min_non_nan_ratio):
                    keep.append(c)
    return keep


def _fit_gmm_2class(Xz_train: np.ndarray, random_state: int) -> GaussianMixture:
    # Conservative/stable (like your FW/DF stages)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="diag",
        random_state=random_state,
        n_init=40,
        max_iter=2500,
        reg_covar=2e-3,
    )
    gmm.fit(Xz_train)
    return gmm


def _soften_proba(proba: np.ndarray, temperature: float | None) -> np.ndarray:
    if temperature is None or temperature <= 1.0:
        return proba
    p = np.power(proba, 1.0 / float(temperature))
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


# -------------------------
# Main API
# -------------------------
def assign_mf_positions(
    df_players: pd.DataFrame,
    min_train_minutes: int = 900,
    proba_temperature: float | None = 1.25,
    random_state: int = 42,
    export_filename: str = "midfielders_mf_posts.csv",
    summary_filename: str = "midfielders_mf_posts_cluster_summary.csv",
) -> MFPositionPack:
    """
    PURE MF only (Pos == 'MF').

    Two-stage MF post assignment:
      Stage 1: DM vs NON_DM  (2-class GMM)
      Stage 2: among NON_DM: AM vs CM (2-class GMM)

    Changes vs your previous version:
      - FEWER but MORE DISCRIMINANT features
      - Stage2 explicitly separates:
          AM = high creation + high final-third involvement
          CM = higher pass volume / switches / long balls + mid-third involvement + some defensive baseline
    """
    df_mf = df_players[df_players["Pos"].astype(str).str.upper().eq("MF")].copy().reset_index(drop=True)
    if df_mf.empty:
        raise ValueError("No PURE MF found: Pos == 'MF'.")

    if "Min" not in df_mf.columns:
        raise ValueError("Expected column 'Min' in df_players.")

    # ---- Aliases (robust to past typos) ----
    rename_map = {}
    if "Clearences" in df_mf.columns and "Clearances" not in df_mf.columns:
        rename_map["Clearences"] = "Clearances"
    if "Tackles_won" in df_mf.columns and "Tackles_Won" not in df_mf.columns:
        rename_map["Tackles_won"] = "Tackles_Won"
    df_mf = df_mf.rename(columns=rename_map)

    # Minutes + train filter
    df_mf["Min_num"] = _parse_minutes_series(df_mf["Min"])
    df_mf["low_min_flag"] = df_mf["Min_num"] < float(min_train_minutes)

    # =====================================================================
    # Stage 1: DM vs NON_DM (compact + discriminant)
    # =====================================================================
    candidates_1 = [
        # deep/zone
        "Touches_Def3rd", "Touches_Att3rd",
        # defense core
        "Interceptions", "Tackles_Interceptions",
        # build-up volume
        "Pass_Att", "Pass_Switch",
        # attacking/creation signals (penalize DM)
        "KP", "xA", "Touches_AttPen", "Touches_final3rd",
    ]
    df_mf = _coerce_numeric(df_mf, candidates_1)
    usable_1 = _available_numeric(df_mf, candidates_1, min_non_nan_ratio=0.60)

    preferred_1 = [
        "Touches_Def3rd",
        "Interceptions", "Tackles_Interceptions",
        "Pass_Att", "Pass_Switch",
        "Touches_Att3rd", "Touches_AttPen", "Touches_final3rd",
        "KP", "xA",
    ]
    feature_cols_1 = [c for c in preferred_1 if c in usable_1]
    if len(feature_cols_1) < 6:
        raise ValueError(f"Too few usable MF Stage1 features. Found: {feature_cols_1}")

    train_1 = df_mf[~df_mf["low_min_flag"]].copy()
    if train_1.shape[0] < 15:
        raise ValueError(
            f"Not enough PURE MF with Min >= {min_train_minutes} to train Stage1. "
            f"Have {train_1.shape[0]}."
        )

    med_1 = train_1[feature_cols_1].median(numeric_only=True)
    X1 = train_1[feature_cols_1].copy().fillna(med_1)

    sc1 = StandardScaler()
    X1z = sc1.fit_transform(X1.values)

    gmm1 = _fit_gmm_2class(X1z, random_state=random_state)
    centers1 = pd.DataFrame(gmm1.means_, columns=feature_cols_1)

    def g1(col: str) -> pd.Series:
        return centers1[col] if col in centers1.columns else 0.0

    # DM score: deep + defending + some build-up, penalize high/box + creation
    score_dm = (
        1.5 * g1("Touches_Def3rd")
        + 1.15 * g1("Touches_DefPen")
        + 1.10 * g1("Interceptions")
        + 1.05 * g1("Tackles_Interceptions")
        + 0.55 * g1("Pass_Switch")
        - 1.10 * g1("Touches_Att3rd")
        - 1.05 * g1("Touches_AttPen")
        - 0.90 * g1("KP")
        - 0.80 * g1("xA")
        - 1.10 * g1("Touches_final3rd")
        - 0.45 * g1("FinalThird_Pass")
    )

    dm_component = int(score_dm.idxmax())
    nondm_component = int([i for i in [0, 1] if i != dm_component][0])
    cluster_map_1 = {dm_component: "DM", nondm_component: "NON_DM"}

    # Predict stage1 for ALL MF
    X1_all = df_mf[feature_cols_1].copy().fillna(med_1)
    X1_all_z = sc1.transform(X1_all.values)
    proba1 = _soften_proba(gmm1.predict_proba(X1_all_z), proba_temperature)

    df_mf["mf_cluster_stage1"] = proba1.argmax(axis=1)
    df_mf["mf_post_stage1"] = pd.Series(df_mf["mf_cluster_stage1"]).map(cluster_map_1).astype(str).values
    df_mf["p_DM"] = proba1[:, dm_component]
    df_mf["p_NON_DM"] = proba1[:, nondm_component]
    df_mf["mf_post_stage1_proba"] = proba1.max(axis=1)

    # =====================================================================
    # Stage 2: AM vs CM among NON_DM only (compact + discriminant)
    # =====================================================================
    df_nondm = df_mf[df_mf["mf_post_stage1"].astype(str).eq("NON_DM")].copy().reset_index(drop=True)
    if df_nondm.empty:
        # degenerate case: everyone DM
        df_mf["mf_post_stage2"] = np.nan
        df_mf["mf_post_stage2_proba"] = np.nan
        df_mf["p_CM"] = np.nan
        df_mf["p_AM"] = np.nan
        df_mf["mf_post"] = "DM"
        df_mf["mf_post_proba"] = df_mf["mf_post_stage1_proba"]
        out_path = PROCESSED_DIR / export_filename
        df_mf.to_csv(out_path, index=False)
        return MFPositionPack(
            df_mf=df_mf,
            feature_cols_stage1=feature_cols_1,
            feature_cols_stage2=[],
            cluster_map_stage1=cluster_map_1,
            cluster_map_stage2={},
            thresholds={"min_train_minutes": float(min_train_minutes), "proba_temperature": float(proba_temperature or 1.0)},
        )

    # CM signals: pass volume/switching + mid-third
    # AM signals: creation + advanced touches + shots (light)
    candidates_2 = [
        "Touches_Mid3rd", "Touches_Att3rd", "Touches_AttPen",
        "Pass_Switch", "Pass_Long_Att",          # keep (discriminant)
        "PrgP",
        "KP", "xA", "PPA", "SCA90", "GCA90",     # add GCA90
        "Sh/90", "xG",
        "Tackles_Interceptions", "Interceptions",
    ]
    df_nondm = _coerce_numeric(df_nondm, candidates_2)
    usable_2 = _available_numeric(df_nondm, candidates_2, min_non_nan_ratio=0.60)

    preferred_2 = [
        "Touches_Mid3rd", "Touches_Att3rd", "Touches_AttPen",
        "KP", "xA", "PPA", "SCA90", "GCA90",
        "Sh/90", "xG",
        "PrgP",
        "Pass_Switch", "Pass_Long_Att",
        "Tackles_Interceptions", "Interceptions",
    ]
    feature_cols_2 = [c for c in preferred_2 if c in usable_2]
    if len(feature_cols_2) < 8:
        raise ValueError(f"Too few usable MF Stage2 features. Found: {feature_cols_2}")

    # Train on NON_DM with minutes filter; fallback to all NON_DM if too few
    train_2 = df_nondm[df_nondm["Min_num"] >= float(min_train_minutes)].copy()
    if train_2.shape[0] < 12:
        train_2 = df_nondm.copy()

    med_2 = train_2[feature_cols_2].median(numeric_only=True)
    X2 = train_2[feature_cols_2].copy().fillna(med_2)

    sc2 = StandardScaler()
    X2z = sc2.fit_transform(X2.values)

    gmm2 = _fit_gmm_2class(X2z, random_state=random_state)
    centers2 = pd.DataFrame(gmm2.means_, columns=feature_cols_2)

    def g2(col: str) -> pd.Series:
        return centers2[col] if col in centers2.columns else 0.0

    # AM score: high creation + high advanced involvement, penalize CM-like pass volume
    score_am = (
        1.25 * g2("KP")
        + 1.10 * g2("xA")
        + 0.95 * g2("PPA")
        + 1.10 * g2("SCA90")
        + 1.15 * g2("GCA90")
        + 0.95 * g2("Touches_Att3rd")
        + 1.25 * g2("Touches_AttPen")
        + 0.55 * g2("Sh/90")
        + 0.45 * g2("xG")
        - 0.55 * g2("Touches_Mid3rd")
        - 0.30 * g2("Pass_Switch")
        - 0.20 * g2("Pass_Long_Att")
        - 0.25 * g2("Tackles_Interceptions")
        - 0.15 * g2("Interceptions")
    )
    am_component = int(score_am.idxmax())
    cm_component = int([i for i in [0, 1] if i != am_component][0])
    cluster_map_2 = {am_component: "AM", cm_component: "CM"}

    # Predict stage2 for all NON_DM
    X2_all = df_nondm[feature_cols_2].copy().fillna(med_2)
    X2_all_z = sc2.transform(X2_all.values)
    proba2 = _soften_proba(gmm2.predict_proba(X2_all_z), proba_temperature)

    df_nondm["mf_cluster_stage2"] = proba2.argmax(axis=1)
    df_nondm["mf_post_stage2"] = pd.Series(df_nondm["mf_cluster_stage2"]).map(cluster_map_2).astype(str).values
    df_nondm["p_AM"] = proba2[:, am_component]
    df_nondm["p_CM"] = proba2[:, cm_component]
    df_nondm["mf_post_stage2_proba"] = proba2.max(axis=1)

    # Merge stage2 back into df_mf
    df_mf = df_mf.merge(
        df_nondm[["Player", "Squad", "mf_post_stage2", "mf_post_stage2_proba", "p_AM", "p_CM"]],
        on=["Player", "Squad"],
        how="left",
    )

    # Final label
    df_mf["mf_post"] = np.where(
        df_mf["mf_post_stage1"].astype(str).eq("DM"),
        "DM",
        df_mf["mf_post_stage2"].astype(str),
    )

    # Final proba (stage used)
    df_mf["mf_post_proba"] = np.where(
        df_mf["mf_post"].astype(str).eq("DM"),
        df_mf["mf_post_stage1_proba"].astype(float),
        df_mf["mf_post_stage2_proba"].astype(float),
    )

    # Optional: short cluster summary export (centers in z-space)
    def _centers_summary(centers: pd.DataFrame, feature_cols: list[str], stage: str, cmap: dict[int, str]) -> pd.DataFrame:
        tmp = centers.copy()
        tmp["stage"] = stage
        tmp["cluster"] = range(len(tmp))
        tmp["mapped"] = tmp["cluster"].map(cmap).astype(str)
        tmp["avg_abs_z"] = tmp[feature_cols].abs().mean(axis=1)
        cols = ["stage", "cluster", "mapped", "avg_abs_z"] + feature_cols
        return tmp[cols].sort_values(["stage", "cluster"])

    summary = pd.concat(
        [
            _centers_summary(centers1, feature_cols_1, "stage1_DM_vs_NONDM", cluster_map_1),
            _centers_summary(centers2, feature_cols_2, "stage2_CM_vs_AM", cluster_map_2),
        ],
        axis=0,
        ignore_index=True,
    )
    (PROCESSED_DIR / summary_filename).write_text(summary.to_csv(index=False), encoding="utf-8")

    # Export (clean)
    out_cols = [
        "Player", "Squad", "Pos", "Min", "Min_num", "low_min_flag",
        "mf_post", "mf_post_proba",
        "mf_post_stage1", "mf_post_stage1_proba", "p_DM", "p_NON_DM",
        "mf_post_stage2", "mf_post_stage2_proba", "p_CM", "p_AM",
    ]
    out_cols = [c for c in out_cols if c in df_mf.columns]
    out_path = PROCESSED_DIR / export_filename
    df_mf[out_cols].sort_values(["Squad", "Player"]).to_csv(out_path, index=False)

    return MFPositionPack(
        df_mf=df_mf,
        feature_cols_stage1=feature_cols_1,
        feature_cols_stage2=feature_cols_2,
        cluster_map_stage1=cluster_map_1,
        cluster_map_stage2=cluster_map_2,
        thresholds={
            "min_train_minutes": float(min_train_minutes),
            "proba_temperature": float(proba_temperature or 1.0),
        },
    )
