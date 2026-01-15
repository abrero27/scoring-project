import pandas as pd
import numpy as np

from src.data_loader import load_outfield_gk_table

# DF
from src.df_positions import assign_df_positions
from src.df_roles_cb import assign_cb_roles
from src.df_roles_fb import assign_fb_roles
from src.df_scoring import score_defenders

# MF
from src.mf_positions import assign_mf_positions
from src.mf_roles_dm import assign_dm_roles
from src.mf_roles_cm import assign_cm_roles
from src.mf_roles_am import assign_am_roles
from src.mf_scoring import score_midfielders

# FW 
from src.fw_positions import assign_fw_positions
from src.fw_roles_all import assign_fw_roles_all
from src.fw_scoring import score_forwards

# Versatility/Radar
from src.role_projection import project_one_player
from src.radar_plot import plot_radar_from_df


def main():
    # 1) Load all blocks from Excel
    df_outfield, df_gk, df_table = load_outfield_gk_table()

    print(f"✅ Outfield players: {df_outfield.shape}")
    print(f"✅ Goalkeepers: {df_gk.shape}")
    print(f"✅ League table: {df_table.shape}\n")

    # 2) Clean squad names for joins (future-proof)
    df_table["Squad"] = df_table["Squad"].astype(str).str.strip().str.lower()
    df_outfield["Squad"] = df_outfield["Squad"].astype(str).str.strip().str.lower()

    # =========================
    # DEFENDERS: POSTS (PURE DF only) — CB / FB_WB (+ hybrid_flag)
    # =========================
    df_pack = assign_df_positions(
        df_players=df_outfield,
        min_train_minutes=900,
        proba_temperature=1.20,
        random_state=42,
        export_filename="defenders_df_posts.csv",
    )

    df_df = df_pack.df_df

    print("\nDF post assignment done ✅ (PURE DF only)")
    print("Used features:", df_pack.feature_cols)
    print("Cluster map:", df_pack.cluster_map)
    print("Thresholds:", df_pack.thresholds, "\n")

    print("DF distribution:")
    print(df_df["df_post"].value_counts(), "\n")

    print("Preview DF (first 20):")
    cols_prev = [
        "Player","Squad","Min","Min_num","df_post","df_post_proba",
        "p_CB","p_FB_WB","low_min_flag"
    ]
    cols_prev = [c for c in cols_prev if c in df_df.columns]
    print(df_df[cols_prev].head(20).to_string(index=False))

    print("\n✅ Exported: data/processed/defenders_df_posts.csv")

    # =========================
    # DEFENDERS: ROLES (CB only)
    # =========================
    cb_pack = assign_cb_roles(
        df_players=df_outfield,
        df_df_posts=df_df[["Player","Squad","df_post"]],
        min_train_minutes=700,
        k_min=3,
        k_max=6,
        proba_temperature=1.20,
        random_state=42,
        export_filename="defenders_cb_roles.csv",
        summary_filename="defenders_cb_roles_cluster_summary.csv",
    )
    df_cb_roles = cb_pack.df_cb_roles

    print("\nCB role model ✅")
    print("Chosen K:", cb_pack.chosen_k)
    print("BIC:", cb_pack.bic)
    print("CB role distribution:")
    print(df_cb_roles["cb_role"].value_counts(), "\n")
    print("Preview CB roles (first 15):")
    print(df_cb_roles[["Player","Squad","Min_num","cb_role","cb_role_proba"]].head(15).to_string(index=False))
    print("\n✅ Exported: data/processed/defenders_cb_roles.csv")

    # =========================
    # DEFENDERS: ROLES (FB/WB only)
    # =========================
    fb_pack = assign_fb_roles(
        df_players=df_outfield,
        df_df_posts=df_df[["Player","Squad","df_post"]],
        min_train_minutes=700,
        k_min=2,
        k_max=8,
        proba_temperature=1.20,
        random_state=42,
        export_filename="defenders_fb_roles.csv",
        summary_filename="defenders_fb_roles_cluster_summary.csv",
    )
    df_fb_roles = fb_pack.df_fb_roles

    print("\nFB/WB role model ✅")
    print("Chosen K:", fb_pack.chosen_k)
    print("BIC:", fb_pack.bic)
    print("FB role distribution:")
    print(df_fb_roles["fb_role"].value_counts(), "\n")
    print("Preview FB roles (first 15):")
    print(df_fb_roles[["Player","Squad","Min_num","fb_role","fb_role_proba"]].head(15).to_string(index=False))
    print("\n✅ Exported: data/processed/defenders_fb_roles.csv")

    # =========================
    # SCORING DF
    # =========================
    weights_df, df_df_scored = score_defenders(
        df_outfield=df_outfield,
        df_df_posts=df_df[["Player","Squad","df_post"]],   # output df_positions
        df_cb_roles=df_cb_roles,                           # output df_roles_cb
        df_fb_roles=df_fb_roles,                           # output df_roles_fb
        min_non_nan_ratio=0.60,
        min_train_minutes=700,
        min_role_n_for_flag=5,
        shrink_k=10,
        export_filename="defenders_df_scored.csv",
        export_weights_filename="defenders_df_weights.csv",
    )

    print("\n✅ DF scoring done.")
    print("Min columns present:", [c for c in ["Min_num", "Min"] if c in df_df_scored.columns])

    # OFFICIAL DF ranking (mix CB + FB because both are 0-100 within their post)
    base = df_df_scored[df_df_scored["minutes_flag"] == "OK"].copy() if "minutes_flag" in df_df_scored.columns else df_df_scored.copy()

    top = (
        base.sort_values("df_post_rating_0100", ascending=False)
        .head(15)
        .copy()
    )

    cols = ["Player","Squad","df_post","cb_role","fb_role"]
    minutes_col = "Min_num" if "Min_num" in df_df_scored.columns else ("Min" if "Min" in df_df_scored.columns else None)
    if minutes_col:
        cols.append(minutes_col)

    cols += [
        "df_post_rating_0100",           # OFFICIAL (CB vs CB) / (FB vs FB)
        "df_role_rating_0100",           # within role (descriptive)
        "df_role_rating_shrunk_0100",    # robust role rating (recommended)
        "n_role","role_sample_flag",
    ]
    if "minutes_flag" in df_df_scored.columns:
        cols.append("minutes_flag")

    cols = [c for c in cols if c in top.columns]

    print("\nTop 15 DF (official ranking):")
    print(top[cols].to_string(index=False))

    # Quick sanity checks: you SHOULD see both posts
    print("\nDF post distribution (scored rows):")
    print(base["df_post"].value_counts(dropna=False))

    # =========================
    # MIDFIELDERS: POSTS (PURE MF only) — 2-stage (DM then CM/AM)
    # =========================

    mf_pack = assign_mf_positions(
        df_outfield,
        min_train_minutes=900,
        proba_temperature=1.25,
        random_state=42,
        export_filename="midfielders_mf_posts.csv",
    )

    df_mf = mf_pack.df_mf

    print("\nMF post assignment done ✅ (2-stage: DM then CM/AM)")
    print("Stage1 features:", mf_pack.feature_cols_stage1)
    print("Stage2 features:", mf_pack.feature_cols_stage2)
    print("Stage1 map:", mf_pack.cluster_map_stage1)
    print("Stage2 map:", mf_pack.cluster_map_stage2)
    print("Thresholds:", mf_pack.thresholds, "\n")

    print("MF distribution:")
    print(df_mf["mf_post"].value_counts(), "\n")

    print("Preview MF (first 25):")
    cols_prev = ["Player","Squad","Min","Min_num","mf_post","mf_post_proba","p_DM","p_CM","p_AM","low_min_flag"]
    cols_prev = [c for c in cols_prev if c in df_mf.columns]
    print(df_mf[cols_prev].head(25).to_string(index=False))

    print("\n✅ Exported: data/processed/midfielders_mf_posts.csv")
    
    # =========================
    # MIDFIELDERS: ROLES (DM only)
    # =========================
    dm_role_pack = assign_dm_roles(
        df_players=df_outfield,
        df_mf_posts=df_mf[["Player", "Squad", "mf_post"]],
        min_train_minutes=700,
        k_min=2,
        k_max=5,
        proba_temperature=1.20,
        random_state=42,
        export_filename="midfielders_dm_roles.csv",
        summary_filename="midfielders_dm_roles_cluster_summary.csv",
    )

    df_dm_roles = dm_role_pack.df_dm_roles

    print("\nDM role model ✅ (DM only)")
    print("Chosen K:", dm_role_pack.chosen_k)
    print("BIC:", dm_role_pack.bic)
    print("\nDM role distribution:")
    print(df_dm_roles["mf_role"].value_counts())
    print("\nPreview (first 20):")
    print(df_dm_roles[["Player","Squad","Min","Min_num","mf_post","mf_role","mf_role_proba"]].head(20).to_string(index=False))

    print("\n✅ Exported: data/processed/midfielders_dm_roles.csv")
    print("✅ Exported: data/processed/midfielders_dm_roles_cluster_summary.csv")

    # =========================
    # MIDFIELDERS: ROLES (CM ONLY)
    # =========================
    cm_pack = assign_cm_roles(
        df_players=df_outfield,
        df_mf_posts=df_mf[["Player","Squad","mf_post"]],  # ton output mf_positions
        min_train_minutes=700,
        k_min=2,
        k_max=4,
        proba_temperature=1.20,
        random_state=42,
        export_filename="midfielders_cm_roles.csv",
        summary_filename="midfielders_cm_roles_cluster_summary.csv",
    )

    df_cm_roles = cm_pack.df_cm_roles

    print("\nCM role model ✅")
    print("Chosen K:", cm_pack.chosen_k)
    print("BIC:", cm_pack.bic)
    print("\nCM role distribution:")
    print(df_cm_roles["cm_role"].value_counts())
    print("\nPreview CM roles:")
    print(df_cm_roles[["Player","Squad","Min","Min_num","cm_role","cm_role_proba"]].head(20).to_string(index=False))
    print("\n✅ Exported: data/processed/midfielders_cm_roles.csv")
    print("✅ Exported: data/processed/midfielders_cm_roles_cluster_summary.csv")
    
    # =========================
    # DEBUG CM CLUSTERS — PLAYERS (train only + all)
    # =========================
    DEBUG_CM = False

    if DEBUG_CM:
        print("\n🔎 CM clusters — players overview")
        df_dbg_cm = df_cm_roles.copy()

        print("\nCluster sizes:")
        print(df_dbg_cm["cm_role_cluster"].value_counts())

        for c in sorted(df_dbg_cm["cm_role_cluster"].unique()):
            sub = df_dbg_cm[df_dbg_cm["cm_role_cluster"] == c].copy()
            print("\n" + "=" * 40)
            print(f"CLUSTER {c} | n={len(sub)}")
            print(
                sub[["Player", "Squad", "Min_num", "cm_role", "cm_role_proba"]]
                .sort_values("Min_num", ascending=False)
                .head(15)
                .to_string(index=False)
            )


    # =========================
    # MIDFIELDERS: ROLES (AM ONLY)
    # =========================
    am_pack = assign_am_roles(
        df_players=df_outfield,
        df_mf_posts=df_mf[["Player","Squad","mf_post"]],
        min_train_minutes=700,
        k_min=2,
        k_max=5,
        proba_temperature=1.20,
        random_state=42,
        export_filename="midfielders_am_roles.csv",
        summary_filename="midfielders_am_roles_cluster_summary.csv",
    )
    df_am_roles = am_pack.df_am_roles

    print("\nAM role model ✅ (AM only)")
    print("Chosen K:", am_pack.chosen_k)
    print("BIC:", am_pack.bic)
    print("\nAM role distribution:")
    print(df_am_roles["mf_role"].value_counts())
    print("\nPreview (first 20):")
    print(df_am_roles[["Player","Squad","Min","Min_num","mf_post","mf_role","mf_role_proba"]].head(20).to_string(index=False))

    print("\n✅ Exported: data/processed/midfielders_am_roles.csv")
    print("✅ Exported: data/processed/midfielders_am_roles_cluster_summary.csv")

    # =========================
    # DEBUG AVANT SCORING MF
    # =========================
    print("\n================ DEBUG MF AVANT SCORING ================\n")

    def _missing_cols(df, cols):
        return [c for c in cols if c not in df.columns]

    print("Missing in df_mf:", _missing_cols(df_mf, ["Player", "Squad", "mf_post"]))
    print("Missing in df_dm_roles:", _missing_cols(df_dm_roles, ["Player", "Squad", "mf_role"]))
    print("Missing in df_am_roles:", _missing_cols(df_am_roles, ["Player", "Squad", "mf_role"]))
    print("Missing in df_cm_roles:", _missing_cols(df_cm_roles, ["Player", "Squad", "cm_role"]))

    df_mf_role_all = pd.concat(
        [
            df_dm_roles[["Player", "Squad", "mf_role"]],
            df_am_roles[["Player", "Squad", "mf_role"]],
        ],
        ignore_index=True,
    )

    df_mf_dbg = (
        df_mf[["Player", "Squad", "mf_post"]]
        .merge(df_mf_role_all, on=["Player", "Squad"], how="left")
        .merge(df_cm_roles[["Player", "Squad", "cm_role"]], on=["Player", "Squad"], how="left")
    )

    print("\nMF post distribution (df_mf):")
    print(df_mf_dbg["mf_post"].value_counts(dropna=False))

    print("\nMF roles (mf_role) distribution (DM + AM only):")
    print(df_mf_dbg["mf_role"].value_counts(dropna=False))

    print("\nCM roles (cm_role) distribution:")
    print(df_mf_dbg["cm_role"].value_counts(dropna=False))

    unexpected_cm = df_mf_dbg.loc[df_mf_dbg["mf_post"] == "CM", "mf_role"].dropna().unique()
    print("\nUnexpected mf_role for CM players:", unexpected_cm)

    unexpected_dm_am = df_mf_dbg.loc[df_mf_dbg["mf_post"].isin(["DM", "AM"]), "cm_role"].dropna().unique()
    print("Unexpected cm_role for DM/AM players:", unexpected_dm_am)

    dm_am_missing = df_mf_dbg[df_mf_dbg["mf_post"].isin(["DM", "AM"]) & df_mf_dbg["mf_role"].isna()]
    cm_missing = df_mf_dbg[
        (df_mf_dbg["mf_post"] == "CM") & df_mf_dbg["cm_role"].isna()
    ]

    print("\nMissing roles counts:")
    print("DM/AM missing mf_role:", len(dm_am_missing))
    print("CM missing cm_role:", len(cm_missing))

    if len(dm_am_missing) > 0:
        print("\nExamples DM/AM missing mf_role:")
        print(dm_am_missing[["Player", "Squad", "mf_post"]].head(10).to_string(index=False))

    if len(cm_missing) > 0:
        print("\nExamples CM missing cm_role:")
        print(cm_missing[["Player", "Squad", "mf_post"]].head(10).to_string(index=False))

    print("\n================ END DEBUG MF ===========================\n")

    # =========================
    # SCORING MF
    # =========================
    weights_mf, df_mf_scored = score_midfielders(
        df_outfield=df_outfield,
        df_mf=df_mf,
        df_dm_roles=df_dm_roles,
        df_cm_roles=df_cm_roles,
        df_am_roles=df_am_roles,
        min_non_nan_ratio=0.60,
        min_train_minutes=700,
        min_role_n_for_flag=5,
        shrink_k=10,
        export_filename="midfielders_mf_scored.csv",
        export_weights_filename="midfielders_mf_weights.csv",
    )

    print("\n✅ MF scoring done.")

    # -------------------------
    # TOP 15 (ranking officiel)
    # -------------------------
    minutes_col = (
        "Min_num" if "Min_num" in df_mf_scored.columns
        else ("Min" if "Min" in df_mf_scored.columns else None)
    )
    print("Min columns present:", [c for c in ["Min_num", "Min"] if c in df_mf_scored.columns])

    # Filter minutes OK if available
    if "minutes_flag" in df_mf_scored.columns:
        base = df_mf_scored[df_mf_scored["minutes_flag"] == "OK"].copy()
    else:
        base = df_mf_scored.copy()

    # OFFICIAL ranking = within-post rating (DM vs DM, CM vs CM, AM vs AM)
    top = (
        base.sort_values("mf_post_rating_0100", ascending=False)
        .head(15)
        .copy()
    )

    cols = ["Player", "Squad", "mf_post", "mf_role", "cm_role"]
    if minutes_col:
        cols.append(minutes_col)

    cols += [
        "mf_post_rating_0100",          # OFFICIAL poste
        "mf_role_rating_0100",          # rôle PUR (dans son rôle)
        "mf_role_rating_shrunk_0100",   # rôle + shrinkage (colonne en plus)
        "n_role",
        "role_sample_flag",
    ]
    if "minutes_flag" in df_mf_scored.columns:
        cols.append("minutes_flag")

    # keep only existing cols
    cols = [c for c in cols if c in top.columns]

    print("\nTop 15 MF (official ranking):")
    print(top[cols].to_string(index=False))

    # quick diagnostics
    if "minutes_flag" in df_mf_scored.columns:
        print("\nminutes_flag distribution:")
        print(df_mf_scored["minutes_flag"].value_counts(dropna=False))

    if "Min_num" in df_mf_scored.columns:
        print("\nMin_num describe:")
        print(df_mf_scored["Min_num"].describe())
    elif "Min" in df_mf_scored.columns:
        print("\nMin describe:")
        print(pd.to_numeric(df_mf_scored["Min"], errors="coerce").describe())

    print("\n✅ Exported: data/processed/midfielders_mf_scored.csv")
    print("✅ Exported: data/processed/midfielders_mf_weights.csv")


    # =========================
    # ATTACKERS: POSTS (PURE FW only) — ST / NON_ST (+ hybrid_flag)
    # =========================
    fw_pack = assign_fw_positions(
        df_players=df_outfield,
        min_train_minutes=900,     # training stable
        prob_threshold=0.60,       # hybrid_flag si max proba < threshold
        hybrid_margin=0.20,        # hybrid_flag si proche 50/50
        proba_temperature=1.30,    # adoucit les proba extrêmes (report)
        random_state=42,
        export_filename="attackers_fw_posts.csv",
    )
    df_fw = fw_pack.df_fw

    print("\nFW post assignment done ✅ (PURE FW only)")
    print("Used features:", fw_pack.feature_cols)
    print("Cluster map:", fw_pack.cluster_map)
    print("Thresholds:", fw_pack.thresholds, "\n")

    print("FW distribution:")
    print(df_fw["fw_post"].value_counts(), "\n")

    print("Hybrid flag distribution (diagnostic):")
    if "hybrid_flag" in df_fw.columns:
        print(df_fw["hybrid_flag"].value_counts(dropna=False), "\n")

    print("Preview FW (first 25):")
    cols_prev = [
        "Player","Squad","fw_post",
        "fw_post_proba","p_ST","p_NON_ST",
        "hybrid_flag","prob_gap",
        "Min","Min_num","low_min_flag",
    ]
    cols_prev = [c for c in cols_prev if c in df_fw.columns]
    print(df_fw[cols_prev].head(25).to_string(index=False))

    print("\n✅ Exported: data/processed/attackers_fw_posts.csv")

    # =========================
    # ATTACKERS: ROLES (PURE FW only)
    # =========================
    fw_role_pack = assign_fw_roles_all(
        df_players=df_outfield,
        df_fw_posts=df_fw[["Player", "Squad", "fw_post"]],  # merge keys
        min_train_minutes=700,       # tu peux tester 600/700/900
        runner_min_minutes=600,      # override runner (ST only)
        k_min=2,
        k_max_st=5,
        k_max_wide=4,
        proba_temperature=1.20,
        random_state=42,
        export_filename="attackers_fw_roles.csv",
        summary_filename="attackers_fw_roles_cluster_summary.csv",
        runner_override_z=0.25,      # override finisher->runner si runner index très haut
    )

    df_fw_roles = fw_role_pack.df_fw_roles
    # =========================
    # ANALYSE Z-SCORE RUNNER (ST only)
    # =========================

    st_runner_check = df_fw_roles[
        (df_fw_roles["fw_post"] == "ST") &
        (df_fw_roles["Min_num"] >= 600)
    ]

    print("\n📊 Runner z-score distribution (ST, Min >= 600):")
    print(st_runner_check["runner_index_z_st"].describe())

    print("\n📌 Runner z-score range:")
    print("Min z:", st_runner_check["runner_index_z_st"].min())
    print("Max z:", st_runner_check["runner_index_z_st"].max())

    print("\nFW role model ✅ (Family-based + runner override)")
    print("Chosen K ST:", fw_role_pack.chosen_k_st)
    print("BIC ST:", fw_role_pack.bic_st)
    print("Chosen K WIDE:", fw_role_pack.chosen_k_wide)
    print("BIC WIDE:", fw_role_pack.bic_wide)

    print("\nFW role distribution:")
    print(df_fw_roles["fw_role"].value_counts(), "\n")

    print("Preview FW roles (first 30):")
    cols = [
        "Player","Squad","fw_post","Min","Min_num","fw_role","alt_role_any",
        "alt_role_any_proba","runner_index_raw","runner_index_z_st"
    ]
    cols = [c for c in cols if c in df_fw_roles.columns]
    print(df_fw_roles[cols].head(30).to_string(index=False))

    print("\n✅ Exported: data/processed/attackers_fw_roles.csv")
    print("✅ Exported: data/processed/attackers_fw_roles_cluster_summary.csv")

    # =========================
    # SCORING FW
    # =========================
    weights_fw, df_fw_scored = score_forwards(
        df_outfield=df_outfield,
        df_fw=df_fw,
        df_fw_roles=df_fw_roles,
        min_non_nan_ratio=0.60,
        min_train_minutes=700,
        min_role_n_for_flag=5,
        shrink_k=10,
        export_filename="attackers_fw_scored.csv",
        export_weights_filename="attackers_fw_weights.csv",
    )

    print("\n✅ FW scoring done.")
    print("Min columns present:", [c for c in ["Min_num", "Min"] if c in df_fw_scored.columns])

    base = df_fw_scored[df_fw_scored["minutes_flag"] == "OK"].copy() if "minutes_flag" in df_fw_scored.columns else df_fw_scored.copy()

    top = (
        base.sort_values("fw_post_rating_0100", ascending=False)
        .head(15)
        .copy()
    )

    cols = ["Player", "Squad", "fw_post", "fw_role"]
    if "Min_num" in top.columns:
        cols.append("Min_num")
    elif "Min" in top.columns:
        cols.append("Min")

    cols += [
        "fw_post_rating_0100",
        "fw_role_rating_0100",
        "fw_role_rating_shrunk_0100",
        "n_role",
        "role_sample_flag",
        "minutes_flag",
    ]
    cols = [c for c in cols if c in top.columns]

    print("\nTop 15 FW (official ranking):")
    print(top[cols].to_string(index=False))

    # =========================
    # BRUNO RADARS (projection)
    # =========================
    df_mf_scored = pd.read_csv("data/processed/midfielders_mf_scored.csv")
    df_df_scored = pd.read_csv("data/processed/defenders_df_scored.csv")
    df_fw_scored = pd.read_csv("data/processed/attackers_fw_scored.csv")

    weights_mf = pd.read_csv("data/processed/midfielders_mf_weights.csv")
    weights_df = pd.read_csv("data/processed/defenders_df_weights.csv")
    weights_fw = pd.read_csv("data/processed/attackers_fw_weights.csv")

    radar_posts_df, radar_roles_df = project_one_player(
        player_name="Bruno Fernandes",
        squad_name="manchester utd",
        df_outfield=df_outfield,
        df_mf_scored=df_mf_scored,
        df_df_scored=df_df_scored,
        df_fw_scored=df_fw_scored,
        weights_mf=weights_mf,
        weights_df=weights_df,
        weights_fw=weights_fw,
        min_train_minutes=700,
    )

    print("\nRadar A (POSTS):")
    print(radar_posts_df)

    print("\nRadar B (MF+FW ROLES):")
    print(radar_roles_df)

    radar_posts_df.to_csv("data/processed/bruno_radar_posts.csv", index=False)
    radar_roles_df.to_csv("data/processed/bruno_radar_roles.csv", index=False)

    print("\n✅ Exported: data/processed/bruno_radar_posts.csv")
    print("✅ Exported: data/processed/bruno_radar_roles.csv")


    plot_radar_from_df(
        radar_posts_df,
        title="Bruno Fernandes — Radar A (Post compatibility)",
        outfile="data/processed/bruno_radar_posts.png",
    )

    plot_radar_from_df(
        radar_roles_df,
        title="Bruno Fernandes — Radar B (MF+FW role compatibility)",
        outfile="data/processed/bruno_radar_roles.png",
    )

    print("✅ Exported: data/processed/bruno_radar_posts.png")
    print("✅ Exported: data/processed/bruno_radar_roles.png")

if __name__ == "__main__":
    main()
