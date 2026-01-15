# Player Scoring Project — Posts, Roles, Ratings & Versatility

This project builds a data-driven football scouting framework that:
1) assigns **posts** inside broad FBref position groups (DF / MF / FW),
2) identifies **roles** inside each post using unsupervised clustering,
3) computes **robust 0–100 ratings** per post and per role using statistically derived weights,
4) optionally estimates **player versatility** by projecting a player onto multiple post/role prototypes and visualizing the result with radar charts.

The key principle is **not to compare heterogeneous profiles** (e.g., full-backs are not compared to centre-backs).  
All comparisons and scores are computed **within the relevant reference group** (post or role).

---

## Repository Structure

# Player Scoring Project — Posts, Roles, Ratings & Versatility

This project builds a data-driven football scouting framework that:
1) assigns **posts** inside broad FBref position groups (DF / MF / FW),
2) identifies **roles** inside each post using unsupervised clustering,
3) computes **robust 0–100 ratings** per post and per role using statistically derived weights,
4) optionally estimates **player versatility** by projecting a player onto multiple post/role prototypes and visualizing the result with radar charts.

The key principle is **not to compare heterogeneous profiles** (e.g., full-backs are not compared to centre-backs).  
All comparisons and scores are computed **within the relevant reference group** (post or role).

---

## Repository Structure

├── main.py
├── environment.yml
├── README.md
├── src/
│ ├── data_loader.py
│ ├── df_positions.py
│ ├── df_roles_cb.py
│ ├── df_roles_fb.py
│ ├── df_scoring.py
│ ├── mf_positions.py
│ ├── mf_roles_dm.py
│ ├── mf_roles_cm.py
│ ├── mf_roles_am.py
│ ├── mf_scoring.py
│ ├── fw_positions.py
│ ├── fw_roles_all.py
│ ├── fw_scoring.py
│ ├── role_projection.py
│ └── radar_plot.py
├── data/
   -- raw (players stats from Fbref)
│ └── processed/ # intermediate outputs created by the pipeline
└── results/ # final artefacts for submission 

---

## Data

The pipeline is designed around an **outfield player table** (FBref-style) containing:
- player identity columns (e.g., `Player`, `Squad`, `Pos`, `Min`)
- per-90 or season-aggregated performance metrics (passing, defending, shooting, etc.)

The script assumes the raw data is loaded via:
- `src/data_loader.py` → `load_outfield_gk_table()`

---

## Methodology (High-level)

### 1) Post Identification (GMM, 2 clusters)
Unsupervised clustering assigns posts within each family:
- **DF** → `CB` vs `FB_WB`
- **MF** → two-stage clustering: `DM` vs `NON_DM`, then `CM` vs `AM`
- **FW** → `ST` vs `NON_ST`

### 2) Role Identification (GMM with K selected by BIC)
Within each post, roles are derived using GMM clustering with K chosen in a range (e.g., 2–6) using BIC.
Examples include:
- CB roles (e.g., `CB_BUILDER`, `CB_DUEL`, `CB_HYBRID`)
- FB roles (e.g., `FB_CLASSIC`, `FB_WIDE`, `FB_INVERTED`)
- MF roles (DM/CM/AM)
- FW roles (ST roles + wide roles)

### 3) Scoring (0–100)
For each post:
- feature importance is **not hand-tuned**
- weights are computed using **ANOVA across roles**, then **effect size (omega squared ω²)**
- raw scores are computed as a **weighted sum of standardized (z-score) features**
- final ratings are converted to **0–100 using percentile ranks** within the reference group

Two ratings are produced:
- **Post rating** (official): within the post (e.g., FB vs FB)
- **Role rating** (descriptive): within the role (e.g., FB_WIDE vs FB_WIDE)

A third rating is added:
- **Shrunk role rating**: role rating shrunk toward post rating to stabilize small-sample roles  
  `role_shrunk = α * role + (1-α) * post`, with `α = n_role / (n_role + k)`

### 4) Versatility (Optional)
A selected player (e.g., Bruno Fernandes) is projected onto:
- a set of **post prototypes** (DF/MF/FW posts)
- a set of **role prototypes** (MF + FW roles)

Distances to each prototype are transformed to **0–100 compatibility scores** and visualized as radar charts.

---

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate player-scoring
