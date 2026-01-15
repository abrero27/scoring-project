# Player Scoring Project – Proposal

## 1. Research question
How can a data-driven scoring system identify football players that are compatible with a club’s tactical identity and financial constraints?

The objective is to go beyond raw statistics and build role-specific, position-aware performance indicators that can be used for recruitment, squad building and player evaluation.

---

## 2. Motivation
Traditional football scouting relies heavily on subjective judgement and raw statistics (goals, assists, tackles, etc.).  
These metrics fail to capture *tactical fit* and *role compatibility*.

This project aims to build a systematic, data-driven framework that:
- separates **positions** from **roles** (e.g., CB Duel vs CB Builder),
- creates **comparable 0–100 scores** across players,
- and measures **versatility** by projecting players across roles.

---

## 3. Data
The dataset is based on event and performance data from FBref / Opta for several hundred players in the Premier League.

Key characteristics:
- Dense, high-quality per-90 statistics
- Minimum minutes filters applied for robustness
- Normalization and outlier clipping for stability

---

## 4. Methodology

The pipeline follows four main stages:

1. **Position identification**  
Players are first assigned to broad positions (DF, MF, FW), then to specific posts (CB, FB, DM, CM, AM, ST, etc.) using clustering (GMM + BIC).

2. **Role classification**  
Within each post, players are clustered into tactical roles (e.g., CB_DUEL, CB_BUILDER, FB_WIDE, FB_INVERTED, etc.).

3. **Scoring model**  
For each role, a small set of highly discriminative features is selected.  
These features are standardized and combined into a **0–100 role score**.

4. **Versatility projection**  
Each player is projected across multiple roles and positions using z-score based transfer functions, allowing us to quantify versatility and role compatibility.

---

## 5. Outputs

The model produces:
- Position scores
- Role scores
- Radar charts for visual interpretation
- Versatility profiles

All results are exported in reproducible CSV and image formats.

---

## 6. Contribution

This project contributes a framework that:
- integrates tactical football knowledge into quantitative modelling,
- produces interpretable player ratings,
- and enables data-driven squad construction under budget constraints.
