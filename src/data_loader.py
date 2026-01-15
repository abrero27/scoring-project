from pathlib import Path
import pandas as pd

BASE_PATH = Path(__file__).resolve().parents[1]
FILE_PATH = BASE_PATH / "data" / "raw" / "Stats joueurs PL 2024-2025.xlsx"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names from FBref/Excel artifacts:
    - remove newlines
    - remove sort arrows ▲ ▼
    - strip extra whitespace
    """
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("▲", "", regex=False)
        .str.replace("▼", "", regex=False)
        .str.strip()
    )
    return df


def prefix_columns(df: pd.DataFrame, prefix: str, keep: list[str]) -> pd.DataFrame:
    """
    Add a prefix to all columns except the ones in keep.
    """
    df = df.copy()
    rename_map = {}
    for c in df.columns:
        if c in keep:
            rename_map[c] = c
        else:
            rename_map[c] = f"{prefix}{c}"
    return df.rename(columns=rename_map)


def load_outfield_players() -> pd.DataFrame:
    """
    Outfield players block
    - Header: Excel row 7
    - Data: Excel rows 8..581 (574 players)
    - Columns: C..FN
    """
    df = pd.read_excel(
        FILE_PATH,
        sheet_name=0,
        header=6,          # Excel row 7
        usecols="C:FN",
        nrows=574,         # rows 8..581
        engine="openpyxl",
    )

    df = df.dropna(how="all")

    # Safety: remove repeated header rows if they appear as data
    if "Player" in df.columns:
        df = df[df["Player"].notna()]
        df = df[df["Player"] != "Player"]

    df = clean_columns(df).reset_index(drop=True)
    return df


def load_goalkeepers() -> pd.DataFrame:
    """
    Goalkeepers (44 rows), two blocks:
    - Standard GK: C..AB
    - (blank) column AC
    - Advanced GK: AD..BC
    Header: Excel row 587
    Data: Excel rows 588..631 (44 GKs)
    """
    # 1) Standard GK
    df_std = pd.read_excel(
        FILE_PATH,
        sheet_name=0,
        header=586,        # Excel row 587
        usecols="C:AB",
        nrows=44,
        engine="openpyxl",
    )

    # 2) Advanced GK (AC is blank)
    df_adv = pd.read_excel(
        FILE_PATH,
        sheet_name=0,
        header=586,
        usecols="AD:BC",
        nrows=44,
        engine="openpyxl",
    )

    df_std = clean_columns(df_std.dropna(how="all")).reset_index(drop=True)
    df_adv = clean_columns(df_adv.dropna(how="all")).reset_index(drop=True)

    # Merge side-by-side
    df_gk = pd.concat([df_std, df_adv], axis=1)
    df_gk = clean_columns(df_gk).reset_index(drop=True)

    # Keep these columns unprefixed (identity + playing time)
    keep = ["Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born", "MP", "Starts", "Min", "90s"]

    # Prefix advanced columns only (except keep)
    adv_cols = df_adv.columns.tolist()
    rename_adv = {}
    for c in adv_cols:
        if c not in keep:
            rename_adv[c] = f"gk_adv_{c}"
    df_gk = df_gk.rename(columns=rename_adv)

    # Prefix everything else (standard GK performance) with gk_, except keep
    df_gk = prefix_columns(df_gk, "gk_", keep=keep)

    return df_gk


def load_league_table() -> pd.DataFrame:
    """
    Premier League table (20 teams)
    - Header: Excel row 636
    - Data: Excel rows 637..656 (20 teams)
    - Columns: C..S
    """
    df = pd.read_excel(
        FILE_PATH,
        sheet_name=0,
        header=635,        # Excel row 636
        usecols="C:S",
        nrows=20,          # rows 637..656
        engine="openpyxl",
    )

    df = clean_columns(df.dropna(how="all")).reset_index(drop=True)

    # Keep these columns as-is, prefix the rest with team_
    keep = ["Rk", "Squad"]
    df = prefix_columns(df, "team_", keep=keep)

    return df

def load_outfield_gk_table():
    """
    Wrapper function that returns:
    - outfield players dataframe
    - goalkeepers dataframe
    - league table dataframe
    """
    df_outfield = load_outfield_players()
    df_gk = load_goalkeepers()
    df_table = load_league_table()
    return df_outfield, df_gk, df_table
