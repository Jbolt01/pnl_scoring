from __future__ import annotations

import io
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, getcontext
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile


PAGE_TITLE = "Competition Scoring"
LEADERBOARD_HEIGHT = 520
DETAIL_HEIGHT = 420
RECONCILIATION_HEIGHT = 220


class RankingScope(str, Enum):
    """Supported leaderboard arrangements."""

    ROUND_TOTAL = "ROUND_TOTAL"
    SYMBOL_PER_ROUND = "SYMBOL_PER_ROUND"


class MissingPolicy(str, Enum):
    """How to treat gaps when computing aggregate ranks."""

    DROP = "DROP"
    FILL_WORST = "FILL_WORST"


@dataclass(frozen=True)
class SidebarState:
    """Aggregated state derived from the sidebar controls."""

    files: Sequence["UploadedFile"]
    ranking_scope: RankingScope
    missing_policy: MissingPolicy
    top_n: int


getcontext().prec = 28
CENT = Decimal("0.01")


def configure_page() -> None:
    """Set baseline Streamlit page configuration."""

    st.set_page_config(
        page_title=PAGE_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )


configure_page()


def _lowercase_map(columns: Iterable[str]) -> Dict[str, str]:
    return {c.lower(): c for c in columns}


def _normalize_decimal(value: Any) -> Decimal:
    """Convert arbitrary numeric-like values to cent-precision decimals."""

    if value is None:
        return Decimal("0.00")
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return Decimal("0.00")
    text = re.sub(r"[,$]", "", text)
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        dec = Decimal(text)
    except InvalidOperation:
        return Decimal("0.00")
    return dec.quantize(CENT, rounding=ROUND_HALF_UP)


def _parse_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _round_sort_key(label: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", label)
    order = int(match.group(1)) if match else 10**9
    return order, label.lower()


def _sort_round_labels(labels: Iterable[str]) -> List[str]:
    return sorted(labels, key=_round_sort_key)


@dataclass(frozen=True)
class ColumnMap:
    """Point-in-time mapping between canonical fields and CSV columns."""

    team: str
    record_type: str
    position_symbol: str
    position_realized: str
    position_unrealized: str
    position_updated_at: Optional[str]
    summary_net: Optional[str]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> ColumnMap:
        lookup = _lowercase_map(frame.columns)
        return cls(
            team=lookup.get("team_name", lookup.get("team", "team")),
            record_type=lookup.get("record_type", "record_type"),
            position_symbol=lookup.get("position_symbol", "position_symbol"),
            position_realized=lookup.get("position_realized_pnl", "position_realized_pnl"),
            position_unrealized=lookup.get("position_unrealized_pnl", "position_unrealized_pnl"),
            position_updated_at=lookup.get("position_updated_at"),
            summary_net=lookup.get("summary_net_pnl"),
        )


@dataclass
class RoundSnapshot:
    """Fully parsed data for a single competition round."""

    label: str
    frame: pd.DataFrame
    columns: ColumnMap


@st.cache_data(show_spinner=False)
def _read_round_csv(file_name: str, content: bytes) -> Tuple[str, pd.DataFrame, ColumnMap]:
    """Read a CSV round file and detect its column mapping."""

    buffer = io.BytesIO(content)
    try:
        frame = pd.read_csv(buffer, low_memory=False)
    except UnicodeDecodeError:
        frame = pd.read_csv(io.BytesIO(content), low_memory=False, encoding="latin-1")
    label = re.sub(r"\.[^.]+$", "", file_name)
    return label, frame, ColumnMap.from_frame(frame)


def _load_rounds(files: Sequence["UploadedFile"]) -> List[RoundSnapshot]:
    snapshots: List[RoundSnapshot] = []
    for uploaded in files:
        label, frame, mapping = _read_round_csv(uploaded.name, uploaded.getvalue())
        snapshots.append(RoundSnapshot(label=label, frame=frame, columns=mapping))
    snapshots.sort(key=lambda snap: _round_sort_key(snap.label))
    return snapshots



def _latest_positions(snapshot: RoundSnapshot) -> pd.DataFrame:
    """Return the final position per (team, symbol) pair for a round."""

    frame = snapshot.frame
    cols = snapshot.columns

    positions = frame[frame[cols.record_type].astype(str).str.lower() == "position"].copy()
    if positions.empty:
        return pd.DataFrame(columns=["team", "symbol", "realized", "unrealized", "net", "timestamp"])

    positions["team"] = positions[cols.team].astype(str).str.strip()
    positions["symbol"] = positions[cols.position_symbol].astype(str).str.strip()
    positions["realized"] = positions[cols.position_realized].apply(_normalize_decimal)
    positions["unrealized"] = positions[cols.position_unrealized].apply(_normalize_decimal)

    if cols.position_updated_at and cols.position_updated_at in positions.columns:
        positions["timestamp"] = _parse_timestamp(positions[cols.position_updated_at])
    else:
        positions["timestamp"] = pd.NaT

    if positions["timestamp"].notna().any():
        positions = positions.sort_values(["team", "symbol", "timestamp"]).reset_index(drop=True)
    else:
        positions = positions.reset_index().rename(columns={"index": "row_order"}).sort_values(
            ["team", "symbol", "row_order"]
        )

    latest = positions.groupby(["team", "symbol"], as_index=False).tail(1)
    latest["net"] = latest["realized"] + latest["unrealized"]

    return latest[["team", "symbol", "realized", "unrealized", "net", "timestamp"]].reset_index(drop=True)


def _team_totals(snapshot: RoundSnapshot) -> pd.DataFrame:
    positions = _latest_positions(snapshot)
    if positions.empty:
        return pd.DataFrame(columns=["team", "end_net"])

    team_totals = positions.groupby("team", as_index=False)["net"].sum()
    team_totals = team_totals.rename(columns={"net": "end_net"})
    return team_totals


def _symbol_totals(snapshot: RoundSnapshot) -> pd.DataFrame:
    positions = _latest_positions(snapshot)
    if positions.empty:
        return pd.DataFrame(columns=["team", "symbol", "end_net"])

    return positions[["team", "symbol", "net"]].rename(columns={"net": "end_net"})


def _summary_totals(snapshot: RoundSnapshot) -> pd.DataFrame:
    cols = snapshot.columns
    if not cols.summary_net:
        return pd.DataFrame(columns=["team", "summary_net"])

    frame = snapshot.frame
    summaries = frame[frame[cols.record_type].astype(str).str.lower() == "summary"].copy()
    if summaries.empty or cols.team not in summaries.columns:
        return pd.DataFrame(columns=["team", "summary_net"])

    summaries["team"] = summaries[cols.team].astype(str).str.strip()
    summaries["summary_net"] = summaries[cols.summary_net].apply(_normalize_decimal)
    return summaries[["team", "summary_net"]]


def _reconciliation_table(snapshot: RoundSnapshot) -> pd.DataFrame:
    team_totals = _team_totals(snapshot)
    summary_totals = _summary_totals(snapshot)

    if team_totals.empty or summary_totals.empty:
        return pd.DataFrame(columns=["team", "end_net", "summary_net", "delta"])

    combined = team_totals.merge(summary_totals, on="team", how="outer")
    combined["end_net"] = combined["end_net"].apply(_normalize_decimal)
    combined["summary_net"] = combined["summary_net"].apply(_normalize_decimal)
    combined["delta"] = combined["end_net"] - combined["summary_net"]
    return combined


def _geometric_mean(values: Iterable[float]) -> float:
    data = [v for v in values if v is not None and not np.isnan(v)]
    if not data:
        return float("nan")
    return float(np.exp(np.mean(np.log(data))))


def _format_decimal(value: Any) -> str:
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    if isinstance(value, (float, int)) and not pd.isna(value):
        return f"{Decimal(value).quantize(CENT):.2f}"
    if pd.isna(value):
        return ""
    return str(value)


def _format_decimal_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    formatted = frame.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].apply(_format_decimal)
    return formatted


def _round_leaderboard(
    per_round_totals: Dict[str, pd.DataFrame],
    missing_policy: MissingPolicy,
) -> pd.DataFrame:
    if not per_round_totals:
        return pd.DataFrame(columns=["team"])

    rounds = _sort_round_labels(per_round_totals.keys())
    teams = sorted({team for df in per_round_totals.values() for team in df.get("team", [])})
    ranks = pd.DataFrame({"team": teams})
    rank_columns: List[str] = []

    for label in rounds:
        totals = per_round_totals[label]
        if totals.empty:
            continue
        working = totals[["team", "end_net"]].copy()
        working["__rank_value"] = working["end_net"].apply(lambda val: float(_normalize_decimal(val)))
        working = working.sort_values("__rank_value", ascending=False).reset_index(drop=True)
        working["rank"] = np.arange(1, len(working) + 1)
        column_name = f"{label} rank"
        ranks = ranks.merge(
            working.set_index("team")["rank"].rename(column_name),
            left_on="team",
            right_index=True,
            how="left",
        )
        rank_columns.append(column_name)

    if missing_policy == MissingPolicy.DROP:
        ranks = ranks.dropna(subset=rank_columns, how="any")
    else:
        for column in rank_columns:
            worst_rank = ranks[column].max(skipna=True)
            if pd.notna(worst_rank):
                ranks[column] = ranks[column].fillna(worst_rank)

    ranks["Geometric mean rank"] = ranks.apply(
        lambda row: _geometric_mean([row[col] for col in rank_columns if col in row]), axis=1
    )

    leaderboard = ranks.copy()
    for label in rounds:
        totals = per_round_totals[label]
        if totals.empty:
            continue
        leaderboard = leaderboard.merge(
            totals.set_index("team")["end_net"].rename(f"{label} total"),
            left_on="team",
            right_index=True,
            how="left",
        )

    total_columns = [col for col in leaderboard.columns if col.endswith(" total")]
    leaderboard = _format_decimal_columns(leaderboard, total_columns)
    leaderboard = leaderboard.sort_values(["Geometric mean rank", "team"]) if not leaderboard.empty else leaderboard
    return leaderboard.reset_index(drop=True)


def _symbol_leaderboard(
    per_round_symbol_totals: Dict[str, pd.DataFrame],
    missing_policy: MissingPolicy,
) -> pd.DataFrame:
    if not per_round_symbol_totals:
        return pd.DataFrame(columns=["team"])

    rounds = _sort_round_labels(per_round_symbol_totals.keys())
    teams = sorted({team for df in per_round_symbol_totals.values() for team in df.get("team", [])})

    unit_identifiers: List[Tuple[str, str]] = []
    for label in rounds:
        frame = per_round_symbol_totals[label]
        if frame.empty:
            continue
        for symbol in sorted(frame["symbol"].unique().tolist()):
            unit_identifiers.append((label, symbol))
    unit_identifiers = sorted(set(unit_identifiers), key=lambda item: (rounds.index(item[0]), item[1]))

    ranks = pd.DataFrame({"team": teams})
    rank_columns: List[str] = []

    for label, symbol in unit_identifiers:
        frame = per_round_symbol_totals[label]
        subset = frame[frame["symbol"] == symbol][["team", "end_net"]].copy()
        if subset.empty:
            continue
        subset["__rank_value"] = subset["end_net"].apply(lambda val: float(_normalize_decimal(val)))
        subset = subset.sort_values("__rank_value", ascending=False).reset_index(drop=True)
        subset["rank"] = np.arange(1, len(subset) + 1)
        column_name = f"{label} :: {symbol} rank"
        ranks = ranks.merge(
            subset.set_index("team")["rank"].rename(column_name),
            left_on="team",
            right_index=True,
            how="left",
        )
        rank_columns.append(column_name)

    if missing_policy == MissingPolicy.DROP:
        ranks = ranks.dropna(subset=rank_columns, how="any")
    else:
        for column in rank_columns:
            worst_rank = ranks[column].max(skipna=True)
            if pd.notna(worst_rank):
                ranks[column] = ranks[column].fillna(worst_rank)

    ranks["Units"] = ranks[rank_columns].notna().sum(axis=1)
    ranks["Geometric mean rank"] = ranks.apply(
        lambda row: _geometric_mean([row[col] for col in rank_columns if col in row]), axis=1
    )

    leaderboard = ranks[["team", "Units", "Geometric mean rank"] + rank_columns]
    leaderboard = leaderboard.sort_values(["Geometric mean rank", "team"]) if not leaderboard.empty else leaderboard
    return leaderboard.reset_index(drop=True)


def _build_leaderboard(
    scope: RankingScope,
    per_round_totals: Dict[str, pd.DataFrame],
    per_round_symbol_totals: Dict[str, pd.DataFrame],
    missing_policy: MissingPolicy,
) -> pd.DataFrame:
    if scope == RankingScope.ROUND_TOTAL:
        return _round_leaderboard(per_round_totals, missing_policy)
    return _symbol_leaderboard(per_round_symbol_totals, missing_policy)



RANKING_SCOPE_OPTIONS: Dict[str, RankingScope] = {
    "Round summary standings": RankingScope.ROUND_TOTAL,
    "Symbol-by-round standings": RankingScope.SYMBOL_PER_ROUND,
}

MISSING_POLICY_OPTIONS: Dict[str, MissingPolicy] = {
    "Drop teams missing data": MissingPolicy.DROP,
    "Treat missing rounds as worst rank": MissingPolicy.FILL_WORST,
}


def _render_sidebar() -> SidebarState:
    with st.sidebar:
        st.header("Inputs")
        files = st.file_uploader(
            label="Round CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help="Upload one CSV export per round. Each file should contain position and summary rows.",
        )

        ranking_label = st.radio(
            label="Leaderboard mode",
            options=list(RANKING_SCOPE_OPTIONS.keys()),
            index=0,
            help="Choose whether to rank by total round performance or by each symbol within every round.",
        )

        missing_label = st.selectbox(
            label="Missing data handling",
            options=list(MISSING_POLICY_OPTIONS.keys()),
            index=0,
            help="Decide how to treat teams with incomplete data when calculating overall standings.",
        )

        top_n = st.slider(
            label="Rows to display",
            min_value=5,
            max_value=100,
            value=50,
            step=5,
            help="Limit the number of leaderboard rows shown on screen.",
        )

    return SidebarState(
        files=files or [],
        ranking_scope=RANKING_SCOPE_OPTIONS[ranking_label],
        missing_policy=MISSING_POLICY_OPTIONS[missing_label],
        top_n=top_n,
    )


def _render_leaderboard(leaderboard: pd.DataFrame, top_n: int) -> None:
    st.subheader("Leaderboard")
    st.dataframe(leaderboard.head(top_n), use_container_width=True, height=LEADERBOARD_HEIGHT)


def _render_round_details(
    round_labels: List[str],
    per_round_symbol_totals: Dict[str, pd.DataFrame],
    per_round_team_totals: Dict[str, pd.DataFrame],
) -> None:
    st.subheader("Round details")
    if not round_labels:
        st.info("No rounds available yet. Upload CSV files to explore results.")
        return

    selected_round = st.selectbox("Select a round", round_labels, index=0)
    symbol_table = per_round_symbol_totals[selected_round]
    team_table = per_round_team_totals[selected_round]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Symbol-level P&L**")
        formatted = _format_decimal_columns(symbol_table, ["end_net"])
        st.dataframe(formatted, use_container_width=True, height=DETAIL_HEIGHT)
    with c2:
        st.markdown("**Team-level P&L**")
        formatted = _format_decimal_columns(team_table, ["end_net"])
        st.dataframe(formatted, use_container_width=True, height=DETAIL_HEIGHT)


def _render_reconciliation(round_labels: List[str], reconciliations: Dict[str, pd.DataFrame]) -> None:
    with st.expander("Reconciliation: snapshot vs. summary"):
        if not round_labels:
            st.caption("Upload rounds to review reconciliation tables.")
            return
        for label in round_labels:
            st.markdown(f"**{label}**")
            table = reconciliations[label]
            formatted = _format_decimal_columns(table, ["end_net", "summary_net", "delta"])
            st.dataframe(formatted, use_container_width=True, height=RECONCILIATION_HEIGHT)


def _render_exports(
    leaderboard: pd.DataFrame,
    per_round_symbol_totals: Dict[str, pd.DataFrame],
    per_round_team_totals: Dict[str, pd.DataFrame],
) -> None:
    st.subheader("Export")
    st.download_button(
        label="Download leaderboard CSV",
        data=leaderboard.to_csv(index=False).encode("utf-8"),
        file_name="leaderboard.csv",
        mime="text/csv",
    )

    symbol_frames: List[pd.DataFrame] = []
    team_frames: List[pd.DataFrame] = []
    for label, symbol_df in per_round_symbol_totals.items():
        if not symbol_df.empty:
            temp = symbol_df.copy()
            temp["round"] = label
            temp = _format_decimal_columns(temp, ["end_net"])
            symbol_frames.append(temp[["round", "team", "symbol", "end_net"]])
    for label, team_df in per_round_team_totals.items():
        if not team_df.empty:
            temp = team_df.copy()
            temp["round"] = label
            temp = _format_decimal_columns(temp, ["end_net"])
            team_frames.append(temp[["round", "team", "end_net"]])

    if symbol_frames:
        symbol_payload = pd.concat(symbol_frames, ignore_index=True).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download per-symbol CSV",
            data=symbol_payload,
            file_name="per_symbol_pnl.csv",
            mime="text/csv",
        )
    if team_frames:
        team_payload = pd.concat(team_frames, ignore_index=True).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download per-team CSV",
            data=team_payload,
            file_name="per_team_pnl.csv",
            mime="text/csv",
        )

    st.caption("All exports retain cent-level precision.")



def main() -> None:
    st.title(PAGE_TITLE)
    st.caption("Review end-of-round positions and standings.")

    sidebar_state = _render_sidebar()
    if not sidebar_state.files:
        st.info("Upload one or more round CSV files to begin.")
        return

    snapshots = _load_rounds(sidebar_state.files)
    if not snapshots:
        st.warning("No readable rounds detected. Confirm the CSV format and try again.")
        return

    round_labels = [snapshot.label for snapshot in snapshots]
    per_round_team_totals: Dict[str, pd.DataFrame] = {}
    per_round_symbol_totals: Dict[str, pd.DataFrame] = {}
    reconciliations: Dict[str, pd.DataFrame] = {}

    for snapshot in snapshots:
        per_round_team_totals[snapshot.label] = _team_totals(snapshot)
        per_round_symbol_totals[snapshot.label] = _symbol_totals(snapshot)
        reconciliations[snapshot.label] = _reconciliation_table(snapshot)

    leaderboard = _build_leaderboard(
        scope=sidebar_state.ranking_scope,
        per_round_totals=per_round_team_totals,
        per_round_symbol_totals=per_round_symbol_totals,
        missing_policy=sidebar_state.missing_policy,
    )

    _render_leaderboard(leaderboard, sidebar_state.top_n)
    _render_round_details(round_labels, per_round_symbol_totals, per_round_team_totals)
    _render_reconciliation(round_labels, reconciliations)
    _render_exports(leaderboard, per_round_symbol_totals, per_round_team_totals)


if __name__ == "__main__":
    main()

