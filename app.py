# app.py
from pathlib import Path
import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
DETAIL_DIR = Path("DETAIL SALES & TRAFFIC")  # folder next to app.py

# =========================
# Data loading
# =========================
def load_latest_detail_sales_traffic_report(folder: Path) -> pd.DataFrame:
    """
    Loads the newest CSV/XLS/XLSX file in DETAIL_DIR.
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder.resolve()}\n\n"
            f"Expected structure:\n"
            f"  your_project/\n"
            f"    app.py\n"
            f"    DETAIL SALES & TRAFFIC/\n"
            f"      your_report.xlsx"
        )

    files = sorted(
        [*folder.glob("*.csv"), *folder.glob("*.xlsx"), *folder.glob("*.xls")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No CSV/XLS/XLSX files found in: {folder.resolve()}")

    latest = files[0]
    if latest.suffix.lower() == ".csv":
        df = pd.read_csv(latest)
    else:
        df = pd.read_excel(latest)

    return df


# =========================
# Column normalization
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps report columns to the table columns shown in your screenshot.
    Uses fuzzy "contains" matching so it works across slightly different exports.

    If something doesn't map correctly, update the pick() tokens or force a mapping below.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def pick(*contains: str):
        """Return the first column whose name contains ALL tokens."""
        tokens = [t.lower() for t in contains]
        for c in df.columns:
            name = c.lower()
            if all(t in name for t in tokens):
                return c
        return None

    asin_col = pick("asin") or "ASIN"
    date_col = pick("date") or pick("week") or "Date"

    # Try to find best matches
    mapping_candidates = {
        asin_col: "ASIN",
        date_col: "Date",
        (pick("session", "total") or pick("sessions")): "Session Total",
        (pick("page", "view", "total") or pick("page views")): "Page Views| Total",
        (pick("featured", "offer") or pick("buy", "box")): "Featured Offer Percentage",
        (pick("units", "ordered") or pick("unit", "ordered")): "Units Ordered",
        (pick("unit", "session") or pick("session", "unit")): "Unit Session Percentage",
        (pick("ordered", "product", "sales") or pick("product", "sales")): "Ordered Product Sales",
        (pick("total", "order", "items") or pick("order", "items")): "Total Order Items",
    }

    rename_dict = {k: v for k, v in mapping_candidates.items() if k and k in df.columns}
    df = df.rename(columns=rename_dict)

    wanted = [
        "ASIN",
        "Date",
        "Session Total",
        "Page Views| Total",
        "Featured Offer Percentage",
        "Units Ordered",
        "Unit Session Percentage",
        "Ordered Product Sales",
        "Total Order Items",
    ]
    keep = [c for c in wanted if c in df.columns]
    df = df[keep].copy()

    # Date formatting
    if "Date" in df.columns:
        parsed = pd.to_datetime(df["Date"], errors="coerce")
        if parsed.notna().any():
            # Linux/Mac: "%-d-%b" -> 7-Dec ; Windows needs "%#d-%b"
            try:
                df["Date"] = parsed.dt.strftime("%-d-%b")
            except ValueError:
                df["Date"] = parsed.dt.strftime("%#d-%b")

    # Percent formatting
    for pct_col in ["Featured Offer Percentage", "Unit Session Percentage"]:
        if pct_col in df.columns:
            s = pd.to_numeric(df[pct_col], errors="coerce")
            if s.notna().any():
                # If values are mostly decimals 0..1, convert to %
                if s.dropna().between(0, 1).mean() > 0.7:
                    df[pct_col] = (s * 100).round(0).astype("Int64").astype(str) + "%"
                else:
                    df[pct_col] = s.round(0).astype("Int64").astype(str) + "%"

    # Currency formatting
    if "Ordered Product Sales" in df.columns:
        s = pd.to_numeric(df["Ordered Product Sales"], errors="coerce")
        if s.notna().any():
            df["Ordered Product Sales"] = s.fillna(0).map(lambda x: f"${x:,.0f}")

    return df


# =========================
# Build "ASIN header row + date rows" like screenshot
# =========================
def build_grouped_display(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Sort (best effort)
    sort_cols = [c for c in ["ASIN", "Date"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    display_rows = []
    for asin, g in df.groupby("ASIN", dropna=False):
        # ASIN header row
        header = {col: "" for col in df.columns}
        header["ASIN"] = str(asin)
        header["__row_type"] = "asin"
        display_rows.append(header)

        # Date rows beneath
        for _, r in g.iterrows():
            row = r.to_dict()
            # Put date in the first column (indented), clear Date column to mimic screenshot
            if "Date" in row:
                row["ASIN"] = f"   {row['Date']}"
                row["Date"] = ""
            row["__row_type"] = "data"
            display_rows.append(row)

    out = pd.DataFrame(display_rows)

    # Keep helper column last
    if "__row_type" in out.columns:
        cols = [c for c in out.columns if c != "__row_type"] + ["__row_type"]
        out = out[cols]

    return out


# =========================
# Render styled table
# =========================
def render_detail_sales_traffic_table(display_df: pd.DataFrame) -> None:
    df_show = display_df.copy()

    def style_rows(row):
        if row.get("__row_type") == "asin":
            # Bold ASIN header + top border separator
            return [
                "font-weight: 700; border-top: 2px solid #2f5f78; background: #ffffff;"
                if c != "__row_type"
                else ""
                for c in row.index
            ]
        return ["" for _ in row.index]

    styler = (
        df_show.style.apply(style_rows, axis=1)
        .set_properties(subset=["ASIN"], **{"white-space": "pre"})  # keep indent spaces
    )

    if "__row_type" in df_show.columns:
        styler = styler.hide(columns=["__row_type"])

    st.dataframe(styler, use_container_width=True, hide_index=True)


# =========================
# Section for "Client Summary"
# =========================
def detail_sales_and_traffic_section() -> None:
    st.markdown("## DETAIL SALES & TRAFFIC")

    with st.expander("Data source (latest file detected)", expanded=False):
        try:
            files = sorted(
                [*DETAIL_DIR.glob("*.csv"), *DETAIL_DIR.glob("*.xlsx"), *DETAIL_DIR.glob("*.xls")],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            st.write(files[0].name if files else "No files found")
        except Exception as e:
            st.write(str(e))

    try:
        raw = load_latest_detail_sales_traffic_report(DETAIL_DIR)
        norm = normalize_columns(raw)

        missing = [c for c in ["ASIN", "Date"] if c not in norm.columns]
        if missing:
            st.error(
                f"Missing required columns after mapping: {missing}\n\n"
                f"Found columns: {list(raw.columns)}\n"
                f"Mapped columns: {list(norm.columns)}\n\n"
                f"If your report uses different headers, update normalize_columns()."
            )
            return

        display = build_grouped_display(norm)
        render_detail_sales_traffic_table(display)

    except Exception as e:
        st.error(f"Could not build DETAIL SALES & TRAFFIC table: {e}")


# =========================
# Main app
# =========================
def main():
    st.set_page_config(page_title="Barrel Aged Dashboard", layout="wide")

    st.title("Barrel Aged Dashboard")

    # Example nav; swap this for your existing routing if you already have it
    page = st.sidebar.selectbox("Page", ["Client Summary", "Other"])

    if page == "Client Summary":
        st.markdown("## Client Summary")
        # Add any other Client Summary components above/below
        detail_sales_and_traffic_section()
    else:
        st.write("Other page content...")


if __name__ == "__main__":
    main()
