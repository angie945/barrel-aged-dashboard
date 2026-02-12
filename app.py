# app.py
from pathlib import Path
from datetime import date, datetime, timedelta
import re
import csv
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

st.markdown("### ✅ DEPLOY CHECK: app.py updated (Detail Sales & Traffic uses DATA_DIR + recursive file search)")
st.caption(f"Running from: {__file__}")

# ------------------------------------------------------------------
# Base paths (works locally + on Streamlit Cloud)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

# ✅ Your repo data folder path (matches your actual repo structure)
# data/Amazon Reports/Weekly Uploads/<CLIENT>/...
DATA_DIR = BASE_DIR / "data" / "Amazon Reports" / "Weekly Uploads"

REPORT_EXTS = {".csv", ".xls", ".xlsx", ".xlsm"}
LOGO_PATH = BASE_DIR / "logo.png"

# ✅ cap at Feb 1, 2026
WTD_MAX_WEEK_ENDING = date(2026, 2, 1)

# ✅ Folder name exactly as it exists in GitHub (per your screenshot)
DETAIL_SALES_TRAFFIC_FOLDER = "Detail Sales & Traffic"

# ✅ UPDATED CLIENT LIST
CLIENTS = [
    "BISI",
    "DLP",
    "Edge Perfection",
    "GTX Performance",
    "Meritool",
    "NP",
    "Rooted",
]

# =============================================================================
# LISTING METRICS TRIGGERS
# =============================================================================
SALES_DROP_PCT_TRIGGER = 0.30
TRAFFIC_DROP_PCT_TRIGGER = 0.25
BUYBOX_MIN_THRESHOLD = 0.85  # 85%

TRIGGER_LOOKUP = {
    "Sales Drop": {
        "Likely Cause": "Suppressed, Buy Box lost, ad issue",
        "Team": "Listings/Ads",
        "Response": "Verify Buy Box, audit ads",
    },
    "Traffic Drop": {
        "Likely Cause": "De-indexing, suppression",
        "Team": "Listings",
        "Response": "Check ranking/indexing",
    },
    "Buy Box Loss": {
        "Likely Cause": "Price/shipping",
        "Team": "Support",
        "Response": "Review price/delivery",
    },
}

# =============================================================================
# Regex helpers
# =============================================================================
_money_re = re.compile(r"\$?\s*([0-9,]+(?:\.[0-9]{1,2})?)")
_date_re = re.compile(r"(\d{4}-\d{2}-\d{2})")
_date_re2 = re.compile(r"(\d{2}-\d{2}-\d{4})")
_date_re3 = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})")


# =============================================================================
# BASIC HELPERS (✅ UPDATED: recursive file discovery)
# =============================================================================
def _iter_report_files(folder: Path) -> List[Path]:
    """
    ✅ Find report files recursively (handles subfolders).
    Streamlit Cloud repo often has extra nesting; this prevents false 'no file found'.
    """
    if not folder or not folder.exists():
        return []
    files = [
        f
        for f in folder.rglob("*")
        if f.is_file()
        and f.suffix.lower() in REPORT_EXTS
        and not f.name.startswith("~$")
    ]
    return files


def latest_report_file(folder: Path):
    files = _iter_report_files(folder)
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def list_report_files(folder: Path) -> List[Path]:
    files = _iter_report_files(folder)
    return sorted(files, key=lambda f: f.stat().st_mtime)


def read_csv_rows(path: Path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.reader(f):
            rows.append([c.strip() for c in r])
    return rows


def parse_currency_only(cell: str):
    if cell is None:
        return None
    s = str(cell).strip()
    if "$" not in s:
        return None
    m = _money_re.search(s.replace(" ", ""))
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def parse_date_from_any_text(text: str) -> Optional[date]:
    if not text:
        return None
    s = str(text).strip()

    m = _date_re.search(s)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except:
            pass

    m = _date_re2.search(s)
    if m:
        try:
            return datetime.strptime(m.group(1), "%m-%d-%Y").date()
        except:
            pass

    m = _date_re3.search(s)
    if m:
        raw = m.group(1)
        parts = raw.split("/")
        if len(parts) == 3:
            mm, dd, yy = parts
            try:
                yy_i = int(yy)
                if yy_i < 100:
                    yy_i += 2000
                return date(int(yy_i), int(mm), int(dd))
            except:
                pass

    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return dt.date()
    except:
        pass

    return None


def find_col(header_row, contains_text: str):
    target = contains_text.lower()
    for i, h in enumerate(header_row):
        if target in str(h).lower():
            return i
    return None


def fmt_currency(x):
    return f"${x:,.2f}" if x is not None else ""


def fmt_diff(x):
    if x is None:
        return ""
    return f"(${abs(x):,.2f})" if x < 0 else f"${x:,.2f}"


def diff_pct(cur, prev):
    if cur is None or prev is None:
        return None, ""
    diff = cur - prev
    if prev == 0:
        return diff, "#DIV/0!"
    return diff, f"{diff / prev:.0%}"  # no decimals


# =============================================================================
# PERCENT PARSING + NORMALIZATION (force NO decimals everywhere)
# =============================================================================
def _pct_to_float(val):
    if val is None:
        return None
    s = str(val).strip()
    if not s or s == "#DIV/0!":
        return None
    if s.endswith("%"):
        try:
            return float(s.replace("%", "").replace(",", "").strip()) / 100.0
        except:
            return None
    try:
        v = float(s.replace(",", ""))
        if -1 <= v <= 1:
            return v
    except:
        return None
    return None


def fmt_pct0(val) -> str:
    """Normalize any percent-ish value to '12%', '-6%', etc."""
    if val is None:
        return ""
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return ""
    if s == "#DIV/0!":
        return s
    p = _pct_to_float(s)
    if p is None:
        return s
    return f"{p:.0%}"


# =============================================================================
# COLOR STYLING — ONLY PERCENT COLUMNS
# =============================================================================
def _colorize_pct(v):
    p = _pct_to_float(v)
    if p is None:
        return ""
    if p < 0:
        return "color: #b91c1c; font-weight: 700;"  # red
    if p > 0:
        return "color: #15803d; font-weight: 700;"  # green
    return "color: #111827;"


def style_percent_columns(df: pd.DataFrame, cols: List[str]):
    styler = df.style
    for c in cols:
        if c in df.columns:
            styler = styler.applymap(_colorize_pct, subset=[c])
    return styler


# =============================================================================
# YTD/MTD EXTRACTION (CSV logic)
# =============================================================================
def _parse_pct_cell(raw):
    if raw is None:
        return ""
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return ""
    if s.endswith("%"):
        return fmt_pct0(s)
    try:
        v = float(s.replace(",", ""))
        if -1 <= v <= 1:
            return f"{v:.0%}"
        if abs(v) <= 100:
            return f"{v:.0f}%"
    except:
        return ""
    return ""


def extract_compare_table_view_through(rows, period_kind: str):
    start = None
    for i, row in enumerate(rows):
        if "compare sales" in " ".join(row).lower() and "table view" in " ".join(row).lower():
            start = i
            break
    if start is None:
        return None, None, "", {"reason": "Could not find 'Compare Sales - Table view'"}

    header_i = None
    header = None
    for j in range(start, min(start + 250, len(rows))):
        cand = rows[j]
        txt = " | ".join(cand).lower()
        if "ordered product sales" in txt:
            header_i = j
            header = cand
            break

    if header is None:
        return None, None, "", {"reason": "Could not find Table view header row"}

    col_sales = None
    for idx, h in enumerate(header):
        if "ordered product sales" in str(h).lower():
            col_sales = idx
            break

    if col_sales is None:
        return None, None, "", {"reason": "Could not locate 'Ordered Product Sales' column"}

    if period_kind == "ytd":
        cur_key = "this year so far"
        prev_key = "last year through"
        pct_key = "% change from last year"
    else:
        cur_key = "this month so far"
        prev_key = "last month through"
        pct_key = "% change from last month"

    cur_row = prev_row = pct_row = None

    for k in range(header_i + 1, min(header_i + 250, len(rows))):
        r = rows[k]
        if not r:
            continue
        label = str(r[0]).strip().lower()

        if cur_row is None and label.startswith(cur_key):
            cur_row = r
        elif prev_row is None and label.startswith(prev_key):
            prev_row = r
        elif pct_row is None and label.startswith(pct_key):
            pct_row = r

        if cur_row is not None and prev_row is not None and pct_row is not None:
            break

    if cur_row is None or prev_row is None:
        return None, None, "", {"reason": "Missing required rows in Table view"}

    cur = parse_currency_only(cur_row[col_sales]) if col_sales < len(cur_row) else None
    prev = parse_currency_only(prev_row[col_sales]) if col_sales < len(prev_row) else None
    pct_str = ""
    if pct_row is not None and col_sales < len(pct_row):
        pct_str = _parse_pct_cell(pct_row[col_sales])

    return cur, prev, pct_str, {"reason": "Extracted from Table view (through rows)"}


def extract_compare_graph_view_money(rows, this_col_options, last_col_options):
    start = None
    for i, row in enumerate(rows):
        joined = " ".join(row).lower()
        if "compare sales" in joined and "graph view" in joined:
            start = i
            break
    if start is None:
        return None, None, {"reason": "Could not find 'Compare Sales - Graph view' section"}

    header_i = None
    header_row = None
    col_this = None
    col_last = None

    for j in range(start, min(start + 250, len(rows))):
        candidate = rows[j]
        cand_text = " | ".join(candidate).lower()

        for a in this_col_options:
            if a.lower() in cand_text:
                tmp_this = find_col(candidate, a)
                if tmp_this is None:
                    continue
                for b in last_col_options:
                    if b.lower() in cand_text:
                        tmp_last = find_col(candidate, b)
                        if tmp_last is None:
                            continue
                        header_i = j
                        header_row = candidate
                        col_this = tmp_this
                        col_last = tmp_last
                        break
            if header_row is not None:
                break
        if header_row is not None:
            break

    if header_row is None:
        return None, None, {"reason": "Could not find header row with expected columns"}

    data_row = None
    for k in range(header_i + 1, min(header_i + 200, len(rows))):
        r = rows[k]
        v1 = r[col_this] if col_this < len(r) else ""
        v2 = r[col_last] if col_last < len(r) else ""
        if "$" in str(v1) or "$" in str(v2):
            data_row = r
            break

    if data_row is None:
        return None, None, {"reason": "Found header row but no data row with $ values"}

    cur = parse_currency_only(data_row[col_this]) if col_this < len(data_row) else None
    prev = parse_currency_only(data_row[col_last]) if col_last < len(data_row) else None
    return cur, prev, {"reason": "Graph view money"}


# =============================================================================
# YTD / MTD (paths use DATA_DIR)
# =============================================================================
def get_ytd_for_client(client: str):
    folder = DATA_DIR / client / "Year"
    fp = latest_report_file(folder)
    if not fp:
        return None, None, "", {"reason": "No Year file found", "file": None}

    rows = read_csv_rows(fp)

    cur, prev, pct_str, dbg = extract_compare_table_view_through(rows, period_kind="ytd")
    if cur is not None and prev is not None:
        return cur, prev, pct_str, {"file": str(fp), "source": "table_view_through", **dbg}

    YTD_THIS = ["This year so far (Ordered product sales)", "This year so far", "This year"]
    YTD_LAST = ["Last year (Ordered product sales)", "Last year"]
    cur2, prev2, dbg2 = extract_compare_graph_view_money(rows, YTD_THIS, YTD_LAST)
    return cur2, prev2, "", {"file": str(fp), "source": "graph_view_money", **dbg2}


def get_mtd_for_client(client: str):
    folder = DATA_DIR / client / "Month"
    fp = latest_report_file(folder)
    if not fp:
        return None, None, "", {"reason": "No Month file found", "file": None}

    rows = read_csv_rows(fp)

    cur, prev, pct_str, dbg = extract_compare_table_view_through(rows, period_kind="mtd")
    if cur is not None and prev is not None:
        return cur, prev, pct_str, {"file": str(fp), "source": "table_view_through", **dbg}

    MTD_THIS = ["This month so far (Ordered product sales)", "This month so far", "This month"]
    MTD_LAST = ["Last month (Ordered product sales)", "Last month"]
    cur2, prev2, dbg2 = extract_compare_graph_view_money(rows, MTD_THIS, MTD_LAST)
    return cur2, prev2, "", {"file": str(fp), "source": "graph_view_money", **dbg2}


# =============================================================================
# WTD (last 6 weeks) — capped
# =============================================================================
def week_ending_sunday(d):
    days_to_sun = 6 - d.weekday()
    return d + pd.Timedelta(days=days_to_sun)


def extract_daily_sales_timeseries(rows):
    start = None
    for i, row in enumerate(rows):
        joined = " ".join(row).lower()
        if "compare sales" in joined and "graph view" in joined:
            start = i
            break
    if start is None:
        return None, {"reason": "Could not find 'Compare Sales - Graph view'"}

    header_i = None
    header = None
    col_time = col_this = col_last = None

    for j in range(start, min(start + 200, len(rows))):
        cand = rows[j]
        txt = " | ".join(cand).lower()
        if "time" in txt and "selected date range" in txt and "same date range" in txt:
            header_i = j
            header = cand
            col_time = find_col(header, "time")

            for idx, h in enumerate(header):
                hlow = str(h).lower()
                if "selected date range" in hlow and "ordered product sales" in hlow:
                    col_this = idx
                if "same date range one year ago" in hlow and "ordered product sales" in hlow:
                    col_last = idx

            if col_time is not None and col_this is not None and col_last is not None:
                break

    if header is None or col_time is None or col_this is None or col_last is None:
        return None, {"reason": "Could not find daily header row for Week report"}

    out = []
    for k in range(header_i + 1, min(header_i + 6000, len(rows))):
        r = rows[k]
        if not r or len(r) <= col_time:
            continue

        d = parse_date_from_any_text(r[col_time])
        if d is None:
            continue

        v_this = r[col_this] if col_this < len(r) else ""
        v_last = r[col_last] if col_last < len(r) else ""

        sales_this = parse_currency_only(v_this)
        sales_last = parse_currency_only(v_last)

        if sales_this is None and sales_last is None:
            continue

        out.append({"date": d, "sales_this": sales_this or 0.0, "sales_last": sales_last or 0.0})

    if not out:
        return None, {"reason": "No daily rows with sales values"}

    df = pd.DataFrame(out)
    return df, {"rows_extracted": len(df)}


def get_wtd_last_6_weeks_for_client(client: str, n_weeks=6, max_week_ending: Optional[date] = None):
    folder = DATA_DIR / client / "Week"
    fp = latest_report_file(folder)
    if not fp:
        return pd.DataFrame(columns=["Week of", "WTD 2026", "WTD 2025", "Year over Year"]), {
            "reason": "No Week file found",
            "file": None,
        }

    rows = read_csv_rows(fp)
    daily_df, dbg = extract_daily_sales_timeseries(rows)
    if daily_df is None or daily_df.empty:
        return pd.DataFrame(columns=["Week of", "WTD 2026", "WTD 2025", "Year over Year"]), {"file": str(fp), **dbg}

    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["week_of"] = daily_df["date"].apply(week_ending_sunday)

    weekly = (
        daily_df.groupby("week_of", as_index=False)[["sales_this", "sales_last"]]
        .sum()
        .sort_values("week_of", ascending=False)
    )

    if max_week_ending is not None:
        cap = pd.to_datetime(max_week_ending)
        weekly = weekly[weekly["week_of"] <= cap]

    weekly = weekly.head(n_weeks)

    out = []
    for _, r in weekly.iterrows():
        this_val = float(r["sales_this"])
        last_val = float(r["sales_last"])
        yoy = ""
        if last_val != 0:
            yoy = f"{((this_val - last_val) / last_val):.0%}"

        out.append(
            {
                "Week of": pd.to_datetime(r["week_of"]).strftime("%d-%b"),
                "WTD 2026": fmt_currency(this_val),
                "WTD 2025": fmt_currency(last_val),
                "Year over Year": yoy,
            }
        )

    return pd.DataFrame(out), {"file": str(fp), **dbg}


# =============================================================================
# DETAIL SALES & TRAFFIC (Excel) — used by Listing Metrics + Client Summary table
# =============================================================================
def _clean_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip())


def _to_percent_0to1(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        if s.endswith("%"):
            return float(s.replace("%", "").strip()) / 100.0
        v = float(s.replace(",", ""))
        if 0 <= v <= 1:
            return v
        if 1 < v <= 100:
            return v / 100.0
        return None
    except:
        return None


def _to_int(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return 0
    try:
        return int(float(s.replace(",", "")))
    except:
        return 0


def _find_detail_folder(client_folder: Path) -> Optional[Path]:
    """
    Exact folder name (per GitHub screenshot):
    data/.../<CLIENT>/Detail Sales & Traffic/
    """
    p = client_folder / DETAIL_SALES_TRAFFIC_FOLDER
    return p if p.exists() and p.is_dir() else None


def _read_detail_sales_traffic_raw(client: str) -> pd.DataFrame:
    client_folder = DATA_DIR / client
    detail_folder = _find_detail_folder(client_folder)
    if not detail_folder:
        return pd.DataFrame()

    fp = latest_report_file(detail_folder)
    if not fp:
        return pd.DataFrame()

    try:
        df = pd.read_excel(fp, sheet_name=0, header=0)
    except Exception as e:
        st.error(f"❌ Failed to read Detail Sales & Traffic Excel for {client}: {fp.name}")
        st.code(str(e))
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df.columns = [_clean_col(c) for c in df.columns]
    return df


def get_detail_metrics_daily(client: str) -> pd.DataFrame:
    df = _read_detail_sales_traffic_raw(client)
    if df.empty:
        return pd.DataFrame()

    col_date = "Date" if "Date" in df.columns else None
    col_asin = "(Parent) ASIN" if "(Parent) ASIN" in df.columns else ("ASIN" if "ASIN" in df.columns else None)

    def find_first(substrings):
        for c in df.columns:
            cl = c.lower()
            if any(s in cl for s in substrings):
                return c
        return None

    col_pageviews = find_first(["page views - total", "page views total", "pageviews - total", "page views", "pageviews"])
    col_order_items = find_first(["total order items", "order items"])
    col_feat = find_first(["featured offer", "buy box"])

    if not (col_date and col_asin and col_pageviews and col_order_items):
        return pd.DataFrame()

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[col_date], errors="coerce")
    out["asin"] = df[col_asin].astype(str).str.strip().str.upper()
    out = out.dropna(subset=["date"])
    out = out[out["asin"].str.match(r"^[A-Z0-9]{8,14}$", na=False)]

    out["total_order_items"] = df.loc[out.index, col_order_items].apply(_to_int)
    out["pageviews_total"] = df.loc[out.index, col_pageviews].apply(_to_int)

    if col_feat is not None:
        out["featured_offer_pct"] = df.loc[out.index, col_feat].apply(_to_percent_0to1)
    else:
        out["featured_offer_pct"] = None

    return out


# =============================================================================
# ✅ DETAIL SALES & TRAFFIC TABLE (Client Summary) — screenshot-style
# =============================================================================
def _find_first_col(df: pd.DataFrame, substrings: List[str]) -> Optional[str]:
    for c in df.columns:
        cl = str(c).lower()
        if any(s in cl for s in substrings):
            return c
    return None


def get_detail_sales_traffic_table(client: str) -> pd.DataFrame:
    df = _read_detail_sales_traffic_raw(client)
    if df.empty:
        return pd.DataFrame()

    col_date = "Date" if "Date" in df.columns else _find_first_col(df, ["date"])
    col_asin = "(Parent) ASIN" if "(Parent) ASIN" in df.columns else ("ASIN" if "ASIN" in df.columns else _find_first_col(df, ["asin"]))

    col_sessions = _find_first_col(df, ["session total", "sessions"])
    col_pageviews = _find_first_col(df, ["page views - total", "page views total", "pageviews - total", "page views", "pageviews"])
    col_feat = _find_first_col(df, ["featured offer", "buy box"])
    col_units = _find_first_col(df, ["units ordered", "units"])
    col_unit_session = _find_first_col(df, ["unit session", "unit session percentage"])
    col_sales = _find_first_col(df, ["ordered product sales", "product sales"])
    col_order_items = _find_first_col(df, ["total order items", "order items"])

    if not (col_date and col_asin):
        return pd.DataFrame()

    out = pd.DataFrame()
    out["ASIN"] = df[col_asin].astype(str).str.strip().str.upper()
    out["Date"] = pd.to_datetime(df[col_date], errors="coerce")
    out = out.dropna(subset=["Date"])
    out = out[out["ASIN"].str.match(r"^[A-Z0-9]{8,14}$", na=False)]

    # Date like 7-Dec
    try:
        out["Date"] = out["Date"].dt.strftime("%-d-%b")
    except Exception:
        out["Date"] = out["Date"].dt.strftime("%#d-%b")

    out["Session Total"] = df.loc[out.index, col_sessions].apply(_to_int) if col_sessions else 0
    out["Page Views| Total"] = df.loc[out.index, col_pageviews].apply(_to_int) if col_pageviews else 0
    out["Units Ordered"] = df.loc[out.index, col_units].apply(_to_int) if col_units else 0
    out["Total Order Items"] = df.loc[out.index, col_order_items].apply(_to_int) if col_order_items else 0

    out["Featured Offer Percentage"] = (
        df.loc[out.index, col_feat].apply(_to_percent_0to1).apply(fmt_pct0) if col_feat else ""
    )
    out["Unit Session Percentage"] = (
        df.loc[out.index, col_unit_session].apply(_to_percent_0to1).apply(fmt_pct0) if col_unit_session else ""
    )

    if col_sales:
        s = df.loc[out.index, col_sales].apply(parse_currency_only)
        out["Ordered Product Sales"] = s.fillna(0).map(lambda x: f"${x:,.0f}")
    else:
        out["Ordered Product Sales"] = "$0"

    out = out[
        [
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
    ].sort_values(["ASIN", "Date"], ascending=True).reset_index(drop=True)

    return out


def build_asin_grouped_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    display_rows = []
    for asin, g in df.groupby("ASIN", dropna=False):
        header = {c: "" for c in df.columns}
        header["ASIN"] = str(asin)
        header["__row_type"] = "asin"
        display_rows.append(header)

        for _, r in g.iterrows():
            row = r.to_dict()
            row["ASIN"] = f"   {row['Date']}"  # date line under ASIN
            row["Date"] = ""                  # hide Date col like screenshot
            row["__row_type"] = "data"
            display_rows.append(row)

    out = pd.DataFrame(display_rows)
    cols = [c for c in out.columns if c != "__row_type"] + ["__row_type"]
    return out[cols]


def debug_detail_sales_traffic(client: str):
    with st.expander("Data source (latest file detected)", expanded=False):
        client_folder = DATA_DIR / client
        detail_folder = client_folder / DETAIL_SALES_TRAFFIC_FOLDER
        st.write("BASE_DIR:", str(BASE_DIR.resolve()))
        st.write("DATA_DIR:", str(DATA_DIR.resolve()))
        st.write("Client folder exists:", client_folder.exists(), str(client_folder))
        st.write("Detail folder exists:", detail_folder.exists(), str(detail_folder))

        fp = latest_report_file(detail_folder) if detail_folder.exists() else None
        st.write("Latest file:", fp.name if fp else "None")


def render_detail_sales_traffic_for_client(client: str):
    df = get_detail_sales_traffic_table(client)
    if df.empty:
        st.info("No Detail Sales & Traffic report found for this client yet.")
        return

    display = build_asin_grouped_display(df)

    def style_rows(row):
        if row.get("__row_type") == "asin":
            return [
                "font-weight: 700; border-top: 2px solid #2f5f78; background: #ffffff;"
                if c != "__row_type"
                else ""
                for c in row.index
            ]
        return ["" for _ in row.index]

    styler = (
        display.style
        .apply(style_rows, axis=1)
        .set_properties(subset=["ASIN"], **{"white-space": "pre"})
        .hide(columns=["__row_type"])
    )

    st.dataframe(styler, use_container_width=True, hide_index=True)


# =============================================================================
# LISTING METRICS: SUMMARY + BREAKDOWN
# =============================================================================
def _pct_drop(cur_v, prev_v) -> Optional[float]:
    if prev_v is None or prev_v == 0:
        return None
    return (prev_v - cur_v) / prev_v


def compute_listing_metrics_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    today = datetime.now().date()
    end_cur = today
    start_cur = today - timedelta(days=6)
    end_prev = start_cur - timedelta(days=1)
    start_prev = end_prev - timedelta(days=6)

    breakdown_rows = []

    for client in CLIENTS:
        daily = get_detail_metrics_daily(client)
        if daily.empty:
            continue

        daily["date"] = pd.to_datetime(daily["date"]).dt.date

        cur = daily[(daily["date"] >= start_cur) & (daily["date"] <= end_cur)]
        prev = daily[(daily["date"] >= start_prev) & (daily["date"] <= end_prev)]
        if cur.empty or prev.empty:
            continue

        cur_agg = cur.groupby("asin", as_index=False).agg(
            cur_sales=("total_order_items", "sum"),
            cur_traffic=("pageviews_total", "sum"),
            cur_bb_x_pv=("featured_offer_pct", lambda s: float(
                pd.to_numeric(s, errors="coerce").fillna(0.0).values
                @ cur.loc[s.index, "pageviews_total"].astype(float).values
            ) if len(s.index) else 0.0),
            cur_pv_for_bb=("featured_offer_pct", lambda s: float(
                cur.loc[s.index, "pageviews_total"].astype(float).values.sum()
            ) if len(s.index) else 0.0),
        )
        prev_agg = prev.groupby("asin", as_index=False).agg(
            prev_sales=("total_order_items", "sum"),
            prev_traffic=("pageviews_total", "sum"),
        )

        merged = cur_agg.merge(prev_agg, on="asin", how="inner")

        merged["cur_buybox_pct"] = merged.apply(
            lambda r: (r["cur_bb_x_pv"] / r["cur_pv_for_bb"])
            if r["cur_pv_for_bb"] and r["cur_pv_for_bb"] > 0
            else None,
            axis=1,
        )

        for _, r in merged.iterrows():
            asin = str(r["asin"]).strip().upper()

            cur_sales = int(r["cur_sales"])
            prev_sales = int(r["prev_sales"])
            sd = _pct_drop(cur_sales, prev_sales)

            cur_tr = int(r["cur_traffic"])
            prev_tr = int(r["prev_traffic"])
            td = _pct_drop(cur_tr, prev_tr)

            bb = r["cur_buybox_pct"]
            bb_pct_str = f"{bb*100:.0f}%" if bb is not None else "N/A"

            if sd is not None and sd >= SALES_DROP_PCT_TRIGGER:
                info = TRIGGER_LOOKUP["Sales Drop"]
                breakdown_rows.append(
                    {
                        "Client": client,
                        "ASIN": asin,
                        "Trigger": "Sales Drop",
                        "Trigger Data": f"Cur: {cur_sales} | Prev: {prev_sales} | Drop: -{sd:.0%}",
                        "Likely Cause": info["Likely Cause"],
                        "Team": info["Team"],
                        "Response": info["Response"],
                    }
                )

            if td is not None and td >= TRAFFIC_DROP_PCT_TRIGGER:
                info = TRIGGER_LOOKUP["Traffic Drop"]
                breakdown_rows.append(
                    {
                        "Client": client,
                        "ASIN": asin,
                        "Trigger": "Traffic Drop",
                        "Trigger Data": f"Cur: {cur_tr} | Prev: {prev_tr} | Drop: -{td:.0%}",
                        "Likely Cause": info["Likely Cause"],
                        "Team": info["Team"],
                        "Response": info["Response"],
                    }
                )

            if bb is not None and bb < BUYBOX_MIN_THRESHOLD:
                info = TRIGGER_LOOKUP["Buy Box Loss"]
                breakdown_rows.append(
                    {
                        "Client": client,
                        "ASIN": asin,
                        "Trigger": "Buy Box Loss",
                        "Trigger Data": f"Cur Buy Box: {bb_pct_str} (threshold < {BUYBOX_MIN_THRESHOLD*100:.0f}%)",
                        "Likely Cause": info["Likely Cause"],
                        "Team": info["Team"],
                        "Response": info["Response"],
                    }
                )

    breakdown = pd.DataFrame(breakdown_rows)
    if breakdown.empty:
        summary = pd.DataFrame(columns=["Client", "# of triggers", "Type of triggers"])
        breakdown_out = pd.DataFrame(columns=["Client", "ASIN", "Trigger", "Trigger Data", "Likely Cause", "Team", "Response"])
        return summary, breakdown_out

    grp = breakdown.groupby("Client")["Trigger"].agg(list).reset_index()
    grp["# of triggers"] = grp["Trigger"].apply(len)
    grp["Type of triggers"] = grp["Trigger"].apply(lambda xs: ", ".join(sorted(set(xs))))
    summary = grp[["Client", "# of triggers", "Type of triggers"]].sort_values("# of triggers", ascending=False).reset_index(drop=True)

    breakdown_out = breakdown[["Client", "ASIN", "Trigger", "Trigger Data", "Likely Cause", "Team", "Response"]].copy()
    breakdown_out = breakdown_out.sort_values(["Client", "Trigger", "ASIN"]).reset_index(drop=True)

    return summary, breakdown_out


def render_listing_metrics_tracker():
    st.header("Listing Metrics Tracker")
    st.caption("Pulls from Detail Sales & Traffic. If a trigger fires, Likely Cause / Team / Response auto-fills from your table.")

    summary_df, breakdown_df = compute_listing_metrics_tables()

    st.subheader("Summary")
    if summary_df.empty:
        st.info("No triggers found (or missing Detail Sales & Traffic files for the date windows).")
        return
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Breakdown")
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Breakdown (CSV)",
        data=breakdown_df.to_csv(index=False).encode("utf-8"),
        file_name=f"listing_metrics_breakdown_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =============================================================================
# SELLER HEALTH
# =============================================================================
def _find_possible_seller_health_folders(client_folder: Path) -> List[Path]:
    if not client_folder.exists():
        return []

    hits = []
    for p in client_folder.iterdir():
        if not p.is_dir():
            continue
        nm = p.name.lower()
        if (
            ("seller" in nm and "health" in nm)
            or ("account" in nm and "health" in nm)
            or ("suppressed" in nm)
            or ("feedback" in nm)
        ):
            hits.append(p)
    return hits


def _read_any_table(fp: Path) -> pd.DataFrame:
    try:
        if fp.suffix.lower() == ".csv":
            return pd.read_csv(fp, encoding="utf-8-sig")
        return pd.read_excel(fp, sheet_name=0)
    except Exception as e:
        st.error(f"❌ Failed to read file: {fp.name}")
        st.code(str(e))
        return pd.DataFrame()


def get_seller_health_history_for_client(client: str) -> pd.DataFrame:
    client_folder = DATA_DIR / client
    folders = _find_possible_seller_health_folders(client_folder)
    if not folders:
        return pd.DataFrame(columns=["Date", "Account Health", "Suppressed listings", "Seller Feedback"])

    all_files = []
    for f in folders:
        all_files.extend(list_report_files(f))  # ✅ now recursive

    if not all_files:
        return pd.DataFrame(columns=["Date", "Account Health", "Suppressed listings", "Seller Feedback"])

    rows = []

    for fp in all_files:
        df = _read_any_table(fp)
        if df.empty:
            continue

        df.columns = [_clean_col(c) for c in df.columns]

        col_client = col_week = col_ah = col_sup = col_fb = None
        for ccol in df.columns:
            cl = str(ccol).lower().strip()
            if cl == "client":
                col_client = ccol
            elif "week" in cl or cl == "date":
                col_week = ccol
            elif "account health" in cl:
                col_ah = ccol
            elif "suppressed" in cl:
                col_sup = ccol
            elif "feedback" in cl:
                col_fb = ccol

        if any(x is None for x in [col_client, col_week, col_ah, col_sup, col_fb]):
            continue

        sub = df[df[col_client].astype(str).str.strip().str.upper() == client.strip().upper()]
        if sub.empty:
            continue

        def _safe_int(v):
            try:
                if pd.isna(v):
                    return 0
                return int(float(str(v).replace(",", "").strip()))
            except:
                return 0

        for _, r in sub.iterrows():
            wk_date = parse_date_from_any_text(str(r[col_week]).strip())
            if wk_date is None:
                continue

            rows.append(
                {
                    "Date": wk_date,
                    "Account Health": _safe_int(r[col_ah]),
                    "Suppressed listings": _safe_int(r[col_sup]),
                    "Seller Feedback": _safe_int(r[col_fb]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Date", "Account Health", "Suppressed listings", "Seller Feedback"])

    out = pd.DataFrame(rows)
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date", ascending=True).drop_duplicates(subset=["Date"]).reset_index(drop=True)
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out


def debug_seller_health(client: str):
    with st.expander("Seller Health debug (folders/files found)", expanded=False):
        client_folder = DATA_DIR / client
        st.write("Client folder:", str(client_folder))
        st.write("Exists:", client_folder.exists())

        folders = _find_possible_seller_health_folders(client_folder)
        st.write("Matched folders:", [f.name for f in folders])

        for f in folders:
            files = _iter_report_files(f)
            st.write(f"Files in {f.name}:", len(files))
            if files:
                newest = max(files, key=lambda p: p.stat().st_mtime)
                st.write("Newest:", newest.name)


# =============================================================================
# PAGES
# =============================================================================
def render_client_summary():
    if LOGO_PATH.exists():
        left, mid, right = st.columns([2, 1, 2])
        with mid:
            st.image(str(LOGO_PATH), width=180)
        st.write("")

    st.header("Client Summary")

    st.subheader("YEAR TO DATE (Ordered Product Sales)")
    ytd_rows = []
    for c in CLIENTS:
        cur, prev, pct_from_report, _ = get_ytd_for_client(c)
        diff, fallback_pct = diff_pct(cur, prev)
        ytd_rows.append(
            {
                "Client": c,
                "This year so far": fmt_currency(cur),
                "Last year": fmt_currency(prev),
                "Difference": fmt_diff(diff),
                "Difference %": fmt_pct0(pct_from_report if pct_from_report else fallback_pct),
            }
        )
    ytd_df = pd.DataFrame(ytd_rows)
    st.dataframe(style_percent_columns(ytd_df, ["Difference %"]), use_container_width=True)

    st.divider()

    st.subheader("MONTH TO DATE (Ordered Product Sales)")
    mtd_rows = []
    for c in CLIENTS:
        cur, prev, pct_from_report, _ = get_mtd_for_client(c)
        diff, fallback_pct = diff_pct(cur, prev)
        mtd_rows.append(
            {
                "Client": c,
                "This month so far": fmt_currency(cur),
                "Last month": fmt_currency(prev),
                "Difference": fmt_diff(diff),
                "Difference %": fmt_pct0(pct_from_report if pct_from_report else fallback_pct),
            }
        )
    mtd_df = pd.DataFrame(mtd_rows)
    st.dataframe(style_percent_columns(mtd_df, ["Difference %"]), use_container_width=True)

    st.divider()

    st.subheader("WEEKLY BREAKDOWN (Last 6 Weeks) — capped at Feb 1")
    for c in CLIENTS:
        st.markdown(f"### {c}")
        wtd_df, _ = get_wtd_last_6_weeks_for_client(c, n_weeks=6, max_week_ending=WTD_MAX_WEEK_ENDING)
        if wtd_df.empty:
            st.info("No Week report found yet.")
        else:
            st.dataframe(style_percent_columns(wtd_df, ["Year over Year"]), use_container_width=True)
        st.write("")

    st.divider()
    st.header("DETAIL SALES & TRAFFIC")

    client_for_detail = st.selectbox(
        "Select client for Detail Sales & Traffic",
        CLIENTS,
        index=0,
        key="detail_client_summary",
    )

    debug_detail_sales_traffic(client_for_detail)
    render_detail_sales_traffic_for_client(client_for_detail)


def render_client_pages():
    st.header("Client Pages")
    client = st.selectbox("Select a client", CLIENTS, index=0)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("YTD (Ordered Product Sales)")
        cur, prev, pct_from_report, _ = get_ytd_for_client(client)
        diff, fallback_pct = diff_pct(cur, prev)
        st.metric("This year so far", fmt_currency(cur), fmt_diff(diff))
        st.caption(
            f"Last year: {fmt_currency(prev)} | Diff %: {fmt_pct0(pct_from_report if pct_from_report else fallback_pct)}"
        )

    with c2:
        st.subheader("MTD (Ordered Product Sales)")
        cur, prev, pct_from_report, _ = get_mtd_for_client(client)
        diff, fallback_pct = diff_pct(cur, prev)
        st.metric("This month so far", fmt_currency(cur), fmt_diff(diff))
        st.caption(
            f"Last month: {fmt_currency(prev)} | Diff %: {fmt_pct0(pct_from_report if pct_from_report else fallback_pct)}"
        )

    st.divider()

    st.subheader("WTD (Last 6 Weeks) — capped at Feb 1")
    wtd_df, _ = get_wtd_last_6_weeks_for_client(client, n_weeks=6, max_week_ending=WTD_MAX_WEEK_ENDING)
    if wtd_df.empty:
        st.info("No Week report found yet.")
    else:
        st.dataframe(style_percent_columns(wtd_df, ["Year over Year"]), use_container_width=True)


def render_seller_health():
    st.header("Seller Health")
    client = st.selectbox("Client", CLIENTS, index=0)

    # ✅ Debug expander to verify detection
    debug_seller_health(client)

    hist = get_seller_health_history_for_client(client)
    if hist.empty:
        st.warning(
            "No Seller Health table found for this client.\n\n"
            "This searches folders that contain: 'seller health', 'account health', 'suppressed', or 'feedback'."
        )
        return

    st.dataframe(hist, use_container_width=True, hide_index=True)


# =============================================================================
# STATIC HTML DASHBOARD (tabs, no filters)
# =============================================================================
def _html_escape(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def _pct_cell_class(val: str) -> str:
    p = _pct_to_float(val)
    if p is None:
        return ""
    if p < 0:
        return "neg"
    if p > 0:
        return "pos"
    return ""


def _df_to_html_table(df: pd.DataFrame, pct_cols: Optional[List[str]] = None) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No data.</p>"

    pct_cols = pct_cols or []
    cols = df.columns.tolist()
    thead = "<tr>" + "".join(f"<th>{_html_escape(c)}</th>" for c in cols) + "</tr>"

    rows_html = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            raw = r[c]
            cls = _pct_cell_class(raw) if c in pct_cols else ""
            cls_attr = f" class='{cls}'" if cls else ""
            tds.append(f"<td{cls_attr}>{_html_escape(raw)}</td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    tbody = "\n".join(rows_html)
    return f"""
    <table class="tbl">
      <thead>{thead}</thead>
      <tbody>{tbody}</tbody>
    </table>
    """


def build_static_clickable_dashboard_html() -> str:
    ytd_rows = []
    for c in CLIENTS:
        cur, prev, pct_from_report, _ = get_ytd_for_client(c)
        diff, fallback_pct = diff_pct(cur, prev)
        ytd_rows.append(
            {
                "Client": c,
                "This year so far": fmt_currency(cur),
                "Last year": fmt_currency(prev),
                "Difference": fmt_diff(diff),
                "Difference %": fmt_pct0(pct_from_report if pct_from_report else fallback_pct),
            }
        )
    ytd_df = pd.DataFrame(ytd_rows)

    mtd_rows = []
    for c in CLIENTS:
        cur, prev, pct_from_report, _ = get_mtd_for_client(c)
        diff, fallback_pct = diff_pct(cur, prev)
        mtd_rows.append(
            {
                "Client": c,
                "This month so far": fmt_currency(cur),
                "Last month": fmt_currency(prev),
                "Difference": fmt_diff(diff),
                "Difference %": fmt_pct0(pct_from_report if pct_from_report else fallback_pct),
            }
        )
    mtd_df = pd.DataFrame(mtd_rows)

    weekly_blocks = []
    for c in CLIENTS:
        wtd_df, _ = get_wtd_last_6_weeks_for_client(c, n_weeks=6, max_week_ending=WTD_MAX_WEEK_ENDING)
        weekly_blocks.append(
            f"""
          <div class="card">
            <h3>{_html_escape(c)}</h3>
            {_df_to_html_table(wtd_df, pct_cols=["Year over Year"]) if not wtd_df.empty else "<p class='muted'>No Week report found yet.</p>"}
          </div>
        """
        )

    summary_view = f"""
      <div class="card">
        <h2>Client Summary</h2>
        <h3>YEAR TO DATE (Ordered Product Sales)</h3>
        {_df_to_html_table(ytd_df, pct_cols=["Difference %"])}
        <h3>MONTH TO DATE (Ordered Product Sales)</h3>
        {_df_to_html_table(mtd_df, pct_cols=["Difference %"])}
      </div>
      <h2>Weekly Breakdown (Last 6 Weeks) — capped at Feb 1</h2>
      {''.join(weekly_blocks)}
    """

    lm_summary, lm_breakdown = compute_listing_metrics_tables()
    listing_view = f"""
      <div class="card">
        <h2>Listing Metrics Tracker</h2>
        <p class="muted">Static snapshot. No filters.</p>
        <h3>Summary</h3>
        {_df_to_html_table(lm_summary)}
        <h3>Breakdown</h3>
        {_df_to_html_table(lm_breakdown)}
      </div>
    """

    seller_blocks = []
    for c in CLIENTS:
        hist = get_seller_health_history_for_client(c)
        seller_blocks.append(
            f"""
          <div class="card">
            <h2>{_html_escape(c)}</h2>
            {_df_to_html_table(hist)}
          </div>
        """
        )
    seller_view = f"""
      <h2>Seller Health</h2>
      <p class="muted">Static snapshot. No filters.</p>
      {''.join(seller_blocks)}
    """

    pages_blocks = []
    for c in CLIENTS:
        cur, prev, pct_from_report, _ = get_ytd_for_client(c)
        diff, fallback_pct = diff_pct(cur, prev)
        ytd_small = pd.DataFrame(
            [
                {
                    "This year so far": fmt_currency(cur),
                    "Last year": fmt_currency(prev),
                    "Difference": fmt_diff(diff),
                    "Diff %": fmt_pct0(pct_from_report if pct_from_report else fallback_pct),
                }
            ]
        )

        cur, prev, pct_from_report, _ = get_mtd_for_client(c)
        diff, fallback_pct = diff_pct(cur, prev)
        mtd_small = pd.DataFrame(
            [
                {
                    "This month so far": fmt_currency(cur),
                    "Last month": fmt_currency(prev),
                    "Difference": fmt_diff(diff),
                    "Diff %": fmt_pct0(pct_from_report if pct_from_report else fallback_pct),
                }
            ]
        )

        wtd_df, _ = get_wtd_last_6_weeks_for_client(c, n_weeks=6, max_week_ending=WTD_MAX_WEEK_ENDING)

        pages_blocks.append(
            f"""
          <div class="card">
            <h2>Client Page: {_html_escape(c)}</h2>
            <div class="two-col">
              <div>
                <h3>YTD</h3>
                {_df_to_html_table(ytd_small, pct_cols=["Diff %"])}
              </div>
              <div>
                <h3>MTD</h3>
                {_df_to_html_table(mtd_small, pct_cols=["Diff %"])}
              </div>
            </div>
            <h3>WTD (Last 6 Weeks)</h3>
            {_df_to_html_table(wtd_df, pct_cols=["Year over Year"]) if not wtd_df.empty else "<p class='muted'>No Week report found yet.</p>"}
          </div>
        """
        )

    pages_view = f"""
      <h2>Client Pages (All Clients)</h2>
      {''.join(pages_blocks)}
    """

    logo_html = ""
    if LOGO_PATH.exists():
        logo_html = f"<img class='logo' src='{_html_escape(str(LOGO_PATH))}' />"

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Static Client Dashboard</title>
  <style>
    * {{
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
    }}
    body {{
      font-family: Arial, Helvetica, sans-serif;
      margin: 0;
      background: #f3f4f6;
      color: #111827;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      background: #111827;
      color: #fff;
      padding: 14px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 999;
    }}
    .topbar .title {{
      font-weight: 800;
      font-size: 14px;
      letter-spacing: .2px;
    }}
    .nav a {{
      color: #fff;
      text-decoration: none;
      margin-left: 14px;
      font-weight: 700;
      font-size: 12px;
      opacity: .9;
    }}
    .nav a.active {{
      text-decoration: underline;
      opacity: 1;
    }}
    .wrap {{
      padding: 16px 18px 28px 18px;
      max-width: 1250px;
      margin: 0 auto;
    }}
    .logo {{
      width: 170px;
      display: block;
      margin: 8px auto 12px auto;
    }}
    .meta {{
      font-size: 12px;
      color: #374151;
      margin-bottom: 10px;
      text-align: center;
    }}
    .view {{
      display: none;
    }}
    .view.active {{
      display: block;
    }}
    .card {{
      background: #ffffff;
      border-radius: 12px;
      padding: 14px 14px;
      margin: 12px 0;
      box-shadow: 0 2px 10px rgba(0,0,0,.06);
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    h2 {{
      margin: 14px 0 8px 0;
      font-size: 18px;
    }}
    h3 {{
      margin: 14px 0 8px 0;
      font-size: 14px;
    }}
    .muted {{
      color: #6b7280;
      font-size: 12px;
    }}
    .tbl {{
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
      margin: 8px 0 14px 0;
    }}
    .tbl th {{
      background: #111827;
      color: #fff;
      text-align: left;
      padding: 8px;
      font-weight: 700;
      border: 1px solid #e5e7eb;
    }}
    .tbl td {{
      padding: 7px 8px;
      border: 1px solid #e5e7eb;
      vertical-align: top;
    }}
    .tbl tr:nth-child(even) td {{
      background: #f9fafb;
    }}
    .tbl td.pos {{
      color: #15803d;
      font-weight: 700;
    }}
    .tbl td.neg {{
      color: #b91c1c;
      font-weight: 700;
    }}
    @media print {{
      body {{ background: #fff; }}
      .topbar {{ position: static; }}
      .wrap {{ max-width: none; }}
      .card {{ box-shadow: none; border: 1px solid #e5e7eb; }}
    }}
  </style>
</head>
<body>

  <div class="topbar">
    <div class="title">Static Client Dashboard (No Filters)</div>
    <div class="nav">
      <a href="#summary" id="nav-summary" onclick="showView('summary'); return false;">Client Summary</a>
      <a href="#pages" id="nav-pages" onclick="showView('pages'); return false;">Client Pages</a>
      <a href="#listing" id="nav-listing" onclick="showView('listing'); return false;">Listing Metrics</a>
      <a href="#seller" id="nav-seller" onclick="showView('seller'); return false;">Seller Health</a>
    </div>
  </div>

  <div class="wrap">
    {logo_html}
    <div class="meta">Generated: {_html_escape(datetime.now().strftime("%Y-%m-%d %H:%M"))}</div>

    <div id="view-summary" class="view active">{summary_view}</div>
    <div id="view-pages" class="view">{pages_view}</div>
    <div id="view-listing" class="view">{listing_view}</div>
    <div id="view-seller" class="view">{seller_view}</div>
  </div>

<script>
  function clearNav() {{
    document.getElementById('nav-summary').classList.remove('active');
    document.getElementById('nav-pages').classList.remove('active');
    document.getElementById('nav-listing').classList.remove('active');
    document.getElementById('nav-seller').classList.remove('active');
  }}

  function showView(which) {{
    document.getElementById('view-summary').classList.remove('active');
    document.getElementById('view-pages').classList.remove('active');
    document.getElementById('view-listing').classList.remove('active');
    document.getElementById('view-seller').classList.remove('active');

    document.getElementById('view-' + which).classList.add('active');

    clearNav();
    document.getElementById('nav-' + which).classList.add('active');
    window.location.hash = '#' + which;
  }}

  (function() {{
    var h = (window.location.hash || '').replace('#','');
    if (h === 'pages') showView('pages');
    else if (h === 'listing') showView('listing');
    else if (h === 'seller') showView('seller');
    else showView('summary');
  }})();
</script>

</body>
</html>
    """.strip()

    return html


# =============================================================================
# SIDEBAR NAV + EXPORTS
# =============================================================================
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Client Summary", "Client Pages", "Listing Metrics Tracker", "Seller Health"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Export")

static_html = build_static_clickable_dashboard_html()
st.sidebar.download_button(
    label="🌐 Download: Static HTML Dashboard (Tabs, No Filters)",
    data=static_html.encode("utf-8"),
    file_name=f"static_dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
    mime="text/html",
    use_container_width=True,
)

# =============================================================================
# RENDER
# =============================================================================
if page == "Client Summary":
    render_client_summary()
elif page == "Client Pages":
    render_client_pages()
elif page == "Listing Metrics Tracker":
    render_listing_metrics_tracker()
else:
    render_seller_health()
