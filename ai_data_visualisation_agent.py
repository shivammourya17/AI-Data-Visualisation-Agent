import io
import csv
import json
import hashlib
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Data Visualization",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Minimal, subtle UI helpers (no heavy theming)
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body { scroll-behavior: smooth; }
    .badge { display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2ff; color:#3730a3; font-weight:600; font-size:0.8rem; }
    .muted { color:#64748b; font-size:0.9rem; }
    .soft { background:#ffffff; border:1px solid #f1f5f9; border-radius:10px; padding:0.75rem 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Helpers and safe defaults
# ------------------------------------------------------------
CHAT_AVAILABLE = hasattr(st, "chat_message") and hasattr(st, "chat_input")


def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "-"
    try:
        size = float(n)
    except Exception:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    for u in units:
        if size < 1024.0:
            return f"{size:3.1f} {u}"
        size /= 1024.0
    return f"{size:.1f} PB"


def detect_delimiter(sample_bytes: bytes, default: str = ",") -> str:
    try:
        sample = sample_bytes[:20000].decode("utf-8", errors="ignore")
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return default


@st.cache_data(show_spinner=False)
def load_csv_cached(
    file_bytes: bytes,
    sep: str,
    encoding: str,
    header: bool,
    parse_dates: bool,
    nrows: Optional[int],
    on_bad_lines: str,
    low_memory: bool,
):
    """Robust CSV loader with fallback for pandas arg differences."""
    if not file_bytes:
        return None
    bio = io.BytesIO(file_bytes)
    base_kwargs = dict(
        sep=sep,
        encoding=encoding,
        low_memory=low_memory,
    )
    base_kwargs["header"] = 0 if header else None
    if parse_dates:
        base_kwargs["infer_datetime_format"] = True
        base_kwargs["dayfirst"] = False
    if nrows is not None:
        base_kwargs["nrows"] = nrows

    # First, try with on_bad_lines if supported
    try:
        df = pd.read_csv(bio, on_bad_lines=on_bad_lines, **base_kwargs)
        return df
    except TypeError:
        # Older pandas may not support on_bad_lines
        pass
    except UnicodeDecodeError:
        # Encoding fallback
        bio.seek(0)
        try:
            df = pd.read_csv(
                bio,
                on_bad_lines=on_bad_lines,
                encoding="latin-1",
                **{k: v for k, v in base_kwargs.items() if k != "encoding"},
            )
            return df
        except Exception:
            pass
    except Exception:
        # Fall through to try without on_bad_lines
        pass

    # Retry without on_bad_lines
    bio.seek(0)
    try:
        df = pd.read_csv(bio, **base_kwargs)
        return df
    except UnicodeDecodeError:
        bio.seek(0)
        df = pd.read_csv(
            bio,
            encoding="latin-1",
            **{k: v for k, v in base_kwargs.items() if k != "encoding"},
        )
        return df


def df_memory_bytes(df: pd.DataFrame) -> int:
    try:
        return int(df.memory_usage(deep=True).sum())
    except Exception:
        return 0


def numeric_columns(df: pd.DataFrame) -> list:
    try:
        return list(df.select_dtypes(include=[np.number]).columns)
    except Exception:
        return []


def categorical_columns(df: pd.DataFrame) -> list:
    try:
        return list(df.select_dtypes(exclude=[np.number]).columns)
    except Exception:
        return []


def df_key_from_bytes(file_bytes: bytes, opts: dict) -> str:
    h = hashlib.sha256()
    h.update(file_bytes)
    h.update(json.dumps(opts, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def compute_summary_cached(df: pd.DataFrame, key: str) -> dict:
    """Compute robust summary; always return safe structures."""
    if df is None or len(df) == 0:
        return {
            "meta": {
                "rows": 0,
                "cols": 0,
                "num_cols": 0,
                "cat_cols": 0,
                "missing_cells": 0,
                "memory": 0,
                "num_col_names": [],
                "cat_col_names": [],
            },
            "desc_all": pd.DataFrame(),
            "desc_num": pd.DataFrame(),
            "missing": pd.DataFrame(),
            "dtypes": pd.DataFrame(),
            "corr": pd.DataFrame(),
            "top_corr_pairs": [],
        }

    num_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)

    # Describe (all) can fail for some mixed objects; guard it
    try:
        desc_all = df.describe(include='all', datetime_is_numeric=True).transpose()
    except Exception:
        desc_all = pd.DataFrame()

    # Describe (numeric)
    try:
        desc_num = df.select_dtypes(include=[np.number]).describe().transpose()
    except Exception:
        desc_num = pd.DataFrame()

    # Missing and dtypes
    try:
        missing = df.isna().sum().to_frame("missing").sort_values("missing", ascending=False)
    except Exception:
        missing = pd.DataFrame()

    try:
        dtypes = df.dtypes.astype(str).to_frame("dtype")
    except Exception:
        dtypes = pd.DataFrame()

    # Correlation (cap number of columns for speed and stability)
    top_corr_pairs = []
    corr = pd.DataFrame()
    corr_cols = num_cols[:30]
    if len(corr_cols) >= 2:
        try:
            corr = df[corr_cols].corr()
            c = corr.where(~np.eye(corr.shape[0], dtype=bool))
            pairs = c.unstack().dropna().sort_values(ascending=False)
            seen = set()
            for (a, b), v in pairs.items():
                tup = tuple(sorted((a, b)))
                if tup in seen:
                    continue
                seen.add(tup)
                top_corr_pairs.append((a, b, float(v)))
                if len(top_corr_pairs) >= 10:
                    break
        except Exception:
            pass

    meta = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "num_cols": len(num_cols),
        "cat_cols": len(cat_cols),
        "missing_cells": int(df.isna().sum().sum()) if hasattr(df, 'isna') else 0,
        "memory": df_memory_bytes(df),
        "num_col_names": num_cols,
        "cat_col_names": cat_cols,
    }

    return {
        "meta": meta,
        "desc_all": desc_all,
        "desc_num": desc_num,
        "missing": missing,
        "dtypes": dtypes,
        "corr": corr,
        "top_corr_pairs": top_corr_pairs,
    }


def build_smart_summary(pre: dict) -> List[str]:
    """Return human-friendly bullet lines; never raise."""
    try:
        meta = pre.get("meta", {}) if isinstance(pre, dict) else {}
        lines = []
        rows = int(meta.get("rows", 0) or 0)
        cols = int(meta.get("cols", 0) or 0)
        lines.append(f"Dataset with {rows} rows and {cols} columns.")
        numc = int(meta.get("num_cols", 0) or 0)
        catc = int(meta.get("cat_cols", 0) or 0)
        if numc or catc:
            lines.append(f"Contains {numc} numeric and {catc} categorical features.")
        miss_cells = int(meta.get("missing_cells", 0) or 0)
        if miss_cells > 0:
            lines.append(f"There are {miss_cells} missing values across the dataset.")
        missing = pre.get("missing", pd.DataFrame()) if isinstance(pre, dict) else pd.DataFrame()
        if isinstance(missing, pd.DataFrame) and not missing.empty and "missing" in missing.columns:
            top_missing = missing[missing["missing"] > 0].head(3)
            if not top_missing.empty:
                items = ", ".join([f"{idx} ({int(val)})" for idx, val in top_missing["missing"].items()])
                lines.append(f"Most missing: {items}.")
        pairs = pre.get("top_corr_pairs", []) if isinstance(pre, dict) else []
        if pairs:
            a, b, v = pairs[0]
            lines.append(f"Strongest numeric correlation: {a} ↔ {b} (r={v:.2f}).")
        tips = []
        if numc >= 2:
            tips.append("Try a correlation heatmap to spot relationships.")
        if miss_cells > 0:
            tips.append("Consider imputing or dropping columns with high missingness.")
        if tips:
            lines.append("Tips: " + " ".join(tips))
        return lines
    except Exception:
        return ["Summary unavailable due to unexpected data format."]


def viz_code_snippet(viz_type: str, params: dict) -> str:
    try:
        if viz_type == "Histogram":
            if params.get("color_by"):
                return (
                    "import plotly.express as px\n"
                    f"fig = px.histogram(df, x='{params['col']}', color='{params['color_by']}', title='Histogram of {params['col']} by {params['color_by']}')\n"
                    "fig.show()\n"
                )
            return (
                "import plotly.express as px\n"
                f"fig = px.histogram(df, x='{params['col']}', title='Histogram of {params['col']}')\n"
                "fig.show()\n"
            )
        if viz_type == "Scatter Plot":
            if params.get("color_by"):
                return (
                    "import plotly.express as px\n"
                    f"fig = px.scatter(df, x='{params['x_col']}', y='{params['y_col']}', color='{params['color_by']}', title='Scatter: {params['x_col']} vs {params['y_col']} by {params['color_by']}')\n"
                    "fig.show()\n"
                )
            return (
                "import plotly.express as px\n"
                f"fig = px.scatter(df, x='{params['x_col']}', y='{params['y_col']}', title='Scatter: {params['x_col']} vs {params['y_col']}')\n"
                "fig.show()\n"
            )
        if viz_type == "Box Plot":
            if params.get("group"):
                return (
                    "import plotly.express as px\n"
                    f"fig = px.box(df, x='{params['group']}', y='{params['col']}', title='Box Plot of {params['col']} by {params['group']}')\n"
                    "fig.show()\n"
                )
            return (
                "import plotly.express as px\n"
                f"fig = px.box(df, y='{params['col']}', title='Box Plot of {params['col']}')\n"
                "fig.show()\n"
            )
        if viz_type == "Correlation Heatmap":
            return (
                "import plotly.express as px\n"
                "import numpy as np\n"
                "corr = df.select_dtypes(include=[np.number]).corr()\n"
                "fig = px.imshow(corr, labels=dict(color='Correlation'), title='Correlation Heatmap')\n"
                "fig.show()\n"
            )
        if viz_type == "Pie Chart":
            return (
                "import plotly.express as px\n"
                f"counts = df['{params['col']}'].value_counts().reset_index()\n"
                f"counts.columns = ['{params['col']}', 'count']\n"
                f"fig = px.pie(counts, names='{params['col']}', values='count', title='Pie of {params['col']}')\n"
                "fig.show()\n"
            )
        if viz_type == "Bar Chart":
            if params.get('agg'):
                return (
                    "import plotly.express as px\n"
                    f"agg = df.groupby('{params['cat_col']}')['{params['val_col']}'].mean().reset_index()\n"
                    f"fig = px.bar(agg, x='{params['cat_col']}', y='{params['val_col']}', title='Bar of mean {params['val_col']} by {params['cat_col']}')\n"
                    "fig.show()\n"
                )
            return (
                "import plotly.express as px\n"
                f"counts = df['{params['col']}'].value_counts().reset_index()\n"
                f"counts.columns = ['{params['col']}', 'count']\n"
                f"fig = px.bar(counts, x='{params['col']}', y='count', title='Bar counts for {params['col']}')\n"
                "fig.show()\n"
            )
        if viz_type == "Line Chart":
            return (
                "import plotly.express as px\n"
                f"fig = px.line(df, x='{params['x_col']}', y='{params['y_col']}', title='Line: {params['y_col']} over {params['x_col']}')\n"
                "fig.show()\n"
            )
        if viz_type == "Violin":
            if params.get('group'):
                return (
                    "import plotly.express as px\n"
                    f"fig = px.violin(df, x='{params['group']}', y='{params['col']}', title='Violin of {params['col']} by {params['group']}')\n"
                    "fig.show()\n"
                )
            return (
                "import plotly.express as px\n"
                f"fig = px.violin(df, y='{params['col']}', title='Violin of {params['col']}')\n"
                "fig.show()\n"
            )
        if viz_type == "Area Chart":
            return (
                "import plotly.express as px\n"
                f"fig = px.area(df, x='{params['x_col']}', y=[{', '.join([repr(c) for c in params.get('y_cols', [])])}], title='Area chart over {params['x_col']}')\n"
                "fig.show()\n"
            )
        return ""
    except Exception:
        return "# Code unavailable due to parameter issue."


# ------------------------------------------------------------
# Session state init
# ------------------------------------------------------------
if "file_bytes" not in st.session_state:
    st.session_state.file_bytes = None
if "file_info" not in st.session_state:
    st.session_state.file_info = {}
if "df" not in st.session_state:
    st.session_state.df = None
if "df_key" not in st.session_state:
    st.session_state.df_key = None
if "precomp" not in st.session_state:
    st.session_state.precomp = None
if "last_fig_html" not in st.session_state:
    st.session_state.last_fig_html = None
if "last_viz_code" not in st.session_state:
    st.session_state.last_viz_code = ""
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Default": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Default"

# ------------------------------------------------------------
# Sidebar: CSV upload and options (CSV only)
# ------------------------------------------------------------
st.sidebar.title("📂 CSV Loader")
st.sidebar.caption("Upload a CSV, then click Load to parse it. CSV-only for reliability.")

up_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"],
)

with st.sidebar.expander("Advanced options", expanded=False):
    sep_choice = st.selectbox("Delimiter", ["Auto", ", (comma)", "; (semicolon)", "\t (tab)", "| (pipe)"])
    encoding_choice = st.selectbox("Encoding", ["utf-8", "latin-1"], index=0)
    header_present = st.checkbox("Header row present", value=True)
    parse_dates = st.checkbox("Try parse dates", value=False)
    load_mode = st.radio("Load mode", ["Quick preview (5,000 rows)", "Full data"], index=0)
    on_bad_lines = st.selectbox("On bad lines", ["skip", "error", "warn"], index=0)
    low_memory = st.checkbox("Low memory mode", value=False)

col_sb1, col_sb2, col_sb3 = st.sidebar.columns([1, 1, 1])
clicked_load = col_sb1.button("Load CSV", use_container_width=True)
clear_data = col_sb2.button("Clear", use_container_width=True)
if col_sb3.button("Clear cache", use_container_width=True):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared.")

# Stage file bytes
if up_file is not None:
    if not up_file.name.lower().endswith(".csv"):
        st.sidebar.error("Please upload a .csv file.")
    else:
        st.session_state.file_bytes = up_file.getvalue()
        st.session_state.file_info = {"name": up_file.name, "size": len(st.session_state.file_bytes)}

# Clear in-memory data
if clear_data:
    st.session_state.df = None
    st.session_state.file_bytes = None
    st.session_state.file_info = {}
    st.session_state.df_key = None
    st.session_state.precomp = None
    st.session_state.last_fig_html = None
    st.session_state.last_viz_code = ""
    try:
        st.toast("Cleared current dataset.")
    except Exception:
        pass

# Load CSV
if clicked_load and st.session_state.file_bytes:
    with st.spinner("Reading CSV…"):
        fb = st.session_state.file_bytes
        if sep_choice.startswith("Auto"):
            sep = detect_delimiter(fb)
        else:
            mapping = {", (comma)": ",", "; (semicolon)": ";", "\t (tab)": "\t", "| (pipe)": "|"}
            sep = mapping.get(sep_choice, ",")
        nrows = 5000 if load_mode.startswith("Quick") else None
        opts = {
            "sep": sep,
            "encoding": encoding_choice,
            "header": header_present,
            "parse_dates": parse_dates,
            "nrows": nrows,
            "on_bad_lines": on_bad_lines,
            "low_memory": low_memory,
        }
        try:
            df = load_csv_cached(
                file_bytes=fb,
                sep=sep,
                encoding=encoding_choice,
                header=header_present,
                parse_dates=parse_dates,
                nrows=nrows,
                on_bad_lines=on_bad_lines,
                low_memory=low_memory,
            )
            st.session_state.df = df
            st.session_state.df_key = df_key_from_bytes(fb, opts)
            st.session_state.precomp = compute_summary_cached(df, st.session_state.df_key)
            try:
                st.toast("CSV loaded ✓")
            except Exception:
                pass
        except Exception as e:
            st.session_state.df = None
            st.session_state.df_key = None
            st.session_state.precomp = None
            st.error(f"Failed to read CSV. Details: {e}")

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("🤖 AI-Powered Data Visualization")
st.caption("Clean UI. CSV-only uploads. Fast summary, instant chat, and reproducible reports.")

# Quick metrics
m1, m2, m3 = st.columns(3)
m1.metric("File", st.session_state.file_info.get("name", "—"))
m2.metric("Size", human_bytes(st.session_state.file_info.get("size")))
if st.session_state.df is not None:
    m3.metric("Rows × Cols", f"{st.session_state.df.shape[0]} × {st.session_state.df.shape[1]}")
else:
    m3.metric("Rows × Cols", "—")

# Tabs
tabs = st.tabs(["📋 Overview", "📈 Visualize", "📊 Statistics", "🧩 Code & Report", "💬 Chat"])

# ------------------------------------------------------------
# Overview
# ------------------------------------------------------------
with tabs[0]:
    if st.session_state.df is None:
        st.info("Upload a .csv file in the sidebar and click Load CSV to begin.")
    else:
        df = st.session_state.df
        pre = st.session_state.precomp or {}
        meta = pre.get("meta", {}) if isinstance(pre, dict) else {}

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Numeric cols", int(meta.get("num_cols", 0) or 0))
        c2.metric("Categorical cols", int(meta.get("cat_cols", 0) or 0))
        c3.metric("Missing cells", int(meta.get("missing_cells", 0) or 0))
        c4.metric("Memory", human_bytes(meta.get("memory")))
        c5.metric("Preview rows", 100)

        st.subheader("Smart summary")
        for line in build_smart_summary(pre):
            st.write(f"• {line}")

        st.subheader("Preview")
        preview_n = st.slider("Rows to preview", min_value=5, max_value=200, value=100, step=5)
        try:
            st.dataframe(df.head(preview_n), use_container_width=True)
        except Exception:
            st.info("Could not render preview.")

        st.subheader("Describe (all)")
        desc_all = pre.get("desc_all", pd.DataFrame()) if isinstance(pre, dict) else pd.DataFrame()
        if isinstance(desc_all, pd.DataFrame) and not desc_all.empty:
            st.dataframe(desc_all, use_container_width=True, height=300)
        else:
            st.info("Describe not available for this dataset.")

# ------------------------------------------------------------
# Visualize (extended)
# ------------------------------------------------------------
with tabs[1]:
    if st.session_state.df is None:
        st.info("Load a dataset to create visualizations.")
    else:
        df = st.session_state.df
        pre = st.session_state.precomp or {}
        num_cols = pre.get("meta", {}).get("num_col_names") or numeric_columns(df)
        cat_cols = pre.get("meta", {}).get("cat_col_names") or categorical_columns(df)

        top_a, top_b = st.columns([2, 1])
        with top_a:
            viz_type = st.selectbox("Chart type", [
                "Histogram",
                "Scatter Plot",
                "Box Plot",
                "Correlation Heatmap",
                "Pie Chart",
                "Bar Chart",
                "Line Chart",
                "Violin",
                "Area Chart",
            ])
        with top_b:
            sample_frac = st.slider("Sample %", 1, 100, 100, help="Use sampling for speed on large data")

        df_viz = df
        if sample_frac < 100:
            try:
                df_viz = df.sample(frac=sample_frac / 100.0, random_state=42)
            except Exception:
                pass

        fig = None
        params = {}

        try:
            if viz_type == "Histogram":
                if not num_cols:
                    st.warning("No numeric columns available.")
                else:
                    ccol = st.selectbox("Column", num_cols)
                    color_by = st.selectbox("Color by (optional)", ["None"] + cat_cols)
                    params = {"col": ccol}
                    if color_by != "None":
                        params["color_by"] = color_by
                        fig = px.histogram(df_viz, x=ccol, color=color_by, title=f"Histogram of {ccol} by {color_by}")
                    else:
                        fig = px.histogram(df_viz, x=ccol, title=f"Histogram of {ccol}")

            elif viz_type == "Scatter Plot":
                if len(num_cols) < 2:
                    st.warning("Need at least two numeric columns.")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        x_col = st.selectbox("X axis", num_cols)
                    with c2:
                        y_col = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols) - 1))
                    color_by = st.selectbox("Color by (optional)", ["None"] + cat_cols)
                    params = {"x_col": x_col, "y_col": y_col}
                    if color_by != "None":
                        params["color_by"] = color_by
                        fig = px.scatter(df_viz, x=x_col, y=y_col, color=color_by, title=f"Scatter: {x_col} vs {y_col} by {color_by}")
                    else:
                        fig = px.scatter(df_viz, x=x_col, y=y_col, title=f"Scatter: {x_col} vs {y_col}")

            elif viz_type == "Box Plot":
                if not num_cols:
                    st.warning("No numeric columns available.")
                else:
                    ccol = st.selectbox("Column", num_cols)
                    group = st.selectbox("Group by (optional)", ["None"] + cat_cols)
                    params = {"col": ccol}
                    if group != "None":
                        params["group"] = group
                        fig = px.box(df_viz, x=group, y=ccol, title=f"Box Plot of {ccol} by {group}")
                    else:
                        fig = px.box(df_viz, y=ccol, title=f"Box Plot of {ccol}")

            elif viz_type == "Correlation Heatmap":
                if len(num_cols) < 2:
                    st.warning("Need at least two numeric columns.")
                else:
                    corr = pre.get("corr")
                    if corr is None or not isinstance(corr, pd.DataFrame) or corr.empty:
                        corr = df_viz.select_dtypes(include=[np.number]).corr()
                    fig = px.imshow(corr, labels=dict(color="Correlation"), title="Correlation Heatmap")

            elif viz_type == "Pie Chart":
                if not cat_cols:
                    st.warning("No categorical columns available.")
                else:
                    ccol = st.selectbox("Category column", cat_cols)
                    params = {"col": ccol}
                    counts = df_viz[ccol].value_counts().reset_index()
                    counts.columns = [ccol, "count"]
                    fig = px.pie(counts, names=ccol, values="count", title=f"Pie of {ccol}")

            elif viz_type == "Bar Chart":
                if not cat_cols:
                    st.warning("No categorical columns available.")
                else:
                    cat_col = st.selectbox("Category column", cat_cols)
                    val_col = st.selectbox("Value column (optional aggregate)", ["None"] + num_cols)
                    if val_col != "None":
                        params = {"cat_col": cat_col, "val_col": val_col, "agg": True}
                        agg = df_viz.groupby(cat_col)[val_col].mean().reset_index()
                        fig = px.bar(agg, x=cat_col, y=val_col, title=f"Mean {val_col} by {cat_col}")
                    else:
                        params = {"col": cat_col}
                        counts = df_viz[cat_col].value_counts().reset_index()
                        counts.columns = [cat_col, "count"]
                        fig = px.bar(counts, x=cat_col, y="count", title=f"Counts for {cat_col}")

            elif viz_type == "Line Chart":
                if not num_cols:
                    st.warning("No numeric columns available.")
                else:
                    x_col = st.selectbox("X axis (prefer datetime)", df.columns.tolist())
                    y_col = st.selectbox("Y axis", num_cols)
                    params = {"x_col": x_col, "y_col": y_col}
                    try:
                        fig = px.line(df_viz.sort_values(by=x_col), x=x_col, y=y_col, title=f"Line: {y_col} over {x_col}")
                    except Exception:
                        fig = px.line(df_viz, x=x_col, y=y_col, title=f"Line: {y_col} over {x_col}")

            elif viz_type == "Violin":
                if not num_cols:
                    st.warning("No numeric columns available.")
                else:
                    ccol = st.selectbox("Column", num_cols)
                    group = st.selectbox("Group by (optional)", ["None"] + cat_cols)
                    params = {"col": ccol}
                    if group != "None":
                        params["group"] = group
                        fig = px.violin(df_viz, x=group, y=ccol, title=f"Violin of {ccol} by {group}")
                    else:
                        fig = px.violin(df_viz, y=ccol, title=f"Violin of {ccol}")

            elif viz_type == "Area Chart":
                if not num_cols:
                    st.warning("No numeric columns available.")
                else:
                    x_col = st.selectbox("X axis (prefer datetime)", df.columns.tolist())
                    y_cols = st.multiselect("Y columns (numeric)", num_cols, default=num_cols[:1])
                    if y_cols:
                        params = {"x_col": x_col, "y_cols": y_cols}
                        try:
                            fig = px.area(df_viz.sort_values(by=x_col), x=x_col, y=y_cols, title=f"Area chart over {x_col}")
                        except Exception:
                            fig = px.area(df_viz, x=x_col, y=y_cols, title=f"Area chart over {x_col}")
        except Exception as e:
            st.error(f"Failed to build the chart: {e}")
            fig = None

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            try:
                st.session_state.last_fig_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
            except Exception:
                st.session_state.last_fig_html = None
            st.session_state.last_viz_code = viz_code_snippet(viz_type, params)

            with st.expander("Show code for this visualization"):
                st.code(st.session_state.last_viz_code or "# No code available.", language="python")

            try:
                html_bytes = pio.to_html(fig, include_plotlyjs="cdn").encode("utf-8")
                st.download_button(
                    label="Download figure (HTML)",
                    data=html_bytes,
                    file_name=f"figure_{viz_type.replace(' ', '_').lower()}.html",
                    mime="text/html",
                )
            except Exception:
                st.info("Could not prepare figure for download.")

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------
with tabs[2]:
    if st.session_state.df is None:
        st.info("Load a dataset to view statistics.")
    else:
        pre = st.session_state.precomp or {}
        missing = pre.get("missing", pd.DataFrame()) if isinstance(pre, dict) else pd.DataFrame()
        dtypes = pre.get("dtypes", pd.DataFrame()) if isinstance(pre, dict) else pd.DataFrame()
        desc_num = pre.get("desc_num", pd.DataFrame()) if isinstance(pre, dict) else pd.DataFrame()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Missing values per column")
            if isinstance(missing, pd.DataFrame) and not missing.empty:
                st.dataframe(missing, use_container_width=True, height=320)
            else:
                st.info("No missing values detected or unavailable.")
        with c2:
            st.subheader("Data types")
            if isinstance(dtypes, pd.DataFrame) and not dtypes.empty:
                st.dataframe(dtypes, use_container_width=True, height=320)
            else:
                st.info("Dtype info unavailable.")

        st.subheader("Describe (numeric)")
        if isinstance(desc_num, pd.DataFrame) and not desc_num.empty:
            st.dataframe(desc_num, use_container_width=True, height=340)
        else:
            st.info("No numeric columns for describe.")

# ------------------------------------------------------------
# Code & Report
# ------------------------------------------------------------
with tabs[3]:
    if st.session_state.df is None:
        st.info("Load a dataset to view code and export a report.")
    else:
        pre = st.session_state.precomp or {}
        st.subheader("Reproducible code (summary)")
        summary_code = (
            "import pandas as pd\n"
            "import numpy as np\n\n"
            "df = pd.read_csv('your_file.csv')\n\n"
            "desc_all = df.describe(include='all', datetime_is_numeric=True).transpose()\n"
            "desc_num = df.select_dtypes(include=[np.number]).describe().transpose()\n"
            "missing = df.isna().sum().to_frame('missing').sort_values('missing', ascending=False)\n"
            "dtypes = df.dtypes.astype(str).to_frame('dtype')\n"
            "num_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n"
            "corr = df[num_cols[:30]].corr() if len(num_cols) >= 2 else None\n"
            "print(desc_all)\nprint(missing)\nprint(dtypes)\nprint(corr)\n"
        )
        st.code(summary_code, language="python")

        if st.session_state.last_viz_code:
            st.subheader("Latest visualization code")
            st.code(st.session_state.last_viz_code, language="python")
        else:
            st.info("Create a visualization to see its code here.")

        st.markdown("---")
        st.subheader("Download HTML report")
        report_title = st.text_input("Report title", value="Data Analysis Report")
        include_fig = st.checkbox("Include latest figure (if available)", value=True)

        def build_html_report(df: pd.DataFrame, title: str, pre: dict, fig_html: Optional[str]) -> bytes:
            style = (
                "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;margin:2rem}"
                "h1,h2,h3{color:#111827}"
                "table{border-collapse:collapse;width:100%;margin-bottom:1.2rem}"
                "th,td{border:1px solid #e5e7eb;padding:6px 8px;font-size:12px}"
                "th{background:#f9fafb}.meta{display:flex;gap:1.5rem;margin-bottom:1rem}.section{margin-top:1.5rem}</style>"
            )
            meta = pre.get("meta", {}) if isinstance(pre, dict) else {}
            desc_html = "<p>Describe not available.</p>"
            da = pre.get("desc_all") if isinstance(pre, dict) else None
            if isinstance(da, pd.DataFrame) and not da.empty:
                try:
                    desc_html = da.to_html()
                except Exception:
                    pass
            miss_html = "<p>No missing values.</p>"
            md = pre.get("missing") if isinstance(pre, dict) else None
            if isinstance(md, pd.DataFrame) and not md.empty:
                try:
                    miss_html = md.to_html()
                except Exception:
                    pass
            dtypes_html = "<p>No dtypes available.</p>"
            dt = pre.get("dtypes") if isinstance(pre, dict) else None
            if isinstance(dt, pd.DataFrame) and not dt.empty:
                try:
                    dtypes_html = dt.to_html()
                except Exception:
                    pass
            corr_pairs = pre.get("top_corr_pairs", []) if isinstance(pre, dict) else []
            if corr_pairs:
                rows = "".join([f"<tr><td>{a}</td><td>{b}</td><td>{v:.3f}</td></tr>" for a, b, v in corr_pairs])
                corr_html = f"<table><tr><th>Col A</th><th>Col B</th><th>r</th></tr>{rows}</table>"
            else:
                corr_html = "<p>No correlations computed.</p>"
            fig_block = f"<div class='section'><h2>Latest Figure</h2>{fig_html}</div>" if fig_html else ""
            html = f"""
            <html><head><meta charset='utf-8'><title>{title}</title>{style}</head>
            <body>
              <h1>{title}</h1>
              <div class='meta'>
                <div><b>Rows</b>: {meta.get('rows','-')}</div>
                <div><b>Columns</b>: {meta.get('cols','-')}</div>
                <div><b>Generated</b>: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
              </div>
              <div class='section'><h2>Describe</h2>{desc_html}</div>
              <div class='section'><h2>Missing Values</h2>{miss_html}</div>
              <div class='section'><h2>Data Types</h2>{dtypes_html}</div>
              <div class='section'><h2>Top Correlations</h2>{corr_html}</div>
              {fig_block}
            </body></html>
            """
            return html.encode("utf-8")

        if st.button("Generate report"):
            try:
                fig_html = st.session_state.last_fig_html if (include_fig and st.session_state.last_fig_html) else None
                html_bytes = build_html_report(st.session_state.df, report_title, pre, fig_html)
                st.download_button(
                    label="Download report (HTML)",
                    data=html_bytes,
                    file_name="report.html",
                    mime="text/html",
                )
                st.success("Report ready. Click the download button above.")
            except Exception as e:
                st.error(f"Failed to generate report: {e}")

# ------------------------------------------------------------
# Chat (instant, local)
# ------------------------------------------------------------
with tabs[4]:
    st.caption("Local, instant chat based on the precomputed summary. No external API calls.")

    # Session controls
    ctrl = st.columns([2, 1, 1, 1])
    with ctrl[0]:
        session_names = list(st.session_state.chat_sessions.keys())
        if not session_names:
            session_names = ["Default"]
            st.session_state.chat_sessions = {"Default": []}
        current_index = session_names.index(st.session_state.current_chat) if st.session_state.current_chat in session_names else 0
        current = st.selectbox("Session", session_names, index=current_index)
        st.session_state.current_chat = current
    with ctrl[1]:
        new_name = st.text_input("New session")
    with ctrl[2]:
        if st.button("New"):
            if new_name and new_name not in st.session_state.chat_sessions:
                st.session_state.chat_sessions[new_name] = []
                st.session_state.current_chat = new_name
                try:
                    st.toast("Session created")
                except Exception:
                    pass
            else:
                st.warning("Provide a unique session name.")
    with ctrl[3]:
        if st.button("Delete"):
            if st.session_state.current_chat != "Default":
                try:
                    del st.session_state.chat_sessions[st.session_state.current_chat]
                    st.session_state.current_chat = "Default"
                    try:
                        st.toast("Session deleted")
                    except Exception:
                        pass
                except Exception:
                    st.warning("Could not delete the session.")
            else:
                st.warning("Cannot delete the Default session.")

    history: List[Dict[str, Any]] = st.session_state.chat_sessions.get(st.session_state.current_chat, [])

    # Utilities to find columns by name heuristics
    def find_best_match_column(name: str, cols: list) -> Optional[str]:
        if not name or not cols:
            return None
        name = name.lower().strip()
        # exact
        for c in cols:
            if c.lower() == name:
                return c
        # contains
        for c in cols:
            if name in c.lower() or c.lower() in name:
                return c
        # token match
        tokens = re.split(r"\W+", name)
        for t in tokens:
            for c in cols:
                if t and t in c.lower():
                    return c
        return None

    def find_column_from_text(text: str, cols: list) -> Optional[str]:
        if not text or not cols:
            return None
        text_low = text.lower()
        # Check for quoted column names first
        q = re.findall(r"['\"](.*?)['\"]", text)
        for cand in q:
            match = find_best_match_column(cand, cols)
            if match:
                return match
        # otherwise try words
        for c in cols:
            n = c.lower()
            if n in text_low:
                return c
        # tokens fallback
        words = re.split(r"\W+", text_low)
        for w in words:
            if not w:
                continue
            match = find_best_match_column(w, cols)
            if match:
                return match
        return None

    def generate_viz_from_text(text: str):
        """Try to interpret the user's message as a viz request. Returns (fig, viz_code, message)."""
        try:
            df = st.session_state.df
            if df is None:
                return None, None, "No dataset loaded."
            pre = st.session_state.precomp or {}
            num_cols = pre.get("meta", {}).get("num_col_names") or numeric_columns(df)
            cat_cols = pre.get("meta", {}).get("cat_col_names") or categorical_columns(df)
            t = (text or "").lower()

            # Histogram
            if any(k in t for k in ["hist", "histogram", "distribution"]):
                col = find_column_from_text(text, num_cols) or (num_cols[0] if num_cols else None)
                if not col:
                    return None, None, "No numeric column found for histogram."
                fig = px.histogram(df, x=col, title=f"Histogram of {col}")
                code = viz_code_snippet("Histogram", {"col": col})
                return fig, code, f"Histogram of {col}."

            # Pie Chart
            if any(k in t for k in ["pie", "pie chart"]):
                col = find_column_from_text(text, cat_cols) or (cat_cols[0] if cat_cols else None)
                if not col:
                    return None, None, "No categorical column found for pie chart."
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, "count"]
                fig = px.pie(counts, names=col, values="count", title=f"Pie of {col}")
                code = viz_code_snippet("Pie Chart", {"col": col})
                return fig, code, f"Pie chart of {col}."

            # Scatter Plot
            if any(k in t for k in ["scatter", " x ", " vs ", "against", "plot "]):
                x = find_column_from_text(text, num_cols)
                y = None
                if x:
                    other = [c for c in num_cols if c != x]
                    y = find_column_from_text(text.replace(x, ""), other) or (other[0] if other else None)
                else:
                    if len(num_cols) >= 2:
                        x, y = num_cols[0], num_cols[1]
                if not x or not y:
                    return None, None, "Could not identify two numeric columns for scatter."
                fig = px.scatter(df, x=x, y=y, title=f"Scatter: {x} vs {y}")
                code = viz_code_snippet("Scatter Plot", {"x_col": x, "y_col": y})
                return fig, code, f"Scatter plot of {x} vs {y}."

            # Bar Chart (counts or aggregate)
            if any(k in t for k in ["bar", "counts", "count of"]):
                cat = find_column_from_text(text, cat_cols)
                val = find_column_from_text(text, num_cols)
                if cat and val:
                    agg = df.groupby(cat)[val].mean().reset_index()
                    fig = px.bar(agg, x=cat, y=val, title=f"Mean {val} by {cat}")
                    code = viz_code_snippet("Bar Chart", {"cat_col": cat, "val_col": val, "agg": True})
                    return fig, code, f"Bar chart of mean {val} by {cat}."
                col = cat or (cat_cols[0] if cat_cols else None)
                if col:
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, "count"]
                    fig = px.bar(counts, x=col, y="count", title=f"Counts for {col}")
                    code = viz_code_snippet("Bar Chart", {"col": col})
                    return fig, code, f"Bar chart of counts for {col}."
                return None, None, "No categorical column found for bar chart."

            # Box / Violin
            if any(k in t for k in ["box", "violin"]):
                col = find_column_from_text(text, num_cols) or (num_cols[0] if num_cols else None)
                group = find_column_from_text(text, cat_cols)
                if not col:
                    return None, None, "No numeric column found for box/violin plot."
                if "violin" in t:
                    if group:
                        fig = px.violin(df, x=group, y=col, title=f"Violin of {col} by {group}")
                        code = viz_code_snippet("Violin", {"col": col, "group": group})
                    else:
                        fig = px.violin(df, y=col, title=f"Violin of {col}")
                        code = viz_code_snippet("Violin", {"col": col})
                    return fig, code, f"Violin plot of {col}."
                else:
                    if group:
                        fig = px.box(df, x=group, y=col, title=f"Box Plot of {col} by {group}")
                        code = viz_code_snippet("Box Plot", {"col": col, "group": group})
                    else:
                        fig = px.box(df, y=col, title=f"Box Plot of {col}")
                        code = viz_code_snippet("Box Plot", {"col": col})
                    return fig, code, f"Box plot of {col}."

            # Correlation heatmap
            if any(k in t for k in ["correl", "correlation", "heatmap", "relationship"]):
                numc = numeric_columns(df)
                if len(numc) < 2:
                    return None, None, "Not enough numeric columns for correlation."
                corr = df[numc[:30]].corr()
                fig = px.imshow(corr, labels=dict(color="Correlation"), title="Correlation Heatmap")
                code = viz_code_snippet("Correlation Heatmap", {})
                return fig, code, "Correlation heatmap."

            # Summary session
            if any(k in t for k in ["summary session", "detailed summary", "detailed overview"]):
                return "SUMMARY", None, "summary session"

            # Suggestions
            if any(k in t for k in ["suggest", "ideas", "what to plot", "recommend"]):
                return None, None, (
                    "Ideas: Histogram (distribution), Scatter (relationships), Box (outliers), "
                    "Pie/Bar (categorical breakdown), Correlation Heatmap (multivariate)."
                )

            return None, None, None
        except Exception as e:
            return None, None, f"Could not create visualization: {e}"

    def answer_user(msg: str) -> Dict[str, Any]:
        """Return a structured assistant response: {content, fig_dict?, tables?, viz_code?}"""
        try:
            df = st.session_state.df
            pre = st.session_state.precomp
            q = (msg or "").strip()
            qlow = q.lower()
            if df is None or pre is None:
                return {"content": "Upload and load a CSV to enable data-aware answers."}

            # Commands that list columns / summary / missing / correlations
            if any(k in qlow for k in ["column", "columns", "fields"]) and 'plot' not in qlow:
                try:
                    cols = list(df.columns[:500])
                except Exception:
                    cols = []
                text = "Columns (first 500):\n- " + "\n- ".join([str(c) for c in cols]) if cols else "No columns available."
                return {"content": text}

            if any(k in qlow for k in ["summary", "describe", "overview"]) and 'plot' not in qlow and 'session' not in qlow:
                num = int(pre.get("meta", {}).get("num_cols", 0) or 0)
                cat = int(pre.get("meta", {}).get("cat_cols", 0) or 0)
                miss = int(pre.get("meta", {}).get("missing_cells", 0) or 0)
                rows = int(pre.get("meta", {}).get("rows", 0) or 0)
                cols = int(pre.get("meta", {}).get("cols", 0) or 0)
                lines = [
                    f"Rows: {rows}, Columns: {cols}",
                    f"Numeric: {num}, Categorical: {cat}",
                    f"Missing cells: {miss}",
                ]
                pairs = pre.get("top_corr_pairs", []) if isinstance(pre, dict) else []
                if pairs:
                    a, b, v = pairs[0]
                    lines.append(f"Strongest correlation: {a} ↔ {b} (r={v:.3f})")
                tips = []
                if num >= 2:
                    tips.append("Try a correlation heatmap to spot relationships.")
                if miss > 0:
                    tips.append("Consider imputing columns with many missing values.")
                if tips:
                    lines.append("Tips: " + " ".join(tips))
                return {"content": "\n".join(lines)}

            if any(k in qlow for k in ["missing", "null", "na"]) and 'plot' not in qlow:
                miss_df = pre.get("missing", pd.DataFrame()) if isinstance(pre, dict) else pd.DataFrame()
                if isinstance(miss_df, pd.DataFrame) and not miss_df.empty and "missing" in miss_df.columns:
                    top_missing = miss_df["missing"].sort_values(ascending=False).head(10)
                    lines = ["Top missing columns:"] + [f"- {idx}: {int(val)}" for idx, val in top_missing.items()]
                    tables = []
                    try:
                        tables.append({"title": "Top missing columns", "html": top_missing.to_frame().to_html()})
                    except Exception:
                        pass
                    return {"content": "\n".join(lines), "tables": tables}
                return {"content": "No missing values detected."}

            if any(k in qlow for k in ["correl", "relationship", "heatmap"]) and 'plot' not in qlow:
                pairs = pre.get("top_corr_pairs", []) if isinstance(pre, dict) else []
                if not pairs:
                    return {"content": "Not enough numeric columns for correlation."}
                lines = ["Top correlations (r):"] + [f"- {a} ↔ {b}: {v:.3f}" for a, b, v in pairs]
                return {"content": "\n".join(lines)}

            # Visualization requests and summary session
            viz_result, viz_code, explanation = generate_viz_from_text(q)
            if viz_result == "SUMMARY":
                lines = build_smart_summary(pre)
                tables = []
                try:
                    da = pre.get("desc_all")
                    if isinstance(da, pd.DataFrame) and not da.empty:
                        tables.append({"title": "Describe (all)", "html": da.to_html()})
                except Exception:
                    pass
                try:
                    md = pre.get("missing")
                    if isinstance(md, pd.DataFrame) and not md.empty:
                        tables.append({"title": "Missing", "html": md.to_html()})
                except Exception:
                    pass
                try:
                    dt = pre.get("dtypes")
                    if isinstance(dt, pd.DataFrame) and not dt.empty:
                        tables.append({"title": "Dtypes", "html": dt.to_html()})
                except Exception:
                    pass
                return {"content": "\n".join([f"• {l}" for l in lines]), "tables": tables}

            if viz_result is not None and hasattr(viz_result, "to_dict"):
                fig = viz_result
                entry = {
                    "content": explanation or "Here is the chart.",
                    "fig_dict": fig.to_dict(),
                }
                if viz_code:
                    entry["viz_code"] = viz_code
                return entry

            if explanation:
                return {"content": explanation}

            # Fallback help
            return {"content": (
                "I can help you explore your data. Try: 'histogram of age', 'scatter price vs quantity', "
                "'bar counts of category', 'correlation heatmap', or ask for 'summary'."
            )}
        except Exception as e:
            return {"content": f"Something went wrong while answering: {e}"}

    # Render chat history
    if CHAT_AVAILABLE:
        for item in history:
            role = item.get("role", "assistant")
            with st.chat_message("user" if role == "user" else "assistant"):
                st.write(item.get("content", ""))
                fig_dict = item.get("fig_dict")
                if fig_dict:
                    try:
                        st.plotly_chart(go.Figure(fig_dict), use_container_width=True)
                    except Exception:
                        pass
                code_txt = item.get("viz_code")
                if code_txt:
                    st.code(code_txt, language="python")
                tables = item.get("tables", [])
                for t in tables:
                    with st.expander(t.get("title", "Table")):
                        try:
                            components.html(t.get("html", ""), height=320, scrolling=True)
                        except Exception:
                            pass

        # Input box
        user_prompt = st.chat_input("Ask about the data or request a chart...")
        if user_prompt:
            # append user message
            history.append({"role": "user", "content": user_prompt})
            # answer
            ans = answer_user(user_prompt)
            assistant_item: Dict[str, Any] = {"role": "assistant", "content": ans.get("content", "")}
            if ans.get("fig_dict"):
                assistant_item["fig_dict"] = ans["fig_dict"]
            if ans.get("viz_code"):
                assistant_item["viz_code"] = ans["viz_code"]
            if ans.get("tables"):
                assistant_item["tables"] = ans["tables"]
            history.append(assistant_item)
            st.session_state.chat_sessions[st.session_state.current_chat] = history
            st.rerun()
    else:
        # Fallback UI if chat components not available
        st.write("Chat components are not available in this Streamlit version. Use the Visualize and Statistics tabs.")
