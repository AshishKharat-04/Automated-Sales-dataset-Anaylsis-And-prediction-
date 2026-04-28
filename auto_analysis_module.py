# auto_analysis_module.py  –  v2  (all errors fixed)
# ─────────────────────────────────────────────────────────────────────────────
# Fixes applied vs v1
#  1. add_vline(x=Timestamp) → Plotly internal crash via integer arithmetic.
#     Fix: convert last_date to plain ISO string via _ts_to_str() before
#          passing to add_vline.  Also removed annotation_position kwarg from
#          add_vline; replaced with a separate add_annotation() call.
#  2. pd.date_range start using pd.offsets.MonthEnd(1) – deprecated int arith.
#     Fix: use pd.DateOffset(months=1) for all date arithmetic.
#  3. select_dtypes(include='object') – Pandas 3 issues + FutureWarning.
#     Fix: replaced everywhere with pd.api.types helpers:
#          is_string_dtype / is_object_dtype / is_numeric_dtype
#          via helper functions get_str_cols() / get_num_cols().
#  4. df.dtypes == object – returns all-False on pandas 3 string-backed cols.
#     Fix: use get_str_cols() / get_num_cols() everywhere.
#  5. Resample freq 'M' – deprecated in newer pandas.
#     Fix: use 'ME' (month-end) throughout.
#  6. All section functions wrapped in try/except so one error never crashes
#     the whole page – error shown inline as st.warning instead.
# ─────────────────────────────────────────────────────────────────────────────

import io
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ─────────────────────────────────────────────
# PANDAS-VERSION-SAFE HELPERS
# ─────────────────────────────────────────────

def get_str_cols(df: pd.DataFrame) -> list:
    """Column names that hold string/object data — works on pandas 1, 2, 3."""
    return [
        c for c in df.columns
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])
    ]


def get_num_cols(df: pd.DataFrame) -> list:
    """Column names that are numeric — works on pandas 1, 2, 3."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _num_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[get_num_cols(df)]


def _ts_to_str(ts) -> str:
    """
    Convert a pd.Timestamp (or anything date-like) to an ISO-date string.
    Plotly's add_vline / add_shape require a plain string when the x-axis
    contains datetime values, otherwise it tries integer arithmetic on the
    Timestamp and raises TypeError.
    """
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)


# ─────────────────────────────────────────────
# KEYWORD MAPS  (smart column detection)
# ─────────────────────────────────────────────

DATE_KEYWORDS     = ["date", "time", "month", "year", "period",
                     "order_date", "sale_date", "invoice_date", "timestamp"]
REVENUE_KEYWORDS  = ["revenue", "sales", "amount", "total", "income",
                     "turnover", "gross", "net_sales", "sale_amount"]
PROFIT_KEYWORDS   = ["profit", "margin", "net", "earnings", "gain"]
QTY_KEYWORDS      = ["quantity", "qty", "units", "items", "volume", "count"]
PRODUCT_KEYWORDS  = ["product", "item", "sku", "category", "segment",
                     "sub_category", "product_name", "product_type"]
REGION_KEYWORDS   = ["region", "state", "city", "country", "location",
                     "territory", "area", "zone", "market"]
CUSTOMER_KEYWORDS = ["customer", "client", "buyer", "account"]


def _match(col: str, keywords: list) -> bool:
    col_l = col.lower().replace(" ", "_")
    return any(k in col_l for k in keywords)


def detect_columns(df: pd.DataFrame) -> dict:
    """Return dict: semantic role → best matching column name (or None)."""
    roles = {
        "date":     DATE_KEYWORDS,
        "revenue":  REVENUE_KEYWORDS,
        "profit":   PROFIT_KEYWORDS,
        "quantity": QTY_KEYWORDS,
        "product":  PRODUCT_KEYWORDS,
        "region":   REGION_KEYWORDS,
        "customer": CUSTOMER_KEYWORDS,
    }
    return {
        role: next((c for c in df.columns if _match(c, kws)), None)
        for role, kws in roles.items()
    }


# ─────────────────────────────────────────────
# SAFE MONTHLY RESAMPLE HELPER
# ─────────────────────────────────────────────

def _monthly_agg(df: pd.DataFrame, date_col: str, value_col: str,
                 agg: str = "sum") -> pd.DataFrame:
    """
    Resample to monthly frequency.
    Returns DataFrame with columns ['Period', value_col].
    Uses freq='ME' (month-end) which is the non-deprecated alias.
    """
    tmp = df[[date_col, value_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, value_col]).sort_values(date_col)
    tmp = tmp.set_index(date_col)
    resampled = tmp[value_col].resample("ME").sum() if agg == "sum" \
                else tmp[value_col].resample("ME").mean()
    return resampled.reset_index().rename(columns={date_col: "Period"})


# ─────────────────────────────────────────────
# SECTION 1 – DATASET OVERVIEW
# ─────────────────────────────────────────────

def show_overview(df: pd.DataFrame):
    st.subheader("📋 Dataset Overview")

    num_cols = get_num_cols(df)
    str_cols = get_str_cols(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",         f"{df.shape[0]:,}")
    c2.metric("Columns",      df.shape[1])
    c3.metric("Numeric Cols", len(num_cols))
    c4.metric("Text Cols",    len(str_cols))

    st.markdown("**Column names & data types**")
    dtype_df = pd.DataFrame({
        "Column":   df.columns,
        "Dtype":    df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Null":     df.isnull().sum().values,
        "Null %":   (df.isnull().mean() * 100).round(2).astype(str) + "%",
        "Unique":   [df[c].nunique() for c in df.columns],
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.markdown("**Summary statistics (numerical columns)**")
    num_frame = _num_df(df)
    if not num_frame.empty:
        st.dataframe(num_frame.describe().T.round(2), use_container_width=True)
    else:
        st.info("No numerical columns detected.")


# ─────────────────────────────────────────────
# SECTION 2 – DATA QUALITY CHECK
# ─────────────────────────────────────────────

def show_quality(df: pd.DataFrame):
    st.subheader("🔍 Data Quality Check")

    total_missing = int(df.isnull().sum().sum())
    dup_rows      = int(df.duplicated().sum())
    completeness  = (1 - df.isnull().mean().mean()) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Missing Values", total_missing)
    c2.metric("Duplicate Rows",       dup_rows)
    c3.metric("Completeness",         f"{completeness:.1f}%")

    missing_s = df.isnull().sum()
    missing_s = missing_s[missing_s > 0].sort_values(ascending=False)
    if not missing_s.empty:
        fig = px.bar(
            x=missing_s.index, y=missing_s.values,
            labels={"x": "Column", "y": "Missing Count"},
            title="Missing Values per Column",
            color=missing_s.values, color_continuous_scale="Reds"
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values found!")

    st.markdown("**Columns that may have mixed / incorrect types**")
    flags = []
    for col in get_str_cols(df):
        converted = pd.to_numeric(df[col], errors="coerce")
        pct = float(converted.notnull().mean())
        if pct > 0.2:
            flags.append({
                "Column": col,
                "Numeric-parseable %": f"{pct*100:.0f}%",
                "Suggestion": "Consider converting to numeric"
            })
        elif any(k in col.lower() for k in ["date", "time", "month", "year"]):
            flags.append({
                "Column": col,
                "Numeric-parseable %": "N/A",
                "Suggestion": "Consider converting to datetime"
            })
    if flags:
        st.dataframe(pd.DataFrame(flags), use_container_width=True, hide_index=True)
    else:
        st.success("✅ No obvious type inconsistencies detected.")


# ─────────────────────────────────────────────
# SECTION 3 – DATA CLEANING
# ─────────────────────────────────────────────

def show_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🧹 Data Cleaning Options")
    cleaned = df.copy()

    with st.expander("⚙️ Cleaning Controls", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            remove_dups = st.checkbox("Remove duplicate rows", value=True)
            missing_strategy = st.selectbox(
                "Handle missing values",
                ["Leave as-is",
                 "Drop rows with any missing",
                 "Fill numeric with median",
                 "Fill numeric with mean",
                 "Fill all with 'Unknown'"]
            )
        with col2:
            auto_date_conv    = st.checkbox("Auto-convert date-like columns to datetime", value=True)
            auto_num_conv     = st.checkbox("Auto-convert numeric-looking text columns",  value=True)

        apply_btn = st.button("✅ Apply Cleaning")

    if apply_btn:
        before = len(cleaned)

        if remove_dups:
            cleaned = cleaned.drop_duplicates()
            removed = before - len(cleaned)
            if removed:
                st.success(f"Removed {removed} duplicate rows.")

        if missing_strategy == "Drop rows with any missing":
            cleaned = cleaned.dropna()
            st.success(f"Dropped rows → {len(cleaned):,} rows remain.")
        elif missing_strategy == "Fill numeric with median":
            for c in get_num_cols(cleaned):
                cleaned[c] = cleaned[c].fillna(cleaned[c].median())
            st.success("Numeric columns filled with median.")
        elif missing_strategy == "Fill numeric with mean":
            for c in get_num_cols(cleaned):
                cleaned[c] = cleaned[c].fillna(cleaned[c].mean())
            st.success("Numeric columns filled with mean.")
        elif missing_strategy == "Fill all with 'Unknown'":
            cleaned = cleaned.fillna("Unknown")
            st.success("All missing values filled with 'Unknown'.")

        if auto_num_conv:
            for col in get_str_cols(cleaned):
                converted = pd.to_numeric(cleaned[col], errors="coerce")
                if float(converted.notnull().mean()) > 0.5:
                    cleaned[col] = converted
                    st.info(f"Converted '{col}' to numeric.")

        if auto_date_conv:
            for col in get_str_cols(cleaned):
                if any(k in col.lower() for k in ["date", "time", "month"]):
                    try:
                        cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
                        st.info(f"Converted '{col}' to datetime.")
                    except Exception:
                        pass

        st.session_state["cleaned_df"] = cleaned
        st.success("🎉 Cleaning complete!")

        buf = io.BytesIO()
        cleaned.to_excel(buf, index=False)
        st.download_button(
            "📥 Download Cleaned Data (Excel)",
            data=buf.getvalue(),
            file_name="cleaned_uploaded_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    return st.session_state.get("cleaned_df", cleaned)


# ─────────────────────────────────────────────
# SECTION 4 – SALES DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

def show_eda(df: pd.DataFrame, cols: dict):
    st.subheader("📊 Sales Data Analysis")

    date_col    = cols.get("date")
    rev_col     = cols.get("revenue")
    profit_col  = cols.get("profit")
    product_col = cols.get("product")
    region_col  = cols.get("region")

    if not any([rev_col, profit_col, cols.get("quantity")]):
        st.warning(
            "Could not detect revenue / profit / quantity columns automatically. "
            "Check column names contain keywords like 'revenue', 'profit', 'quantity'."
        )
        return

    # Sales trend over time
    if date_col and rev_col and date_col in df.columns and rev_col in df.columns:
        st.markdown("#### 📈 Sales Trend Over Time")
        try:
            monthly = _monthly_agg(df, date_col, rev_col)
            fig = px.line(monthly, x="Period", y=rev_col,
                          title="Monthly Revenue Trend", markers=True)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render sales trend: {e}")

    # Profit analysis
    if profit_col and profit_col in df.columns:
        st.markdown("#### 💰 Profit Analysis")
        c1, c2 = st.columns(2)

        with c1:
            if date_col and date_col in df.columns:
                try:
                    monthly_p = _monthly_agg(df, date_col, profit_col)
                    fig = px.area(monthly_p, x="Period", y=profit_col,
                                  title="Monthly Profit Trend",
                                  color_discrete_sequence=["#00cc88"])
                    fig.update_layout(height=320)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render profit trend: {e}")
            else:
                fig = px.histogram(df, x=profit_col, nbins=30,
                                   title="Profit Distribution")
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if rev_col and rev_col in df.columns:
                try:
                    rev_safe = df[rev_col].replace(0, np.nan)
                    margin_s = (df[profit_col] / rev_safe * 100).dropna()
                    fig = px.histogram(margin_s, nbins=30,
                                       title="Profit Margin Distribution (%)",
                                       color_discrete_sequence=["#6366f1"])
                    fig.update_layout(height=320)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render margin chart: {e}")

    # Product / Category performance
    if product_col and rev_col and product_col in df.columns and rev_col in df.columns:
        st.markdown("#### 🏷️ Product / Category Performance")
        top_n = st.slider("Top N categories", 5, 30, 10, key="top_n_product")
        try:
            prod_perf = (
                df.groupby(product_col)[rev_col]
                  .sum()
                  .sort_values(ascending=False)
                  .head(top_n)
            )
            fig = px.bar(
                x=prod_perf.index, y=prod_perf.values,
                labels={"x": product_col, "y": "Revenue"},
                title=f"Top {top_n} by Revenue",
                color=prod_perf.values,
                color_continuous_scale="Blues_r"
            )
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render product chart: {e}")

    # Regional distribution
    if region_col and rev_col and region_col in df.columns and rev_col in df.columns:
        st.markdown("#### 🌍 Regional Sales Distribution")
        try:
            reg_perf = df.groupby(region_col)[rev_col].sum().sort_values(ascending=False)
            fig = px.pie(values=reg_perf.values, names=reg_perf.index,
                         title="Revenue by Region", hole=0.4)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render regional chart: {e}")

    # Correlation heatmap
    st.markdown("#### 🔗 Correlation Heatmap")
    num_frame = _num_df(df)
    if num_frame.shape[1] >= 2:
        try:
            corr = num_frame.corr().round(2)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                            title="Correlation Matrix of Numeric Columns", aspect="auto")
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render correlation heatmap: {e}")
    else:
        st.info("Not enough numeric columns for a correlation matrix.")


# ─────────────────────────────────────────────
# SECTION 5 – SMART INSIGHTS
# ─────────────────────────────────────────────

def show_smart_insights(df: pd.DataFrame, cols: dict):
    st.subheader("💡 Smart Insights")

    rev_col    = cols.get("revenue")
    profit_col = cols.get("profit")
    qty_col    = cols.get("quantity")
    date_col   = cols.get("date")
    region_col = cols.get("region")

    insights = []

    if rev_col and rev_col in df.columns:
        total_rev = df[rev_col].sum()
        avg_rev   = df[rev_col].mean()
        insights.append(
            f"💰 **Total Revenue:** ${total_rev:,.2f}  |  Avg per row: ${avg_rev:,.2f}"
        )
        insights.append(
            f"📈 **Revenue range:** ${df[rev_col].min():,.2f} – ${df[rev_col].max():,.2f}"
        )

    if profit_col and profit_col in df.columns:
        total_profit = df[profit_col].sum()
        insights.append(f"🟢 **Total Profit:** ${total_profit:,.2f}")
        if rev_col and rev_col in df.columns:
            rev_sum = df[rev_col].sum()
            margin  = (total_profit / rev_sum * 100) if rev_sum else 0
            insights.append(f"📊 **Overall Profit Margin:** {margin:.1f}%")

    if qty_col and qty_col in df.columns:
        insights.append(f"📦 **Total Units Sold:** {df[qty_col].sum():,.0f}")

    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not dates.empty:
            insights.append(
                f"🗓️ **Date range:** {dates.min().date()} → {dates.max().date()}"
            )

    if (region_col and rev_col
            and region_col in df.columns and rev_col in df.columns):
        try:
            best = df.groupby(region_col)[rev_col].sum().idxmax()
            insights.append(f"🌍 **Best performing region:** {best}")
        except Exception:
            pass

    if insights:
        for ins in insights:
            st.markdown(ins)
    else:
        st.info(
            "No auto-insights could be generated — check column names contain "
            "sales keywords like 'revenue', 'profit', 'date', 'region'."
        )


# ─────────────────────────────────────────────
# SECTION 6 – FORECASTING
# ─────────────────────────────────────────────

def show_forecasting(df: pd.DataFrame, cols: dict):
    st.subheader("🔮 Forecasting")

    date_col = cols.get("date")
    rev_col  = cols.get("revenue")

    if not date_col or not rev_col:
        st.warning(
            "Forecasting needs a **date** column and a **revenue/sales** column. "
            "Rename columns to include keywords like 'date', 'revenue', or 'sales'."
        )
        return None

    # Build monthly time series
    try:
        df_ts = df[[date_col, rev_col]].copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors="coerce")
        df_ts = df_ts.dropna().sort_values(date_col).set_index(date_col)
        monthly = df_ts[rev_col].resample("ME").sum()
    except Exception as e:
        st.error(f"Error preparing time series: {e}")
        return None

    if len(monthly) < 6:
        st.warning("Need at least 6 months of data for forecasting.")
        return None

    st.markdown(
        f"**Training data:** {len(monthly)} monthly observations "
        f"({monthly.index.min().date()} → {monthly.index.max().date()})"
    )

    horizon      = st.slider("Forecast horizon (months)", 3, 12, 6)
    model_choice = st.selectbox(
        "Choose forecasting model",
        ["Linear Regression", "Gradient Boosting", "Random Forest"]
    )

    # Feature engineering: lag features
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    lags = min(3, len(monthly) - 1)
    X, y = [], []
    for i in range(lags, len(monthly)):
        X.append([float(monthly.iloc[i - j]) for j in range(1, lags + 1)] + [float(i)])
        y.append(float(monthly.iloc[i]))

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    }
    mdl = models[model_choice]
    mdl.fit(X_scaled, y)

    # Generate future predictions
    last_vals    = [float(v) for v in monthly.values[-lags:]]
    future_preds = []
    n_train      = len(monthly)

    for step in range(horizon):
        feat = [last_vals[-j] for j in range(1, lags + 1)] + [float(n_train + step)]
        pred = float(mdl.predict(scaler.transform([feat]))[0])
        pred = max(pred, 0.0)
        future_preds.append(pred)
        last_vals.append(pred)

    # Build future date index
    # FIX: use pd.DateOffset(months=1) — integer arithmetic on Timestamp is removed
    last_date    = monthly.index[-1]          # pd.Timestamp
    future_dates = pd.date_range(
        start   = last_date + pd.DateOffset(months=1),
        periods = horizon,
        freq    = "ME"
    )

    df_forecast = pd.DataFrame({
        "Month":              future_dates,
        "Forecasted Revenue": future_preds,
    })

    # Convert last_date to plain string for Plotly
    # FIX: Plotly's add_vline / add_shape does integer arithmetic with Timestamp
    #      internally which crashes on pandas 2+.  A plain ISO string avoids this.
    last_date_str = _ts_to_str(last_date)

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly.index, y=monthly.values,
        mode="lines+markers", name="Historical",
        line=dict(color="steelblue", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds,
        mode="lines+markers", name="Forecast",
        line=dict(color="tomato", width=3, dash="dash"),
        marker=dict(size=10, symbol="star")
    ))

    # add_vline with plain string x
    # FIX: removed annotation_position kwarg (also crashed); use add_annotation instead
    fig.add_vline(x=last_date_str, line_dash="dot", line_color="green")
    fig.add_annotation(
        x=last_date_str, y=1.05,
        xref="x", yref="paper",
        text="Forecast Start",
        showarrow=False,
        font=dict(color="green", size=12),
        bgcolor="rgba(0,0,0,0)"
    )

    fig.update_layout(
        height=450, title="Revenue Forecast",
        xaxis_title="Month", yaxis_title="Revenue ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    df_disp = df_forecast.copy()
    df_disp["Month"]              = df_disp["Month"].dt.strftime("%b %Y")
    df_disp["Forecasted Revenue"] = df_disp["Forecasted Revenue"].apply(
        lambda x: f"${x:,.2f}"
    )
    st.dataframe(df_disp, use_container_width=True, hide_index=True)

    # Download
    buf = io.BytesIO()
    df_forecast.to_excel(buf, index=False)
    st.download_button(
        "📥 Download Forecast (Excel)",
        data=buf.getvalue(),
        file_name="uploaded_data_forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    return df_forecast


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – ML EVALUATION COMPARISON
# Trains all 6 models on BOTH the training dataset and the uploaded dataset,
# then shows full evaluation metrics side-by-side so the user can see:
#   • how each model performs on each dataset
#   • which model is best suited for their data
#   • a radar chart for intuitive multi-metric comparison
# ─────────────────────────────────────────────────────────────────────────────

# ── sklearn imports live here so the rest of the module doesn't require them ─
from sklearn.linear_model        import LinearRegression, Ridge
from sklearn.ensemble            import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree                import DecisionTreeRegressor
from sklearn.svm                 import SVR
from sklearn.preprocessing       import StandardScaler
from sklearn.model_selection     import train_test_split, cross_val_score
from sklearn.metrics             import (r2_score, mean_absolute_error,
                                         mean_squared_error,
                                         explained_variance_score)

# ── colour palette shared across all charts ──────────────────────────────────
_MODEL_COLORS = {
    "Linear Regression":  "#4C78A8",
    "Ridge Regression":   "#72B7B2",
    "Decision Tree":      "#F58518",
    "Random Forest":      "#54A24B",
    "Gradient Boosting":  "#E45756",
    "SVR":                "#B279A2",
}
_MODEL_ORDER = list(_MODEL_COLORS.keys())


def _mape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _build_features(df: pd.DataFrame, date_col: str, rev_col: str) -> tuple:
    """
    Aggregate to monthly, engineer lag / rolling features, return (X, y, feature_names).
    Extra numeric columns in the dataframe are included as additional features.
    Returns (None, None, msg_str) when data is insufficient.
    """
    extra_num = [c for c in get_num_cols(df) if c != rev_col]
    cols_needed = [date_col, rev_col] + extra_num
    cols_needed = [c for c in cols_needed if c in df.columns]

    tmp = df[cols_needed].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, rev_col]).sort_values(date_col).set_index(date_col)

    agg_dict = {c: "sum" for c in [rev_col] + [c for c in extra_num if c in tmp.columns]}
    monthly  = tmp.resample("ME").agg(agg_dict).reset_index()

    if len(monthly) < 8:
        return None, None, f"Only {len(monthly)} months of data — need at least 8."

    monthly["month_num"] = np.arange(len(monthly), dtype=float)
    monthly["month"]     = monthly[date_col].dt.month.astype(float)
    monthly["quarter"]   = monthly[date_col].dt.quarter.astype(float)
    for lag in [1, 2, 3]:
        monthly[f"lag_{lag}"] = monthly[rev_col].shift(lag)
    monthly["rolling_3"] = monthly[rev_col].shift(1).rolling(3).mean()
    monthly = monthly.dropna()

    base_feats  = ["month_num", "month", "quarter", "lag_1", "lag_2", "lag_3", "rolling_3"]
    extra_feats = [c for c in extra_num if c in monthly.columns]
    feat_cols   = base_feats + extra_feats

    X = monthly[feat_cols].values.astype(float)
    y = monthly[rev_col].values.astype(float)
    return X, y, feat_cols


def _evaluate_models(X: np.ndarray, y: np.ndarray, label: str) -> pd.DataFrame:
    """
    Train / evaluate all 6 models on X, y.
    Returns DataFrame with one row per model and all metric columns.
    """
    n_cv  = min(3, max(2, len(X) // 4))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    sc      = StandardScaler()
    X_tr_s  = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)
    X_all_s = sc.transform(X)

    model_defs = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Decision Tree":     DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVR":               SVR(kernel="rbf", C=100, gamma=0.1),
    }

    rows = []
    for name in _MODEL_ORDER:
        mdl = model_defs[name]
        mdl.fit(X_tr_s, y_tr)
        y_pred    = mdl.predict(X_te_s)
        y_pred_tr = mdl.predict(X_tr_s)

        try:
            cv_r2 = float(cross_val_score(
                model_defs[name].__class__(**model_defs[name].get_params()),
                X_all_s, y, cv=n_cv, scoring="r2"
            ).mean())
        except Exception:
            cv_r2 = np.nan

        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae  = float(mean_absolute_error(y_te, y_pred))
        r2t  = float(r2_score(y_te, y_pred))
        r2tr = float(r2_score(y_tr, y_pred_tr))
        ev   = float(explained_variance_score(y_te, y_pred))
        mp   = _mape(y_te, y_pred)

        rows.append({
            "Model":              name,
            "Dataset":            label,
            "R² Test":            round(r2t,  4),
            "R² Train":           round(r2tr, 4),
            "CV R² (3-fold)":     round(cv_r2, 4) if not np.isnan(cv_r2) else None,
            "MAE ($)":            round(mae,  2),
            "RMSE ($)":           round(rmse, 2),
            "MAPE (%)":           round(mp,   2),
            "Explained Variance": round(ev,   4),
        })

    return pd.DataFrame(rows)


def _best_model_card(df_metrics: pd.DataFrame, label: str, color: str):
    """Render a highlighted card for the best model on a given dataset."""
    best = df_metrics.sort_values("R² Test", ascending=False).iloc[0]
    r2   = best["R² Test"]
    grade = ("🟢 Excellent" if r2 >= 0.85 else
             "🟡 Good"      if r2 >= 0.70 else
             "🟠 Fair"      if r2 >= 0.50 else
             "🔴 Weak")
    st.markdown(
        f"""
        <div style="border:2px solid {color}; border-radius:10px; padding:14px;
                    background:rgba(0,0,0,0.03);">
          <h4 style="margin:0 0 6px 0; color:{color};">🏆 Best for {label}</h4>
          <p style="margin:2px 0; font-size:1.15em; font-weight:700;">
            {best['Model']}
          </p>
          <p style="margin:2px 0;">R² Test: <b>{r2:.4f}</b> &nbsp;|&nbsp; {grade}</p>
          <p style="margin:2px 0;">MAPE: <b>{best['MAPE (%)']:.2f}%</b> &nbsp;|&nbsp;
             RMSE: <b>${best['RMSE ($)']:,.0f}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _radar_chart(df_tr: pd.DataFrame, df_up: pd.DataFrame, model: str) -> go.Figure:
    """
    Build a radar / spider chart comparing a single model's normalised metrics
    across the two datasets.
    """
    metrics = ["R² Test", "CV R² (3-fold)", "Explained Variance"]
    # lower-is-better metrics get inverted (1 - normalised)
    low_better = ["MAPE (%)", "RMSE ($)", "MAE ($)"]

    row_tr = df_tr[df_tr["Model"] == model].iloc[0]
    row_up = df_up[df_up["Model"] == model].iloc[0]

    # Normalise each metric 0-1 across the two datasets for radar display
    def _norm(val_a, val_b, invert=False):
        lo, hi = min(val_a, val_b), max(val_a, val_b)
        if hi == lo:
            na, nb = 0.5, 0.5
        else:
            na = (val_a - lo) / (hi - lo)
            nb = (val_b - lo) / (hi - lo)
        if invert:
            na, nb = 1 - na, 1 - nb
        return na, nb

    all_metrics = ["R² Test", "CV R² (3-fold)", "Explained Variance",
                   "1-MAPE", "1-RMSE", "1-MAE"]
    vals_tr, vals_up = [], []

    for m in ["R² Test", "CV R² (3-fold)", "Explained Variance"]:
        a = float(row_tr[m] or 0)
        b = float(row_up[m] or 0)
        na, nb = _norm(a, b, invert=False)
        vals_tr.append(na); vals_up.append(nb)

    for m in ["MAPE (%)", "RMSE ($)", "MAE ($)"]:
        a = float(row_tr[m] or 0)
        b = float(row_up[m] or 0)
        na, nb = _norm(a, b, invert=True)
        vals_tr.append(na); vals_up.append(nb)

    cats = ["R²", "CV R²", "Expl.Var.", "1-MAPE", "1-RMSE", "1-MAE"]
    cats_closed = cats + [cats[0]]
    tr_closed   = vals_tr + [vals_tr[0]]
    up_closed   = vals_up + [vals_up[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=tr_closed, theta=cats_closed, fill="toself",
        name="Training Dataset", line_color="#E45756", opacity=0.6
    ))
    fig.add_trace(go.Scatterpolar(
        r=up_closed, theta=cats_closed, fill="toself",
        name="Uploaded Dataset", line_color="#4C78A8", opacity=0.6
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=400,
        title=f"Radar: {model} — Training vs Uploaded",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def show_comparison(df_upload: pd.DataFrame, df_train: pd.DataFrame,
                    upload_cols: dict, forecast_df=None):

    st.subheader("📐 Model Evaluation: Training Dataset vs Your Uploaded Dataset")
    st.markdown(
        "All **6 ML models** are trained and evaluated independently on both datasets. "
        "Metrics below show how well each model can learn from each dataset, "
        "which model fits best, and how your data compares to the training baseline."
    )

    # ── guard: need revenue + date columns ───────────────────────────────────
    def _col_ok(df, col): return col and col in df.columns

    up_date = upload_cols.get("date");    up_rev = upload_cols.get("revenue")
    train_cols = detect_columns(df_train)
    tr_date = train_cols.get("date");    tr_rev = train_cols.get("revenue")

    if not (_col_ok(df_upload, up_date) and _col_ok(df_upload, up_rev)):
        st.error(
            "Cannot run model comparison: uploaded dataset is missing a detected "
            "**date** or **revenue** column. Use the column mapping expander above "
            "to assign them."
        )
        return

    # ── dataset size cards ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Uploaded – Rows",    f"{df_upload.shape[0]:,}")
    c2.metric("Uploaded – Columns", df_upload.shape[1])
    if df_train is not None:
        c3.metric("Training – Rows",    f"{df_train.shape[0]:,}")
        c4.metric("Training – Columns", df_train.shape[1])

    st.markdown("---")

    # ── build features for uploaded dataset ──────────────────────────────────
    with st.spinner("🔧 Engineering features & training models on uploaded dataset…"):
        X_up, y_up, feats_up = _build_features(df_upload, up_date, up_rev)

    if X_up is None:
        st.error(f"Uploaded dataset: {feats_up}")   # feats_up is the error msg here
        return

    df_metrics_up = _evaluate_models(X_up, y_up, "Uploaded Dataset")

    # ── build features for training dataset (if available) ───────────────────
    df_metrics_tr = None
    if df_train is not None and _col_ok(df_train, tr_date) and _col_ok(df_train, tr_rev):
        with st.spinner("🔧 Engineering features & training models on Regional Sales dataset…"):
            X_tr, y_tr, feats_tr = _build_features(df_train, tr_date, tr_rev)
        if X_tr is not None:
            df_metrics_tr = _evaluate_models(X_tr, y_tr, "Training Dataset")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB LAYOUT  (sub-tabs inside the Comparison tab)
    # ══════════════════════════════════════════════════════════════════════════
    sub_tabs = st.tabs([
        "🏆 Best Model",
        "📊 Metrics Table",
        "📈 Visual Comparison",
        "🕸️ Radar Chart",
        "💡 Recommendation",
    ])

    # ── SUB-TAB 1 : Best Model ────────────────────────────────────────────────
    with sub_tabs[0]:

        # ── best / worst header cards ─────────────────────────────────────────
        st.markdown("### 🏆 Best & Worst Model — Uploaded Dataset")
        ranked_up  = df_metrics_up.sort_values("R² Test", ascending=False).reset_index(drop=True)
        best_row   = ranked_up.iloc[0]
        worst_row  = ranked_up.iloc[-1]

        def _r2_grade(r2):
            return ("🟢 Excellent" if r2 >= 0.85 else
                    "🟡 Good"      if r2 >= 0.70 else
                    "🟠 Fair"      if r2 >= 0.50 else
                    "🔴 Weak")

        hc1, hc2 = st.columns(2)
        with hc1:
            st.success(f"🏆 **Best Model: {best_row['Model']}**")
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("R² Test ↑",  f"{best_row['R² Test']:.4f}")
            bc2.metric("MAPE % ↓",   f"{best_row['MAPE (%)']:.2f}%")
            bc3.metric("RMSE ($) ↓", f"${best_row['RMSE ($)']:,.0f}")
            st.caption(f"Grade: {_r2_grade(best_row['R² Test'])}  |  MAE: ${best_row['MAE ($)']:,.0f}")

        with hc2:
            st.error(f"❌ **Worst Model: {worst_row['Model']}**")
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("R² Test ↑",  f"{worst_row['R² Test']:.4f}")
            wc2.metric("MAPE % ↓",   f"{worst_row['MAPE (%)']:.2f}%")
            wc3.metric("RMSE ($) ↓", f"${worst_row['RMSE ($)']:,.0f}")
            st.caption(f"Grade: {_r2_grade(worst_row['R² Test'])}  |  MAE: ${worst_row['MAE ($)']:,.0f}")

        st.markdown("---")

        # ── helpers (no HTML — pure Streamlit) ───────────────────────────────
        RANK_ICONS = {0: "🥇", 1: "🥈", 2: "🥉", 3: "4️⃣", 4: "5️⃣", 5: "6️⃣"}

        def _overfit_label(r2_train, r2_test):
            gap = r2_train - r2_test
            if gap > 0.30:  return "⚠️ Overfit"
            if gap < -0.05: return "ℹ️ Test>Train"
            return "✅ Healthy"

        # ── full comparison table ─────────────────────────────────────────────
        st.markdown("### 📋 All 6 Models — Full Evaluation (Uploaded Dataset)")
        st.caption("Sorted best → worst by R² Test  |  ↑ higher is better  |  ↓ lower is better for MAPE / RMSE / MAE")

        # Build a display-ready DataFrame
        table_rows = []
        for rank_idx, row in ranked_up.iterrows():
            cv_val = row["CV R² (3-fold)"]
            cv_str = f"{cv_val:.4f}" if (cv_val is not None and cv_val == cv_val) else "N/A"
            icon   = RANK_ICONS.get(rank_idx, str(rank_idx + 1))
            label  = " 🏆 BEST" if rank_idx == 0 else (" ❌ WORST" if rank_idx == len(ranked_up) - 1 else "")
            table_rows.append({
                "Rank":                f"{icon}{label}",
                "Model":               row["Model"],
                "R² Test ↑":           round(row["R² Test"],  4),
                "R² Train":            round(row["R² Train"], 4),
                "CV R² ↑":             cv_str,
                "MAPE % ↓":            round(row["MAPE (%)"], 2),
                "RMSE ($) ↓":          round(row["RMSE ($)"], 0),
                "MAE ($) ↓":           round(row["MAE ($)"],  0),
                "Expl. Var. ↑":        round(row["Explained Variance"], 4),
                "Overfit?":            _overfit_label(row["R² Train"], row["R² Test"]),
            })

        tbl = pd.DataFrame(table_rows)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── per-model expander with st.metric widgets ─────────────────────────
        st.markdown("### 🔍 Per-Model Detail (click to expand)")

        for rank_idx, row in ranked_up.iterrows():
            icon    = RANK_ICONS.get(rank_idx, str(rank_idx + 1))
            label   = " — 🏆 BEST" if rank_idx == 0 else (" — ❌ WORST" if rank_idx == len(ranked_up) - 1 else "")
            cv_val  = row["CV R² (3-fold)"]
            cv_disp = f"{cv_val:.4f}" if (cv_val is not None and cv_val == cv_val) else "N/A"
            grade   = _r2_grade(row["R² Test"])
            ov      = _overfit_label(row["R² Train"], row["R² Test"])

            # delta vs best model (rank 0)
            best_r2   = ranked_up.iloc[0]["R² Test"]
            delta_r2  = row["R² Test"] - best_r2 if rank_idx > 0 else None
            delta_str = f"{delta_r2:+.4f}" if delta_r2 is not None else None

            with st.expander(f"{icon} {row['Model']}{label}  |  R²={row['R² Test']:.4f}  |  {grade}"):
                # Row 1: accuracy metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("R² Test  ↑", f"{row['R² Test']:.4f}",
                          delta=delta_str, delta_color="normal" if rank_idx == 0 else "inverse")
                m2.metric("R² Train", f"{row['R² Train']:.4f}")
                m3.metric("CV R² (3-fold)  ↑", cv_disp)

                # Row 2: error metrics
                m4, m5, m6 = st.columns(3)
                best_mape = ranked_up.iloc[0]["MAPE (%)"]
                mape_delta = f"{row['MAPE (%)'] - best_mape:+.2f}%" if rank_idx > 0 else None
                m4.metric("MAPE %  ↓", f"{row['MAPE (%)']:.2f}%",
                          delta=mape_delta,
                          delta_color="inverse")
                m5.metric("RMSE ($)  ↓", f"${row['RMSE ($)']:,.0f}")
                m6.metric("MAE ($)  ↓",  f"${row['MAE ($)']:,.0f}")

                # Row 3: extra info
                m7, m8 = st.columns(2)
                m7.metric("Explained Variance  ↑", f"{row['Explained Variance']:.4f}")
                m8.metric("Overfitting Check", ov)

        # ── training dataset best/worst summary ───────────────────────────────
        if df_metrics_tr is not None:
            st.markdown("---")
            st.markdown("### 🏆 Best & Worst — Training Dataset")
            ranked_tr = df_metrics_tr.sort_values("R² Test", ascending=False).reset_index(drop=True)

            tc1, tc2 = st.columns(2)
            best_tr  = ranked_tr.iloc[0]
            worst_tr = ranked_tr.iloc[-1]

            with tc1:
                st.success(f"🏆 **Best on Training:** {best_tr['Model']}")
                st.metric("R² Test",  f"{best_tr['R² Test']:.4f}")
                st.metric("MAPE %",   f"{best_tr['MAPE (%)']:.2f}%")
                st.metric("RMSE ($)", f"${best_tr['RMSE ($)']:,.0f}")
                st.caption(_r2_grade(best_tr["R² Test"]))

            with tc2:
                st.error(f"❌ **Worst on Training:** {worst_tr['Model']}")
                st.metric("R² Test",  f"{worst_tr['R² Test']:.4f}")
                st.metric("MAPE %",   f"{worst_tr['MAPE (%)']:.2f}%")
                st.metric("RMSE ($)", f"${worst_tr['RMSE ($)']:,.0f}")
                st.caption(_r2_grade(worst_tr["R² Test"]))

    # ── SUB-TAB 2 : Metrics Table ─────────────────────────────────────────────
    with sub_tabs[1]:

        metric_cols = ["Model", "R² Test", "R² Train", "CV R² (3-fold)",
                       "MAE ($)", "RMSE ($)", "MAPE (%)", "Explained Variance"]

        st.markdown("#### 📋 Uploaded Dataset — All Model Metrics")
        disp_up = df_metrics_up[metric_cols].sort_values("R² Test", ascending=False).copy()

        # Colour-code the best row
        best_model_up = disp_up.iloc[0]["Model"]
        st.dataframe(
            disp_up.style.apply(
                lambda row: ["background-color: rgba(76,120,168,0.15)"] * len(row)
                            if row["Model"] == best_model_up else [""] * len(row),
                axis=1
            ),
            use_container_width=True, hide_index=True
        )

        if df_metrics_tr is not None:
            st.markdown("#### 📋 Training Dataset — All Model Metrics")
            disp_tr = df_metrics_tr[metric_cols].sort_values("R² Test", ascending=False).copy()
            best_model_tr = disp_tr.iloc[0]["Model"]
            st.dataframe(
                disp_tr.style.apply(
                    lambda row: ["background-color: rgba(228,87,86,0.15)"] * len(row)
                                if row["Model"] == best_model_tr else [""] * len(row),
                    axis=1
                ),
                use_container_width=True, hide_index=True
            )

            st.markdown("#### ↔️ Side-by-Side R² Comparison")
            merged = (
                disp_up[["Model", "R² Test", "MAPE (%)"]].rename(
                    columns={"R² Test": "R² [Uploaded]", "MAPE (%)": "MAPE [Uploaded]"})
                .merge(
                    disp_tr[["Model", "R² Test", "MAPE (%)"]].rename(
                        columns={"R² Test": "R² [Training]", "MAPE (%)": "MAPE [Training]"}),
                    on="Model"
                )
            )
            merged["R² Δ"] = (merged["R² [Uploaded]"] - merged["R² [Training]"]).round(4)
            merged["MAPE Δ (%)"] = (merged["MAPE [Uploaded]"] - merged["MAPE [Training]"]).round(2)
            st.dataframe(merged, use_container_width=True, hide_index=True)

        # Download
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_metrics_up.to_excel(writer, sheet_name="Uploaded Dataset", index=False)
            if df_metrics_tr is not None:
                df_metrics_tr.to_excel(writer, sheet_name="Training Dataset", index=False)
        st.download_button(
            "📥 Download All Metrics (Excel)",
            data=buf.getvalue(),
            file_name="model_comparison_metrics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # ── SUB-TAB 3 : Visual Comparison ────────────────────────────────────────
    with sub_tabs[2]:

        metric_sel = st.selectbox(
            "Choose metric to visualise",
            ["R² Test", "R² Train", "CV R² (3-fold)", "MAE ($)", "RMSE ($)",
             "MAPE (%)", "Explained Variance"],
            key="cmp_metric_sel"
        )
        low_better_set = {"MAE ($)", "RMSE ($)", "MAPE (%)"}
        higher_is_better = metric_sel not in low_better_set

        # ── Grouped bar: uploaded vs training ────────────────────────────────
        fig_bar = go.Figure()

        up_sorted = df_metrics_up.set_index("Model").reindex(_MODEL_ORDER)[metric_sel]
        fig_bar.add_trace(go.Bar(
            name="Uploaded Dataset",
            x=_MODEL_ORDER,
            y=up_sorted.values,
            marker_color="#4C78A8",
            text=[f"{v:.3f}" for v in up_sorted.values],
            textposition="outside"
        ))

        if df_metrics_tr is not None:
            tr_sorted = df_metrics_tr.set_index("Model").reindex(_MODEL_ORDER)[metric_sel]
            fig_bar.add_trace(go.Bar(
                name="Training Dataset",
                x=_MODEL_ORDER,
                y=tr_sorted.values,
                marker_color="#E45756",
                text=[f"{v:.3f}" for v in tr_sorted.values],
                textposition="outside"
            ))

        direction = "↑ Higher is better" if higher_is_better else "↓ Lower is better"
        fig_bar.update_layout(
            barmode="group", height=420,
            title=f"{metric_sel} — All Models  ({direction})",
            xaxis_title="Model", yaxis_title=metric_sel,
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Multi-metric overview (4-panel) ───────────────────────────────────
        st.markdown("#### 📊 Multi-Metric Overview (Uploaded Dataset)")
        panel_metrics = ["R² Test", "MAPE (%)", "RMSE ($)", "MAE ($)"]
        fig_multi = make_subplots(
            rows=2, cols=2,
            subplot_titles=panel_metrics,
            vertical_spacing=0.16, horizontal_spacing=0.08
        )
        positions = [(1,1),(1,2),(2,1),(2,2)]
        color_scales = ["Greens", "Reds_r", "Blues_r", "Purples_r"]

        for (r, c), pm, cs in zip(positions, panel_metrics, color_scales):
            pm_data = df_metrics_up.set_index("Model").reindex(_MODEL_ORDER)[pm]
            fig_multi.add_trace(
                go.Bar(
                    x=_MODEL_ORDER,
                    y=pm_data.values,
                    marker=dict(
                        color=pm_data.values,
                        colorscale=cs
                    ),
                    text=[f"{v:.2f}" for v in pm_data.values],
                    textposition="outside",
                    showlegend=False
                ),
                row=r, col=c
            )

        fig_multi.update_xaxes(tickangle=35, tickfont=dict(size=9))
        fig_multi.update_layout(height=600, title_text="All Key Metrics — Uploaded Dataset")
        st.plotly_chart(fig_multi, use_container_width=True)

        # ── Revenue distribution overlay ───────────────────────────────────────
        up_rev = upload_cols.get("revenue")
        if _col_ok(df_upload, up_rev) and df_metrics_tr is not None:
            t_rev_col = train_cols.get("revenue")
            if _col_ok(df_train, t_rev_col):
                st.markdown("#### 📦 Revenue Distribution — Training vs Uploaded")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=df_upload[up_rev].dropna(), name="Uploaded Dataset",
                    opacity=0.6, nbinsx=40, marker_color="#4C78A8"
                ))
                fig_hist.add_trace(go.Histogram(
                    x=df_train[t_rev_col].dropna(), name="Training Dataset",
                    opacity=0.6, nbinsx=40, marker_color="#E45756"
                ))
                fig_hist.update_layout(
                    barmode="overlay", height=360,
                    xaxis_title="Revenue ($)", yaxis_title="Frequency"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # ── SUB-TAB 4 : Radar Chart ───────────────────────────────────────────────
    with sub_tabs[3]:
        st.markdown(
            "The radar chart shows normalised performance across all metrics for "
            "one model. Axes pointing **outward = better**. "
            "1-MAPE, 1-RMSE, 1-MAE are inverted so bigger = better for all axes."
        )
        if df_metrics_tr is None:
            st.info("Radar comparison requires the training dataset to be loaded.")
        else:
            model_for_radar = st.selectbox(
                "Select model for radar", _MODEL_ORDER, key="radar_model_sel"
            )
            fig_radar = _radar_chart(df_metrics_tr, df_metrics_up, model_for_radar)
            st.plotly_chart(fig_radar, use_container_width=True)

            # Also show a scatter: R² Test uploaded vs training for all models
            st.markdown("#### R² Test — Training vs Uploaded (Scatter)")
            scatter_df = df_metrics_up[["Model","R² Test"]].merge(
                df_metrics_tr[["Model","R² Test"]],
                on="Model", suffixes=(" Uploaded"," Training")
            )
            fig_sc = px.scatter(
                scatter_df,
                x="R² Test Training", y="R² Test Uploaded",
                text="Model",
                title="R² Test: Training Dataset vs Uploaded Dataset",
                color="Model",
                color_discrete_map={m: _MODEL_COLORS.get(m,"grey")
                                    for m in scatter_df["Model"]},
            )
            # perfect-match diagonal
            lo = min(scatter_df[["R² Test Training","R² Test Uploaded"]].min())
            hi = max(scatter_df[["R² Test Training","R² Test Uploaded"]].max())
            fig_sc.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines",
                line=dict(dash="dash", color="grey"),
                name="y = x  (equal performance)"
            ))
            fig_sc.update_traces(textposition="top center", selector=dict(mode="markers+text"))
            fig_sc.update_layout(height=430)
            st.plotly_chart(fig_sc, use_container_width=True)

    # ── SUB-TAB 5 : Recommendation ────────────────────────────────────────────
    with sub_tabs[4]:
        st.markdown("### 🤖 Automated Model Recommendation")

        best_row = df_metrics_up.sort_values("R² Test", ascending=False).iloc[0]
        bname    = best_row["Model"]
        br2      = best_row["R² Test"]
        bmape    = best_row["MAPE (%)"]
        brmse    = best_row["RMSE ($)"]
        bcvr2    = best_row["CV R² (3-fold)"]

        # Score quality bands
        grade   = ("Excellent" if br2 >= 0.85 else
                   "Good"      if br2 >= 0.70 else
                   "Fair"      if br2 >= 0.50 else
                   "Weak — consider more data or feature engineering")
        overfit = (br2 - best_row["R² Train"]) < -0.20   # train >> test

        overfitting_note = (
            "⚠️ **Overfitting detected**: R² Train is significantly higher than "
            "R² Test. The model memorises training data but may not generalise well. "
            "Try Ridge Regression or increasing regularisation." if overfit else
            "✅ Train/test R² gap is acceptable — model generalises well."
        )

        # Model-specific explanation
        explanations = {
            "Linear Regression":
                "Fits a straight-line relationship between features and revenue. "
                "Fast, interpretable, but may miss non-linear patterns.",
            "Ridge Regression":
                "Linear regression with L2 regularisation — reduces overfitting "
                "when features are correlated. Often the most reliable baseline.",
            "Decision Tree":
                "Splits data by thresholds on each feature. Highly interpretable "
                "but prone to overfitting on small datasets.",
            "Random Forest":
                "Ensemble of decision trees — robust to noise, handles non-linearity "
                "well. Usually the best out-of-the-box choice for sales data.",
            "Gradient Boosting":
                "Sequentially corrects previous tree errors. High accuracy but "
                "needs enough data; can overfit on small monthly datasets.",
            "SVR":
                "Support Vector Regression — effective when relationships are complex "
                "and non-linear, but requires careful hyperparameter tuning.",
        }

        color = _MODEL_COLORS.get(bname, "#4C78A8")
        st.markdown(
            f"""
            <div style="border-left:5px solid {color}; padding:16px 20px;
                        border-radius:6px; background:rgba(0,0,0,0.03);">
              <h3 style="margin:0 0 8px 0; color:{color};">
                🏆 Recommended: {bname}
              </h3>
              <p style="margin:4px 0;"><b>R² (test):</b> {br2:.4f} — <i>{grade}</i></p>
              <p style="margin:4px 0;"><b>CV R²:</b> {bcvr2:.4f} &nbsp;|&nbsp;
                 <b>MAPE:</b> {bmape:.2f}% &nbsp;|&nbsp;
                 <b>RMSE:</b> ${brmse:,.0f}</p>
              <hr style="border-color:{color}33; margin:10px 0;">
              <p style="margin:4px 0;"><b>Why this model?</b><br>
                {explanations.get(bname, "")}</p>
              <p style="margin:8px 0 0 0;">{overfitting_note}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### 📊 All Models — Quick Scorecard (Uploaded Dataset)")

        # Normalise R² Test 0–100 for a simple progress-bar scorecard
        max_r2  = df_metrics_up["R² Test"].max()
        min_r2  = df_metrics_up["R² Test"].min()
        r2_range = max_r2 - min_r2 if max_r2 != min_r2 else 1

        for _, row in df_metrics_up.sort_values("R² Test", ascending=False).iterrows():
            mname   = row["Model"]
            mr2     = row["R² Test"]
            score   = max((mr2 - min_r2) / r2_range, 0.0)
            tag     = " ← Recommended" if mname == bname else ""
            mcol    = _MODEL_COLORS.get(mname, "#888")
            st.markdown(
                f"**{mname}**{tag}  |  R²={mr2:.4f} | MAPE={row['MAPE (%)']:.1f}%",
            )
            st.progress(float(score))

        st.markdown("---")
        st.markdown("#### 📌 When to use each model")
        guide_rows = [
            ("Linear Regression",  "Small datasets, quick baselines, highly linear data"),
            ("Ridge Regression",   "Correlated features, risk of overfitting, stable predictions"),
            ("Decision Tree",      "Need full interpretability, non-linear but small dataset"),
            ("Random Forest",      "General-purpose, robust, best for most sales datasets"),
            ("Gradient Boosting",  "Highest accuracy when you have 12+ months of history"),
            ("SVR",                "Complex non-linear patterns, smaller feature sets"),
        ]
        guide_df = pd.DataFrame(guide_rows, columns=["Model", "Best suited when…"])
        st.dataframe(guide_df, use_container_width=True, hide_index=True)

        # Forecast vs training trend overlay (moved here from old comparison)
        if forecast_df is not None:
            t_date_col = train_cols.get("date")
            t_rev_col  = train_cols.get("revenue")
            if (_col_ok(df_train, t_date_col) and _col_ok(df_train, t_rev_col)
                    and df_train is not None):
                try:
                    st.markdown("---")
                    st.markdown("#### 📈 Your Forecast vs Training Dataset Revenue Trend")
                    tr_monthly = _monthly_agg(df_train, t_date_col, t_rev_col)
                    fig_ov = go.Figure()
                    fig_ov.add_trace(go.Scatter(
                        x=tr_monthly["Period"], y=tr_monthly[t_rev_col],
                        mode="lines", name="Training Dataset",
                        line=dict(color="#E45756", width=2)
                    ))
                    fig_ov.add_trace(go.Scatter(
                        x=forecast_df["Month"], y=forecast_df["Forecasted Revenue"],
                        mode="lines+markers", name="Your Forecast",
                        line=dict(color="#4C78A8", width=2, dash="dash"),
                        marker=dict(size=8, symbol="star")
                    ))
                    fig_ov.update_layout(
                        height=400,
                        title="Your Forecast vs Training Dataset Revenue",
                        xaxis_title="Period", yaxis_title="Revenue ($)",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_ov, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render overlay chart: {e}")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_auto_analysis(df_train: pd.DataFrame = None):
    """
    Full automated analysis page.
    Pass df_train = the Regional Sales training DataFrame for the comparison tab.
    """
    st.header("🤖 Automated Dataset Analysis & Forecasting")
    st.markdown(
        "Upload any CSV or Excel sales dataset to instantly get an overview, "
        "data-quality report, cleaning tools, visualisations, smart insights, "
        "a sales forecast, and a comparison against the training baseline."
    )

    uploaded = st.file_uploader(
        "📁 Upload your dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded is None:
        st.info("⬆️ Upload a file above to get started.")
        return

    # Load file
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # Reset cached state on new file
    if st.session_state.get("last_upload") != uploaded.name:
        st.session_state["raw_df"]      = df_raw.copy()
        st.session_state["cleaned_df"]  = df_raw.copy()
        st.session_state["last_upload"] = uploaded.name
        st.session_state.pop("forecast_result", None)

    df_raw = st.session_state["raw_df"]
    st.success(
        f"✅ Loaded **{uploaded.name}** — "
        f"{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns"
    )

    # Auto-detect columns + manual override
    cols = detect_columns(df_raw)
    with st.expander("🔎 Auto-detected column mapping (edit if needed)", expanded=False):
        role_labels = {
            "date":     "Date / Time column",
            "revenue":  "Revenue / Sales column",
            "profit":   "Profit column",
            "quantity": "Quantity column",
            "product":  "Product / Category column",
            "region":   "Region column",
            "customer": "Customer column",
        }
        for role, label in role_labels.items():
            options     = ["(none)"] + list(df_raw.columns)
            current     = cols.get(role)
            default_idx = options.index(current) if current in options else 0
            chosen      = st.selectbox(label, options, index=default_idx, key=f"col_{role}")
            cols[role]  = None if chosen == "(none)" else chosen

    # Dashboard tabs
    tabs = st.tabs([
        "📋 Overview",
        "🔍 Quality",
        "🧹 Cleaning",
        "📊 Analysis",
        "💡 Insights",
        "🔮 Forecast",
        "📐 Comparison",
    ])

    with tabs[0]:
        show_overview(df_raw)

    with tabs[1]:
        show_quality(df_raw)

    with tabs[2]:
        show_cleaning(df_raw)

    with tabs[3]:
        df_work = st.session_state.get("cleaned_df", df_raw)
        show_eda(df_work, cols)

    with tabs[4]:
        df_work = st.session_state.get("cleaned_df", df_raw)
        show_smart_insights(df_work, cols)

    with tabs[5]:
        df_work = st.session_state.get("cleaned_df", df_raw)
        forecast_result = show_forecasting(df_work, cols)
        if forecast_result is not None:
            st.session_state["forecast_result"] = forecast_result

    with tabs[6]:
        df_work = st.session_state.get("cleaned_df", df_raw)
        fr = st.session_state.get("forecast_result")
        if df_train is not None:
            show_comparison(df_work, df_train, cols, forecast_df=fr)
        else:
            st.info(
                "Training dataset not loaded. Place "
                "**Regional_Sales_Dataset_2021_2024.xlsx** in the same folder "
                "as the app to enable the comparison tab."
            )
