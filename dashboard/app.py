import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json
import os

from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Demand Forecasting System",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

METRICS_FILE = (
    BASE_DIR /
    "outputs" /
    "model_metrics.json"
)


# ============================================================
# API
# ============================================================

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 AI Demand Forecasting System"
)

st.markdown(
    """
    **ML-powered demand forecasting for stores and products**

    Predict future product demand using historical sales,
    time-based features, lag features and rolling statistics.
    """
)

st.divider()


# ============================================================
# CHECK API
# ============================================================

try:

    response = requests.get(
        f"{API_URL}/",
        timeout=5
    )

    if response.status_code != 200:

        st.error(
            "FastAPI backend is not responding correctly."
        )

        st.stop()

except requests.exceptions.RequestException:

    st.error(
        """
        ❌ Cannot connect to the FastAPI backend.

        Start the backend first:

        `uvicorn api.main:app --reload --port 8000`
        """
    )

    st.stop()


# ============================================================
# GET STORES
# ============================================================

try:

    stores_response = requests.get(
        f"{API_URL}/stores",
        timeout=5
    )

    stores = stores_response.json()["stores"]

except Exception as error:

    st.error(
        f"Unable to load stores: {error}"
    )

    st.stop()


# ============================================================
# GET ITEMS
# ============================================================

try:

    items_response = requests.get(
        f"{API_URL}/items",
        timeout=5
    )

    items = items_response.json()["items"]

except Exception as error:

    st.error(
        f"Unable to load products: {error}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Forecast Settings"
)


selected_store = st.sidebar.selectbox(
    "Select Store",
    stores
)


selected_item = st.sidebar.selectbox(
    "Select Product",
    items
)


forecast_days = st.sidebar.slider(
    "Forecast Horizon",
    min_value=1,
    max_value=30,
    value=7
)


st.sidebar.divider()


st.sidebar.markdown(
    """
    ### Model

    **XGBoost Regression**

    Features include:

    • Historical lag values  
    • Rolling averages  
    • Rolling standard deviation  
    • Day of week  
    • Month  
    • Weekend indicator  
    • Store  
    • Product
    """
)


forecast_button = st.sidebar.button(
    "🔮 Forecast Demand",
    use_container_width=True
)


# ============================================================
# LANDING STATE
# ============================================================

if not forecast_button:

    st.subheader(
        "🚀 Ready to Forecast"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Stores Available",
            len(stores)
        )

    with col2:

        st.metric(
            "Products Available",
            len(items)
        )

    with col3:

        st.metric(
            "Maximum Forecast",
            "30 Days"
        )

    st.info(
        """
        Select a store, product and forecast horizon
        from the sidebar, then click **Forecast Demand**.
        """
    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE ON LANDING PAGE
    # --------------------------------------------------------

    if METRICS_FILE.exists():

        st.divider()

        st.subheader(
            "🎯 Model Performance"
        )

        try:

            with open(
                METRICS_FILE,
                "r"
            ) as file:

                metrics = json.load(file)

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "MAE",
                    f"{metrics['MAE']:.2f}"
                )

            with col2:

                st.metric(
                    "RMSE",
                    f"{metrics['RMSE']:.2f}"
                )

            with col3:

                st.metric(
                    "MAPE",
                    f"{metrics['MAPE']:.2f}%"
                )

        except Exception:

            pass


# ============================================================
# FORECAST
# ============================================================

if forecast_button:

    payload = {

        "store": int(selected_store),

        "item": int(selected_item),

        "days": int(forecast_days)

    }


    # --------------------------------------------------------
    # CALL FORECAST API
    # --------------------------------------------------------

    with st.spinner(
        "Generating demand forecast..."
    ):

        try:

            forecast_response = requests.post(
                f"{API_URL}/forecast",
                json=payload,
                timeout=60
            )

        except requests.exceptions.RequestException as error:

            st.error(
                f"Forecast API error: {error}"
            )

            st.stop()


    # --------------------------------------------------------
    # API ERROR
    # --------------------------------------------------------

    if forecast_response.status_code != 200:

        try:

            error_detail = (
                forecast_response.json()
            )

        except Exception:

            error_detail = (
                forecast_response.text
            )

        st.error(
            f"Forecast request failed: {error_detail}"
        )

        st.stop()


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    forecast_data = (
        forecast_response.json()
    )


    forecast_list = (
        forecast_data["forecast"]
    )


    forecast_df = pd.DataFrame(
        forecast_list
    )


    forecast_df["date"] = pd.to_datetime(
        forecast_df["date"]
    )


    forecast_df["predicted_demand"] = (
        pd.to_numeric(
            forecast_df[
                "predicted_demand"
            ]
        )
    )


    total_demand = (
        forecast_data[
            "total_predicted_demand"
        ]
    )


    average_demand = (
        forecast_data[
            "average_daily_demand"
        ]
    )


    # ========================================================
    # GET HISTORICAL DATA
    # ========================================================

    try:

        history_response = requests.get(
            f"{API_URL}/history/"
            f"{selected_store}/"
            f"{selected_item}"
            f"?days=30",
            timeout=10
        )


        if history_response.status_code == 200:

            history_data = (
                history_response.json()
            )

            history_df = pd.DataFrame(
                history_data["history"]
            )

            history_df["date"] = pd.to_datetime(
                history_df["date"]
            )

            history_df["sales"] = pd.to_numeric(
                history_df["sales"]
            )

        else:

            history_df = pd.DataFrame()


    except Exception:

        history_df = pd.DataFrame()


    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.subheader(
        f"🔮 Demand Forecast — "
        f"Store {selected_store}, "
        f"Product {selected_item}"
    )


    # ========================================================
    # KPI SECTION
    # ========================================================

    peak_demand = (
        forecast_df[
            "predicted_demand"
        ].max()
    )


    lowest_demand = (
        forecast_df[
            "predicted_demand"
        ].min()
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Forecast Horizon",
            f"{forecast_days} Days"
        )


    with col2:

        st.metric(
            "Total Expected Demand",
            f"{total_demand:,.0f}"
        )


    with col3:

        st.metric(
            "Average Daily Demand",
            f"{average_demand:,.2f}"
        )


    with col4:

        st.metric(
            "Peak Daily Demand",
            f"{peak_demand:,.0f}"
        )


    st.divider()


    # ========================================================
    # HISTORICAL + FORECAST CHART
    # ========================================================

    st.subheader(
        "📊 Historical Demand + Future Forecast"
    )


    if not history_df.empty:

        historical_plot = history_df[
            ["date", "sales"]
        ].copy()


        historical_plot[
            "type"
        ] = "Historical Demand"


        forecast_plot = forecast_df[
            [
                "date",
                "predicted_demand"
            ]
        ].copy()


        forecast_plot = (
            forecast_plot.rename(
                columns={
                    "predicted_demand":
                    "sales"
                }
            )
        )


        forecast_plot[
            "type"
        ] = "Forecast"


        combined_df = pd.concat(
            [
                historical_plot,
                forecast_plot
            ],
            ignore_index=True
        )


        fig = px.line(
            combined_df,
            x="date",
            y="sales",
            color="type",
            markers=True,
            title=(
                "Historical Demand vs "
                "Future Forecast"
            )
        )


        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Demand",
            hovermode="x unified"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        # ----------------------------------------------------
        # FALLBACK FORECAST CHART
        # ----------------------------------------------------

        fig = px.line(
            forecast_df,
            x="date",
            y="predicted_demand",
            markers=True,
            title="Predicted Daily Demand"
        )


        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Predicted Demand",
            hovermode="x unified"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # FORECAST INSIGHTS
    # ========================================================

    st.subheader(
        "💡 Forecast Insights"
    )


    peak_row = forecast_df.loc[
        forecast_df[
            "predicted_demand"
        ].idxmax()
    ]


    lowest_row = forecast_df.loc[
        forecast_df[
            "predicted_demand"
        ].idxmin()
    ]


    col1, col2 = st.columns(2)


    with col1:

        st.success(
            f"""
            **Peak Demand**

            📅 {peak_row["date"].strftime("%d %B %Y")}

            📦 Expected demand:
            **{peak_row["predicted_demand"]:,.0f} units**
            """
        )


    with col2:

        st.warning(
            f"""
            **Lowest Demand**

            📅 {lowest_row["date"].strftime("%d %B %Y")}

            📦 Expected demand:
            **{lowest_row["predicted_demand"]:,.0f} units**
            """
        )


    # ========================================================
    # FORECAST TABLE
    # ========================================================

    st.subheader(
        "📋 Detailed Forecast"
    )


    display_df = forecast_df.copy()


    display_df["date"] = (
        display_df["date"]
        .dt.strftime("%Y-%m-%d")
    )


    display_df["predicted_demand"] = (
        display_df[
            "predicted_demand"
        ].round(2)
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    csv_data = forecast_df.to_csv(
        index=False
    )


    st.download_button(
        label="⬇️ Download Forecast CSV",

        data=csv_data,

        file_name=(
            f"store_{selected_store}_"
            f"item_{selected_item}_"
            f"forecast.csv"
        ),

        mime="text/csv"
    )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.divider()


    st.subheader(
        "🎯 Model Performance"
    )


    if METRICS_FILE.exists():

        try:

            with open(
                METRICS_FILE,
                "r"
            ) as file:

                metrics = json.load(file)


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "MAE",
                    f"{metrics['MAE']:.2f}"
                )


            with col2:

                st.metric(
                    "RMSE",
                    f"{metrics['RMSE']:.2f}"
                )


            with col3:

                st.metric(
                    "MAPE",
                    f"{metrics['MAPE']:.2f}%"
                )


            st.caption(
                "Metrics are calculated on the "
                "held-out test period."
            )


        except Exception as error:

            st.warning(
                f"Could not load metrics: {error}"
            )


    else:

        st.warning(
            """
            Model performance metrics are not available.

            Run:

            `python -m src.evaluate`
            """
        )


    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.caption(
        "AI Demand Forecasting System • "
        "XGBoost + FastAPI + Streamlit"
    )