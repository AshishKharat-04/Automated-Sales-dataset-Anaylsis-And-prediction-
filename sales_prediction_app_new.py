# TO RUN THIS, USE BELOW COMMAND
#python -m streamlit run sales_prediction_app_new.py --server.port 8080
# Place auto_analysis_module.py in the same folder as this file.


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1e1e1e;
        border: 1px solid #2e2e2e;
        padding: 16px;
        border-radius: 12px;
        color: white;
    }

    div[data-testid="metric-container"] label {
        color: #b0b0b0;
    }

    div[data-testid="metric-container"] div {
        color: white;
        font-weight: 600;
    }

    h1, h2, h3 {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# Load data
from pathlib import Path
from auto_analysis_module import run_auto_analysis, detect_columns

# Load data
@st.cache_data
def load_data():
    base_dir = Path(__file__).parent

    try:
        df_comparison = pd.read_csv(base_dir / 'model_comparison.csv')
        df_predictions = pd.read_csv(base_dir / 'all_model_predictions.csv')
        df_forecast = pd.read_csv(base_dir / 'forecast_6months.csv')
        df_monthly = pd.read_csv(base_dir / 'monthly_aggregated_data.csv')

        df_monthly['Year-Month'] = pd.to_datetime(df_monthly['Year-Month'])
        return df_comparison, df_predictions, df_forecast, df_monthly

    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run the analysis notebook first to generate the required CSV files.")
        return None, None, None, None


@st.cache_data
def load_training_dataset():
    """Load the Regional Sales Dataset used as the ML training baseline."""
    base_dir = Path(__file__).parent
    path_xlsx = base_dir / "Regional_Sales_Dataset_2021_2024.xlsx"
    path_csv  = base_dir / "Regional_Sales_Dataset_2021_2024.csv"
    try:
        if path_xlsx.exists():
            return pd.read_excel(path_xlsx)
        elif path_csv.exists():
            return pd.read_csv(path_csv)
    except Exception:
        pass
    return None


df_comparison, df_predictions, df_forecast, df_monthly = load_data()
df_training = load_training_dataset()

if df_comparison is not None:
    
    # Sidebar
    st.sidebar.header("🎯 Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["📈 Model Comparison", "🎯 Predictions", "🔮 Forecast",
         "📊 Data Insights", "🤖 Auto Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About this Dashboard:**
    
    Interactive sales prediction system using multiple ML models.
    
    - Compare 6 different models
    - View predictions vs actuals
    - See 6-month forecast
    - Analyze performance metrics
    - 🆕 Upload & analyze your own dataset
    """)
    
    # PAGE 1: Model Comparison
    if page == "📈 Model Comparison":
        st.header("🤖 Machine Learning Model Comparison")
        
        # Sort by R² Score
        df_comp_sorted = df_comparison.sort_values('R² Score', ascending=False)
        
        # Best model highlight
        best_model = df_comp_sorted.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏆 Best Model", best_model['Model'])
        with col2:
            st.metric("R² Score", f"{best_model['R² Score']:.4f}")
        with col3:
            st.metric("RMSE", f"${best_model['RMSE']:,.0f}")
        with col4:
            st.metric("MAPE", f"{best_model['MAPE (%)']:.2f}%")
        
        st.markdown("---")
        
        # Full comparison table
        st.subheader("📊 Complete Model Performance Metrics")
        
        # Format the dataframe for display
        df_display = df_comp_sorted.copy()
        df_display['MAE'] = df_display['MAE'].apply(lambda x: f"${x:,.2f}")
        df_display['MSE'] = df_display['MSE'].apply(lambda x: f"{x:,.2f}")
        df_display['RMSE'] = df_display['RMSE'].apply(lambda x: f"${x:,.2f}")
        df_display['R² Score'] = df_display['R² Score'].apply(lambda x: f"{x:.4f}")
        df_display['MAPE (%)'] = df_display['MAPE (%)'].apply(lambda x: f"{x:.2f}%")
        df_display['R² Train'] = df_display['R² Train'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Visualization
        st.subheader("📈 Visual Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            fig1 = px.bar(
                df_comp_sorted, 
                x='Model', 
                y='R² Score',
                title='R² Score Comparison (Higher is Better)',
                color='R² Score',
                color_continuous_scale='Greens',
                text='R² Score'
            )
            fig1.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig1.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # RMSE comparison
            fig2 = px.bar(
                df_comp_sorted, 
                x='Model', 
                y='RMSE',
                title='RMSE Comparison (Lower is Better)',
                color='RMSE',
                color_continuous_scale='Reds_r',
                text='RMSE'
            )
            fig2.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig2.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Metrics overview
        st.subheader("📊 All Metrics Overview")
        
        fig3 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MAE ($)', 'RMSE ($)', 'R² Score', 'MAPE (%)'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # MAE
        fig3.add_trace(
            go.Bar(x=df_comp_sorted['Model'], y=df_comp_sorted['MAE'], 
                   marker_color='skyblue', name='MAE'),
            row=1, col=1
        )
        
        # RMSE
        fig3.add_trace(
            go.Bar(x=df_comp_sorted['Model'], y=df_comp_sorted['RMSE'], 
                   marker_color='lightcoral', name='RMSE'),
            row=1, col=2
        )
        
        # R² Score
        fig3.add_trace(
            go.Bar(x=df_comp_sorted['Model'], y=df_comp_sorted['R² Score'], 
                   marker_color='lightgreen', name='R²'),
            row=2, col=1
        )
        
        # MAPE
        fig3.add_trace(
            go.Bar(x=df_comp_sorted['Model'], y=df_comp_sorted['MAPE (%)'], 
                   marker_color='gold', name='MAPE'),
            row=2, col=2
        )
        
        fig3.update_layout(height=700, showlegend=False, title_text="Comprehensive Metrics Dashboard")
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Metric explanations
        with st.expander("ℹ️ Understanding the Metrics"):
            st.markdown("""
            **R² Score (R-squared):**
            - Measures how well the model explains variance in the data
            - Range: 0 to 1 (higher is better)
            - 1.0 = perfect predictions, 0.0 = no better than average
            
            **RMSE (Root Mean Squared Error):**
            - Average prediction error in dollars
            - Lower is better
            - More sensitive to large errors
            
            **MAE (Mean Absolute Error):**
            - Average absolute difference between actual and predicted
            - Lower is better
            - Easier to interpret than RMSE
            
            **MAPE (Mean Absolute Percentage Error):**
            - Average error as a percentage
            - Lower is better
            - Good for comparing across different scales
            
            **MSE (Mean Squared Error):**
            - Average of squared errors
            - Lower is better
            - Heavily penalizes large errors
            """)
    
    # PAGE 2: Predictions
    elif page == "🎯 Predictions":
        st.header("🎯 Model Predictions Analysis")
        
        # Model selector
        available_models = [col.replace('_Prediction', '') for col in df_predictions.columns if col.endswith('_Prediction')]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_models = st.multiselect(
                "Select Models to Compare:",
                available_models,
                default=available_models[:2] if len(available_models) >= 2 else available_models
            )
        
        with col2:
            show_stats = st.checkbox("Show Statistics", value=True)
        
        if selected_models:
            st.markdown("---")
            
            # Actual vs Predicted scatter plot
            st.subheader("📊 Actual vs Predicted Revenue")
            
            fig = go.Figure()
            
            # Perfect prediction line
            min_val = df_predictions['Actual'].min()
            max_val = df_predictions['Actual'].max()
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], 
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            # Add each selected model
            colors = px.colors.qualitative.Plotly
            for idx, model in enumerate(selected_models):
                pred_col = f'{model}_Prediction'
                if pred_col in df_predictions.columns:
                    fig.add_trace(go.Scatter(
                        x=df_predictions['Actual'],
                        y=df_predictions[pred_col],
                        mode='markers',
                        name=model,
                        marker=dict(size=6, opacity=0.6, color=colors[idx % len(colors)])
                    ))
            
            fig.update_layout(
                height=600,
                xaxis_title='Actual Revenue ($)',
                yaxis_title='Predicted Revenue ($)',
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics for selected models
            if show_stats:
                st.markdown("---")
                st.subheader("📈 Prediction Statistics")
                
                for model in selected_models:
                    pred_col = f'{model}_Prediction'
                    if pred_col in df_predictions.columns:
                        with st.expander(f"📊 {model} - Detailed Stats"):
                            col1, col2, col3 = st.columns(3)
                            
                            actual = df_predictions['Actual']
                            predicted = df_predictions[pred_col]
                            
                            # Calculate metrics
                            mae = np.mean(np.abs(actual - predicted))
                            rmse = np.sqrt(np.mean((actual - predicted)**2))
                            r2 = 1 - (np.sum((actual - predicted)**2) / np.sum((actual - actual.mean())**2))
                            
                            with col1:
                                st.metric("MAE", f"${mae:,.2f}")
                            with col2:
                                st.metric("RMSE", f"${rmse:,.2f}")
                            with col3:
                                st.metric("R² Score", f"{r2:.4f}")
                            
                            # Residual plot
                            residuals = actual - predicted
                            fig_resid = px.scatter(
                                x=predicted, y=residuals,
                                labels={'x': 'Predicted Revenue', 'y': 'Residuals'},
                                title=f'Residual Plot - {model}'
                            )
                            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                            fig_resid.update_layout(height=300)
                            st.plotly_chart(fig_resid, use_container_width=True)
            
            # Prediction distribution
            st.markdown("---")
            st.subheader("📊 Prediction Error Distribution")
            
            fig_dist = go.Figure()
            
            for model in selected_models:
                pred_col = f'{model}_Prediction'
                if pred_col in df_predictions.columns:
                    errors = df_predictions['Actual'] - df_predictions[pred_col]
                    fig_dist.add_trace(go.Histogram(
                        x=errors,
                        name=model,
                        opacity=0.6,
                        nbinsx=30
                    ))
            
            fig_dist.update_layout(
                height=400,
                xaxis_title='Prediction Error ($)',
                yaxis_title='Frequency',
                barmode='overlay',
                title='Distribution of Prediction Errors'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        else:
            st.warning("Please select at least one model to view predictions.")
    
    # PAGE 3: Forecast
    elif page == "🔮 Forecast":
        st.header("🔮 6-Month Revenue Forecast")
        
        # Forecast metrics
        total_forecast = df_forecast['Forecasted Revenue'].sum()
        avg_monthly = df_forecast['Forecasted Revenue'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Forecasted Revenue", f"${total_forecast:,.2f}")
        with col2:
            st.metric("Average Monthly Revenue", f"${avg_monthly:,.2f}")
        with col3:
            # Compare to historical average
            hist_avg = df_monthly['Revenue'].mean()
            growth = ((avg_monthly - hist_avg) / hist_avg) * 100
            st.metric("vs Historical Avg", f"{growth:+.1f}%")
        
        st.markdown("---")
        
        # Forecast table
        st.subheader("📅 Monthly Forecast Breakdown")
        
        df_forecast_display = df_forecast.copy()
        df_forecast_display['Forecasted Revenue'] = df_forecast_display['Forecasted Revenue'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(df_forecast_display, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Forecast visualization
        st.subheader("📈 Forecast Visualization")
        
        # Combine historical and forecast
        fig = go.Figure()
        
        # Historical data (last 12 months)
        hist_recent = df_monthly.tail(12)
        fig.add_trace(go.Scatter(
            x=hist_recent['Year-Month'],
            y=hist_recent['Revenue'],
            mode='lines+markers',
            name='Historical Revenue',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Forecast data
        forecast_dates = pd.to_datetime(df_forecast['Month'])
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=df_forecast['Forecasted Revenue'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=10, symbol='star')
        ))
        
        # Add vertical line
        # Safe forecast start marker (FIXED)
        forecast_start = pd.to_datetime(df_monthly['Year-Month'].max())

        fig.add_shape(
        type="line",
        x0=forecast_start,
        x1=forecast_start,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="green", width=2, dash="dot")
        )

        fig.add_annotation(
            x=forecast_start,
            y=1.02,
            xref="x",
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            font=dict(color="green", size=12)
        )

        
        # Trend analysis
        st.markdown("---")
        st.subheader("📊 Forecast Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Month-over-month change
            forecast_values = df_forecast['Forecasted Revenue'].values
            mom_changes = [(forecast_values[i] - forecast_values[i-1]) / forecast_values[i-1] * 100 
                          for i in range(1, len(forecast_values))]
            
            fig_mom = go.Figure()
            fig_mom.add_trace(go.Bar(
                x=df_forecast['Month'][1:],
                y=mom_changes,
                marker_color=['green' if x > 0 else 'red' for x in mom_changes],
                text=[f"{x:+.1f}%" for x in mom_changes],
                textposition='outside'
            ))
            fig_mom.update_layout(
                title='Month-over-Month Growth Rate',
                xaxis_title='Month',
                yaxis_title='Growth (%)',
                height=350
            )
            st.plotly_chart(fig_mom, use_container_width=True)
        
        with col2:
            # Cumulative forecast
            cumulative = df_forecast['Forecasted Revenue'].cumsum()
            
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=df_forecast['Month'],
                y=cumulative,
                mode='lines+markers',
                fill='tozeroy',
                line=dict(color='purple', width=3),
                marker=dict(size=10)
            ))
            fig_cum.update_layout(
                title='Cumulative Forecasted Revenue',
                xaxis_title='Month',
                yaxis_title='Cumulative Revenue ($)',
                height=350
            )
            st.plotly_chart(fig_cum, use_container_width=True)
        
        # Download forecast
        st.markdown("---")
        csv = df_forecast.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Data (CSV)",
            data=csv,
            file_name='6_month_forecast.csv',
            mime='text/csv'
        )
    
    # PAGE 4: Data Insights
    elif page == "📊 Data Insights":
        st.header("📊 Historical Data Insights")
        
        # Time period selector
        years = sorted(df_monthly['Year'].unique())
        selected_year = st.selectbox("Select Year:", ['All'] + [str(y) for y in years])
        
        if selected_year != 'All':
            df_monthly_filtered = df_monthly[df_monthly['Year'] == int(selected_year)]
        else:
            df_monthly_filtered = df_monthly
        
        st.markdown("---")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Revenue", f"${df_monthly_filtered['Revenue'].sum():,.0f}")
        with col2:
            st.metric("Total Profit", f"${df_monthly_filtered['Profit'].sum():,.0f}")
        with col3:
            st.metric("Avg Monthly Revenue", f"${df_monthly_filtered['Revenue'].mean():,.0f}")
        with col4:
            profit_margin = (df_monthly_filtered['Profit'].sum() / df_monthly_filtered['Revenue'].sum()) * 100
            st.metric("Profit Margin", f"{profit_margin:.1f}%")
        
        st.markdown("---")
        
        # Revenue and Profit trend
        st.subheader("📈 Revenue & Profit Trends")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=df_monthly_filtered['Year-Month'], y=df_monthly_filtered['Revenue'],
                      name="Revenue", line=dict(color='blue', width=3)),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=df_monthly_filtered['Year-Month'], y=df_monthly_filtered['Profit'],
                      name="Profit", line=dict(color='green', width=3)),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig.update_yaxes(title_text="Profit ($)", secondary_y=True)
        fig.update_layout(height=500, hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Monthly performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Orders & Quantity")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_monthly_filtered['Year-Month'],
                y=df_monthly_filtered['Orders'],
                name='Orders',
                marker_color='coral'
            ))
            fig.update_layout(height=350, xaxis_title='Date', yaxis_title='Number of Orders')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📦 Total Quantity Sold")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_monthly_filtered['Year-Month'],
                y=df_monthly_filtered['Quantity'],
                name='Quantity',
                marker_color='gold'
            ))
            fig.update_layout(height=350, xaxis_title='Date', yaxis_title='Quantity')
            st.plotly_chart(fig, use_container_width=True)
        
        # Year-over-year comparison
        if selected_year == 'All' and len(years) > 1:
            st.markdown("---")
            st.subheader("📅 Year-over-Year Comparison")
            
            yearly_data = df_monthly.groupby('Year').agg({
                'Revenue': 'sum',
                'Profit': 'sum',
                'Orders': 'sum',
                'Quantity': 'sum'
            }).reset_index()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Revenue by Year', 'Profit by Year', 'Orders by Year', 'Quantity by Year')
            )
            
            fig.add_trace(go.Bar(x=yearly_data['Year'], y=yearly_data['Revenue'], marker_color='steelblue'), row=1, col=1)
            fig.add_trace(go.Bar(x=yearly_data['Year'], y=yearly_data['Profit'], marker_color='seagreen'), row=1, col=2)
            fig.add_trace(go.Bar(x=yearly_data['Year'], y=yearly_data['Orders'], marker_color='coral'), row=2, col=1)
            fig.add_trace(go.Bar(x=yearly_data['Year'], y=yearly_data['Quantity'], marker_color='gold'), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # PAGE 5: Auto Analysis (new)
    elif page == "🤖 Auto Analysis":
        run_auto_analysis(df_train=df_training)

else:
    st.error("Could not load data files. Please run the analysis notebook first.")
    st.info("""
    Required files:
    - model_comparison.csv
    - all_model_predictions.csv
    - forecast_6months.csv
    - monthly_aggregated_data.csv
    
    These are generated by running the Sales Analysis notebook.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Regional Sales Prediction Dashboard</strong> | Built with Streamlit & Plotly</p>
    <p>© 2026 - Sales Analytics System</p>
</div>
""", unsafe_allow_html=True)
