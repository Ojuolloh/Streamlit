import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px

# Set page configuration must be the very first Streamlit command
st.set_page_config(
    page_title='Customers Churn Dashboard',
    page_icon="🏂",
    layout='wide',
)

def dashboard_page():
    # Your Streamlit dashboard logic here
    st.title("Dashboard Page")
    alt.themes.enable("dark")

    # read data table for dashboard
    df = pd.read_csv("data/traindata.csv")

    col1, col2 = st.columns(2)
    with col1:
        pass
    with col2:
        choose_options = st.selectbox('Choose Dashboard', options=('KPI', 'EDA'), key='dashboard_selectbox')

    #  understand the churn rate of customers
    total_customers = len(df["Churn"])
    churn_customer = len(df[df['Churn'] == 'Yes'])
    non_churn_customer = len(df[df['Churn'] == 'No'])
    churn_rate = churn_customer / total_customers * 100

    # create four columns
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    def create_kpi():
        # Fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="Total Customer ⏳",
            value=round(total_customers),
        )
        kpi2.metric(
            label="Churn Customers 💍",
            value=round(churn_customer),
        )
        kpi3.metric(
            label="Non Churn Customer",
            value=int(non_churn_customer),
        )
        kpi4.metric(
            label="Churn Rate",
            value=int(churn_rate),
        )

        # create two columns for charts
        fig_col1, fig_col2, fig_col3 = st.columns(3)

        with fig_col1:
            fig1 = px.histogram(data_frame=df, x='TotalCharges')
            st.write(fig1)

        fig2 = px.histogram(data_frame=df, x='MonthlyCharges')
        st.write(fig2)

        st.markdown("#### Density Heatmap for Numerical columns")
        num_df = df.select_dtypes(include='number')
        fig3 = px.density_heatmap(data_frame=num_df)
        st.write(fig3)

        st.markdown("#### Box plot for Numerical columns")
        fig2 = px.box(data_frame=num_df)
        st.write(fig2)

    def create_eda():
        eda_col1, eda_col2 = st.columns(2)
        with eda_col1:
            st.subheader("Frequently used internet service")
            fig_eda1 = px.histogram(data_frame=df, x='PaymentMethod', color='Churn', color_discrete_sequence=['orange', 'blue'])
            fig_eda1.update_layout(barmode='group')
            st.write(fig_eda1)

        with eda_col2:
            st.markdown('####  Gender with Highest Churn')
            fig_eda1 = px.bar(data_frame=df, x='gender', color='Churn', color_discrete_sequence=['orange', 'blue'])
            fig_eda1.update_layout(barmode='group')
            st.write(fig_eda1)

        eda_col3, eda_col4 = st.columns(2)
        with eda_col3:
            st.markdown('####  Contract Type with Highest Churn')
            fig_eda1 = px.bar(data_frame=df, x='Contract', color='Churn', color_discrete_sequence=['orange', 'blue'])
            fig_eda1.update_layout(barmode='group')
            st.write(fig_eda1)

    if choose_options == 'KPI':
        create_kpi()
    else:
        create_eda()

