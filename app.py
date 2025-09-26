import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


@st.cache_data
def load_data():
    df = pd.read_excel("Dataset.xlsx", sheet_name="01 Churn-Dataset")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df

data = load_data()

st.sidebar.title("🔍 Filters")
contract_type = st.sidebar.multiselect("Contract Type", data["Contract"].unique(), default=data["Contract"].unique())
internet_type = st.sidebar.multiselect("Internet Service", data["InternetService"].unique(), default=data["InternetService"].unique())

filtered_data = data[(data["Contract"].isin(contract_type)) & (data["InternetService"].isin(internet_type))]

st.title("📊 Customer Churn Analysis Dashboard")


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📌 Overview", "🥧 Pie Chart", "🔵 Bubble Chart", "📈 Line Trend",
    "🔔 Bell Curve", "🌳 Treemap", "☀️ Sunburst", "📊 Correlations"
])


with tab1:
    st.subheader("Dataset Overview")
    st.write("Filtered Data Preview:")
    st.dataframe(filtered_data.head(10))
    st.metric("Total Customers", len(filtered_data))
    st.metric("Churned Customers", filtered_data["Churn"].value_counts().get("Yes", 0))

with tab2:
    churn_count = filtered_data["Churn"].value_counts().reset_index()
    churn_count.columns = ["Churn", "Count"]
    fig = px.pie(churn_count, names="Churn", values="Count",
                 color="Churn", color_discrete_sequence=px.colors.qualitative.Set3,
                 title="Churn Percentage")
    st.plotly_chart(fig, use_container_width=True)


with tab3:
    fig = px.scatter(filtered_data, x="MonthlyCharges", y="TotalCharges",
                     size="tenure", color="Churn",
                     hover_data=["Contract", "PaymentMethod"],
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     title="Bubble Chart: Charges vs Tenure")
    st.plotly_chart(fig, use_container_width=True)


with tab4:
    tenure_churn = filtered_data.groupby("tenure")["Churn"].value_counts().unstack().fillna(0).reset_index()
    fig = px.line(tenure_churn, x="tenure", y=["Yes", "No"],
                  title="Churn Trend over Tenure",
                  labels={"value": "Customer Count", "tenure": "Tenure (Months)"})
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    fig, ax = plt.subplots()
    sns.kdeplot(filtered_data[filtered_data["Churn"] == "Yes"]["MonthlyCharges"], shade=True, label="Churn = Yes", ax=ax)
    sns.kdeplot(filtered_data[filtered_data["Churn"] == "No"]["MonthlyCharges"], shade=True, label="Churn = No", ax=ax)
    ax.set_title("Bell Curve of Monthly Charges by Churn")
    ax.set_xlabel("Monthly Charges")
    ax.set_ylabel("Density")
    st.pyplot(fig)


with tab6:
    treemap_data = filtered_data.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
    fig = px.treemap(treemap_data, path=["Contract", "Churn"], values="Count",
                     color="Churn", color_discrete_sequence=px.colors.qualitative.Pastel,
                     title="Treemap of Churn by Contract")
    st.plotly_chart(fig, use_container_width=True)


with tab7:
    fig = px.sunburst(treemap_data, path=["Contract", "Churn"], values="Count",
                      color="Churn", color_discrete_sequence=px.colors.qualitative.Prism,
                      title="Sunburst of Churn by Contract")
    st.plotly_chart(fig, use_container_width=True)


with tab8:
    st.subheader("Correlation Heatmap")
    num_cols = filtered_data.select_dtypes(include=np.number).columns
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(filtered_data[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    le = LabelEncoder()
    encoded = filtered_data.copy()
    for col in encoded.select_dtypes(include="object").columns:
        encoded[col] = le.fit_transform(encoded[col])

    correlations = encoded.corr()["Churn"].sort_values(ascending=False)
    st.subheader("Feature Correlation with Churn")
    st.bar_chart(correlations)
