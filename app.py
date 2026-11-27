import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")
st.title("Employee Attrition Prediction and HR Analytics System")

# Load data and the saved model
risk = pd.read_csv("employee_risk_scores.csv")
bundle = joblib.load("model.pkl")
model, scaler = bundle["model"], bundle["scaler"]
columns, num_cols = bundle["columns"], bundle["num_cols"]

# Top reasons come straight from the model (same as your importance cell)
importance = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
reasons = importance.head(10).reset_index()
reasons.columns = ["Factor", "Importance"]

tab1, tab2 = st.tabs(["Dashboard", "Predict one employee"])

# ---------------- Tab 1: Dashboard ----------------
with tab1:
    # Sidebar filters
    st.sidebar.header("Filters")
    dept = st.sidebar.multiselect("Department", risk["Department"].unique(),
                                  default=list(risk["Department"].unique()))
    level = st.sidebar.multiselect("Risk level", ["High", "Medium", "Low"],
                                   default=["High", "Medium", "Low"])
    view = risk[risk["Department"].isin(dept) & risk["RiskLevel"].isin(level)]

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", len(view))
    c2.metric("High risk", int((view["RiskLevel"] == "High").sum()))
    c3.metric("Actual attrition rate", f"{(view['Attrition'] == 'Yes').mean() * 100:.1f}%")
    c4.metric("Avg leave probability", f"{view['LeaveProbability'].mean() * 100:.1f}%")

    # Charts
    left, right = st.columns(2)
    with left:
        st.subheader("Employees by risk level")
        counts = view["RiskLevel"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
        counts.columns = ["RiskLevel", "Employees"]
        st.plotly_chart(px.bar(counts, x="RiskLevel", y="Employees", color="RiskLevel",
                               color_discrete_map={"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}),
                        width="stretch")
    with right:
        st.subheader("Top 10 reasons for attrition")
        st.plotly_chart(px.bar(reasons.sort_values("Importance"), x="Importance", y="Factor",
                               orientation="h"), width="stretch")

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("High risk by department")
        dep = view[view["RiskLevel"] == "High"]["Department"].value_counts().reset_index()
        dep.columns = ["Department", "Employees"]
        st.plotly_chart(px.bar(dep, x="Department", y="Employees"), width="stretch")
    with right2:
        st.subheader("Average leave probability by job role")
        role = view.groupby("JobRole")["LeaveProbability"].mean().sort_values().reset_index()
        st.plotly_chart(px.bar(role, x="LeaveProbability", y="JobRole", orientation="h"),
                        width="stretch")

    # Employee table
    st.subheader("Highest-risk employees")
    cols = ["Department", "JobRole", "MonthlyIncome", "OverTime", "JobSatisfaction",
            "YearsSinceLastPromotion", "LeaveProbability", "RiskLevel"]
    st.dataframe(view.sort_values("LeaveProbability", ascending=False)[cols].head(50),
                 width="stretch")

# ---------------- Tab 2: Predict one employee ----------------
with tab2:
    st.subheader("Check the attrition risk of one employee")

    a, b, c = st.columns(3)
    age = a.slider("Age", int(risk["Age"].min()), int(risk["Age"].max()), 35)
    gender = a.selectbox("Gender", sorted(risk["Gender"].unique()))
    department = a.selectbox("Department", sorted(risk["Department"].unique()))
    job_role = b.selectbox("Job role", sorted(risk["JobRole"].unique()))
    income = b.number_input("Monthly income", int(risk["MonthlyIncome"].min()),
                            int(risk["MonthlyIncome"].max()), int(risk["MonthlyIncome"].median()))
    overtime = b.selectbox("Overtime", ["No", "Yes"])
    years_company = c.slider("Years at company", 0, int(risk["YearsAtCompany"].max()), 5)
    years_promo = c.slider("Years since last promotion", 0, int(risk["YearsSinceLastPromotion"].max()), 2)
    distance = c.slider("Distance from home", int(risk["DistanceFromHome"].min()),
                        int(risk["DistanceFromHome"].max()), 10)
    d, e, f = st.columns(3)
    job_sat = d.slider("Job satisfaction (1-4)", 1, 4, 3)
    wlb = e.slider("Work-life balance (1-4)", 1, 4, 3)
    perf = f.slider("Performance rating", int(risk["PerformanceRating"].min()),
                    int(risk["PerformanceRating"].max()), int(risk["PerformanceRating"].min()))

    if st.button("Predict risk"):
        row = pd.DataFrame([{
            "Age": age, "Gender": 1 if gender == "Male" else 0, "MonthlyIncome": income,
            "OverTime": 1 if overtime == "Yes" else 0, "YearsAtCompany": years_company,
            "YearsSinceLastPromotion": years_promo, "JobSatisfaction": job_sat,
            "WorkLifeBalance": wlb, "DistanceFromHome": distance, "PerformanceRating": perf,
            "Department_" + department: 1, "JobRole_" + job_role: 1,
        }])
        row = row.reindex(columns=columns, fill_value=0).astype(float)   # same columns as training
        row[num_cols] = scaler.transform(row[num_cols])
        prob = model.predict_proba(row)[0, 1]

        # top 15% = High, next 30% = Medium
        high_cut = risk["LeaveProbability"].quantile(0.85)
        med_cut = risk["LeaveProbability"].quantile(0.55)
        level = "High" if prob >= high_cut else "Medium" if prob >= med_cut else "Low"

        m1, m2 = st.columns(2)
        m1.metric("Leave probability", f"{prob * 100:.1f}%")
        m2.metric("Risk level", level)