import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="ML Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #1f2937;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 0.4rem 1rem;
}
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
# 🚀 Machine Learning Model Trainer
### Train, Evaluate and Visualize ML Models Easily
""")
st.divider()

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ==================================================
    # DATASET OVERVIEW
    # ==================================================
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df, use_container_width=True)

    # ==================================================
    # ANALYSIS SECTION
    # ==================================================
    with st.expander("🔍 Descriptive Analysis", expanded=False):

        st.write("### Summary Statistics")
        st.write(df.describe(include="all").transpose())

        if df.select_dtypes(include=['object']).shape[1] > 0:
            st.write("### Categorical Value Counts")
            for col in df.select_dtypes(include=['object']).columns:
                st.write(f"**{col}**")
                st.write(df[col].value_counts())

        if df.select_dtypes(include=[np.number]).shape[1] > 1:
            st.write("### Correlation Matrix")
            corr = df.select_dtypes(include=[np.number]).corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr,
                annot=True,
                cmap="viridis",
                linewidths=0.5,
                fmt=".2f",
                ax=ax
            )
            plt.xticks(rotation=45)
            st.pyplot(fig)

    st.divider()

    # ==================================================
    # SIDEBAR CONFIGURATION
    # ==================================================
    st.sidebar.header("⚙️ Model Configuration")

    task_type = st.sidebar.radio(
        "Problem Type",
        ("Classification", "Regression")
    )

    if task_type == "Classification":
        model_list = [
            "Gaussian Naive Bayes",
            "Logistic Regression",
            "KNN",
            "Decision Tree",
            "Random Forest",
            "SVM"
        ]
    else:
        model_list = [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "KNN",
            "Decision Tree",
            "Random Forest",
            "SVM"
        ]

    model_name = st.sidebar.selectbox("Select Model", model_list)

    # Target selection
    if task_type == "Classification":
        possible_targets = df.select_dtypes(include=['object']).columns.tolist()
    else:
        possible_targets = df.select_dtypes(include=[np.number]).columns.tolist()

    if not possible_targets:
        st.warning("No suitable target column found.")
        st.stop()

    target = st.sidebar.selectbox("Select Target Variable", possible_targets)

    features = st.sidebar.multiselect(
        "Select Features",
        df.columns.drop(target),
        default=list(df.columns.drop(target))
    )

    if not features:
        st.warning("Please select at least one feature.")
        st.stop()

    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100

    train_button = st.sidebar.button("🚀 Train Model")

    # ==================================================
    # DATA PREPARATION
    # ==================================================
    X = df[features].copy()
    y = df[target].copy()

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # ==================================================
    # MODEL TRAINING
    # ==================================================
    if train_button:

        st.subheader("📈 Model Results")

        # -------------------------------
        # CLASSIFICATION
        # -------------------------------
        if task_type == "Classification":

            if model_name == "Gaussian Naive Bayes":
                model = GaussianNB()

            elif model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)

            elif model_name == "KNN":
                k = st.sidebar.slider("K Value", 1, 15, 5)
                model = KNeighborsClassifier(n_neighbors=k)

            elif model_name == "Decision Tree":
                depth = st.sidebar.slider("Max Depth", 1, 20, 5)
                model = DecisionTreeClassifier(max_depth=depth)

            elif model_name == "Random Forest":
                trees = st.sidebar.slider("Number of Trees", 10, 200, 100)
                model = RandomForestClassifier(n_estimators=trees)

            else:
                model = SVC()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            st.metric("Accuracy", f"{acc:.4f}")

            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                linewidths=1,
                cbar=False,
                ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        # -------------------------------
        # REGRESSION
        # -------------------------------
        else:

            if model_name == "Linear Regression":
                model = LinearRegression()

            elif model_name == "Ridge Regression":
                alpha = st.sidebar.slider("Alpha", 0.01, 10.0, 1.0)
                model = Ridge(alpha=alpha)

            elif model_name == "Lasso Regression":
                alpha = st.sidebar.slider("Alpha", 0.01, 10.0, 1.0)
                model = Lasso(alpha=alpha)

            elif model_name == "KNN":
                k = st.sidebar.slider("K Value", 1, 15, 5)
                model = KNeighborsRegressor(n_neighbors=k)

            elif model_name == "Decision Tree":
                depth = st.sidebar.slider("Max Depth", 1, 20, 5)
                model = DecisionTreeRegressor(max_depth=depth)

            elif model_name == "Random Forest":
                trees = st.sidebar.slider("Number of Trees", 10, 200, 100)
                model = RandomForestRegressor(n_estimators=trees)

            else:
                model = SVR()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score", f"{r2:.4f}")
            col2.metric("MAE", f"{mae:.4f}")
            col3.metric("RMSE", f"{rmse:.4f}")

            st.write("### Predicted vs Actual")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot(
                [y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                '--'
            )
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Predicted vs Actual")
            st.pyplot(fig)

else:
    st.info("📂 Please upload a CSV file to begin.")