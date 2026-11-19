from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Student Score Predictor", page_icon="🎓", layout="wide")


@st.cache_data
def demo_data() -> pd.DataFrame:
    """Create clearly synthetic practice data with a repeatable random seed."""
    rng = np.random.default_rng(17)
    n = 240
    study_hours = rng.uniform(0, 20, n)
    attendance = rng.uniform(50, 100, n)
    previous_score = rng.uniform(35, 100, n)
    tutoring_hours = rng.poisson(2, n)
    noise = rng.normal(0, 7, n)
    final_score = np.clip(
        18 + 1.1 * study_hours + 0.22 * attendance + 0.32 * previous_score
        + 1.5 * tutoring_hours + noise,
        0,
        100,
    )
    return pd.DataFrame({
        "study_hours_per_week": study_hours.round(1),
        "attendance_percent": attendance.round(1),
        "previous_score": previous_score.round(1),
        "tutoring_hours": tutoring_hours,
        "final_score": final_score.round(1),
    })


@st.cache_data
def read_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(data))


st.title("🎓 Student Performance Predictor")
st.write("Explore how input variables relate to a numeric student outcome, then evaluate a simple machine-learning model on held-out data.")
st.warning("This is an educational demo, not a tool for grading, admissions, or decisions about individual students. The built-in practice data is synthetic and does not describe real students.")

with st.sidebar:
    st.header("Choose data")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    use_demo = st.checkbox("Use synthetic practice data", value=uploaded is None)

try:
    if uploaded is not None and not use_demo:
        df = read_csv(uploaded.getvalue())
        data_source = uploaded.name
    else:
        df = demo_data()
        data_source = "Synthetic practice data"
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if df.empty:
    st.error("This file has no rows.")
    st.stop()

# Convert number-like text columns, while leaving other categories available as inputs.
for col in df.columns:
    if df[col].dtype == "object":
        cleaned = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        if cleaned.notna().sum() >= max(3, int(0.7 * df[col].notna().sum())):
            df[col] = cleaned

numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
if len(numeric_columns) < 2:
    st.error("Please include a numeric outcome column and at least one numeric input column.")
    st.stop()

with st.sidebar:
    st.header("Set up prediction")
    target = st.selectbox("Numeric outcome to predict", numeric_columns, index=numeric_columns.index("final_score") if "final_score" in numeric_columns else len(numeric_columns) - 1)
    candidates = [col for col in df.columns if col != target]
    default_features = [c for c in candidates if c in ["study_hours_per_week", "attendance_percent", "previous_score", "tutoring_hours"]]
    if not default_features:
        default_features = candidates[: min(5, len(candidates))]
    features = st.multiselect("Input columns", candidates, default=default_features)
    test_fraction = st.slider("Hold-out test share", min_value=0.15, max_value=0.4, value=0.25, step=0.05)

if len(features) == 0:
    st.info("Choose one or more input columns in the sidebar.")
    st.stop()

model_data = df[[*features, target]].copy()
model_data[target] = pd.to_numeric(model_data[target], errors="coerce")
model_data = model_data.dropna(subset=[target])
if len(model_data) < 10:
    st.error("At least 10 rows with a numeric outcome are needed for a meaningful train/test split.")
    st.stop()
if model_data[target].nunique() < 2:
    st.error("The selected outcome needs at least two different numeric values.")
    st.stop()

X = model_data[features]
y = model_data[target]
numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
categorical_features = [c for c in features if c not in numeric_features]

transformers = []
if numeric_features:
    transformers.append(("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric_features))
if categorical_features:
    transformers.append(("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features))

preprocessor = ColumnTransformer(transformers)
model = Pipeline([
    ("prepare", preprocessor),
    ("regression", LinearRegression()),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_fraction, random_state=17
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
baseline = np.full(len(y_test), y_train.mean())
mae = mean_absolute_error(y_test, predictions)
baseline_mae = mean_absolute_error(y_test, baseline)
rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else float("nan")

st.subheader("Model results")
st.caption(f"Data: {data_source} · {len(model_data)} rows used · {len(features)} input columns · Linear regression")
metric_cols = st.columns(4)
metric_cols[0].metric("Mean absolute error", f"{mae:.2f}", help="Typical absolute distance between prediction and actual value, in the outcome's units. Lower is better.")
metric_cols[1].metric("RMSE", f"{rmse:.2f}", help="Penalizes larger errors more than MAE. Lower is better.")
metric_cols[2].metric("R²", f"{r2:.2f}", help="How much variation is explained on this test split. Can be negative; higher is better.")
metric_cols[3].metric("Mean-only baseline MAE", f"{baseline_mae:.2f}", help="Error from always predicting the training-set average.")

if mae < baseline_mae:
    st.success("On this held-out split, the model had lower MAE than the mean-only baseline. This is an initial comparison, not proof it will predict other students well.")
else:
    st.info("On this held-out split, the model did not beat the mean-only baseline by MAE. Try different relevant inputs or more representative data.")

results = pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": predictions})
fig = px.scatter(results, x="Actual", y="Predicted", title="Predicted vs. actual outcomes", labels={"Actual": "Actual outcome", "Predicted": "Predicted outcome"})
lo = float(min(results.min()))
hi = float(max(results.max()))
fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line={"dash": "dash", "color": "gray"})
st.plotly_chart(fig, use_container_width=True)

with st.expander("See test-set predictions"):
    st.dataframe(results.round(2), use_container_width=True, hide_index=True)

with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

st.caption("Model quality depends on data quality and the test split. Associations do not establish that an input causes an outcome. Avoid sensitive personal data and never use this demo to make decisions about real students.")
