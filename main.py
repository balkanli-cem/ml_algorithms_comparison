import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# --- DATASET LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF',
                'Neighborhood', 'ExterQual', 'KitchenQual', 'GarageType', 'SaleCondition', 'SalePrice']

    df = df[features].dropna()

    ordinal_cols = ['ExterQual', 'KitchenQual']
    for col in ordinal_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, columns=['Neighborhood', 'GarageType', 'SaleCondition'], drop_first=True)

    df['SalePrice'] = np.log1p(df['SalePrice'])

    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

# --- TRAIN MODELS ---
@st.cache_resource
def train_models():
    lr = LinearRegression().fit(X_train, y_train)
    dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train, y_train)
    return lr, dt, xgb

lr_model, dt_model, xgb_model = train_models()

# --- EVALUATE MODELS ---
def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R² Score": r2_score(y_test, y_pred)
    }

model_results = pd.DataFrame([
    evaluate_model("Linear Regression", lr_model),
    evaluate_model("Decision Tree", dt_model),
    evaluate_model("XGBoost", xgb_model)
])

# --- STREAMLIT APP ---
st.set_page_config(page_title="House Price Model Comparison", layout="centered")

st.title("🏠 House Price Prediction: Model Comparison")

st.markdown("Compare Linear Regression, Decision Tree, and XGBoost for house price prediction.")

# --- MODEL SELECTION ---
model_choice = st.selectbox("📊 Choose a model to view performance:", model_results['Model'])
selected = model_results[model_results['Model'] == model_choice].iloc[0]

st.metric("RMSE", f"{selected['RMSE']:.4f}")
st.metric("R² Score", f"{selected['R² Score']:.4f}")

# --- COMPARISON CHART ---
st.markdown("### 📈 Performance Comparison")
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].bar(model_results['Model'], model_results['RMSE'], color='skyblue')
axs[0].set_title('RMSE by Model')
axs[1].bar(model_results['Model'], model_results['R² Score'], color='salmon')
axs[1].set_title('R² Score by Model')
axs[1].set_ylim(0, 1)
st.pyplot(fig)

# --- EXPANDABLE MODEL SUMMARIES ---
with st.expander("📄 Linear Regression Summary"):
    st.write("Best performer with RMSE = 0.1618 and R² = 0.8137.")

with st.expander("📄 Decision Tree Summary"):
    st.write("Overfits the training data. RMSE = 0.2250, R² = 0.6396.")

with st.expander("📄 XGBoost Summary"):
    st.write("Performed well but slightly behind linear regression. RMSE = 0.1665, R² = 0.8026.")

# --- PREDICTION FORM ---
st.markdown("### 🧮 Predict House Price (Linear Regression)")

with st.form("prediction_form"):
    OverallQual = st.slider("Overall Quality", 1, 10, 5)
    GrLivArea = st.number_input("Living Area (sq ft)", 300, 6000, 1500)
    GarageCars = st.slider("Garage (Cars)", 0, 4, 2)
    TotalBsmtSF = st.number_input("Basement Area", 0, 4000, 800)
    FirstFlrSF = st.number_input("1st Floor Area", 300, 4000, 1000)

    ExterQual = st.selectbox("Exterior Quality", ['Po', 'Fa', 'TA', 'Gd', 'Ex'])
    KitchenQual = st.selectbox("Kitchen Quality", ['Po', 'Fa', 'TA', 'Gd', 'Ex'])

    # These must match dummy column suffixes in your dataset
    Neighborhood = st.selectbox("Neighborhood", sorted([col.split('_')[1] for col in X_train.columns if col.startswith('Neighborhood_')]))
    GarageType = st.selectbox("Garage Type", sorted([col.split('_')[1] for col in X_train.columns if col.startswith('GarageType_')]))
    SaleCondition = st.selectbox("Sale Condition", sorted([col.split('_')[1] for col in X_train.columns if col.startswith('SaleCondition_')]))

    submit = st.form_submit_button("Predict")

if submit:
    ordinal_map = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}

    input_dict = {
        'OverallQual': OverallQual,
        'GrLivArea': GrLivArea,
        'GarageCars': GarageCars,
        'TotalBsmtSF': TotalBsmtSF,
        '1stFlrSF': FirstFlrSF,
        'ExterQual': ordinal_map[ExterQual],
        'KitchenQual': ordinal_map[KitchenQual],
    }

    # Handle one-hot encodings
    for col in X_train.columns:
        if col.startswith("Neighborhood_"):
            input_dict[col] = 1 if col == f"Neighborhood_{Neighborhood}" else 0
        elif col.startswith("GarageType_"):
            input_dict[col] = 1 if col == f"GarageType_{GarageType}" else 0
        elif col.startswith("SaleCondition_"):
            input_dict[col] = 1 if col == f"SaleCondition_{SaleCondition}" else 0
        elif col not in input_dict:
            input_dict[col] = 0

    input_df = pd.DataFrame([input_dict])
    prediction_log = lr_model.predict(input_df)[0]
    price = np.expm1(prediction_log)
    st.success(f"🏷️ Estimated House Price: €{price:,.0f}")