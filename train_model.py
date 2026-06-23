# ============================================================
#   CarPrice AI — Model Training Script
#   CodeAlpha Internship | Task 1: Car Price Prediction
# ============================================================

# --- Step 1: Libraries Import ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   CarPrice AI — Model Training Started")
print("=" * 55)

# --- Step 2: Load Dataset ---
print("\n📂 Step 1: Loading Dataset...")
df = pd.read_csv("car_data_task.csv")
print(f"   ✅ Dataset loaded! Shape: {df.shape}")
print(f"   Total Cars: {df.shape[0]}, Total Features: {df.shape[1]}")

# --- Step 3: Basic Info ---
print("\n📊 Step 2: Dataset Overview...")
print(df.head(3).to_string())

# --- Step 4: Data Preprocessing ---
print("\n🔧 Step 3: Data Preprocessing...")

# Calculate Car Age
df['Car_Age'] = 2024 - df['Year']
print("   ✅ Car_Age column added (2024 - Year)")

# Encode Categorical Columns
le = LabelEncoder()
df['Fuel_Type_enc']    = le.fit_transform(df['Fuel_Type'])
df['Selling_type_enc'] = le.fit_transform(df['Selling_type'])
df['Transmission_enc'] = le.fit_transform(df['Transmission'])
print("   ✅ Categorical columns encoded")

# Save label classes for Streamlit app
fuel_classes         = list(df['Fuel_Type'].unique())
selling_classes      = list(df['Selling_type'].unique())
transmission_classes = list(df['Transmission'].unique())

# --- Step 5: Feature Selection ---
print("\n🎯 Step 4: Feature Engineering...")
features = [
    'Present_Price', 'Driven_kms', 'Car_Age',
    'Fuel_Type_enc', 'Selling_type_enc', 'Transmission_enc', 'Owner'
]
target = 'Selling_Price'

X = df[features]
y = df[target]
print(f"   ✅ Features selected: {features}")
print(f"   ✅ Target: {target}")

# --- Step 6: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📦 Step 5: Train/Test Split...")
print(f"   Training Samples : {len(X_train)}")
print(f"   Testing  Samples : {len(X_test)}")

# --- Step 7: Train Multiple Models ---
print("\n🤖 Step 6: Training Models...")

models = {
    "Linear Regression"     : LinearRegression(),
    "Random Forest"         : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"     : GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost"               : xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
}

results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {"R2 Score": round(r2, 4),
                     "MAE": round(mae, 4),
                     "RMSE": round(rmse, 4)}
    trained_models[name] = model
    print(f"   ✅ {name:<25} | R²: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

# --- Step 8: Best Model Selection ---
print("\n🏆 Step 7: Selecting Best Model...")
best_model_name = max(results, key=lambda x: results[x]["R2 Score"])
best_model      = trained_models[best_model_name]
best_r2         = results[best_model_name]["R2 Score"]
print(f"   🥇 Best Model: {best_model_name} (R² = {best_r2})")

# --- Step 9: Save Model & Metadata ---
print("\n💾 Step 8: Saving Model...")
os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/best_model.pkl")
print("   ✅ Best model saved → model/best_model.pkl")

# Save model results for app
results_df = pd.DataFrame(results).T.reset_index()
results_df.columns = ["Model", "R2 Score", "MAE", "RMSE"]
results_df.to_csv("model/model_results.csv", index=False)
print("   ✅ Model results saved → model/model_results.csv")

# Save metadata
metadata = {
    "best_model_name"     : best_model_name,
    "best_r2"             : best_r2,
    "features"            : features,
    "fuel_classes"        : fuel_classes,
    "selling_classes"     : selling_classes,
    "transmission_classes": transmission_classes,
    "total_cars"          : int(df.shape[0]),
    "year_min"            : int(df['Year'].min()),
    "year_max"            : int(df['Year'].max()),
}
import json
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print("   ✅ Metadata saved → model/metadata.json")

# Save processed dataset for app
df.to_csv("model/processed_data.csv", index=False)
print("   ✅ Processed data saved → model/processed_data.csv")

# --- Step 10: Feature Importance ---
print("\n📈 Step 9: Generating Feature Importance Chart...")
if hasattr(best_model, 'feature_importances_'):
    fi = pd.DataFrame({
        'Feature'   : features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=True)

    plt.figure(figsize=(8, 5))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    plt.barh(fi['Feature'], fi['Importance'], color=colors)
    plt.xlabel('Importance Score')
    plt.title(f'Feature Importance — {best_model_name}')
    plt.tight_layout()
    plt.savefig("model/feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Feature importance chart saved → model/feature_importance.png")

# --- Done ---
print("\n" + "=" * 55)
print("   🎉 Training Complete! Summary:")
print("=" * 55)
for name, res in results.items():
    marker = "🥇" if name == best_model_name else "  "
    print(f"   {marker} {name:<25} R²: {res['R2 Score']}")
print(f"\n   Best Model → {best_model_name}")
print(f"   Accuracy   → {best_r2 * 100:.2f}%")
print("\n   ✅ Files saved in 'model/' folder")
print("   ▶  Now run: streamlit run app.py")
print("=" * 55)