import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import gaussian_kde
from sklearn.model_selection import learning_curve

df = pd.read_csv('us_tornado_dataset_1950_2021.csv')

features = ['mag', 'len', 'wid']
target = 'inj'

# Clean the data: remove rows with missing or invalid target
df_model = df[df[target] >= 0]

# Create feature matrix X and target vector y
X = df_model[features]
y = df_model[target]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Evaluate XGBoost
rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2 = r2_score(y_test, y_pred_xgb)
print(f"XGBoost --> RMSE: {rmse:.2f}, R²: {r2:.2f}")





## BELL CURVE (KDE) of log(1 + Injuries)
y_test_log = np.log1p(y_test)
y_pred_log = np.log1p(y_pred_xgb)

# Clean non-finite values
valid_mask = np.isfinite(y_test_log) & np.isfinite(y_pred_log)
y_test_log_clean = y_test_log[valid_mask]
y_pred_log_clean = y_pred_log[valid_mask]

# Plot KDEs
kde_actual = gaussian_kde(y_test_log_clean)
kde_pred = gaussian_kde(y_pred_log_clean)

plt.figure(figsize=(8, 5))
x_vals = np.linspace(0, max(y_test_log_clean.max(), y_pred_log_clean.max()), 200)
plt.plot(x_vals, kde_actual(x_vals), label='Actual Injuries (log)', linewidth=2)
plt.plot(x_vals, kde_pred(x_vals), label='XGBoost Predictions (log)', linewidth=2)
plt.fill_between(x_vals, kde_actual(x_vals), alpha=0.3)
plt.fill_between(x_vals, kde_pred(x_vals), alpha=0.3)
plt.title("Log-Transformed KDE: XGBoost")
plt.xlabel("log(1 + Injuries)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()


## FEATURE IMPORTANCE PLOT 
#"XGBoost builds many trees using features like length, width, and magnitude. Each time it splits the data, it picks the feature that best improves predictions by reducing error. The feature importance plot shows which features contributed the most to making better splits. In our case, magnitude helped the most to reduce the error, so it’s ranked highest."
feature_importances = xgb.feature_importances_
sorted_idx = np.argsort(feature_importances)
sorted_features = X_train.columns[sorted_idx]
sorted_importances = feature_importances[sorted_idx]
palette = sns.color_palette("viridis", len(sorted_features))

plt.figure(figsize=(8, 5))
sns.barplot(x=sorted_importances, y=sorted_features, palette=palette)
plt.title("Feature Importance (XGBoost)", fontsize=14)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## LEARNING CURVE
train_sizes, train_scores, val_scores = learning_curve(
    XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    X_train, y_train,
    cv=5,
    scoring='neg_root_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_rmse = -train_scores.mean(axis=1)
val_rmse = -val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_rmse, label='Training RMSE', marker='o')
plt.plot(train_sizes, val_rmse, label='Validation RMSE', marker='o')
plt.title('Learning Curve (XGBoost)')
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')
plt.legend()
plt.tight_layout()
plt.show()