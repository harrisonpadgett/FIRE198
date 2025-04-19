import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor #a gradient boosting library that uses multiple decision trees and corrects the error of the previous one as it goes down

# LOADING AND PREPARING DATASET
df = pd.read_csv("seismic_data.csv")

# Define target and features
target_column = "Predicted Damage Index (0–1 Scale)"
X = df.drop(columns=target_column)
y = df[target_column]

# Define categorical and numerical columns
cat_cols = ["Seismic Zone", "Soil Type", "Structural Material", "Foundation Type", "Lateral Load Resisting System"]
num_cols = [col for col in X.columns if col not in cat_cols]



# PREPROCESSING PIPELINES
# standardizing the numeric features
numeric_transformer = StandardScaler()

# One-hot encode categorical features (converting into a binary vector)
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Combine preprocessing for numeric and categorical columns (aplying the transformations to the cat_cols and num_cols)
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])



# MODEL PIPELINE
# Full pipeline (preprocessing + regressor)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor( 
        n_estimators=100, #100 desision trees
        max_depth=5,
        learning_rate=0.1, #how much each tree contributes to the entire model
        random_state=42
    ))
])



# SPLITTING TRAIN AND TEST DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)



# EVALUATION
# Predict on test data
y_pred = pipeline.predict(X_test) #predicts damage index for test set

# Print evaluation metrics
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred))) #how far off the predictions are from the actual values.
print("R² Score:", r2_score(y_test, y_pred)) #how much variation of target is shown by input features



# 1)FEATURE IMPORTANCE PLOT
# Extract trained model and feature names
model = pipeline.named_steps["regressor"] #fetches XGBooster from the pipeline
ohe_feature_names = pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(cat_cols) #fetches ohe feature names
feature_names = np.concatenate([num_cols, ohe_feature_names])
importances = model.feature_importances_ #fetches feature importance

# Validate shapes
assert len(importances) == len(feature_names)

# Exclude the most dominant feature(prevents one feature from dominating the plot)
max_idx = np.argmax(importances)
sorted_idx = np.argsort(importances)
sorted_idx = sorted_idx[sorted_idx != max_idx][-15:]  # Top 15 excluding the top one

# Plot top feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=feature_names[sorted_idx])
plt.title("Top Feature Importances (Excluding Collapse Probability)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



# ACTUAL VS PREDICTED PLOT
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([0, 1], [0, 1], '--r')  # Diagonal reference line
plt.xlabel("Actual Damage Index")
plt.ylabel("Predicted Damage Index")
plt.title("Predicted vs. Actual Damage Index")
plt.tight_layout()
plt.show()



# RESIDUAL DISTRIBUTION (diff between actual and predicted values)
residuals = y_test - y_pred
sns.histplot(residuals, kde=True) #kernel density estimate curves the histogram
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()



# CORRELATION MATRIX OF THE NUMERIC FEATURES
numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.show()