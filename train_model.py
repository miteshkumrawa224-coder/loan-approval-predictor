import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

print("Loading data...")
df = pd.read_csv("loan_approval_data.csv")

# Clean data
df = df.drop("Applicant_ID", axis=1)

# Clean Target
cat_imp_y = SimpleImputer(strategy="most_frequent")
df[["Loan_Approved"]] = cat_imp_y.fit_transform(df[["Loan_Approved"]])
y = df["Loan_Approved"].apply(lambda x: 1 if x == "Yes" else 0)

X = df.drop("Loan_Approved", axis=1)

print("Imputing missing values...")
# Separate categorical columns and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

num_imp = SimpleImputer(strategy="mean")
X[numerical_cols] = num_imp.fit_transform(X[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
X[categorical_cols] = cat_imp.fit_transform(X[categorical_cols])

# Education Level to binary using LabelEncoder (matching notebook)
le_edu = LabelEncoder()
X["Education_Level"] = le_edu.fit_transform(X["Education_Level"])

# One hot encoding for other categorical columns
ohe_cols = [c for c in categorical_cols if c != "Education_Level"]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(X[ohe_cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=X.index)

X = pd.concat([X.drop(columns=ohe_cols), encoded_df], axis=1)

# Feature engineering
X["DTI_Ratio_sq"] = X["DTI_Ratio"] ** 2
X["Credit_Score_aq"] = X["Credit_Score"] ** 2
X["Applicant_Income_log"] = np.log1p(X["Applicant_Income"])
X = X.drop(columns=["DTI_Ratio", "Credit_Score"])

# Ensure X_train gets the right columns
ohe_feature_names = list(X.columns)

# Splitting
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
print("Training Logistic Regression Model...")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save artifacts
print("Saving models and preprocessors...")
os.makedirs("models", exist_ok=True)
joblib.dump(num_imp, 'models/num_imputer.pkl')
joblib.dump(cat_imp, 'models/cat_imputer.pkl')
joblib.dump(le_edu, 'models/le_edu.pkl')
joblib.dump(ohe, 'models/ohe.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(log_model, 'models/log_model.pkl')
joblib.dump(ohe_cols, 'models/ohe_cols.pkl')
joblib.dump(ohe_feature_names, 'models/ohe_feature_names.pkl')

print("Done! Artifacts saved successfully.")
