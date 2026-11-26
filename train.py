"""Rebuilds churn_model.joblib and churn_model_calibrated.joblib from the IBM Telco dataset."""
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

df = pd.read_csv(URL)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True).drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

num = X.select_dtypes(include="number").columns
cat = X.select_dtypes(exclude="number").columns
pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
spw = (y_train == 0).sum() / (y_train == 1).sum()

# Best parameters found by RandomizedSearchCV
clf = XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=1.0,
                    colsample_bytree=1.0, min_child_weight=3, scale_pos_weight=spw,
                    eval_metric="logloss", random_state=42)
best = Pipeline([("pre", pre), ("clf", clf)]).fit(X_train, y_train)
cal = CalibratedClassifierCV(best, method="sigmoid", cv=5).fit(X_train, y_train)

joblib.dump(best, "churn_model.joblib")
joblib.dump(cal, "churn_model_calibrated.joblib")
print("Test ROC-AUC:", round(roc_auc_score(y_test, cal.predict_proba(X_test)[:, 1]), 3))
