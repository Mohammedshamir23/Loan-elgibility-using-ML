# ============================================
# SMARTBANK SIMPLE TRAIN MODEL
# NO CROSS VALIDATION
# ============================================

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- LOAD DATA ----------------
data = pd.read_csv("loan_data.csv")

data.drop(['Loan_ID', 'Credit_History'], axis=1, inplace=True)

data['Loan_Status'] = data['Loan_Status'].map({'Y':1, 'N':0})
data['Dependents'] = data['Dependents'].replace('3+', '3')

# Convert income to thousands
data['ApplicantIncome'] = data['ApplicantIncome'] / 1000
data['CoapplicantIncome'] = data['CoapplicantIncome'] / 1000

# ---------------- HANDLE MISSING ----------------
categorical_cols = [
    'Gender','Married','Dependents',
    'Education','Self_Employed','Property_Area'
]

numerical_cols = [
    'ApplicantIncome','CoapplicantIncome',
    'LoanAmount','Loan_Amount_Term'
]

cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

num_imputer = SimpleImputer(strategy='median')
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

data['Dependents'] = data['Dependents'].astype(int)

# ---------------- FEATURE ENGINEERING ----------------
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['LogTotalIncome'] = np.log(data['TotalIncome'] + 1)
data['LogLoanAmount'] = np.log(data['LoanAmount'] + 1)

data['EMI'] = data['LoanAmount'] / data['Loan_Amount_Term']
data['EMI_to_Income'] = data['EMI'] / (data['TotalIncome'] + 1)

data['IncomePerDependent'] = data['TotalIncome'] / (data['Dependents'] + 1)

data['HighIncomeFlag'] = (data['TotalIncome'] > 10).astype(int)
data['ShortTermFlag'] = (data['Loan_Amount_Term'] < 180).astype(int)
data['LargeLoanFlag'] = (data['LoanAmount'] > 250).astype(int)

# ---------------- ENCODING ----------------
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

# ---------------- SPLIT ----------------
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- ACCURACY ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy,4))

# ---------------- SAVE ----------------
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(le_dict, open("encoder.pkl","wb"))
pickle.dump(X.columns.tolist(), open("feature_names.pkl","wb"))

print("Model Saved Successfully ✅")