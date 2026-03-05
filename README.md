💼  Loan-elgibility-using-ML
This project is a Machine Learning based loan eligibility prediction system that predicts whether a loan application should be Approved or Rejected based on applicant financial and personal details.
The system uses Random Forest Classification and includes a Streamlit web application that allows users to input applicant details and instantly receive a loan decision, risk grade, and EMI estimate.
The project simulates a real-world banking loan evaluation process without using credit score.

🎯 Project Objectives
Predict loan approval using machine learning
Analyze applicant financial capability
Provide risk grading (A–D) for applications
Calculate estimated EMI
Display decision explanation for transparency
Provide a bank-style interactive UI
🧠 Machine Learning Model
The project uses Random Forest Classifier, an ensemble learning algorithm that builds multiple decision trees and combines their predictions.
Why Random Forest?
Handles complex financial relationships
Reduces overfitting
Works well with mixed numerical and categorical data
Provides stable predictions

📊 Dataset Features
The dataset contains historical loan application records.
Input Features
Gender
Married
Dependents
Education
Self Employed
Applicant Income
Coapplicant Income
Loan Amount
Loan Amount Term
Property Area
Target Variable
Loan_Status
Y → Approved
N → Rejected

⚙️ Data Preprocessing
Several preprocessing steps were applied before training the model.
Steps Performed
Removed unnecessary columns (Loan_ID, Credit_History)
Handled missing values using median and mode imputation
Converted categorical variables to numeric using Label Encoding
Converted Dependents (3+) into numeric value
Standardized income and loan amount values in thousands
Created additional financial features

🔧 Feature Engineering
To improve model performance, additional features were created:
Total Income
Log Total Income
Log Loan Amount
EMI (Estimated Monthly Installment)
Income per Dependent
High Income Flag
Large Loan Flag
Short Term Loan Flag
These features help the model better understand financial risk patterns.

📈 Model Training
The dataset was split into:
80% Training Data
20% Testing Data
The model learns patterns between applicant features and loan approval outcomes.
Model performance is evaluated using prediction accuracy.

🖥️ Web Application
A Streamlit web interface was built to allow users to interact with the model.
Features
Secure login authentication
Manual input of financial data
Income and loan values entered in thousands
Instant loan prediction
Risk grade display
EMI estimation
Decision explanation

📊 Example Prediction
Input
Applicant Income: 60 (₹60,000)
Coapplicant Income: 20 (₹20,000)
Loan Amount: 150 (₹1,50,000)
Loan Term: 360 months

Output
Loan Status: Approved
Risk Grade: A
EMI: ₹416 per month
Explanation:
Strong income
Manageable loan amount
Long repayment term

🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Streamlit
Matplotlib

📂 Project Structure
Loan-Eligibility-Prediction
│
├── dataset
│   └── loan_data.csv
│
├── train_model.py
├── loan_app.py
├── model.pkl
├── requirements.txt
└── README.md

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Train the Model
python train_model.py
This will generate the trained model file.

3️⃣ Run the Web Application
streamlit run loan_app.py

Open the browser and interact with the loan prediction system.

🚀 Future Improvements
Deploy application on cloud
Integrate bank API
Add fraud detection
Improve model accuracy using advanced ensemble methods

📌 Conclusion

This project demonstrates how machine learning can assist financial institutions in making faster and more reliable loan approval decisions.
By combining data preprocessing, feature engineering, and ensemble learning, the system simulates a simplified AI-driven banking loan evaluation system.
