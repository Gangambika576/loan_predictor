import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load the data
df = pd.read_csv('loan_data.csv')

# 2. IQR Technique (Removing Outliers as per your image instructions)
Q1 = df['Income'].quantile(0.25)
Q3 = df['Income'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering the data
df = df[(df['Income'] >= lower_bound) & (df['Income'] <= upper_bound)]

# 3. Train the AI
X = df[['Income', 'Loan_Amount', 'Credit_History']] # Input
y = df['Loan_Status'] # Goal

model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# 4. Save the "Brain"
pickle.dump(model, open('loan_model.pkl', 'wb'))
print("✅ Step 2: AI Model 'loan_model.pkl' is ready!")