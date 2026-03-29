import pandas as pd
import numpy as np

# Creating fake data for 100 people
data = {
    'Applicant_Name': ['Person ' + str(i) for i in range(1, 101)],
    'Income': np.random.randint(2000, 10000, 100),
    'Loan_Amount': np.random.randint(100, 500, 100),
    'Credit_History': np.random.choice([0, 1], 100, p=[0.3, 0.7])
}

df = pd.DataFrame(data)

# Logic: Approve if Credit History is 1 (Good) and Income is over 3500
df['Loan_Status'] = np.where((df['Credit_History'] == 1) & (df['Income'] > 3500), 'Approved', 'Rejected')

# Save it
df.to_csv('loan_data.csv', index=False)
print("✅ Step 1: 'loan_data.csv' created successfully!")