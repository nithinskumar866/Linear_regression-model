import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Generate Dataset
np.random.seed(42)
n = 200
exp = np.random.normal(5, 2.5, n).clip(0)
certs = np.random.randint(0, 6, n)
projects = np.random.randint(1, 11, n)
edu_levels = np.random.choice(['Bachelors', 'Masters', 'PhD'], n, p=[0.6, 0.3, 0.1])
edu_map = {'Bachelors': 1, 'Masters': 2, 'PhD': 3}
edu_num = [edu_map[e] for e in edu_levels]

salary = (
    25000 + exp*4000 + certs*3000 + projects*1500 + np.array(edu_num)*5000 +
    np.random.normal(0, 3000, n)
)

df = pd.DataFrame({
    "YearsExperience": exp,
    "Certifications": certs,
    "ProjectsHandled": projects,
    "EducationLevel": edu_levels,
    "EducationNumeric": edu_num,
    "Salary": salary.round(2)
})

df.drop(columns=["EducationLevel"]).to_csv("dataset/job_salary_linear_dataset.csv", index=False)

# Train and save model
X = df[["YearsExperience", "Certifications", "ProjectsHandled", "EducationNumeric"]]
y = df["Salary"]
model = LinearRegression().fit(X, y)
joblib.dump(model, "model/linear_model.pkl")
