# Experiment 201: Student Performance Analysis
import pandas as pd

df = pd.read_csv(r"C:\Users\HP\Downloads\ML Lab Exp's\Exp-1\student_performance_100_samples.csv")

print(df.head(5))

# b) Gender distribution by department
gender_distr = pd.crosstab(df['Department'], df['Gender'])
print(gender_distr)

# c) Total marks and grades
df['Total_Marks'] = df['Mid1'] + df['Mid2']
print(df['Total_Marks'])
print(df[['Mid1', 'Mid2']].mean())

grades = []
for i in range(len(df)):
    avg = df['Total_Marks'].iloc[i] / 2
    if avg >= 90:
        grades.append('A+')
    elif avg >= 80:
        grades.append('A')
    elif avg >= 70:
        grades.append('B')
    elif avg >= 60:
        grades.append('C')
    elif avg >= 50:
        grades.append('D')
    else:
        grades.append('F')

df['Grade'] = grades
print(df[['Name', 'Grade']])

# d) Scholarship eligibility
scholarship = []
for i in range(len(df)):
    cgpa = df['CGPA'].iloc[i]
    scholarship.append('Eligible' if cgpa >= 9 else 'Not Eligible')

df['Scholarship'] = scholarship
print(df[['Name', 'Scholarship']])
