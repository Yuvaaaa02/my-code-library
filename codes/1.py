# 1. Import libraries and load data
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("student_performance_100_samples(1).csv")

# Display data
print(df.head())

# 2. Data cleaning
# Check dataset information
print(df.info())

# Check duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Display cleaned data
print(df.head())

# 3. Handle missing values
# Check missing values
print(df.isnull().sum())

# Fill missing numerical values with mean
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# Check again
print("Missing values after handling:")
print(df.isnull().sum())

# 4. Visualization
# Histogram of CGPA
plt.figure()
plt.hist(df["CGPA"], bins=10)
plt.xlabel("CGPA")
plt.ylabel("Number of Students")
plt.title("CGPA Distribution")
plt.show()

# Attendance vs CGPA
plt.figure()
plt.scatter(df["Attendance"], df["CGPA"])
plt.xlabel("Attendance")
plt.ylabel("CGPA")
plt.title("Attendance vs CGPA")
plt.show()

# Department-wise student count
plt.figure()
df["Department"].value_counts().plot(kind="bar")
plt.xlabel("Department")
plt.ylabel("Number of Students")
plt.title("Students by Department")
plt.show()