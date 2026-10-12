# Experiment 205: Decision Tree Classification
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import metrics

df = pd.read_csv(r"C:\Users\HP\Downloads\ML Lab Exp's\Exp-5\SAHeart.csv")
df.head(5)
df = df.drop('row.names', axis=1)
X = df.drop('chd', axis=1)
y = df['chd']
X = pd.get_dummies(X, drop_first=True, dtype=int)

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.8, random_state=42
)
model = DecisionTreeClassifier(random_state=42)
model.fit(train_X, train_y)
y_pred = model.predict(test_X)
print(classification_report(test_y, y_pred))

# b) Confusion matrix and ROC-AUC
y_pred = model.predict(test_X)
y_prob = model.predict_proba(test_X)[:, 1]
cm = metrics.confusion_matrix(test_y, y_pred)

plt.figure(figsize=(6, 4))
sn.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['No CHD', 'CHD'],
    yticklabels=['No CHD', 'CHD']
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.show()

print(metrics.classification_report(test_y, y_pred))
fpr, tpr, thresholds = metrics.roc_curve(test_y, y_prob)
auc_score = metrics.roc_auc_score(test_y, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (AUC = %.3f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve")
plt.legend(loc="lower right")
plt.show()
print("ROC-AUC Score:", auc_score)
