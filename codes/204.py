# Experiment 204: Logistic Regression for German Credit Risk
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv(r"C:\Users\HP\Downloads\ML Lab Exp's\Exp-4\GermanCredit.csv")
features = list(df.columns)
features.remove('credit_risk')
y = pd.to_numeric(df['credit_risk']).astype(int)
encoded_df = pd.get_dummies(df[features], drop_first=True).astype(float)
X = sm.add_constant(encoded_df)

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.8, random_state=42
)
model = sm.Logit(train_y, train_X).fit()
print(model.summary2())

# b) Confusion matrix and classification report
predicted_prob = model.predict(test_X.astype(float))
predicted = (predicted_prob > 0.5).astype(int)

def draw_cm(actual, predicted):
    cm = metrics.confusion_matrix(actual, predicted, labels=[1, 0])
    sn.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=['Bad credit', 'Good credit'],
        yticklabels=['Bad credit', 'Good credit']
    )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

draw_cm(test_y, predicted)
print(metrics.classification_report(test_y, predicted, zero_division=0))

# c) ROC-AUC curve
fpr, tpr, thresholds = metrics.roc_curve(test_y, predicted_prob, drop_intermediate=False)
auc_score = metrics.roc_auc_score(test_y, predicted_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area=%0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc="lower right")
plt.show()

# d) Precision, recall, and confusion matrix at threshold 0.5
pred_class_opt = (predicted_prob >= 0.5).astype(int)
cm = metrics.confusion_matrix(test_y, pred_class_opt)
precision = metrics.precision_score(test_y, pred_class_opt, zero_division=0)
recall = metrics.recall_score(test_y, pred_class_opt, zero_division=0)
print("Precision:", precision)
print("Recall:", recall)

sn.heatmap(
    cm, annot=True, fmt='d',
    xticklabels=['Good credit', 'Bad credit'],
    yticklabels=['Good credit', 'Bad credit']
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()
