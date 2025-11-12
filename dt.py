import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris

iris = load_iris()

iris
iris.data

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn import tree

classifier = DecisionTreeClassifier()
classifier.fit(iris.data, iris.target)

plt.figure(figsize=(15,10))
tree.plot_tree(classifier, filled=True)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, recall_score, confusion_matrix, classification_report

df = pd.read_csv("Loan Eligibility Prediction.csv")

df.head()
df.tail()
df.info()
df.describe()

df.hist(figsize=(17,16), color='skyblue')
plt.suptitle("Histogram")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("🔍 Correlation Heatmap", fontsize=14)
plt.show()

df.isnull().sum()

df = df.dropna(subset=["Loan_Status"])

for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Loan_Status", axis=1)
y = df_encoded["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("accuracy_score", accuracy_score(y_test, y_pred) * 100)
print("confusion_matrix", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(15,10))
tree.plot_tree(classifier, feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True)
