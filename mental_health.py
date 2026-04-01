# Mental Health Data Analysis Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. Load Dataset
# -----------------------------
df=pd.read_csv("mental_health.csv")
print(df.head())
print(df.info())
print(df.describe())

# -----------------------------
# 2. Data Quality Check
# -----------------------------
from sklearn.preprocessing import LabelEncoder
print("Missing values:")
print(df.isnull().sum())

print("\nDuplicates:", df.duplicated().sum())

print("\nUnique values per columns:")
for col in df.columns:
    print(col, ":", df[col].unique())

# -----------------------------
# 3. Exploratory Data Analysis (EDA)
# -----------------------------
print("\nGrowing Stress distribution:")
print(df['Growing_Stress'].value_counts())

# -----------------------------
# 4. Growing Stress Distribution
# -----------------------------
sns.countplot(x='Growing_Stress', data=df)
plt.title("Growing Stress Distribution")
plt.show()

# -----------------------------
# 5. Growing Stress by Age Group
# -----------------------------
sns.countplot(x='Age', hue='Growing_Stress', data=df, palette='pastel')
plt.title("Growing Stress by Age Group")
plt.show()

# -----------------------------
# 6. Growing Stress by Gender
# -----------------------------
sns.countplot(x='Gender', hue='Growing_Stress', data=df, palette='pastel')
plt.title("Growing Stress by Gender")
plt.show()

# -----------------------------
# 7. Growing Stress by Occupation
# -----------------------------
sns.countplot(x='Occupation', hue='Growing_Stress', data=df, palette='pastel')
plt.title("Growing Stress by Occupation")
plt.show()

# -----------------------------
# 8. Growing Stress by Mental Health History
# -----------------------------
sns.countplot(x='Mental_Health_History', hue='Growing_Stress', data=df, palette='pastel')
plt.title("Growing Stress by Mental Health History")
plt.show()

# -----------------------------
# 9. Growing Stress by Mood Swings
# -----------------------------
sns.countplot(x='Mood_Swings', hue='Growing_Stress', data=df, palette='pastel')
plt.title("Growing Stress by Mood Swings")
plt.show()

# -----------------------------
# 10. Data Preprocessing (Label Encoding)
# -----------------------------
le=LabelEncoder()
df_encoded=df.copy()

for col in df_encoded.columns:
    df_encoded[col]=le.fit_transform(df_encoded[col])
print("\nAfter encoding:")
print(df_encoded.head())

# -----------------------------
# 11. Standard Deviation per Column
# -----------------------------
print("Standart Deviation per column:")
print(df_encoded.std().sort_values(ascending=False))

# -----------------------------
# 12. Correlation with Target
# ----------------------------
correlations=df_encoded.corr(numeric_only=True)['Growing_Stress'].sort_values(ascending=False)
print(correlations)

correlations.drop('Growing_Stress').head(10).plot(kind='bar', color='pink')
plt.title("Feature Correlation with Growing Stress")
plt.axhline(0, color='black', linewidth=0.8)
plt.show()

# -----------------------------
# 13. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(12,6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 14. Binarize Target Variable
# -----------------------------
df['Stress_Binary']=df['Growing_Stress'].apply(lambda x: 0 if x== 'No' else 1)
print(df['Stress_Binary'].value_counts())

# -----------------------------
# 15. Data Preprocessing (Binary Target)
# -----------------------------
df_binary=df.drop('Growing_Stress',axis=1).copy()
for col in df_binary.columns:
    if df_binary[col].dtype=='object':
        df_binary[col]=le.fit_transform(df_binary[col])
x=df_binary.drop('Stress_Binary', axis=1)
y=df_binary['Stress_Binary']

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)

# -----------------------------
# 16. Train Random Forest (Binary)
# -----------------------------
model_binary=RandomForestClassifier(random_state=42)
model_binary.fit(x_train, y_train)

y_pred_binary=model_binary.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print(classification_report(y_test, y_pred_binary))

# -----------------------------
# 17. Confusion Matrix (Random Forest)
# -----------------------------
sns.heatmap(confusion_matrix(y_test, y_pred_binary), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Binary)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# -----------------------------
# 18. Handle Class Imbalance (SMOTE)
# -----------------------------
from imblearn.over_sampling import SMOTE 
smote=SMOTE(random_state=42)

x_train_smote, y_train_smote= smote.fit_resample(x_train, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_smote.value_counts().to_dict())

# -----------------------------
# 19. Train Random Forest (SMOTE)
# -----------------------------
model_smote= RandomForestClassifier(random_state=42)
model_smote.fit(x_train_smote, y_train_smote)

y_pred_smote=model_smote.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred_smote))
print(classification_report(y_test, y_pred_smote))

# -----------------------------
# 20. Confusion Matrix (SMOTE)
# -----------------------------
sns.heatmap(confusion_matrix(y_test, y_pred_smote), annot=True, fmt='d',cmap='Blues' )
plt.title('Confusion Matrix (SMOTE)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# -----------------------------
# 21. Train Logistic Regression
# -----------------------------
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(random_state=42, max_iter=1000)

model_lr.fit(x_train, y_train)

y_pred_lr = model_lr.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# -----------------------------
# 22. Confusion Matrix (Logistic Regression)
# -----------------------------
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Logistic Regression)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show() 

# -----------------------------
# 23. Train Logistic Regression (SMOTE)
# -----------------------------
model_lr_smote=LogisticRegression(random_state=42, max_iter=1000)
model_lr_smote.fit(x_train_smote, y_train_smote)

y_pred_lr_smote=model_smote.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred_lr_smote))
print(classification_report(y_test, y_pred_lr_smote, zero_division=0))

# -----------------------------
# 24. Confusion Matrix (Logistic Regression + SMOTE)
# -----------------------------
sns.heatmap(confusion_matrix(y_test, y_pred_lr_smote), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Logistic Regression + SMOTE)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# -----------------------------
# 25. Feature Importance
# -----------------------------




