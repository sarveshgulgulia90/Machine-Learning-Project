import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,f1_score, accuracy_score, precision_score,recall_score, roc_curve, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
df = pd.read_csv('TMDB_movie_dataset_v11.csv')
df = df.dropna(subset=['budget', 'revenue', 'genres'])  
df = df[df['budget'] > 0]
df['ROI'] = df['revenue'] / df['budget']
def classify_success(roi):
    if roi >= 2.5: return 'Blockbuster'
    elif roi >= 2:return 'Super Hit'
    elif roi >= 1.5:return 'Hit'
    elif roi >= 1.25: return 'Average'
    elif roi >= 1:return 'Flop'
    else:return 'Disaster'
df['success_category'] = df['ROI'].apply(classify_success)
features = ['budget', 'runtime', 'vote_average', 'vote_count', 'popularity', 'original_language', 'genres']
X = df[features]
y = df['success_category']
y_encoded = y.astype('category').cat.codes
class_names = dict(enumerate(y.astype('category').cat.categories))
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
categorical_cols = ['original_language', 'genres']
numerical_cols = ['budget', 'runtime', 'vote_average', 'vote_count', 'popularity']
preprocessor = ColumnTransformer([('num', StandardScaler(), numerical_cols),('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])
pipeline = Pipeline([('preprocessor', preprocessor),('classifier', DecisionTreeClassifier(random_state=42))])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Accuracy:", accuracy)
print("Precision (weighted):", precision)
print("Recall (weighted):", recall)
print("F1 Score (weighted):", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names.values()))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',xticklabels=class_names.values(), yticklabels=class_names.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Decision Tree Confusion Matrix')
plt.show()
y_test_bin = label_binarize(y_test, classes=np.unique(y_encoded))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
plt.figure(figsize=(10, 6))
for i in range(len(class_names)):
    plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Decision Tree - Multi-class")
plt.legend(loc="lower right")
plt.grid()
plt.show()
actual_labels = [class_names[i] for i in y_test]
predicted_labels = [class_names[i] for i in y_pred]
pred_df = pd.DataFrame({'Actual': actual_labels,'Predicted': predicted_labels})
if 'title' in df.columns:
    pred_df['Movie_Title'] = df.loc[X_test.index, 'title'].values
pred_df.to_csv("decision_tree_movie_predictions_with_genres.csv", index=False)
print("✅ Predictions saved to decision_tree_movie_predictions_with_genres.csv")
