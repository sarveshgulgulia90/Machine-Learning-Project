import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
df = pd.read_csv('TMDB_movie_dataset_v11.csv')
df = df.dropna(subset=['budget', 'revenue'])
df = df[df['budget'] > 0]
df['ROI'] = df['revenue'] / df['budget']
def classify_success(roi):
    if roi >= 2.5: return 'Blockbuster'
    elif roi >= 2: return 'Super Hit'
    elif roi >= 1.5: return 'Hit'
    elif roi >= 1.25: return 'Average'
    elif roi >= 1: return 'Flop'
    else: return 'Disaster'
df['success_category'] = df['ROI'].apply(classify_success)
features = ['budget', 'runtime', 'vote_average', 'vote_count', 'popularity', 'original_language']
X = df[features]
y = df['success_category']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['budget', 'runtime', 'vote_average', 'vote_count', 'popularity']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['original_language'])])
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=5, random_state=42))])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Random Forest F1 Score (weighted):", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu',xticklabels=class_names, yticklabels=class_names)
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
plt.figure(figsize=(10, 6))
for i in range(len(class_names)):
    plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Random Forest - ROC Curve (Multi-Class)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()
pred_df = pd.DataFrame({
    'Movie_Title': df.loc[X_test.index, 'title'] if 'title' in df.columns else np.nan,
    'Actual': label_encoder.inverse_transform(y_test),
    'Predicted': label_encoder.inverse_transform(y_pred)
})
pred_df.to_csv("random_forest_predictions.csv", index=False)
print("Saved predictions to random_forest_predictions.csv")
