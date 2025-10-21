import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# -----------------------------------------------------------
# 1. Dataset Creation
# -----------------------------------------------------------
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y
print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------------------------------------
# 2. Data Split & Scaling
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------
# 3. Train SVM (Linear Kernel)
# -----------------------------------------------------------
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# -----------------------------------------------------------
# 4. Train Decision Tree
# -----------------------------------------------------------
tree_model = DecisionTreeClassifier(max_depth=10, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))

# -----------------------------------------------------------
# 5. Hyperparameter Tuning (SVM)
# -----------------------------------------------------------
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
grid = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train_scaled, y_train)
best_svm = grid.best_estimator_
y_pred_best_svm = best_svm.predict(X_test_scaled)
print("Best SVM Params:", grid.best_params_)
print("Tuned SVM Accuracy:", accuracy_score(y_test, y_pred_best_svm))

# -----------------------------------------------------------
# 6. Random Forest
# -----------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, n_jobs=-1)
print("Random Forest CV Mean Accuracy:", cv_scores.mean())

# -----------------------------------------------------------
# 7. Voting Ensemble
# -----------------------------------------------------------
voting = VotingClassifier(
    estimators=[('svm', best_svm), ('tree', tree_model), ('rf', rf_model)],
    voting='hard',
    n_jobs=-1
)
voting.fit(X_train_scaled, y_train)
ensemble_acc = voting.score(X_test_scaled, y_test)
print("Voting Ensemble Accuracy:", ensemble_acc)

# -----------------------------------------------------------
# 8. Confusion Matrices
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_best_svm), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Tuned SVM Confusion Matrix')
sns.heatmap(confusion_matrix(y_test, y_pred_tree), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Decision Tree Confusion Matrix')
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Reds', ax=axes[2])
axes[2].set_title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 9. Accuracy Summary
# -----------------------------------------------------------
print("\nAccuracy Summary")
print(f"SVM: {accuracy_score(y_test, y_pred_svm):.3f}")
print(f"Tuned SVM: {accuracy_score(y_test, y_pred_best_svm):.3f}")
print(f"Decision Tree: {accuracy_score(y_test, y_pred_tree):.3f}")
print(f"Random Forest: {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"Voting Ensemble: {ensemble_acc:.3f}")

# -----------------------------------------------------------
# 10. Save Models for Deployment
# -----------------------------------------------------------
joblib.dump(best_svm, "best_svm_model.pkl")
joblib.dump(tree_model, "decision_tree_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ All models saved successfully! Ready for deployment.")
