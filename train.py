from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib




data = pd.read_csv("./Data/credit_data.csv", index_col=0)
df = pd.DataFrame(data)

df['Credit amount'] = np.log(df['Credit amount'])
X = df.drop('Risk_bad', 1).values
y = df['Risk_bad'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('XGB', XGBClassifier()))


results = []
names = []
scoring = 'recall'

for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Comparaison des algorithmes
fig = plt.figure(figsize=(11,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "n_estimators": [3, 5, 10, 25, 50, 150],
    "max_features": [4, 7, 15, 20]
}

# model with best parameters

model = RandomForestClassifier(random_state=2)
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
grid_search.fit(X_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)

# Final model with best parameters
rf = RandomForestClassifier(max_depth=None, max_features=10, n_estimators=15, random_state=2)
rf.fit(X_train, y_train)

# Saving the model

joblib.dump(rf, 'modele_entreine.pkl')