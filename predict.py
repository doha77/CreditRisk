
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score
import joblib
from sklearn.model_selection import train_test_split


model = joblib.load('modele_entreine.pkl')


data = pd.read_csv("./Data/credit_data.csv", index_col=0)
df = pd.DataFrame(data)


df['Credit amount'] = np.log(df['Credit amount'])
X = df.drop('Risk_bad', 1).values
y = df['Risk_bad'].values

# Séparation des données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Prédiction sur les données de test
y_pred = model.predict(X_test)

# Evaluation des résultats
print(accuracy_score(y_test, y_pred))
print("\n")
print(confusion_matrix(y_test, y_pred))
print("\n")
print(fbeta_score(y_test, y_pred, beta=2))
