# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Créer un Toy Dataset
data = {
    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    'Glucose': [85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140],
    'Outcome': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0: Pas de diabète, 1: Diabète
}

# Convertir en DataFrame
df = pd.DataFrame(data)

# 2. Séparation des variables indépendantes (X) et de la cible (y)
X = df.drop('Outcome', axis=1)  # Variables indépendantes (Age et Glucose)
y = df['Outcome']  # Variable cible (Outcome)

# 3. Diviser les données en ensemble d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normaliser les données (Standardisation)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Créer et entraîner le modèle de régression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# 7. Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrécision du modèle : {accuracy * 100:.2f}%")

# 8. Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion :")
print(conf_matrix)

# 9. Affichage de la matrice de confusion sous forme graphique
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Pas de Diabète', 'Diabète'], yticklabels=['Pas de Diabète', 'Diabète'])
plt.title("Matrice de Confusion")
plt.xlabel('Prédiction')
plt.ylabel('Réel')
plt.show()

# 10. Visualisation des résultats
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Age', y='Glucose', hue='Outcome', palette='coolwarm', s=100, marker='o')
plt.title('Age vs Glucose avec Prédiction du Diabète')
plt.xlabel('Age')
plt.ylabel('Glucose')
plt.legend(title='Outcome')
plt.show()
