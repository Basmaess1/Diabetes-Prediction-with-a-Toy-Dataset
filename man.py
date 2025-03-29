import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Créer un Toy Dataset
data = {
    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    'Glucose': [85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140],
    'Outcome': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0: Pas de diabète, 1: Diabète
}

# Convertir en numpy arrays
X = np.array([data['Age'], data['Glucose']]).T  # Variables indépendantes
y = np.array(data['Outcome'])  # Variable cible

# 2. Diviser les données en ensemble d'entraînement (80%) et de test (20%)
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        random.seed(random_state)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 3. Normalisation (Standardisation)
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

X_train = standardize(X_train)
X_test = standardize(X_test)

# 4. Implémentation de la régression logistique
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n + 1)  # Ajouter un biais
    X = np.c_[np.ones(m), X]  # Ajouter une colonne de 1 pour le biais
    
    for _ in range(epochs):
        z = np.dot(X, theta)
        predictions = sigmoid(z)
        gradient = np.dot(X.T, (predictions - y)) / m
        theta -= lr * gradient
    
    return theta

def predict(X, theta):
    X = np.c_[np.ones(X.shape[0]), X]  # Ajouter la colonne de biais
    return (sigmoid(np.dot(X, theta)) >= 0.5).astype(int)

# Entraînement du modèle
theta = train_logistic_regression(X_train, y_train, lr=0.1, epochs=1000)

# 5. Prédictions et évaluation
y_pred = predict(X_test, theta)

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

acc = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {acc * 100:.2f}%")

# 6. Matrice de confusion
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion :")
print(conf_matrix)

# 7. Visualisation des données
plt.figure(figsize=(8, 6))
for i, label in enumerate(y):
    plt.scatter(X[i, 0], X[i, 1], c='red' if label == 1 else 'blue', s=100, marker='o')
plt.xlabel('Age')
plt.ylabel('Glucose')
plt.title('Age vs Glucose avec Prédiction du Diabète')
plt.show()
