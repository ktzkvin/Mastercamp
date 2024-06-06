#%% import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%

# Charger un échantillon des données pour travailler sur une taille plus gérable
df = 'data/valeursfoncieres-2023.txt'


# Lire un échantillon des données pour travailler sur une taille plus gérable
chunk_size = 10000  # Nombre de lignes à lire par chunk
data_chunks = pd.read_csv(df, sep='|', chunksize=chunk_size, decimal=',')

# Combiner quelques morceaux pour former un dataset plus petit
df = pd.concat([chunk for chunk in data_chunks])

# Afficher les premières lignes pour vérification
print(df.head())

# Suppression des colonnes non pertinentes
columns_to_keep = [
    'No disposition', 'Date mutation', 'Nature mutation', 'Valeur fonciere',
    'No voie', 'B/T/Q', 'Type de voie', 'Code voie', 'Voie',
    'Code postal', 'Commune', 'Code departement', 'Code commune',
    'Section', 'No plan', 'Type local', 'Surface reelle bati', 'Surface terrain'
]
df = df[columns_to_keep]

# Renommer les colonnes pour un accès plus facile
df.columns = [
    'numero_disposition', 'date_mutation', 'nature_mutation', 'valeur_fonciere',
    'numero_voie', 'btq', 'type_voie', 'code_voie', 'voie',
    'code_postal', 'commune', 'code_departement', 'code_commune',
    'section', 'numero_plan', 'type_local', 'surface_reelle_bati', 'surface_terrain'
]

# Assurez-vous que toutes les valeurs de 'valeur_fonciere' sont des chaînes de caractères
df['valeur_fonciere'] = df['valeur_fonciere'].astype(str)

# Remplacer les virgules et convertir en numérique
df['valeur_fonciere'] = pd.to_numeric(df['valeur_fonciere'].str.replace(',', ''), errors='coerce')

# Convertir les autres colonnes en types numériques si nécessaire
df['surface_reelle_bati'] = pd.to_numeric(df['surface_reelle_bati'], errors='coerce')
df['surface_terrain'] = pd.to_numeric(df['surface_terrain'], errors='coerce')

# Suppression des valeurs manquantes
df.dropna(inplace=True)

# Filtrage des valeurs incorrectes
df = df[df['valeur_fonciere'] > 0]
df = df[df['surface_reelle_bati'] > 0]

# Afficher les premières lignes après nettoyage
print(df.head())



#%%
import matplotlib.pyplot as plt

# Statistiques descriptives
print(df.describe())

# Histogramme des valeurs foncières
plt.figure(figsize=(10, 5))
plt.hist(df['valeur_fonciere'], bins=50, alpha=0.5, label='Valeur fonciere')
plt.title('Distribution des Valeurs Foncières')
plt.xlabel('Valeur fonciere (€)')
plt.ylabel('Fréquence')
plt.show()

# Histogramme des surfaces réelles bâties
plt.figure(figsize=(10, 5))
plt.hist(df['surface_reelle_bati'], bins=50, alpha=0.5, label='Surface reelle bati')
plt.title('Distribution des Surfaces Réelles Bâties')
plt.xlabel('Surface reelle bati (m²)')
plt.ylabel('Fréquence')
plt.show()

# Diagramme de dispersion: prix vs surface
plt.figure(figsize=(10, 5))
plt.scatter(df['surface_reelle_bati'], df['valeur_fonciere'], alpha=0.5)
plt.title('Prix vs Surface')
plt.xlabel('Surface reelle bati (m²)')
plt.ylabel('Valeur fonciere (€)')
plt.show()

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Séparation des données
X = df[['surface_reelle_bati', 'prix_m2']]
y = df['valeur_fonciere']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation du modèle
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R²:', r2_score(y_test, y_pred))

# Affichage des coefficients du modèle
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)


