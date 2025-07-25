# Projet : NoSQL

**Nom :** MOUGNI Asma, CHETIOUI Khaoula, ACKRICHE Thinhinane  
**Binôme :** GROUPE 5  
**Classe :** 3IABD2  
**Date :** 01/07/2025

---

## ✨ Présentation du Projet Annuel

### 📘 C'est quoi SignAI ?
**SignAI** est un projet de reconnaissance de la langue des signes (l'alphabet signé plus précisément), dont le but est de classifier les images renseigné.

---

## Objectif du projet NoSQL
Le projet NoSQL va permettre de structurer la gestion des données de SignAI.
1. Importation des données
2. Chargement + Entrainement   
3. Démonstration


### 📦 Importation des données dans MongoDB
Le script suivant permet d’importer des images classées par lettre (A, B, C) dans MongoDB en utilisant GridFS :
```
import os
from pymongo import MongoClient
import gridfs
from datetime import datetime

# Connexion à MongoDB
client = MongoClient("mongodb://localhost:27017")

db = client["letters"]  # Création de la base "letters"
fs = gridfs.GridFS(db)  # GridFS pour stocker les images

metadata_col = db["metadata"]  # Collection pour les métadonnées

DATASET_PATH = "dataset"

# Compteur
total_uploaded = 0

# Parcours des classes (A, B, C)
for class_label in os.listdir(DATASET_PATH):
class_dir = os.path.join(DATASET_PATH, class_label)

    if not os.path.isdir(class_dir):
        continue

    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)

        # Lecture de l'image
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Stockage dans GridFS
        file_id = fs.put(
            image_data,
            filename=image_name,
            class_label=class_label,
            upload_date=datetime.utcnow()
        )

        # Métadonnées associées
        metadata = {
            "filename": image_name,
            "class": class_label,
            "file_id": file_id,
            "path": image_path,
            "upload_date": datetime.utcnow()
        }

        metadata_col.insert_one(metadata)
        total_uploaded += 1

print(f" {total_uploaded} images importées dans la base 'letters'.")
```


### 🤖 Chargement & Entraînement du modèle Rust
Le but est de récupérer les données depuis MongoDB pour entraîner le modèle Rust via une interface Python.

- Charger les images depuis GridFS
- Les convertir en matrices d’entrée normalisées (grayscale 28x28 → 784 float)
- Associer chaque image à sa classe
- Envoyer les données à la bibliothèque Rust via FFI (ctypes ou cffi)
- Lancer l'entraînement et stocker le modèle
```python
from pymongo import MongoClient
import gridfs
from PIL import Image
import io
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from ctypes import CDLL, POINTER, c_double, c_uint32, c_void_p, c_size_t
import time

client = MongoClient("mongodb://localhost:27017")
db = client["signai"]
fs = gridfs.GridFS(db)

labels_voulus = {"A", "B"}
X, y = [], []

for file in db.fs.files.find({"label": {"$in": list(labels_voulus)}}):
    label = file.get("label")
    if label not in labels_voulus:
        continue

    grid_out = fs.get(file["_id"])
    image_data = grid_out.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB").resize((64, 64))
    array = np.asarray(image).astype("float32") / 255.0
    X.append(array.flatten())
    y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  # A=0, B=1
print("Données chargées :", X.shape)
print("Répartition :", dict(zip(encoder.classes_, np.bincount(y_encoded))))

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0) + 1e-8
X = (X - X_mean) / X_std

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
y_train_u32 = y_train.astype(np.uint32)

lib = CDLL("./target/release/mymodel.dll")
lib.create_mlp_classifier.restype = c_void_p
lib.train_mlp_classifier.argtypes = [
    c_void_p, POINTER(c_double), POINTER(c_uint32), c_size_t, c_size_t
]
lib.predict_mlp_classifier.argtypes = [c_void_p, POINTER(c_double), c_size_t]
lib.predict_mlp_classifier.restype = c_uint32

n_samples, n_features = X_train.shape
n_classes = len(np.unique(y_encoded))
n_hidden = 32
learning_rate = 0.01
epochs = 50

# Préparation pointeurs 
X_ptr = X_train.ctypes.data_as(POINTER(c_double))
y_ptr = y_train_u32.ctypes.data_as(POINTER(c_uint32))
 
model = lib.create_mlp_classifier(
    c_size_t(n_features),
    c_size_t(n_hidden),
    c_size_t(n_classes),
    c_double(learning_rate),
    c_size_t(epochs)
)

print("Entrainement du MLP...")
start = time.time()
lib.train_mlp_classifier(model, X_ptr, y_ptr, c_size_t(n_samples), c_size_t(n_features))
print("Entraînement terminé en", round(time.time() - start, 2), "secondes")

y_pred = np.zeros(len(X_test), dtype=np.uint32)
for i in range(len(X_test)):
    xi_ptr = X_test[i].ctypes.data_as(POINTER(c_double))
    y_pred[i] = lib.predict_mlp_classifier(model, xi_ptr, c_size_t(n_features))

print("\nÉvaluation du MLP (binaire A/B) :")
print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))

```


### 🧰 Technologies utilisées
MongoDB	: Stockage des images (NoSQL + GridFS)  
Python	: Scripts d’import et de communication  
Rust	: Implémentation des modèles ML  
Docker	: Conteneurisation de MongoDB  
FastAPI	: API pour prédiction   
MongoDB Compass	: Visualisation de la base

# Nous vous remercions par avance pour l’attention portée à ce rapport.