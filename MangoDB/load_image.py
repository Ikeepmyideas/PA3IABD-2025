from pymongo import MongoClient
import gridfs
from PIL import Image
import io
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Connexion MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["signai"]
fs = gridfs.GridFS(db)

# Labels à charger
labels_voulus = {"A", "B", "C"}

X = []
y = []

# Lire les images stockées avec label in (A, B, C)
for file in db.fs.files.find({"label": {"$in": list(labels_voulus)}}):
    grid_out = fs.get(file["_id"])
    image_data = grid_out.read()

    # Ouvrir image, redimensionner à 64x64, convertir RGB
    image = Image.open(io.BytesIO(image_data)).convert("RGB").resize((64, 64))
    array = np.asarray(image).astype("float32") / 255.0  # Normalisé
    vector = array.flatten()  # (64 * 64 * 3 = 12288 valeurs)

    X.append(vector)
    y.append(file["label"])

X = np.array(X)
y = np.array(y)

# Encoder les labels (A → 0, B → 1, C → 2)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("X shape :", X.shape)
print("y_encoded :", y_encoded[:10])


# Séparer données pour test rapide
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Entraîner un modèle simple
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Évaluer
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["A", "B", "C"]))

