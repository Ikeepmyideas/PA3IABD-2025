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
lib.train_mlp_classifier(model, X_ptr, y_ptr, c_size_t(n_samples), c_size_t(n_features))
print("Entrainement terminé en")

y_pred = np.zeros(len(X_test), dtype=np.uint32)
for i in range(len(X_test)):
    xi_ptr = X_test[i].ctypes.data_as(POINTER(c_double))
    y_pred[i] = lib.predict_mlp_classifier(model, xi_ptr, c_size_t(n_features))

print("\nÉvaluation du MLP (binaire A/B) :")
print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))
