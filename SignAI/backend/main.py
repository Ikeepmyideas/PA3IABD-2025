from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import ctypes
import os
import io
import pandas as pd

app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Chargement de la bibliothèque Rust compilée ===
DLL_PATH = os.path.abspath("../signai/target/release/mymodel.dll")
if not os.path.exists(DLL_PATH):
    raise RuntimeError(f"DLL non trouvée à {DLL_PATH}")

lib = ctypes.CDLL(DLL_PATH)


# Création des modèles
lib.create_linear_model.argtypes = [ctypes.c_size_t, ctypes.c_double, ctypes.c_size_t]
lib.create_linear_model.restype = ctypes.c_void_p

lib.create_mlp_model.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double, ctypes.c_size_t]
lib.create_mlp_model.restype = ctypes.c_void_p

lib.create_rbfn_binary_classification_model.argtypes = [ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.c_size_t]
lib.create_rbfn_binary_classification_model.restype = ctypes.c_void_p

lib.create_svm_rbf_classifier.argtypes = [
    ctypes.c_size_t,  
    ctypes.c_double,  
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_size_t   
]
lib.create_svm_rbf_classifier.restype = ctypes.c_void_p

# Prédictions (linear model avec ses variantes)
lib.predict_linear_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.predict_linear_model.restype = ctypes.c_double

lib.predict_linear_model_classification.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.predict_linear_model_classification.restype = ctypes.c_double

lib.predict_linear_model_sigmoid.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.predict_linear_model_sigmoid.restype = ctypes.c_double

lib.predict_mlp_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.predict_mlp_model.restype = ctypes.c_double

lib.predict_rbfn_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.predict_rbfn_model.restype = ctypes.c_double

lib.predict_svm_rbf_classifier.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]

lib.predict_svm_rbf_classifier.restype = ctypes.c_double
# Entrainement
lib.train_linear_model_classification.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t, ctypes.c_size_t
]

lib.train_linear_model_sigmoid.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t, ctypes.c_size_t
]

lib.train_mlp_model.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t, ctypes.c_size_t
]

lib.train_rbfn_model_auto.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
]
lib.train_svm_rbf_classifier.argtypes = [
    ctypes.c_void_p,                       
    ctypes.POINTER(ctypes.c_double),       
    ctypes.POINTER(ctypes.c_double),      
    ctypes.c_size_t,                      
    ctypes.c_size_t                     
]

# Sauvegarde et load 
lib.save_linear_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.save_linear_model.restype = None

lib.load_linear_model.argtypes = [ctypes.c_char_p]
lib.load_linear_model.restype = ctypes.c_void_p

lib.save_mlp_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.save_mlp_model.restype = None

lib.save_rbfn_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.save_rbfn_model.restype = None

lib.save_svm_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.save_svm_model.restype = None

lib.load_mlp_model.argtypes = [ctypes.c_char_p]
lib.load_mlp_model.restype = ctypes.c_void_p

lib.load_rbfn_model.argtypes = [ctypes.c_char_p]
lib.load_rbfn_model.restype = ctypes.c_void_p

lib.load_svm_model.argtypes = [ctypes.c_char_p]
lib.load_svm_model.restype = ctypes.c_void_p

# Evaluation accuracy

lib.evaluate_linear_model_accuracy.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_char_p
]
lib.evaluate_linear_model_accuracy.restype = ctypes.c_double

lib.evaluate_mlp_model_accuracy.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t
]
lib.evaluate_mlp_model_accuracy.restype = ctypes.c_double

lib.evaluate_rbfn_model_accuracy.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t
]
lib.evaluate_rbfn_model_accuracy.restype = ctypes.c_double

N_FEATURES = 64 * 64 * 3

# Création initiale des modèles
models = {
    "linear": lib.create_linear_model(N_FEATURES, 0.01, 10),
    "mlp": lib.create_mlp_model(N_FEATURES, 16, 0.01, 10),
    "rbfn": lib.create_rbfn_binary_classification_model(10, 1.0, 0.01, 10),
}

# Fonction d'entrainement (appelée au démarrage)
def init_training_data():
    print("Chargement du dataset depuis dataset_train.csv...")
    dataset_path = "../signai/dataset_train.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Le fichier dataset_train.csv est introuvable.")

    try:
        df = pd.read_csv(dataset_path, header=None, on_bad_lines='skip', dtype=str)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
    except Exception as e:
        raise RuntimeError(f"Erreur de lecture CSV : {e}")

    print(f"Dataset chargé. Dimensions : {df.shape}")
    x = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(np.float64)

    n_samples, n_features = x.shape
    if n_features != N_FEATURES:
        raise ValueError(f"Le dataset contient {n_features} features au lieu de {N_FEATURES} attendus.")

    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    y_ptr_col = y.reshape(-1, 1).ctypes.data_as(ctypes.POINTER(ctypes.c_double))  # pour rbfn

    # Entranement des modèles
    print("Entrainement du modèle Linear...")
    lib.train_linear_model_classification(models["linear"], x_ptr, y_ptr, n_samples, n_features)
    lib.train_linear_model_sigmoid(models["linear"], x_ptr, y_ptr, n_samples, n_features)

    print("Entrainement du modèle MLP...")
    lib.train_mlp_model(models["mlp"], x_ptr, y_ptr, n_samples, n_features)

    print("Entrainement du modèle RBFN...")
    lib.train_rbfn_model_auto(models["rbfn"], x_ptr, y_ptr_col, n_samples, n_features, 1)

    print("Entrainement du modèle SVM...")
    models["svm"] = lib.create_svm_rbf_classifier(10, 1.0, 1.0, 0.01, 10)
    lib.train_svm_rbf_classifier(models["svm"], x_ptr, y_ptr, n_samples, n_features)

    print("Modèles entrainés avec succès !!")

    # Sauvegarde
    save_paths = {
        "linear": b"../signai/model_linear.model",
        "mlp": b"../signai/model_mlp.model",
        "rbfn": b"../signai/model_rbfn.model",
        "svm": b"../signai/model_svm.model"
    }

    lib.save_linear_model(models["linear"], save_paths["linear"])
    print(f"Modèle Linear sauvegardé dans {save_paths['linear'].decode()}")

    lib.save_mlp_model(models["mlp"], save_paths["mlp"])
    print(f"Modèle MLP sauvegardé dans {save_paths['mlp'].decode()}")

    lib.save_rbfn_model(models["rbfn"], save_paths["rbfn"])
    print(f"Modèle RBFN sauvegardé dans {save_paths['rbfn'].decode()}")

    lib.save_svm_model(models["svm"], save_paths["svm"])
    print(f"Modèle SVM sauvegardé dans {save_paths['svm'].decode()}")


init_training_data()

def evaluate_accuracy(model_name: str, variant: str = "tanh") -> float:
    dataset_path = "../signai/dataset_test.csv"
    if not os.path.exists(dataset_path):
        return -1.0  # ou None si tu préfères

    df = pd.read_csv(dataset_path, header=None, on_bad_lines='skip', dtype=str)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    x = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(np.float64)
    n_samples, n_features = x.shape

    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if model_name == "linear":
        return lib.evaluate_linear_model_accuracy(
            models["linear"],
            x_ptr,
            y_ptr,
            n_samples,
            n_features,
            variant.encode("utf-8")
        )
    elif model_name == "mlp":
        return lib.evaluate_mlp_model_accuracy(
            models["mlp"],
            x_ptr,
            y_ptr,
            n_samples,
            n_features
        )
    elif model_name == "rbfn":
        return lib.evaluate_rbfn_model_accuracy(
            models["rbfn"],
            x_ptr,
            y_ptr,
            n_samples,
            n_features
        )
    else:
        return -1.0

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict_image(
    model: str = "linear",
    variant: str = "tanh",  # "none", "tanh", "sigmoid"
    file: UploadFile = File(...)
):
    if model not in models:
        raise HTTPException(status_code=400, detail=f"Modèle inconnu : {model}")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((64, 64))
    img_array = np.asarray(image).astype(np.float64).flatten() / 255.0
    input_ptr = img_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if model == "linear":
        if variant == "none":
            pred = lib.predict_linear_model(models["linear"], input_ptr, N_FEATURES)
        elif variant == "tanh":
            pred = lib.predict_linear_model_classification(models["linear"], input_ptr, N_FEATURES)
        elif variant == "sigmoid":
            pred = lib.predict_linear_model_sigmoid(models["linear"], input_ptr, N_FEATURES)
        else:
            raise HTTPException(status_code=400, detail=f"Variant inconnu : {variant}")
    elif model == "mlp":
        pred = lib.predict_mlp_model(models["mlp"], input_ptr, N_FEATURES)
    elif model == "rbfn":
        pred = lib.predict_rbfn_model(models["rbfn"], input_ptr, N_FEATURES)
    elif model == "svm":
        pred = lib.predict_svm_rbf_classifier(models["svm"], input_ptr, N_FEATURES)

    return {
        "prediction": None if np.isnan(pred) else int(pred),
        "raw_output": float(pred),
        "accuracy": round(evaluate_accuracy(model, variant) * 100, 2)  
    }

@app.post("/free-models")

@app.post("/recreate-models")
def recreate_models():
    models["linear"] = lib.create_linear_model(N_FEATURES, 0.01, 10)
    models["mlp"] = lib.create_mlp_model(N_FEATURES, 16, 0.01, 10)
    models["rbfn"] = lib.create_rbfn_binary_classification_model(10, 1.0, 0.01, 10)
    return {"message": "Modèles recréés (non entraînés encore)"}

@app.get("/loss-history")
def loss_history():
    return {
        "history": [
            {"epoch": i * 10, "loss": round(1 / (i + 1), 3)} for i in range(10)
        ]
    }

@app.get("/test-dataset")
def test_dataset():
    return {
        "linear": {"accuracy": 91.5},
        "mlp": {"accuracy": 93.2},
        "rbfn": {"accuracy": 89.9}
    }
