from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import ctypes
import os
import platform
from pathlib import Path
from collections import Counter  
from fastapi import UploadFile, File
import shutil
from fastapi.responses import Response
import json
from datetime import datetime

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

RESULTS_DIR = "result_train"
os.makedirs(RESULTS_DIR, exist_ok=True)


DATASET_PATH = "../dataset"
IMAGE_SIZE = (64, 64)
SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Chargement de la bibliothèque Rust ===
def load_library():
    system = platform.system()
    if system == "Windows":
        libname = "../target/release/mymodel.dll"
    elif system == "Darwin":
        libname = "../target/release/libmymodel.dylib"
    else:
        libname = "../target/release/libmymodel.so"

    libpath = os.path.abspath(libname)
    if not os.path.exists(libpath):
        raise FileNotFoundError(f"Librairie Rust introuvable : {libpath}")

    return ctypes.cdll.LoadLibrary(libpath)

lib = load_library()


lib.create_multiclass.restype = ctypes.c_void_p
lib.train_multiclass.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_bool
]
lib.predict_multiclass.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint)
]

lib.create_mlp_classifier.restype = ctypes.c_void_p
lib.create_mlp_classifier.argtypes = [
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_size_t
]
lib.train_mlp_classifier.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_uint),
    ctypes.c_size_t,
    ctypes.c_size_t
]
lib.predict_mlp_classifier.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]
lib.predict_mlp_classifier.restype = ctypes.c_uint

lib.evaluate_mlp_classifier_mse.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_uint),
    ctypes.c_size_t,
    ctypes.c_size_t
]
lib.evaluate_mlp_classifier_mse.restype = ctypes.c_float

lib.save_mlp_classifier.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.load_mlp_classifier.argtypes = [ctypes.c_char_p]
lib.load_mlp_classifier.restype = ctypes.c_void_p

lib.get_mlp_loss_pointer.argtypes = [ctypes.c_void_p]
lib.get_mlp_loss_pointer.restype = ctypes.POINTER(ctypes.c_float)

lib.get_mlp_loss_length.argtypes = [ctypes.c_void_p]
lib.get_mlp_loss_length.restype = ctypes.c_size_t
lib.evaluate_mlp_classifier_mse.restype = ctypes.c_float

# === RBFN (multi-classe) ===
lib.create_rbfn_multiclass_model.argtypes = [
    ctypes.c_double,  # sigma
    ctypes.c_double,  # learning_rate
    ctypes.c_size_t,  # epochs
    ctypes.c_size_t,  # n_classes
]
lib.create_rbfn_multiclass_model.restype = ctypes.c_void_p

lib.train_rbfn_model_auto.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]

lib.predict_rbfn_model.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
]
lib.predict_rbfn_model.restype = ctypes.c_double

lib.get_rbfn_loss_pointer.argtypes = [ctypes.c_void_p]
lib.get_rbfn_loss_pointer.restype = ctypes.POINTER(ctypes.c_float)

lib.get_rbfn_loss_length.argtypes = [ctypes.c_void_p]
lib.get_rbfn_loss_length.restype = ctypes.c_size_t


lib.create_svm_rbf_multiclass.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_size_t
]
lib.create_svm_rbf_multiclass.restype = ctypes.c_void_p

lib.train_svm_rbf_multiclass.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.c_size_t
]

lib.predict_svm_rbf_multiclass.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]
lib.predict_svm_rbf_multiclass.restype = ctypes.c_size_t

lib.destroy_svm_rbf_multiclass.argtypes = [ctypes.c_void_p]

lib.get_svm_loss_pointer.restype = ctypes.POINTER(ctypes.c_double)
lib.get_svm_loss_pointer.argtypes = [ctypes.c_void_p]

lib.get_svm_loss_length.restype = ctypes.c_size_t
lib.get_svm_loss_length.argtypes = [ctypes.c_void_p]


class CreateInput(BaseModel):
    n_features: int
    learning_rate: float
    max_epochs: int
    activation_type: int
    n_classes: int

#class DatasetRequest(BaseModel):
   # labels: List[str] = []
class DatasetRequest(BaseModel):
    labels: List[str] = []
    learning_rate: float = 0.01
    epochs: int = 50
    activation_type: int = 2
    sigma: float = 1.0
    n_hidden: int = 8

class PredictInput(BaseModel):
    x: List[List[float]]

class MLPCreateInput(BaseModel):
    n_inputs: int
    n_hidden: int
    learning_rate: float
    epochs: int
    is_regression: bool = False
    use_activation: bool = True

class MLPClassifierCreateInput(BaseModel):
    n_inputs: int
    n_hidden: int
    n_classes: int
    learning_rate: float
    epochs: int

class MLPInput(BaseModel):
    x: List[float]
class MLPClassifierCreateInput(BaseModel):
    n_inputs: int
    n_hidden: int
    n_classes: int
    learning_rate: float
    epochs: int

class SVMTrainRequest(BaseModel):
    labels: List[str]
    gamma: float
    c: float
    learning_rate: float 
    epochs: int
class SVMCreateRequest(BaseModel):
    gamma: float
    c: float
    learning_rate: float
    epochs: int


model_ptr = None
mlp_clf_ptr = None
rbfn_ptr = None
svm_ptr = None

label_to_index = {}


def load_full_dataset(base_path=DATASET_PATH, image_size=IMAGE_SIZE, filter_labels: List[str] = None):
    global label_to_index
    label_to_index = {}

    def load_split(split):
        X, y = [], []
        split_path = Path(base_path) / split

        for label_dir in sorted(split_path.iterdir(), key=lambda x: x.name.lower()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            if filter_labels and label not in filter_labels:
                continue

            label_index = label_to_index.setdefault(label, len(label_to_index))

            for img_path in label_dir.glob("*"):
                if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                try:
                    img = Image.open(img_path).convert("L")
                    img = img.resize(image_size)
                    arr = np.array(img).astype(np.float32) / 255.0
                    X.append(arr.flatten())
                    y.append(label_index)
                except Exception as e:
                    print(f"Erreur avec {img_path}: {e}")

        print(f"🔍 Split '{split}' - {len(X)} images chargées.")
        return X, y

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    print("🎯 Label mapping utilisé :", label_to_index)
    print("📊 Répartition des y_train :", Counter(y_train))
    return X_train, y_train, X_test, y_test, label_to_index

@app.post("/linear/create/")
def create_linear_model(params: CreateInput):
    global model_ptr
    model_ptr = lib.create_multiclass(
        params.n_classes,
        params.n_features,
        ctypes.c_float(params.learning_rate),
        params.max_epochs,
        params.activation_type
    )
    if not model_ptr:
        raise HTTPException(status_code=500, detail="Erreur création modèle multiclasses.")
    print("Modèle multiclasses créé")
    return {"status": "created"}
@app.post("/mlp/create/")
def create_mlp_classifier(params: MLPClassifierCreateInput):
    global mlp_clf_ptr
    mlp_clf_ptr = lib.create_mlp_classifier(
        params.n_inputs,
        params.n_hidden,
        params.n_classes,
        ctypes.c_double(params.learning_rate),
        params.epochs
    )
    if not mlp_clf_ptr:
        raise HTTPException(status_code=500, detail="Erreur création MLPClassifier.")
    return {"status": "MLPClassifier created"}

@app.post("/linear/train/")
def train_multiclass_model_from_dataset(req: DatasetRequest):
    global model_ptr, label_to_index
    if not model_ptr:
        raise HTTPException(status_code=400, detail="Modèle non initialisé.")

    X_train, y_train, _, _, label_map = load_full_dataset(filter_labels=req.labels)
    if not X_train:
        raise HTTPException(status_code=400, detail="Pas de données d'entraînement.")

    x = np.array(X_train, dtype=np.float32)
    y = np.array(y_train, dtype=np.uint32)

    print("Dimensions X_train:", x.shape)
    print("Dimensions y_train:", y.shape)

    lib.train_multiclass(
        model_ptr,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        x.shape[0],
        x.shape[1],
        True
    )

    label_to_index = label_map
    return {"status": "trained", "labels": label_map}

@app.post("/linear/predict/")
def predict_multiclass(data: PredictInput):
    print("📥 Requête reçue :", data)
    global model_ptr, label_to_index
    if not model_ptr:
        raise HTTPException(status_code=400, detail="Modèle non initialisé.")

    x = np.array(data.x, dtype=np.float32)

    if x.ndim != 2:
        raise HTTPException(status_code=400, detail="Format attendu: liste de listes.")
    
    n_samples, n_features = x.shape
    if n_features != 64 * 64:
        raise HTTPException(
            status_code=400,
            detail=f"Chaque échantillon doit avoir {64*64} features (actuellement {n_features})"
        )

    out = np.zeros(n_samples, dtype=np.uint32)

    lib.predict_multiclass(
        model_ptr,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_samples,
        n_features,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
    )

    index_to_label = {v: k for k, v in label_to_index.items()}
    label_preds = [index_to_label.get(p, "?") for p in out]

    return {"predictions": label_preds}

@app.post("/mlp/train/")
def train_mlp_classifier(req: DatasetRequest):
    global mlp_clf_ptr, label_to_index
    if not mlp_clf_ptr:
        raise HTTPException(status_code=400, detail="Modèle MLPClassifier non initialisé.")

    X_train, y_train, _, _, label_map = load_full_dataset(filter_labels=req.labels)

    if not X_train:
        raise HTTPException(status_code=400, detail="Dataset vide ou invalide.")

    print("Entrainement MLPClassifier")
    print("y_train:", Counter(y_train))
    print("label_to_index:", label_map)

    x = np.array(X_train, dtype=np.float64)
    y = np.array(y_train, dtype=np.uint32)

    lib.train_mlp_classifier(
        mlp_clf_ptr,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        x.shape[0],
        x.shape[1]
    )

    # Obtenir la perte globale
    loss = lib.evaluate_mlp_classifier_mse(
        mlp_clf_ptr,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        x.shape[0],
        x.shape[1]
    )

    # Obtenir les pertes par époque depuis Rust
    length = lib.get_mlp_loss_length(mlp_clf_ptr)
    loss_ptr = lib.get_mlp_loss_pointer(mlp_clf_ptr)
    loss_array = np.ctypeslib.as_array(loss_ptr, shape=(length,))
    loss_per_epoch = loss_array.tolist()

    label_to_index = label_map
    run_id = f"mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_path = os.path.join(RESULTS_DIR, run_id)

    log_data = {
        "model": "mlp",
        "loss_per_epoch": loss_per_epoch,
        "mse_loss": float(loss),
        "labels": label_map,
        "config": {
            "learning_rate": req.learning_rate,
            "epochs": req.epochs,
            "n_hidden": req.n_hidden,
            "labels_used": req.labels,
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(save_path, "w") as f:
        json.dump(log_data, f, indent=2)


    return {
        "status": "trained",
        "labels": label_map,
        "mse_loss": float(loss),
        "loss_per_epoch": loss_per_epoch
    }


@app.post("/mlp/predict/")
def predict_mlp_classifier(input: MLPInput):
    global mlp_clf_ptr, label_to_index
    if not mlp_clf_ptr:
        raise HTTPException(status_code=400, detail="Modèle non initialisé.")

    x_np = np.array(input.x, dtype=np.float64)
    pred_index = lib.predict_mlp_classifier(
        mlp_clf_ptr,
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(input.x)
    )
    print(f"👉 Index brut prédit : {pred_index}")


    index_to_label = {v: k for k, v in label_to_index.items()}
    label = index_to_label.get(pred_index, f"classe {pred_index}")
    return {"prediction_index": pred_index, "label": label}

@app.post("/model/save/")
def save_current_model(name: str):
    global mlp_clf_ptr
    if not mlp_clf_ptr:
        raise HTTPException(status_code=400, detail="Aucun modèle à sauvegarder.")

    path = os.path.join(MODEL_DIR, f"{name}.bin")
    lib.save_mlp_classifier(mlp_clf_ptr, path.encode("utf-8"))

    return {"status": "saved", "filename": f"{name}.bin"}

@app.get("/dataset/stats/")
def get_dataset_stats():
    base_path = Path(DATASET_PATH) / "train"
    stats = {}

    for label_dir in sorted(base_path.iterdir(), key=lambda x: x.name.lower()):
        if not label_dir.is_dir():
            continue
        count = len([f for f in label_dir.glob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS])
        stats[label_dir.name] = count

    return {"train_distribution": stats}
@app.get("/test-image")
def send_test_image():
    from fastapi.responses import Response 

    test_path = Path(DATASET_PATH) / "train" / "A"
    images = list(test_path.glob("*.png")) + list(test_path.glob("*.jpg")) + list(test_path.glob("*.jpeg"))

    if not images:
        raise HTTPException(status_code=404, detail="Pas d'image de test trouvée dans train/A.")

    with open(images[0], "rb") as f:
        return Response(content=f.read(), media_type="image/png")


@app.post("/rbfn_multiclass/create/")
def create_rbf_model(sigma: float = 1.0, learning_rate: float = 0.01, epochs: int = 50, n_classes: int = 3):
    global rbfn_ptr
    rbfn_ptr = lib.create_rbfn_multiclass_model(sigma, learning_rate, epochs, n_classes)
    if not rbfn_ptr:
        raise HTTPException(status_code=500, detail="Erreur création RBF.")
    return {"status": "RBF model created"}
@app.post("/rbf/train/")
def train_rbf_model(req: DatasetRequest):
    global rbfn_ptr, label_to_index
    if not rbfn_ptr:
        raise HTTPException(status_code=400, detail="Modèle RBF non initialisé.")

    X_train, y_train, _, _, label_map = load_full_dataset(filter_labels=req.labels)
    if not X_train:
        raise HTTPException(status_code=400, detail="Dataset vide ou invalide.")

    n_classes = len(label_map)
    y_onehot = np.zeros((len(y_train), n_classes), dtype=np.float64)
    for i, yi in enumerate(y_train):
        y_onehot[i, yi] = 1.0

    x = np.array(X_train, dtype=np.float64)
    y = y_onehot

    lib.train_rbfn_model_auto(
        rbfn_ptr,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        x.shape[0],
        x.shape[1],
        y.shape[1]
    )

    length = lib.get_rbfn_loss_length(rbfn_ptr)
    loss_ptr = lib.get_rbfn_loss_pointer(rbfn_ptr)
    loss_array = np.ctypeslib.as_array(loss_ptr, shape=(length,))
    loss_per_epoch = loss_array.tolist()

    label_to_index = label_map

    run_id = f"rbf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_path = os.path.join(RESULTS_DIR, run_id)

    log_data = {
        "model": "rbf",
        "loss_per_epoch": loss_per_epoch,
        "labels": label_map,
        "config": {
            "sigma": sigma,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "labels_used": req.labels,
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(save_path, "w") as f:
        json.dump(log_data, f, indent=2)


    return {
        "status": "trained",
        "labels": label_map,
        "loss_per_epoch": loss_per_epoch
    }


@app.post("/rbfn/predict/")
def predict_rbf_model(input: MLPInput): 
    global rbfn_ptr, label_to_index
    if not rbfn_ptr:
        raise HTTPException(status_code=400, detail="Modèle RBF non initialisé.")

    x_np = np.array(input.x, dtype=np.float64)
    pred_index = int(lib.predict_rbfn_model(
        rbfn_ptr,
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(input.x)
    ))

    index_to_label = {v: k for k, v in label_to_index.items()}
    label = index_to_label.get(pred_index, f"classe {pred_index}")
    return {"prediction_index": pred_index, "label": label}

@app.post("/svm/create/")
def create_svm_model(params: SVMCreateRequest):
    global svm_ptr
    svm_ptr = lib.create_svm_rbf_multiclass(
        params.gamma, 
        params.c, 
        params.learning_rate, 
        params.epochs
    )
    if not svm_ptr:
        raise HTTPException(status_code=500, detail="Erreur création SVM.")
    return {"status": "SVM créé"}
@app.post("/svm/train/")
def train_svm_model(req: SVMTrainRequest):
    global svm_ptr, label_to_index
    if not svm_ptr:
        raise HTTPException(status_code=400, detail="Modèle SVM non initialisé.")

    # Charger le dataset filtré
    X_train, y_train, _, _, label_map = load_full_dataset(filter_labels=req.labels)
    if not X_train:
        raise HTTPException(status_code=400, detail="Dataset vide ou invalide.")

    x = np.array(X_train, dtype=np.float64)
    y = np.array(y_train, dtype=np.uintp)  # ctypes.c_size_t compatible

    print("⏳ Début appel Rust SVM...")
    start = datetime.now()

    lib.train_svm_rbf_multiclass(
        svm_ptr,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        x.shape[0],
        x.shape[1]
    )

    print("Fin appel Rust SVM", datetime.now() - start)

    loss_length = lib.get_svm_loss_length(svm_ptr)
    loss_ptr = lib.get_svm_loss_pointer(svm_ptr)
    loss_array = np.ctypeslib.as_array(loss_ptr, shape=(loss_length,))
    loss_per_epoch = loss_array.tolist()

    label_to_index = label_map

    run_id = f"svm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_path = os.path.join(RESULTS_DIR, run_id)

    log_data = {
        "model": "svm",
        "loss_per_epoch": loss_per_epoch,
        "labels": label_map,
        "config": {
            "gamma": req.gamma,
            "c": req.c,
            "learning_rate": req.learning_rate,
            "epochs": req.epochs,
            "labels_used": req.labels,
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(save_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return {
        "status": "trained",
        "labels": label_map,
        "loss_per_epoch": loss_per_epoch
    }


@app.post("/svm/predict/")
def predict_svm_model(input: MLPInput):
    global svm_ptr, label_to_index
    if not svm_ptr:
        raise HTTPException(status_code=400, detail="Modèle SVM non initialisé.")

    x_np = np.array(input.x, dtype=np.float64)
    pred_index = int(lib.predict_svm_rbf_multiclass(
        svm_ptr,
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(input.x)
    ))

    index_to_label = {v: k for k, v in label_to_index.items()}
    label = index_to_label.get(pred_index, f"classe {pred_index}")
    return {"prediction_index": pred_index, "label": label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
