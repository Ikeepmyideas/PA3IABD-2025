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
from pydantic import BaseModel, Extra
from typing import List
from sklearn.preprocessing import StandardScaler

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

RESULTS_DIR = "result_train"
os.makedirs(RESULTS_DIR, exist_ok=True)


DATASET_PATH = "../dataset"
IMAGE_SIZE = (32, 32)
SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


lib.save_mlp_classifier.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.load_mlp_classifier.argtypes = [ctypes.c_char_p]
lib.load_mlp_classifier.restype = ctypes.c_void_p



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
lib.train_rbfn_model_auto.restype = None

lib.predict_rbfn_model.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
]
lib.predict_rbfn_model.restype = ctypes.c_double


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

lib.create_softmax_model.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double, ctypes.c_size_t, ctypes.c_double]
lib.create_softmax_model.restype = ctypes.c_void_p
lib.train_softmax_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_size_t]
lib.predict_softmax_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.predict_softmax_model.restype = ctypes.c_size_t

lib.save_rbfn_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.save_rbfn_model.restype = None

lib.load_rbfn_model.argtypes = [ctypes.c_char_p]
lib.load_rbfn_model.restype = ctypes.c_void_p

lib.get_deep_mlp_train_losses_ptr.argtypes = [ctypes.c_void_p]
lib.get_deep_mlp_train_losses_ptr.restype = ctypes.POINTER(ctypes.c_double)

lib.get_deep_mlp_train_losses_len.argtypes = [ctypes.c_void_p]
lib.get_deep_mlp_train_losses_len.restype = ctypes.c_size_t

lib.get_deep_mlp_test_losses_ptr.argtypes = [ctypes.c_void_p]
lib.get_deep_mlp_test_losses_ptr.restype = ctypes.POINTER(ctypes.c_double)

lib.get_deep_mlp_test_losses_len.argtypes = [ctypes.c_void_p]
lib.get_deep_mlp_test_losses_len.restype = ctypes.c_size_t

lib.get_deep_mlp_train_accuracies_ptr.argtypes = [ctypes.c_void_p]
lib.get_deep_mlp_train_accuracies_ptr.restype = ctypes.POINTER(ctypes.c_double)

lib.get_deep_mlp_train_accuracies_len.argtypes = [ctypes.c_void_p]
lib.get_deep_mlp_train_accuracies_len.restype = ctypes.c_size_t

lib.get_deep_mlp_test_accuracies_ptr.argtypes = [ctypes.c_void_p]
lib.get_deep_mlp_test_accuracies_ptr.restype = ctypes.POINTER(ctypes.c_double)

lib.get_deep_mlp_test_accuracies_len.argtypes = [ctypes.c_void_p]
lib.get_deep_mlp_test_accuracies_len.restype = ctypes.c_size_t


# === Deep MLP ===
lib.create_deep_mlp_classifier.argtypes = [
    ctypes.c_size_t,  # n_inputs
    ctypes.c_size_t,  # hidden_units
    ctypes.c_size_t,  # n_classes
    ctypes.c_double,  # learning_rate
    ctypes.c_size_t,  # epochs
    ctypes.c_size_t,  # activation_id (0=ReLU, 1=Tanh)
    ctypes.c_size_t,  # batch_size
    ctypes.c_double,  # lambda
    ctypes.c_size_t,  # nb_hidden_layers
]
lib.create_deep_mlp_classifier.restype = ctypes.c_void_p

lib.train_deep_mlp_classifier.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.c_size_t
]
lib.train_deep_mlp_classifier.restype = None

lib.predict_deep_mlp_classifier.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]
lib.predict_deep_mlp_classifier.restype = ctypes.c_size_t

class SoftmaxCreateInput(BaseModel):
    n_features: int
    n_classes: int
    learning_rate: float
    epochs: int
    l2_lambda: float


class CreateInput(BaseModel):
    n_features: int
    n_classes: int
    learning_rate: float
    max_epochs: int
    l2_lambda: float = 0.0
#class DatasetRequest(BaseModel):
   # labels: List[str] = []

class DatasetRequest(BaseModel):
    labels: List[str]
    learning_rate: float 
    epochs: int 
    activation_type: int 
    sigma: float
    n_hidden: int

    class Config:
        extra = Extra.allow  

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
class MLPDeepCreateInput(BaseModel):
    n_features: int
    hidden_units: int
    n_layers: int
    n_classes: int
    learning_rate: float
    epochs: int
    batch_size: int 
    lambda_: float 
    activation_type: int  # 0: ReLU, 1: Tanh
class MLPDeepTrainInput(BaseModel):
    n_classes: int
    hidden_units: int
    n_layers: int
    activation_type: int
    learning_rate: float
    epochs: int
    batch_size: int
    lambda_: float

model_ptr = None
mlp_clf_ptr = None
rbfn_ptr = None
svm_ptr = None
softmax_ptr = None
mlp_deep_ptr = None

label_to_index = {}
scaler = None


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

        return X, y

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    return X_train, y_train, X_test, y_test, label_to_index

@app.post("/linear/create/")
def create_softmax_model_api(params: CreateInput):
    global softmax_ptr
    softmax_ptr = lib.create_softmax_model(
        params.n_features,
        params.n_classes,
        ctypes.c_double(params.learning_rate),
        params.max_epochs,
        ctypes.c_double(params.l2_lambda)
    )
    if not softmax_ptr:
        raise HTTPException(status_code=500, detail="Erreur création softmax.")
    return {"status": "created"}

@app.post("/linear/train/")
def train_softmax_model_api(req: DatasetRequest):
    global softmax_ptr, label_to_index, scaler
    if not softmax_ptr:
        raise HTTPException(status_code=400, detail="Modèle non initialisé.")

    X_train, y_train, _, _, label_map = load_full_dataset(filter_labels=req.labels)
    if not X_train:
        raise HTTPException(status_code=400, detail="Pas de données d'entraînement.")

    x = np.array(X_train, dtype=np.float64)
    y = np.array(y_train, dtype=np.uintp)

    # Normalize like in original notebook
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    lib.train_softmax_model(
        softmax_ptr,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        x.shape[0],
        x.shape[1]
    )

    label_to_index = label_map
    return {"status": "trained", "labels": label_map}

@app.post("/linear/predict/")
def predict_softmax_model_api(input: PredictInput):
    global softmax_ptr, label_to_index, scaler
    if not softmax_ptr:
        raise HTTPException(status_code=400, detail="Modèle non initialisé.")

    x_np = np.array(input.x, dtype=np.float64)
    if scaler is not None:
        x_np = scaler.transform(x_np)

    if x_np.ndim != 2:
        raise HTTPException(status_code=400, detail="Format attendu: liste de listes")

    results = []
    index_to_label = {v: k for k, v in label_to_index.items()}

    for x in x_np:
        pred_index = lib.predict_softmax_model(
            softmax_ptr,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(x)
        )
        results.append({
            "prediction_index": pred_index,
            "label": index_to_label.get(pred_index, f"classe {pred_index}")
        })

    return {"predictions": results}

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

    label_to_index = label_map
    run_id = f"mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_path = os.path.join(RESULTS_DIR, run_id)

    log_data = {
        "model": "mlp",
       
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


    label_to_index = label_map
    run_id = f"rbf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_path = os.path.join(RESULTS_DIR, run_id)

    log_data = {
        "model": "rbf",
        "labels": label_map,
        "config": {
            "sigma": req.sigma,
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
    }

@app.post("/rbfn/predict/")
def predict_rbf_model(input: MLPInput): 
    global rbfn_ptr, label_to_index
    if not rbfn_ptr:
        raise HTTPException(status_code=400, detail="Modèle RBF non initialisé.")

    if len(input.x) != 32 * 32:
        raise HTTPException(status_code=400, detail=f"Image incorrecte. Attendu: 64x64 ({32*32} valeurs)")

    x_np = np.array(input.x, dtype=np.float64)

    pred_raw = lib.predict_rbfn_model(
        rbfn_ptr,
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(input.x)
    )

    pred_index = max(0, int(round(pred_raw)))

    index_to_label = {v: k for k, v in label_to_index.items()}
    label = index_to_label.get(pred_index, f"classe {pred_index}")

    return {
        "prediction_index": pred_index,
        "label": label
    }

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

    X_train, y_train, _, _, label_map = load_full_dataset(filter_labels=req.labels)
    if not X_train:
        raise HTTPException(status_code=400, detail="Dataset vide ou invalide.")

    x = np.array(X_train, dtype=np.float64)
    y = np.array(y_train, dtype=np.uintp) 

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

    label_to_index = label_map

    run_id = f"svm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_path = os.path.join(RESULTS_DIR, run_id)

    log_data = {
        "model": "svm",
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

class ModelName(BaseModel):
    name: str
    model_type: str 

@app.post("/model/save/")
def save_current_model(data: ModelName):
    global mlp_clf_ptr, softmax_ptr, rbfn_ptr, svm_ptr

    name = data.name
    if not name.endswith(".model"):
        name += ".model"

    path = os.path.join(MODEL_DIR, name)

    if mlp_clf_ptr:
        lib.save_mlp_classifier(mlp_clf_ptr, path.encode("utf-8"))
        model_type = "mlp"

    elif softmax_ptr:
        if not hasattr(lib, "save_softmax_model"):
            raise HTTPException(status_code=501, detail="Fonction Rust 'save_softmax_model' manquante.")
        lib.save_softmax_model(softmax_ptr, path.encode("utf-8"))
        model_type = "softmax"

    elif rbfn_ptr:
        if not hasattr(lib, "save_rbfn_model"):
            raise HTTPException(status_code=501, detail="Fonction Rust 'save_rbfn_model' manquante.")
        lib.save_rbfn_model(rbfn_ptr, path.encode("utf-8"))
        model_type = "rbf"

    elif svm_ptr:
        if not hasattr(lib, "save_svm_model"):
            raise HTTPException(status_code=501, detail="Fonction Rust 'save_svm_model' manquante.")
        lib.save_svm_model(svm_ptr, path.encode("utf-8"))
        model_type = "svm"

    else:
        raise HTTPException(status_code=400, detail="Aucun modèle à sauvegarder.")

    return {"status": "saved", "filename": name, "model_type": model_type}

class ModelLoadRequest(BaseModel):
    name: str
    model_type: str


@app.post("/model/load/")
def load_model(req: ModelLoadRequest):
    global svm_ptr, mlp_clf_ptr, rbfn_ptr, softmax_ptr

    filename = req.name if req.name.endswith(".model") else f"{req.name}.model"
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Modèle introuvable")

    model_type = req.model_type.lower().strip()
    name_lower = filename.lower()
    if not model_type:
        if "svm" in name_lower:
            model_type = "svm"
        elif "mlp" in name_lower:
            model_type = "mlp"
        elif "rbf" in name_lower:
            model_type = "rbf"
        elif "linear" in name_lower or "softmax" in name_lower:
            model_type = "linear"
        else:
            raise HTTPException(status_code=400, detail="Impossible de déterminer le type de modèle à partir du nom")

    if model_type == "svm":
        if not hasattr(lib, "load_svm_model"):
            raise HTTPException(status_code=501, detail="load_svm_model non défini dans la lib Rust")
        svm_ptr = lib.load_svm_model(path.encode("utf-8"))
        if not svm_ptr:
            raise HTTPException(status_code=500, detail="Échec du chargement SVM")
        return {"status": "ok", "type": "svm"}

    elif model_type == "mlp":
        if not hasattr(lib, "load_mlp_classifier"):
            raise HTTPException(status_code=501, detail="load_mlp_classifier non défini dans la lib Rust")
        mlp_clf_ptr = lib.load_mlp_classifier(path.encode("utf-8"))
        if not mlp_clf_ptr:
            raise HTTPException(status_code=500, detail="Échec du chargement MLP")
        return {"status": "ok", "type": "mlp"}

    elif model_type == "linear":
        if not hasattr(lib, "load_softmax_model"):
            raise HTTPException(status_code=501, detail="load_softmax_model non défini dans la lib Rust")
        softmax_ptr = lib.load_softmax_model(path.encode("utf-8"))
        if not softmax_ptr:
            raise HTTPException(status_code=500, detail="Échec du chargement Softmax")
        return {"status": "ok", "type": "linear"}

    elif model_type == "rbf":
        if not hasattr(lib, "load_rbfn_model"):
            raise HTTPException(status_code=501, detail="load_rbfn_model non défini dans la lib Rust")
        rbfn_ptr = lib.load_rbfn_model(path.encode("utf-8"))
        if not rbfn_ptr:
            raise HTTPException(status_code=500, detail="Échec du chargement RBF")
        return {"status": "ok", "type": "rbf"}

    else:
        raise HTTPException(status_code=400, detail=f"Type de modèle non pris en charge: {model_type}")

@app.get("/model/list/")
def list_models():
    if not os.path.exists(MODEL_DIR):
        return {"models": []}
    return {
        "models": [f for f in os.listdir(MODEL_DIR) if f.endswith(".model")]
    }
@app.post("/mlp_deep/create/")
def create_mlp_deep_classifier(params: MLPDeepCreateInput):
    global mlp_deep_ptr

    if params.hidden_units <= 0 or params.n_layers <= 0:
        raise HTTPException(status_code=400, detail="hidden_units ou n_layers invalide")

    mlp_deep_ptr = lib.create_deep_mlp_classifier(
        params.n_features,
        params.hidden_units,
        params.n_classes,
        ctypes.c_double(params.learning_rate),
        params.epochs,
        params.activation_type,
        params.batch_size,
        ctypes.c_double(params.lambda_),
        params.n_layers
    )

    if not mlp_deep_ptr:
        raise HTTPException(status_code=500, detail="Erreur création MLP Deep")

    return {"status": "mlp_deep created"}


@app.post("/mlp_deep/train/")
def train_mlp_deep_classifier(req: MLPDeepTrainInput):
    print(req)

    global mlp_deep_ptr, label_to_index
    if not mlp_deep_ptr:
        raise HTTPException(status_code=400, detail="Modèle MLPDeep non initialisé")

    # Chargement des données
    X_train, y_train, X_test, y_test, label_map = load_full_dataset()
    if not X_train:
        raise HTTPException(status_code=400, detail="Données d'entraînement introuvables")

    x_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.uint32)
    x_test = np.array(X_test, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.uint32)

    # Entraînement du modèle
    lib.train_deep_mlp_classifier(
        mlp_deep_ptr,
        x_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        len(X_train),
        x_test.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_test.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        len(X_test),
        x_train.shape[1]
    )

    # Récupération des courbes
    def get_array(ptr_func, len_func):
        ptr = ptr_func(mlp_deep_ptr)
        length = len_func(mlp_deep_ptr)
        return [ptr[i] for i in range(length)] if ptr and length > 0 else []

    train_losses = get_array(lib.get_deep_mlp_train_losses_ptr, lib.get_deep_mlp_train_losses_len)
    test_losses = get_array(lib.get_deep_mlp_test_losses_ptr, lib.get_deep_mlp_test_losses_len)
    train_accuracies = get_array(lib.get_deep_mlp_train_accuracies_ptr, lib.get_deep_mlp_train_accuracies_len)
    test_accuracies = get_array(lib.get_deep_mlp_test_accuracies_ptr, lib.get_deep_mlp_test_accuracies_len)

    # Sauvegarde des paramètres
    label_to_index = label_map
    run_id = f"mlpdeep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_path = os.path.join(RESULTS_DIR, run_id)

    log_data = {
        "model": "mlp_deep",
        "labels": label_map,
        "config": {
            "learning_rate": req.learning_rate,
            "epochs": req.epochs,
            "activation": req.activation_type,
            "n_layers": req.n_layers,
            "batch_size": req.batch_size,
            "lambda": req.lambda_,
            "labels_used": "all",
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(save_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return {
        "status": "trained",
        "labels": label_map,
        "loss_train": train_losses,
        "loss_test": test_losses,
        "acc_train": train_accuracies,
        "acc_test": test_accuracies,
    }

@app.post("/mlp_deep/predict/")
def predict_mlp_deep(input: MLPInput):
    global mlp_deep_ptr, label_to_index
    if not mlp_deep_ptr:
        raise HTTPException(status_code=400, detail="Modèle MLPDeep non initialisé")

    x_np = np.array(input.x, dtype=np.float64)
    pred_index = lib.predict_deep_mlp_classifier(
        mlp_deep_ptr,
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(input.x)
    )

    index_to_label = {v: k for k, v in label_to_index.items()}
    label = index_to_label.get(pred_index, f"classe {pred_index}")
    return {"prediction_index": pred_index, "label": label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
