from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import ctypes
import numpy as np
import os
import platform

# === Chargement de la bibliothèque Rust ===
def load_library():
    system = platform.system()
    if system == "Windows":
        libname = "ml_models.dll"
    elif system == "Darwin":
        libname = "libml_models.dylib"
    else:
        libname = "libml_models.so"
    return ctypes.cdll.LoadLibrary(os.path.abspath(libname))

lib = load_library()

# === Définition des types Rust <-> Python ===
class TrainInput(BaseModel):
    x: List[List[float]]
    y: List[float]
    show_logs: bool = False

class PredictInput(BaseModel):
    x: List[List[float]]

# === Gestion des modèles ===
# Pour simplifier, on garde un modèle global en mémoire (une vraie app ferait mieux)
model_ptr = None

# === FastAPI app ===
app = FastAPI()

@app.post("/linear/create/")
def create_linear_model(n_features: int, learning_rate: float, max_epochs: int, activation_type: int):
    global model_ptr
    lib.create_linear.restype = ctypes.c_void_p
    model_ptr = lib.create_linear(n_features, ctypes.c_float(learning_rate), max_epochs, activation_type)
    if not model_ptr:
        raise HTTPException(status_code=500, detail="Erreur lors de la création du modèle.")
    return {"status": "created", "ptr": str(model_ptr)}

@app.post("/linear/train/")
def train_linear_model(data: TrainInput):
    global model_ptr
    if not model_ptr:
        raise HTTPException(status_code=400, detail="Modèle non initialisé.")
    
    x = np.array(data.x, dtype=np.float32)
    y = np.array(data.y, dtype=np.float32)
    
    if x.shape[0] != len(y):
        raise HTTPException(status_code=400, detail="Nombre d'exemples incohérent entre x et y.")

    lib.train_linear.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_bool
    ]

    lib.train_linear(model_ptr,
                     x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                     y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                     x.shape[0],
                     x.shape[1],
                     data.show_logs)
    return {"status": "trained"}

@app.post("/linear/predict/")
def predict_linear_model(data: PredictInput):
    global model_ptr
    if not model_ptr:
        raise HTTPException(status_code=400, detail="Modèle non initialisé.")
    
    x = np.array(data.x, dtype=np.float32)
    n_samples, n_features = x.shape

    out = np.zeros(n_samples, dtype=np.float32)

    lib.predict_linear.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float)
    ]

    lib.predict_linear(model_ptr,
                       x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       n_samples,
                       n_features,
                       out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    return {"predictions": out.tolist()}

@app.post("/linear/destroy/")
def destroy_linear_model():
    global model_ptr
    if model_ptr:
        lib.destroy_linear.argtypes = [ctypes.c_void_p]
        lib.destroy_linear(model_ptr)
        model_ptr = None
        return {"status": "destroyed"}
    else:
        return {"status": "no model to destroy"}
