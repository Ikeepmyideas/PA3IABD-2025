import os
from pymongo import MongoClient
import gridfs
from datetime import datetime

# Connexion à MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["letters"]  # Création de la base "letters"
fs = gridfs.GridFS(db)  # GridFS pour stocker les images
metadata_col = db["metadata"]  # Collection pour les métadonnées

# Dossier contenant le dataset
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
