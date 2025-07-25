from pymongo import MongoClient
import gridfs
import os
import mimetypes

# Connexion à MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["signai"]
fs = gridfs.GridFS(db)

dataset_path = "../dataset/train"

labels_voulus = {"A", "B", "C"}

extensions_valides = (".png", ".jpg", ".jpeg")

for label in os.listdir(dataset_path):
    if label not in labels_voulus:
        continue 

    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path):
        continue

    for filename in os.listdir(label_path):
        file_path = os.path.join(label_path, filename)

        if not filename.lower().endswith(extensions_valides):
            continue  

        with open(file_path, "rb") as f:
            data = f.read()

        content_type, _ = mimetypes.guess_type(filename)
        if content_type is None:
            content_type = "application/octet-stream"

        if fs.exists({"filename": filename, "label": label}):
            print(f"Déjà importé : {filename}")
            continue

        fs.put(
            data,
            filename=filename,
            content_type=content_type,
            label=label
        )

print("Import terminé pour A, B, C ")
