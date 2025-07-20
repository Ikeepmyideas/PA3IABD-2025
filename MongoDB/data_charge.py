from pymongo import MongoClient
import gridfs
from PIL import Image
import io

def load_images_from_mongodb(limit=None):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["letters"]
    fs = gridfs.GridFS(db)
    metadata = db["metadata"]

    images, labels = [], []

    cursor = metadata.find()
    if limit:
        cursor = cursor.limit(limit)

    for doc in cursor:
        file_id = doc["file_id"]
        label = doc["class"]

        binary_data = fs.get(file_id).read()
        image = Image.open(io.BytesIO(binary_data)).convert("RGB")

        images.append(image)
        labels.append(label)

    return images, labels
