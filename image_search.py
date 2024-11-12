import os

import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from pymilvus import (
    MilvusClient,
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# -- Create a Milvus Collection
client = MilvusClient(db_name="default", uri="http://milvus-standalone:19530")
client.create_collection(
    collection_name="image_embeddings",
    vector_field_name="vector",
    dimension=512,
    auto_id=True,
    enable_dynamic_field=True, # default: True
    metric_type="COSINE",
)

# -- Insert the Embeddings to Milvus
extractor = FeatureExtractor("resnet34")

# db_images = "./db_images"
# for dirpath, foldername, filenames in os.walk(db_images):
#     for filename in filenames:
#         filepath = dirpath + "/" + filename
#         image_embedding = extractor(filepath)
#         client.insert(
#             "image_embeddings",
#             {"vector": image_embedding, "filename": filepath}
#         )

# -- Search Image
query_image = "./query_images/0.jpg"
results = client.search(
    "image_embeddings",
    data=[extractor(query_image)],
    output_fields=["filename"],
    search_params={"metric_type": "COSINE"},
    limit=2,
)

images = []
for result in results:
    for hit in result:
        print(hit)
        filename = hit["entity"]["filename"]
        img = Image.open(filename)
        images.append(img)

# -- Save Found Images
for idx, image in enumerate(images):
    image.save(f"{idx}.png")
