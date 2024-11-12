## Milvus

### Docker Compose

```bash
sudo docker-compose up -d
```

- stop and delete

```bash
# Stop Milvus
$ sudo docker-compose down

# Delete service data
$ sudo rm -rf volumes
```

### Import

```python
import numpy as np
from pymilvus import (
    connections, # Milvus 네트워크 접속 기능 제공
    utility, # Milvus 주요 리소스(Collection, Index, User ..)를 관리하기 위한 유틸리티 기능 제공
    FieldSchema, # Collection의 각 필드 타입(VarChar, Double, FloatVector..)을 정의하기 위한 스키마
    CollectionSchema, # 여러개의 필드를 하나의 Row Document 로 관리하기 위한 스키마
    DataType, # 각 필드의 데이터타입을 정의
    Collection, # RDB의 테이블과 비슷한 스키마를 기반으로 데이터를 저장하는 공간
)
```

### Connect

```python
connections.connect(db_name="default", host="milvus-standalone", port="19530")
```

### Create Collection

```python
# -- create collection
# +-+------------+------------+------------------+------------------------------+
# | | field name | field type | other attributes |       field description      |
# +-+------------+------------+------------------+------------------------------+
# |1|    "pk"    |   VarChar  |  is_primary=True |      "primary field"         |
# | |            |            |   auto_id=False  |                              |
# +-+------------+------------+------------------+------------------------------+
# |2|  "random"  |    Double  |                  |      "a double field"        |
# +-+------------+------------+------------------+------------------------------+
# |3|"embeddings"| FloatVector|     dim=8        |  "float vector with dim 8"   |
# +-+------------+------------+------------------+------------------------------+
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="random", dtype=DataType.DOUBLE),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]
schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong")
```

### Insert Data

```python
# -- case 1
rng = np.random.default_rng(seed=19530)
entities = [
    # provide the pk field because `auto_id` is set to False
    [str(i) for i in range(num_entities)],
    rng.random(num_entities).tolist(),  # field random, only supports list
    rng.random((num_entities, dim), np.float32),    # field embeddings, supports numpy.ndarray and list
]

insert_result = hello_milvus.insert(entities)

# -- case 2
row = {
    "pk": "19530",
    "random": 0.5,
    "embeddings": rng.random((1, dim), np.float32)[0]
}
hello_milvus.insert(row)

hello_milvus.flush()
```

### Create Index

```python
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}
hello_milvus.create_index("embeddings", index)
```

### search, query, and hybrid search

```python
# load the data in `hello_milvus` into memory
hello_milvus.load()

# search based on vector similarity
vectors_to_search = entities[-1][-2:]
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}, # 가장 가까운 클러스터 10개에 대해 검색
}
result = hello_milvus.search(
    data=vectors_to_search, 
    anns_field="embeddings",
    param=search_params,
    limit=3, # top-k
    output_fields=["random"]
)

for hits in result:
    for hit in hits:
        print(f"hit: {hit}, random field: {hit.entity.get('random')}")

"""
hit: id: 2998, distance: 0.0, entity: {'random': 0.9728033590489911}, random field: 0.9728033590489911
hit: id: 999, distance: 0.09934990108013153, entity: {'random': 0.9519034206569449}, random field: 0.9519034206569449
hit: id: 1310, distance: 0.10135538130998611, entity: {'random': 0.26669865443188623}, random field: 0.26669865443188623
hit: id: 2999, distance: 0.0, entity: {'random': 0.02316334456872482}, random field: 0.02316334456872482
hit: id: 2502, distance: 0.13083189725875854, entity: {'random': 0.9289998713260136}, random field: 0.9289998713260136
hit: id: 2669, distance: 0.1590736359357834, entity: {'random': 0.6080847854541138}, random field: 0.6080847854541138
"""

# query based on scalar filtering(boolean, int, etc.)
result = hello_milvus.query(expr="random > 0.5", output_fields=["random", "embeddings"])

"""
-{'random': 0.6378742006852851, 'embeddings': [np.float32(0.8367804), np.float32(0.20963514), np.float32(0.6766955), np.float32(0.39746654), np.float32(0.8180806), np.float32(0.1201905), np.float32(0.9467144), np.float32(0.6947491)], 'pk': '0'}
"""

# pagination
r1 = hello_milvus.query(expr="random > 0.5", limit=4, output_fields=["random"])
r2 = hello_milvus.query(expr="random > 0.5", offset=1, limit=3, output_fields=["random"])
```