import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

PERSIST_DIR = "./vector_sql_schema3"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class LocalEF(embedding_functions.EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def __call__(self, texts):
        return self.model.encode(texts).tolist()


# 1) Load your schema.json (the one you pasted)
with open("schema.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

# 2) Init Chroma
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(
    name="vector_DB",
    embedding_function=LocalEF(),
    metadata={"hnsw:space": "cosine"},
)

docs = []
metas = []
ids = []

# 3) Loop over tables (actor, film, rental, payment, ...)
for table_name, tdef in schema.items():
    table_desc = tdef.get("description", "") or ""

    # ---- add table-level embedding ----
    docs.append(f"TABLE {table_name}: {table_desc}")
    metas.append(
        {
            "type": "table",
            "table": table_name,
            "description": table_desc,
        }
    )
    ids.append(f"table::{table_name}")

    # 4) Get fields (columns) from "fields"
    fields = tdef.get("fields", {})  # 👈 هنا نستخدم fields وليس columns

    if isinstance(fields, dict):
        for col_name, cdef in fields.items():
            col_desc = cdef.get("description", "") or ""
            # ---- add column-level embedding ----
            docs.append(f"COLUMN {table_name}.{col_name}: {col_desc}")
            metas.append(
                {
                    "type": "column",
                    "table": table_name,
                    "column": col_name,
                    "description": col_desc,
                }
            )
            ids.append(f"col::{table_name}::{col_name}")

# 5) Insert into Chroma
collection.add(
    documents=docs,
    metadatas=metas,
    ids=ids,
)

print(f"✅ Seeded {len(docs)} embeddings into Chroma from schema.json")

