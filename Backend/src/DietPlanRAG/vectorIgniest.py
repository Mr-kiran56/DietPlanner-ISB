import os
import time
import requests
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from fileProcessing import file_process

# ================= ENV =================
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError("❌ PINECONE_API_KEY not loaded")

if not NVIDIA_API_KEY:
    raise RuntimeError("❌ NVIDIA_API_KEY not loaded")

# ================= CONFIG =================
INDEX_NAME = "diet-plan"
DIMENSION = 1024
METRIC = "cosine"

BATCH_SIZE = 30
MAX_RETRIES = 3
RETRY_SLEEP = 5  # seconds

DVC_MARKER = "dvc_artifacts/vector_ingestion.done"


# ================= UTILS =================
def write_dvc_marker(message: str):
    os.makedirs("dvc_artifacts", exist_ok=True)
    with open(DVC_MARKER, "w") as f:
        f.write(message)


# ================= PINECONE =================
def ensure_index_exists(pc: Pinecone):
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME in existing_indexes:
        print(f"✅ Pinecone index '{INDEX_NAME}' already exists")
        return

    print(f"⚠️ Pinecone index '{INDEX_NAME}' not found. Creating...")

    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print(f"✅ Pinecone index '{INDEX_NAME}' created")


# ================= INGESTION =================
def ingest_in_batches(docs, vectorstore):
    total = len(docs)
    batch_no = 1

    for i in range(0, total, BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"🔹 Batch {batch_no} | Docs {i}–{i + len(batch)}")
                vectorstore.add_documents(batch)
                break
            except requests.exceptions.ConnectionError as e:
                print(f"⚠️ Connection error (attempt {attempt}/{MAX_RETRIES})")
                if attempt == MAX_RETRIES:
                    raise RuntimeError("❌ NVIDIA embedding failed repeatedly") from e
                time.sleep(RETRY_SLEEP)

        batch_no += 1


def ingest_documents(document_chunks, filename: str):
    embeddings = NVIDIAEmbeddings(api_key=NVIDIA_API_KEY)

    namespace = filename.replace("/", "_").replace(".", "_").lower()

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Ensure index exists
    ensure_index_exists(pc)

    index = pc.Index(INDEX_NAME)

    # -------- DVC SAFE SKIP --------
    stats = index.describe_index_stats()
    if namespace in stats.get("namespaces", {}):
        print(f"✅ Namespace '{namespace}' already exists. Skipping ingestion.")
        write_dvc_marker("Vector ingestion skipped (already exists).")
        return

    print(f"🚀 Starting ingestion for namespace: {namespace}")

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace
    )

    ingest_in_batches(document_chunks, vectorstore)

    print(f"✅ Documents ingested into namespace: {namespace}")

    write_dvc_marker("Vector ingestion completed successfully.")


# ================= ENTRY =================
def main():
    pdf_path = "DeitPDFS/medical_diet_guidelines.pdf"

    chunks = file_process(pdf_path)
    ingest_documents(chunks, pdf_path)


if __name__ == "__main__":
    main()
