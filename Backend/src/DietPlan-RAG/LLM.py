import os
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.prompt_template import prompt_temp

# ---------------- LOAD ENV ----------------
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print("API Keys Loaded")

# ---------------- PINECONE ----------------
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "indexsample"   # ONE index, multiple namespaces

if INDEX_NAME not in pc.list_indexes().names():
    raise RuntimeError(" Pinecone index not found. Create it first.")

print(f"Index '{INDEX_NAME}' ready")


def generate_diet(
    document_chunks,
    filename: str,
    ml_prediction: str,
    Intents: dict,
):
    """
    Generate questions using RAG strictly from uploaded document
    """

    print("\nInitializing Embeddings & LLM...")

    # -------- Embeddings --------
    embeddings = NVIDIAEmbeddings()

    # -------- Namespace (CRITICAL FIX) --------
    namespace = filename.replace(".", "_").lower()
    print(f"Using namespace: {namespace}")

    # -------- Vector Store (ISOLATED) --------
    vectorstore = PineconeVectorStore.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )
    print("Vector store created")

    # -------- Retriever --------
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6,
            "namespace": namespace
        }
    )

    # -------- LLM --------
    llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        temperature=0.1,
        api_key=NVIDIA_API_KEY
    )

    print("LLM initialized")

    # -------- Prompt --------
    prompt = prompt_temp()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # -------- RAG Chain --------
    rag_chain = (
        {
            "context": retriever | format_docs,
            "ml_prediction": lambda _: ml_prediction,
            "Intents": lambda _: Intents,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # -------- Invoke --------
    print(f"\nGenerating DietPlan...")
    response = rag_chain.invoke(
        "Generate questions ONLY from the provided document. "
        "use too little external knowledge. Ignore unrelated topics."
    )

    print("Generation complete")
    return response


# ================= SAVE OUTPUT =================
def save_output(text, output_file="generated_questions.txt"):
    if not text:
        return
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved output to {output_file}")

