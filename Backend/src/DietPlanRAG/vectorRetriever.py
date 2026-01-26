import os
from pinecone import Pinecone
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv


load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

INDEX_NAME = "diet-plan"

def get_retriever():
    """
    Pinecone Retriever that automatically finds your data namespace.
    """
    # 1. Setup Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    
    # 2. Find the Namespace (Where your data is hiding!)
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    
    # Get the first available namespace from the stats
    # (Your logs showed: 'b:_protfilo_my-protfilo_src_assets_kiran_bio_pdf')
    namespaces = list(stats.get('namespaces', {}).keys())
    target_namespace = namespaces[0] if namespaces else ""
    
    print(f"--> [DEBUG] Using Pinecone Namespace: '{target_namespace}'")

    # 3. Connect to VectorStore
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
    
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=target_namespace  # <--- CRITICAL FIX
    )
    
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )