from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from .promptTemplate import unified_medical_prompt
from .vectorRetriever import get_retriever


def generate_diet(
    context: str,           # ← Add this
    payload: dict,          # ← Rename from user_payload
    days: int               # ← Add this
):
    """
    Generate diet plan from context and payload.
    """
    # Remove the retriever logic since you're passing context directly
    
    llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        temperature=0.1,
        max_tokens=7000
    )

    prompt = unified_medical_prompt()

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "context": context,
        "payload": payload,
        "days": days
    })


