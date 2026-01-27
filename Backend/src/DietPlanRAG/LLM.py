from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from .promptTemplate import unified_medical_prompt
from .vectorRetriever import get_retriever


def build_retriever_query(payload: dict) -> str:
    disease = payload.get("ml_prediction", {}).get("predicted_disease", "")
    confidence = payload.get("ml_prediction", {}).get("confidence", "")
    intents = payload.get("detected_intents", [])
    intent_names = [i["intent"] for i in intents if isinstance(i, dict)]
    prefs = payload.get("user_preferences", {})
    profile = payload.get("patient_profile", {})

    key_metrics = []
    for k, v in profile.items():
        if isinstance(v, (int, float)):
            key_metrics.append(k)

    return f"""
    Evidence-based medical nutrition guidelines for {disease} and confidence of disease {confidence}.
    Dietary recommendations addressing patient metrics: {', '.join(key_metrics)}.
    Relevant to intents: {', '.join(intent_names)}.

    Food type preference: {prefs.get('food_type', 'any')}.
    Budget considerations: {prefs.get('budget', 'standard')}.
    """




def generate_diet(payload: dict):
    days = payload["user_preferences"]["days"]

    llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        temperature=0.05,
        max_tokens=7000,
        max_completion_tokens=4096
    )

    retriever = get_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retrieval_query = build_retriever_query(payload)

    docs = retriever.invoke(retrieval_query)
    context = format_docs(docs)

    chain = (
        unified_medical_prompt()
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "context": context,
        "payload": payload,
        "days": days
    })
