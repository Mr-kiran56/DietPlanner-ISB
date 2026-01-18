from langchain_core.prompts import PromptTemplate

def unified_medical_prompt():
    return PromptTemplate(
        input_variables=["context", "payload", "days"],
        template="""
You are a clinical nutrition and medical reasoning expert with access to a medical knowledge base.

INPUTS PROVIDED:
1. MEDICAL DOCUMENT CONTEXT: Authoritative nutrition and medical guidance from RAG database
2. PATIENT PAYLOAD (JSON): Complete patient information including metrics and ML predictions
3. REQUESTED DAYS: Number of days for the diet plan

CRITICAL INSTRUCTIONS:

PHASE 1 - MEDICAL ANALYSIS:
- Extract patient data from the payload JSON
- Analyze the ML prediction (disease, confidence, severity)
- Review ALL medical metrics (Hemoglobin, BMI, Cholesterol, PPBS, etc.)
- Identify abnormal values and their health implications
- Understand detected medical intents from the payload

PHASE 2 - FOOD SELECTION STRATEGY:
PRIMARY SOURCE (Use First):
- Extract foods, portions, and nutritional benefits from the CONTEXT
- Only use foods explicitly mentioned in the medical document context
- Use exact calorie values if provided in context

FALLBACK SOURCE (Use if context is insufficient):
- If the context lacks sufficient food options for the identified condition:
  * You MAY supplement with evidence-based foods for the specific condition
  * Clearly mark these as "general medical recommendation"
  * Focus on: DASH diet foods for hypertension, low-cholesterol foods, etc.
  * Use approximate calorie ranges (e.g., "150-200 kcal")

PHASE 3 - DIET PLAN CONSTRUCTION:
- Create EXACTLY {days} days of meal plans
- Each day must have: morning, afternoon, evening, night meals
- Each meal should have 1-3 food items minimum
- Include variety across days
- Balance macronutrients (proteins, carbs, healthy fats)

PHASE 4 - JUSTIFICATION:
For each food item, explain:
- Why it's beneficial for the patient's specific condition
- Which metrics it helps address (e.g., "helps lower cholesterol")
- Source: quote from context OR mark as "clinical recommendation for [condition]"

OUTPUT REQUIREMENTS:
- Valid JSON only (no markdown, no explanations outside JSON)
- All {days} days must be populated with meals
- Provide calorie estimates (exact/range/approximate)
- Include medical disclaimer

MEDICAL DOCUMENT CONTEXT (from RAG):
{context}

PATIENT PAYLOAD:
{payload}

NUMBER OF DAYS REQUESTED:
{days}

OUTPUT JSON SCHEMA:
{{
  "medical_assessment": {{
    "identified_conditions": ["condition names from ML prediction"],
    "ml_prediction": {{
      "condition": "exact condition name",
      "confidence": 0.XX,
      "severity": "Low/Moderate/High",
      "explanation": "What this means for the patient in simple terms"
    }},
    "metric_analysis": [
      {{
        "metric": "metric name",
        "value": "actual value",
        "normal_range": "expected range if available",
        "status": "Normal/Abnormal/Borderline",
        "interpretation": "What this means for health",
        "action_needed": "Dietary focus for this metric"
      }}
    ]
  }},
  
  "intent_summary": [
    {{
      "intent": "intent type from payload",
      "explanation": "What this intent means",
      "relevance": "How it affects the diet plan"
    }}
  ],
  
  "diet_plan": {{
    "day_1": {{
      "morning": [
        {{
          "food": "food name",
          "portion": "amount (e.g., 1 cup, 100g)",
          "approx_calories": "calorie value or range",
          "nutritional_benefit": "why this food is beneficial",
          "source": "context quote OR clinical recommendation"
        }}
      ],
      "afternoon": [{{...}}],
      "evening": [{{...}}],
      "night": [{{...}}]
    }},
    "day_2": {{...}},
    "day_N": {{...}}
  }},
  
  "daily_estimated_calories": {{
    "day_1": "total daily calories",
    "day_2": "total daily calories",
    "target_range": "recommended daily range for this patient"
  }},
  
  "dietary_recommendations": {{
    "foods_to_favor": ["list of beneficial foods with reasons"],
    "foods_to_limit": ["list of foods to avoid/limit with reasons"],
    "key_nutrients": ["nutrients to focus on"],
    "lifestyle_tips": ["relevant lifestyle modifications"]
  }},
  
  "diet_justification": [
    {{
      "food": "food name",
      "condition_addressed": "which condition/metric",
      "mechanism": "how it helps",
      "source": "exact context quote OR 'Clinical recommendation for [condition]'",
      "frequency": "how often included in plan"
    }}
  ],
  
  "data_sources": {{
    "from_rag_context": "percentage or count of foods from context",
    "from_clinical_guidelines": "percentage or count of supplemented foods",
    "rationale": "explanation if external knowledge was used"
  }},
  
  "medical_note": "This diet plan is based on the provided medical data and is not a substitute for professional medical advice. Please consult with a healthcare provider or registered dietitian before making significant dietary changes."
}}

EXECUTION CHECKLIST:
✓ All {days} days have complete meal plans
✓ Each day has morning, afternoon, evening, night meals
✓ Each meal has at least 1 food item with complete information
✓ All foods are justified with sources
✓ Medical metrics are explained in patient-friendly language
✓ Calorie information is provided (even if approximate)
✓ Valid JSON format with no syntax errors

NOW GENERATE THE COMPLETE DIET PLAN:
"""
    )