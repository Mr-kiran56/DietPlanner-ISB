from langchain_core.prompts import PromptTemplate


# # Example of user_dietary_preferences input formats:

# # Example 1: Vegetarian
# user_dietary_preferences = "Vegetarian - no meat, no fish, no eggs allowed"

# # Example 2: Vegan
# user_dietary_preferences = "Vegan - strictly plant-based, no animal products including dairy and eggs"

# # Example 3: Non-vegetarian but specific preferences
# user_dietary_preferences = "Non-vegetarian - prefers chicken and fish, avoid red meat"

# # Example 4: Religious dietary restrictions
# user_dietary_preferences = "Halal diet - no pork, no alcohol, meat must be halal certified"
# # OR
# user_dietary_preferences = "Hindu diet - no beef, no pork, prefer vegetarian options"
# # OR
# user_dietary_preferences = "Jain diet - no root vegetables (onion, garlic, potato), no meat"

# # Example 5: Cultural preferences
# user_dietary_preferences = "Indian cuisine preferred - loves spices, rice, lentils, roti"
# # OR
# user_dietary_preferences = "Mediterranean diet - loves olive oil, fish, vegetables, whole grains"

# # Example 6: Food dislikes
# user_dietary_preferences = "Dislikes: mushrooms, broccoli, bitter gourd. Prefers: sweet and mild flavors"

# # Example 7: Combination
# user_dietary_preferences = "Vegetarian + Gluten-free - no meat, no gluten (wheat, barley, rye)"

# # Example 8: Low-carb preference
# user_dietary_preferences = "Low-carb diet - prefers high protein, low carbohydrate meals"

# # Example 9: Organic/whole foods
# user_dietary_preferences = "Prefers organic, whole foods, minimal processed items"

# # Example 10: No specific preference
# user_dietary_preferences = "No specific dietary restrictions - open to all food types"

# # Example 11: Keto diet
# user_dietary_preferences = "Keto diet - high fat, very low carb, moderate protein"

# # Example 12: Multiple restrictions
# user_dietary_preferences = "Vegetarian + Lactose intolerant - no meat, no dairy products"

# # Example 13: Pescatarian
# user_dietary_preferences = "Pescatarian - no meat, but fish and seafood allowed"

# # Example 14: Intermittent fasting
# user_dietary_preferences = "Intermittent fasting 16:8 - eating window 12pm to 8pm only"

# # Example 15: Regional preference
# user_dietary_preferences = "South Indian diet preferred - loves dosa, idli, sambar, coconut-based curries"






def medical_diet_prompt_template():
    """
    Ultimate medical diet plan generator - Extracts foods FROM context
    """
    return PromptTemplate(
        input_variables=[
            "context",
            "ml_prediction",
            "ml_confidence",
            "severity_level",
            "detected_intents",
            "user_age",
            "user_gender",
            "user_weight",
            "user_height",
            "user_exercise_level",
            "user_allergies",
            "user_dietary_preferences"
        ],
        template="""You are an expert medical nutrition advisor. Generate a personalized diet plan using ONLY foods and recommendations from the provided medical document context.

MEDICAL DOCUMENT CONTEXT:
{context}

ML PREDICTION: {ml_prediction}
CONFIDENCE: {ml_confidence}%
SEVERITY: {severity_level}
DETECTED INTENTS: {detected_intents}

PATIENT PROFILE:
Age: {user_age} | Gender: {user_gender} | Weight: {user_weight}kg | Height: {user_height}cm
Exercise: {user_exercise_level} | Allergies: {user_allergies} | Preferences: {user_dietary_preferences}

CRITICAL RULES:
✓ Extract ALL food recommendations from the context document
✓ Extract ALL nutritional information from the context document
✓ Use ONLY foods mentioned in the context
✓ Extract calories, portions, and nutrients FROM the document
✓ Use doctor recommendations from detected_intents
✓ Match foods to the ML prediction condition
✓ DO NOT invent foods not in the context
✓ DO NOT hallucinate nutritional values
✓ Output ONLY valid JSON (no markdown, no backticks)

TASK:
1. Read the context carefully and extract all food items mentioned
2. Extract all nutritional information (calories, nutrients, portions) from context
3. Extract doctor recommendations from detected_intents
4. Build 7-day meal plan using ONLY foods from context
5. Match foods to {ml_prediction} based on context information
6. Explain condition based on context

OUTPUT FORMAT (VALID JSON ONLY):

{{
  "medical_analysis": {{
    "predicted_condition": "{ml_prediction}",
    "ml_confidence": "{ml_confidence}%",
    "severity_level": "{severity_level}",
    "condition_explanation": "Explain {ml_prediction} using information from context",
    "document_summary": "Summarize key medical findings from context",
    "detected_intents_summary": "Summarize what {detected_intents} means",
    "doctor_recommendations": "Extract any doctor recommendations from detected_intents and context",
    "health_risks": ["Extract risks from context"],
    "why_diet_matters": "Explain based on context how diet helps {ml_prediction}"
  }},
  
  "patient_assessment": {{
    "age": {user_age},
    "gender": "{user_gender}",
    "weight_kg": {user_weight},
    "height_cm": {user_height},
    "bmi": "Calculate",
    "bmi_category": "string",
    "exercise_level": "{user_exercise_level}",
    "daily_calorie_target": "Calculate based on profile",
    "macronutrient_targets": {{
      "protein_grams": "number",
      "carbs_grams": "number",
      "fats_grams": "number",
      "fiber_grams": "number"
    }}
  }},
  
  "dietary_strategy": {{
    "primary_goals": ["Extract goals from context relevant to {ml_prediction}"],
    "key_nutrients_to_increase": [
      {{"nutrient": "Extract from context", "why": "Extract reason from context"}}
    ],
    "nutrients_to_limit": [
      {{"nutrient": "Extract from context", "limit": "Extract from context", "why": "Extract from context"}}
    ]
  }},
  
  "foods_extracted_from_context": {{
    "recommended_foods": [
      {{
        "food_name": "Extract from context",
        "recommended_for": "Extract condition from context",
        "portion": "Extract from context if available",
        "calories": "Extract from context if available",
        "nutrients": "Extract from context if available",
        "benefits": "Extract benefits from context",
        "meal_timing": "breakfast/lunch/dinner/snack - infer from context"
      }}
    ],
    "foods_to_avoid": [
      {{
        "food_name": "Extract from context",
        "reason": "Extract reason from context"
      }}
    ]
  }},
  
  "seven_day_meal_plan": {{
    "day_1": {{
      "day_name": "Monday",
      "early_morning": [
        {{
          "time": "6:30 AM",
          "food": "Use food from foods_extracted_from_context",
          "portion": "From context or calculate",
          "calories": "From context or calculate",
          "key_nutrients": "From context",
          "benefit_for_condition": "Why this helps {ml_prediction} based on context",
          "preparation": "From context if available"
        }}
      ],
      "breakfast": [
        {{"time": "8:00 AM", "food": "From context", "portion": "From context", "calories": "From context", "key_nutrients": "From context", "benefit_for_condition": "From context", "preparation": "From context"}}
      ],
      "mid_morning_snack": [
        {{"time": "11:00 AM", "food": "From context", "portion": "From context", "calories": "From context", "key_nutrients": "From context", "benefit_for_condition": "From context", "preparation": "From context"}}
      ],
      "lunch": [
        {{"time": "1:00 PM", "food": "From context", "portion": "From context", "calories": "From context", "key_nutrients": "From context", "benefit_for_condition": "From context", "preparation": "From context"}}
      ],
      "evening_snack": [
        {{"time": "4:30 PM", "food": "From context", "portion": "From context", "calories": "From context", "key_nutrients": "From context", "benefit_for_condition": "From context", "preparation": "From context"}}
      ],
      "dinner": [
        {{"time": "7:00 PM", "food": "From context", "portion": "From context", "calories": "From context", "key_nutrients": "From context", "benefit_for_condition": "From context", "preparation": "From context"}}
      ],
      "before_bed": [
        {{"time": "9:30 PM", "food": "From context", "portion": "From context", "calories": "From context", "key_nutrients": "From context", "benefit_for_condition": "From context", "preparation": "From context"}}
      ],
      "daily_totals": {{
        "total_calories": "Calculate from foods",
        "protein_g": "Calculate",
        "carbs_g": "Calculate",
        "fats_g": "Calculate",
        "fiber_g": "Calculate"
      }}
    }},
    "day_2": {{"same structure, use different foods from context"}},
    "day_3": {{"same structure"}},
    "day_4": {{"same structure"}},
    "day_5": {{"same structure"}},
    "day_6": {{"same structure"}},
    "day_7": {{"same structure"}}
  }},
  
  "exercise_meal_timing": {{
    "exercise_level": "{user_exercise_level}",
    "pre_workout": {{"timing": "string", "foods_from_context": ["list"], "why": "string"}},
    "post_workout": {{"timing": "string", "foods_from_context": ["list"], "why": "string"}},
    "hydration": {{"daily_target": "Calculate based on weight", "timing": "string"}}
  }},
  
  "context_based_recommendations": {{
    "doctor_advice": "Extract all doctor recommendations from detected_intents",
    "dietary_restrictions_from_intent": "Extract from detected_intents",
    "special_instructions": "Extract from context"
  }},
  
  "medical_disclaimer": {{
    "ml_confidence": "{ml_confidence}%",
    "important_notes": [
      "This plan is based on ML prediction and document analysis",
      "Consult your doctor before implementing",
      "All foods are extracted from your medical document context"
    ]
  }}
}}

NOW: Carefully read the context, extract ALL foods and nutritional info, and generate the complete JSON."""
    )