import pandas as pd
import numpy as np
import re


def calculate_body_fat(bmi, age, gender):
    if bmi is None or age is None or gender is None:
        return None
    sex = 1 if gender.lower().startswith("m") else 0
    body_fat = (1.2 * bmi) + (0.23 * age) - (10.8 * sex) - 5.4
    return round(body_fat, 1)


def estimate_height(age, gender):
    if pd.isna(age) or pd.isna(gender):
        return None
    base_height = 170 if gender.lower().startswith("m") else 158
    if age > 40:
        base_height -= ((age - 40) // 10) * 0.5
    return round(base_height, 1)


def estimate_weight(bmi, height_cm, diabetes, activity, cholesterol, systolic_bp):
    if pd.isna(bmi) or pd.isna(height_cm):
        return None

    height_m = height_cm / 100
    weight = bmi * (height_m ** 2)

    if diabetes == "Yes":
        weight *= 1.025
    if activity == "Low":
        weight *= 1.025
    if cholesterol and cholesterol > 200:
        weight *= 1.01
    if systolic_bp and systolic_bp > 140:
        weight *= 1.01

    return round(weight, 1)


def calculate_bmi(weight, height_cm):
    if weight and height_cm:
        return round(weight / ((height_cm / 100) ** 2), 1)
    return None



def safe_search(pattern, text, cast=None, group=1):
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    value = match.group(group).strip()
    return cast(value) if cast else value


def simple_preprocess(df):
    replacements = {
        "Mede": "Moderate",
        "low": "Low",
        "Mitd": "Moderate",
        "Normat": "Normal"
    }

    for col in ["Activity", "Anemia"]:
        if col in df.columns:
            df[col] = df[col].replace(replacements)

    return df



def ocr_max_text(image_path):
    import cv2
    import pytesseract
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Failed to read image file: {image_path}")

    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.fastNlMeansDenoising(img, h=35)

    texts = []
    for block in [11, 21, 31]:
        th = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block, 10
        )
        text = pytesseract.image_to_string(th, config="--oem 3 --psm 6")
        texts.append(text)

    return max(texts, key=len).strip()




def pdfDataExtraction(file_path):
    import fitz
    
    pdf = fitz.open(file_path)
    text = ""

    for page in pdf:
        text += page.get_text()

    data = {
        "Name": safe_search(r'Name\s*[:\-]?\s*(.+)', text),
        "Gender": safe_search(r'Gender\s*[:\- ]?\s*(Male|Female|M|F)', text),
        "Age": safe_search(r'Age\s*[:\- ]?\s*(\d+)', text, int),
        "Blood Group": safe_search(r'Blood\s*Group\s*[:\- ]?\s*(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)', text),
        "Height (cm)": safe_search(r'Height\s*\(cm\)\s*[:\- ]?\s*(\d+)', text, int),
        "Weight (kg)": safe_search(r'Weight\s*\(kg\)\s*[:\- ]?\s*(\d+)', text, int),
        "BMI": safe_search(r'BMI\s*[:\- ]?\s*([\d.]+)', text, float),
        "Body Fat (%)": safe_search(r'Body\s*Fat\s*\(%\)\s*[:\- ]?\s*([\d.]+)', text, float),
        "Systolic BP": safe_search(r'Systolic\s*BP\s*[:\- ]?\s*(\d+)', text, int),
        "Diastolic BP": safe_search(r'Diastolic\s*BP\s*[:\- ]?\s*(\d+)', text, int),
        "Cholesterol": safe_search(r'Cholesterol\s*[:\- ]?\s*(\d+)', text, int),
        "Hemoglobin": safe_search(r'Hemoglobin\s*[:\- ]?\s*([\d.]+)', text, float),
        "PPBS": safe_search(r'PPBS\s*[:\- ]?\s*(\d+)', text, int),
        "Diabetes": safe_search(r'Diabetes\s*[:\- ]?\s*(Yes|No)', text),
        "Smoker": safe_search(r'Smoker\s*[:\- ]?\s*(Yes|No)', text),
        "Activity": safe_search(r'Activity\s*[:\- ]?\s*(\w+)', text),
        "Anemia": safe_search(r'Anemia\s*[:\- ]?\s*(\w+)', text)
    }

    return pd.DataFrame([data]),text


def imageDataExtraction(file_path):
    
    

    text = ocr_max_text(file_path)

    data = {
        "Name": safe_search(r'Name\s*[:\-]?\s*(.+)', text),
        "Gender": safe_search(r'Gender\s*[:\- ]?\s*(Male|Female|M|F)', text),
        "Age": safe_search(r'Age\s*[:\- ]?\s*(\d+)', text, int),
        "Blood Group": safe_search(r'Blood\s*Group\s*[:\- ]?\s*(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)', text),
        "Height (cm)": safe_search(r'Height\s*\(cm\)\s*[:\- ]?\s*(\d+)', text, int),
        "Weight (kg)": safe_search(r'Weight\s*\(kg\)\s*[:\- ]?\s*(\d+)', text, int),
        "BMI": safe_search(r'BMI\s*[:\- ]?\s*([\d.]+)', text, float),
        "Body Fat (%)": safe_search(r'Body\s*Fat\s*\(%\)\s*[:\- ]?\s*([\d.]+)', text, float),
        "Systolic BP": safe_search(r'Systolic\s*BP\s*[:\- ]?\s*(\d+)', text, int),
        "Diastolic BP": safe_search(r'Diastolic\s*BP\s*[:\- ]?\s*(\d+)', text, int),
        "Cholesterol": safe_search(r'Cholesterol\s*[:\- ]?\s*(\d+)', text, int),
        "Hemoglobin": safe_search(r'Hemoglobin\s*[:\- ]?\s*([\d.]+)', text, float),
        "PPBS": safe_search(r'PPBS\s*[:\- ]?\s*(\d+)', text, int),
        "Diabetes": safe_search(r'Diabetes\s*[:\- ]?\s*(Yes|No)', text),
        "Smoker": safe_search(r'Smoker\s*[:\- ]?\s*(Yes|No)', text),
        "Activity": safe_search(r'Activity\s*[:\- ]?\s*(\w+)', text),
        "Anemia": safe_search(r'Anemia\s*[:\- ]?\s*(\w+)', text)
    }

    return pd.DataFrame([data]),text




def textDataExtraction(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    data = {
        "Name": safe_search(r'PATIENT\s*[:\-]\s*([A-Za-z ]+)', text),
        "Gender": safe_search(r'Gender\s*[:\- ]?\s*(Male|Female|M|F)', text),
        "Age": safe_search(r'Age\s*[:\- ]?\s*(\d+)', text, int),
        "Blood Group": safe_search(r'Blood\s*Group\s*[:\- ]?\s*(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)', text),
        "Height (cm)": safe_search(r'Height\s*\(cm\)\s*[:\- ]?\s*(\d+)', text, int),
        "Weight (kg)": safe_search(r'Weight\s*\(kg\)\s*[:\- ]?\s*(\d+)', text, int),
        "BMI": safe_search(r'BMI\s*[:\- ]?\s*([\d.]+)', text, float),
        "Body Fat (%)": safe_search(r'Body\s*Fat\s*\(%\)\s*[:\- ]?\s*([\d.]+)', text, float),
        "Systolic BP": safe_search(r'BP\s*[:\- ]?\s*(\d{2,3})\D+(\d{2,3})', text, int, 1),
        "Diastolic BP": safe_search(r'BP\s*[:\- ]?\s*(\d{2,3})\D+(\d{2,3})', text, int, 2),
        "Cholesterol": safe_search(r'Cholesterol\s*[:\- ]?\s*(\d+)', text, int),
        "Hemoglobin": safe_search(r'Hemoglobin\s*[:\- ]?\s*([\d.]+)', text, float),
        "PPBS": safe_search(r'PPBS\s*[:\- ]?\s*(\d+)', text, int),
        "Diabetes": safe_search(r'Diabetes\s*[:\- ]?\s*(Yes|No)', text),
        "Smoker": safe_search(r'Smoker\s*[:\- ]?\s*(Yes|No)', text),
        "Activity": safe_search(r'Activity\s*[:\- ]?\s*(.+)', text),
        "Anemia": safe_search(r'Anemia\s*(Status)?\s*[:\- ]?\s*(.+)', text, group=2)
    }

    return pd.DataFrame([data]),text




def DataExtraction(file_path):
    if file_path.endswith(".pdf"):
        df,text = pdfDataExtraction(file_path)
    elif file_path.lower().endswith((".jpg", ".jpeg", ".png")):
        df,text= imageDataExtraction(file_path)
    elif file_path.endswith(".txt"):
        df,text = textDataExtraction(file_path)
    else:
        raise ValueError("Unsupported file type")

    df = simple_preprocess(df)
    row = df.iloc[0]

    if pd.isna(row["Body Fat (%)"]) and row["BMI"]:
        df.at[0, "Body Fat (%)"] = calculate_body_fat(row["BMI"], row["Age"], row["Gender"])

    if pd.isna(row["Height (cm)"]):
        df.at[0, "Height (cm)"] = estimate_height(row["Age"], row["Gender"])

    if pd.isna(row["Weight (kg)"]) and row["BMI"] and row["Height (cm)"]:
        df.at[0, "Weight (kg)"] = estimate_weight(
            row["BMI"], row["Height (cm)"], row["Diabetes"],
            row["Activity"], row["Cholesterol"], row["Systolic BP"]
        )

    if pd.isna(row["BMI"]) and row["Weight (kg)"] and row["Height (cm)"]:
        df.at[0, "BMI"] = calculate_bmi(row["Weight (kg)"], row["Height (cm)"])

    return df,text


