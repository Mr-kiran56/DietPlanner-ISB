import os
import re
import pandas as pd
import fitz
import spacy
from PIL import ImageFilter,Image,ImageEnhance
import pytesseract
import cv2
import numpy as np
import pytesseract
from PIL import Image


#predicted calculations
def calculate_body_fat(bmi, age, gender):
    if bmi is None or age is None or gender is None:
        return None
    sex = 1 if gender.lower().startswith("m") else 0
    body_fat = (1.2 * bmi) + (0.23 * age) - (10.8 * sex) - 5.4
    return round(body_fat, 1)


def estimate_height(age, gender):
    """Estimate height based on age and gender."""
    if pd.isna(age) or pd.isna(gender):
        return None
    gender = gender.lower()
    base_height = 170 if gender.startswith('m') else 158
    if age > 40:
        reduction = ((age - 40) // 10) * 0.5
        base_height -= reduction
    return round(base_height, 1)



def estimate_weight(bmi, height_cm, diabetes, activity, cholesterol, systolic_bp):
    """Estimate weight based on BMI, height, and health factors."""
    if pd.isna(bmi) or pd.isna(height_cm):
        return None
    height_m = height_cm / 100
    weight = bmi * (height_m ** 2)

    if diabetes and diabetes.lower() == "yes":
        weight *= 1.025
    if activity and activity.lower() == "low":
        weight *= 1.025
    if cholesterol and cholesterol > 200:
        weight *= 1.01
    if systolic_bp and systolic_bp > 140:
        weight *= 1.01

    return round(weight, 1)


def calculate_bmi(weight, height_cm):
    if weight > 0 and height_cm > 0:
        height_m = height_cm / 100
        return round(weight / (height_m ** 2), 1)
    return None


def simple_preprocess(df1):

    if "Unnamed: 0" in df1.columns:
        df1.drop(columns="Unnamed: 0",inplace=True)
    if df1['Activity']=="Mede":
       df1['Activity'].replace('Mede','Moderate',inplace=True)
    if df1['Activity']=="low":
       df1['Activity'].replace('low','Low',inplace=True)
    if df1['Anemia']=='Mild':
       df1['Anemia'].replace('Mild','Moderate',inplace=True)
    if df1['Anemia']=='Mitd':
       df1['Anemia'].replace('Mitd','Moderate',inplace=True)
    if df1['Anemia']=='Mede':
       df1['Anemia'].replace('Mede','Moderate',inplace=True)
    if df1['Anemia']=='Normat':
       df1['Anemia'].replace('Normat','Normal',inplace=True)


def ocr_max_text(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=35)

    texts = []
    for block in [11, 21, 31]:
        th = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block, 10
            )
        kernel = np.ones((2,2), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

        config = r"""
                --oem 3
                --psm 6
                -c preserve_interword_spaces=1
            """
        text = pytesseract.image_to_string(th, config=config)
        texts.append(text)

    return max(texts, key=len).strip()


def safe_search(pattern, text, cast=None, group=1):
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    value = match.group(group).strip()
    return cast(value) if cast else value


#PDF file data Extracting
def pdfDataExtraction(file_path):
    pdf = fitz.open(file_path)
    for page in pdf:
        text += page.get_text()

    name_match = re.search(r'Name\s*[:\-\n]?\s*(.+)', text)
    name = name_match.group(1).strip() if name_match else None

    data = {
                "Name": name,
                "Gender": re.search(r'Gender\s*[:\- ]?\s*(Male|Female|M|F)', text, re.IGNORECASE).group(1),
                "Age": int(re.search(r'Age\s*[:\- ]?\s*(\d+)', text).group(1)),
                "Blood Group": re.search(r'Blood\s*Group\s*[:\- ]?\s*(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)', text, re.IGNORECASE).group(1),
                "Height (cm)": int(re.search(r'Height\s*\(cm\)\s*[:\- ]?\s*(\d+)', text).group(1)),
                "Weight (kg)": int(re.search(r'Weight\s*\(kg\)\s*[:\- ]?\s*(\d+)', text).group(1)),
                "BMI": float(re.search(r'BMI\s*[:\- ]?\s*([\d.]+)', text).group(1)),
                "Body Fat (%)": float(re.search(r'Body\s*Fat\s*\(%\)\s*[:\- ]?\s*([\d.]+)', text).group(1)),
                "Systolic BP": int(re.search(r'Systolic\s*BP\s*[:\- ]?\s*(\d+)', text).group(1)),
                "Diastolic BP": int(re.search(r'Diastolic\s*BP\s*[:\- ]?\s*(\d+)', text).group(1)),
                "Cholesterol": int(re.search(r'Cholesterol\s*[:\- ]?\s*(\d+)', text).group(1)),
                "Hemoglobin": float(re.search(r'Hemoglobin\s*[:\- ]?\s*([\d.]+)', text).group(1)),
                "PPBS": int(re.search(r'PPBS\s*[:\- ]?\s*(\d+)', text).group(1)),
                "Diabetes": re.search(r'Diabetes\s*[:\- ]?\s*(Yes|No)', text, re.IGNORECASE).group(1),
                "Smoker": re.search(r'Smoker\s*[:\- ]?\s*(Yes|No)', text, re.IGNORECASE).group(1),
                "Activity": re.search(r'Activity\s*[:\- ]?\s*(.+)', text).group(1),
                "Anemia": re.search(r'Anemia\s*[:\- ]?\s*(.+)', text).group(1)
            }


            
    df = pd.DataFrame(data)
    print(df)




# Scanned Images Preprocessing Using OCR-Tesseract


def imageDataExtraction(file_path):
    pytesseract.pytesseract.tesseract_cmd = "/bin/tesseract"
    
    text = ocr_max_text(file_path)

    data = {
                    "Name": safe_search(r'Name\s*[:\-\n]?\s*(.+)', text),
                    "Gender": safe_search(r'Gender\s*[:\- ]?\s*(Male|Female|M|F)', text),
                    "Age": safe_search(r'Age\s*[:\- ]?\s*(\d+)', text, int),

                    "Blood Group": (
                        safe_search(r'Blood\s*Group\s*[:\- ]?\s*(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)', text)
                        or safe_search(r'Blood\s*Group\s*[:\- ]?\s*(\d+)', text)
                    ),

                    "Height (cm)": safe_search(r'Height\s*\(cm\)\s*[:\- ]?\s*(\d+)', text, int),
                    "Weight (kg)": safe_search(r'Weight\s*\(kg\)\s*[:\- ]?\s*(\d+)', text, int),
                    "BMI": safe_search(r'BMI\s*[:\- ]?\s*([\d.]+)', text, float) or safe_search(r'BMI\s*[:\- ]?\s*(\d+)', text, float),
                    "Body Fat (%)": safe_search(r'Body\s*Fat\s*\(%\)\s*[:\- ]?\s*([\d.]+)', text, float) or safe_search(r'Body\s*Fat\s*[:\- ]?\s*(\\d+)', text, float),
                    "Systolic BP": safe_search(r'Systolic\s*BP\s*[:\- ]?\s*(\d+)', text, int),
                    "Diastolic BP": safe_search(r'Diastolic\s*BP\s*[:\- ]?\s*(\d+)', text, int),
                    "Cholesterol": safe_search(r'Cholesterol\s*[:\- ]?\s*(\d+)', text, int),
                    "Hemoglobin": safe_search(r'Hemoglobin\s*[:\- ]?\s*([\d.]+)', text, float) or safe_search(r'Hemoglobin\s*[:\- ]?\s*(\d+)', text, float),
                    "PPBS": safe_search(r'PPBS\s*[:\- ]?\s*(\d+)', text, int),
                    "Diabetes": safe_search(r'Diabetes\s*[:\- ]?\s*(Yes|No)', text),
                    "Smoker": safe_search(r'Smoker\s*[:\- ]?\s*(Yes|No)', text),
                    "Activity": safe_search(r'Activity\s*[:\- ]?\s*(\w+)', text),
                    "Anemia": safe_search(r'Anemia\s*[:\- ]?\s*(\w+)', text)
                }


    df1 = pd.DataFrame(data)
    print(df1)





# text-based file processing 

def textDataExtraction(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

        data = {
                "Name": safe_search(r'PATIENT\s*[:\-]\s*([A-Za-z ]+)', text),

                "Gender": safe_search(r'Gender\s*[:\- ]?\s*(Male|Female|M|F)', text),
                "Age": safe_search(r'Age\s*[:\- ]?\s*(\d+)', text, int),

                "Blood Group": safe_search(
                    r'Blood\s*Group\s*[:\- ]?\s*(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)', text
                ),

                "Height (cm)": safe_search(r'Height\s*\(cm\)\s*[:\- ]?\s*(\d+)', text, int),
                "Weight (kg)": safe_search(r'Weight\s*\(kg\)\s*[:\- ]?\s*(\d+)', text, int),
                "BMI": safe_search(r'BMI\s*[:\- ]?\s*([\d.]+)', text, float),
                "Body Fat (%)": safe_search(r'Body\s*Fat\s*\(%\)\s*[:\- ]?\s*([\d.]+)', text, float),


                "Systolic BP": safe_search(
                    r'BP\s*[:\- ]?\s*(\d{2,3})\D+(\d{2,3})', text, int, group=1
                ),
                "Diastolic BP": safe_search(
                    r'BP\s*[:\- ]?\s*(\d{2,3})\D+(\d{2,3})', text, int, group=2
                ),

                "Cholesterol": safe_search(r'Cholesterol\s*[:\- ]?\s*(\d+)', text, int),
                "Hemoglobin": safe_search(r'Hemoglobin\s*[:\- ]?\s*([\d.]+)', text, float),
                "PPBS": safe_search(r'PPBS\s*[:\- ]?\s*(\d+)', text, int),

                "Diabetes": safe_search(r'Diabetes\s*[:\- ]?\s*(Yes|No)', text),
                "Smoker": safe_search(r'Smoker\s*[:\- ]?\s*(Yes|No)', text),
                "Activity": safe_search(r'Activity\s*[:\- ]?\s*(.+)', text),
                "Anemia": safe_search(r'Anemia\s*(Status)?\s*[:\- ]?\s*(.+)', text, group=2)
            }

        df3= pd.DataFrame(data)
        print(data)

