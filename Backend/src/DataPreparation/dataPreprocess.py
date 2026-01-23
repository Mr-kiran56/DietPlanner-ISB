import numpy as np
import pandas as pd
import os
import yaml
import logging
from numpy import dtype
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "raw_csv"/"raw_data.csv"

# logging configuration
logger = logging.getLogger('dataPreprocess')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    try:
        with open(params_path,'r')as file:
            params=yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s',params_path)
        return params
    except FileNotFoundError:
        logger.error('File not Found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def assign_single_label(row):

    sys = row['Systolic BP']
    dia = row['Diastolic BP']
    ppbs = row['PPBS']
    bmi = row['BMI']
    chol = row['Cholesterol']
    hb = row['Hemoglobin']
    smoker = row['Smoker']

    # Hypertension (highest priority)
    if sys >= 160 or dia >= 100:
        return 3
    elif sys >= 140 or dia >= 90:
        return 2
    elif sys >= 120 or dia >= 80:
        return 1

    # Diabetes
    if ppbs >= 200:
        return 5
    elif ppbs >= 140:
        return 4

    # Metabolic Syndrome
    if bmi >= 30 and chol >= 240 and ppbs >= 140:
        return 10

    # Cardiovascular risk
    if chol >= 240 or (chol >= 200 and smoker == 'Yes'):
        return 7

    # Obesity
    if bmi >= 30:
        return 6

    # Anemia
    if hb < 10:
        return 9
    elif hb < 12:
        return 8

    return 0

# from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
# import pandas as pd


def Preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop useless cols
    df = df.drop(columns=["Blood Group", "Name"], errors="ignore")

    # ===== Activity =====
    if 'Activity' in df.columns:
        df['Activity'] = df['Activity'].astype(object)

        ordinal = OrdinalEncoder(
            categories=[["Low", "Moderate", "High"]],
            handle_unknown="use_encoded_value",
            unknown_value=np.nan
        )

        mask = df['Activity'].notna()
        if mask.any():
            df.loc[mask, 'Activity'] = ordinal.fit_transform(
                df.loc[mask, ['Activity']]
            ).ravel()

        df['Activity'] = pd.to_numeric(df['Activity'], errors="coerce")

    # ===== Anemia =====
    if 'Anemia' in df.columns:
        df['Anemia'] = df['Anemia'].astype(object)

        anemia_encoder = OrdinalEncoder(
            categories=[["Normal", "Mild", "Moderate", "Severe"]],
            handle_unknown="use_encoded_value",
            unknown_value=np.nan
        )

        mask = df['Anemia'].notna()
        if mask.any():
            df.loc[mask, 'Anemia'] = anemia_encoder.fit_transform(
                df.loc[mask, ['Anemia']]
            ).ravel()

        df['Anemia'] = pd.to_numeric(df['Anemia'], errors="coerce")

    # ===== Label Encoding =====
    for col in ['Gender', 'Diabetes', 'Smoker']:
        if col in df.columns:
            df[col] = df[col].astype(object)
            if df[col].notna().any():
                le = LabelEncoder()
                df.loc[df[col].notna(), col] = le.fit_transform(
                    df.loc[df[col].notna(), col]
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:

    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise



def main(filename):
    try:
        PARAMS_PATH = PROJECT_ROOT / "params.yaml"
        params = load_params(params_path=PARAMS_PATH)

        test_size = params['data_ingestion']['test_size']
        logger.debug("Starting data preprocessing...")

        data = pd.read_csv(filename)

        df = Preprocess_data(data, assign_single_label)

        train_processed_data, test_processed_data = train_test_split(
            df, test_size=test_size, random_state=42
        )

        save_data(train_processed_data, test_processed_data, data_path='./data')

    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main(DATA_PATH)

        




   
