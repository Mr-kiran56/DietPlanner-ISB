import numpy as np
import pandas as pd
import os
import yaml
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "originalCSV" / "Medical-Report-Data-FINAL.csv"

#logging Configuration
logger=logging.getLogger('dataIgnestion')
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler=logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter=logging.Formatter('%(asctime)s-%(name)s -%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_url: str)->pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logger.debug('Data loaded from %s ',data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexcepted error occured while loading the data: %s ',e)
        raise



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()  # avoid chained assignment issues

        # ---------- Drop junk column ----------
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns='Unnamed: 0')

        # ---------- AGE ----------
        if 'Age' in df.columns:
            df.loc[df['Age'] > 150, 'Age'] = df.loc[df['Age'] > 150, 'Age'] // 10
            df['Age'] = df['Age'].astype('Int64')  # allows NaN

        # ---------- BMI ----------
        if 'BMI' in df.columns:
            df.loc[df['BMI'] > 46, 'BMI'] = df.loc[df['BMI'] > 46, 'BMI'] // 10
            df['BMI'] = df['BMI'].astype(float)

        # ---------- SYSTOLIC BP ----------
        if 'Systolic BP' in df.columns:
            df.loc[df['Systolic BP'] < 20, 'Systolic BP'] = (
                df.loc[df['Systolic BP'] < 20, 'Systolic BP'] * 10
            )

        # ---------- ANEMIA TEXT FIX ----------
        if 'Anemia' in df.columns:
            df['Anemia'] = df['Anemia'].replace({
                'Dietary iron': 'Mild',
                'Iron therapy needed': 'Moderate'
            }).astype(object)

        return df

    
    except KeyError as e:

        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(csv_path:str,data_csv:pd.DataFrame)->None:
    try:
        raw_data_csv=os.path.join(csv_path,'raw_csv')

        os.makedirs(raw_data_csv,exist_ok=True)

        data_csv.to_csv(os.path.join(raw_data_csv,"raw_data.csv"),index=False)

        logger.debug('New CSV data saved to %s', csv_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        df = load_data(DATA_PATH)

        
        final_df = preprocess_data(df)


        save_data(PROJECT_ROOT / "data", final_df)

        
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()


















