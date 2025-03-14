import sys
import os
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    save_dir: str=os.path.join('artifacts')# Directory where artifacts (e.g., the CSV file) will be saved.
    data_path: str=os.path.join('Notebook', 'dataset', 'Bank_Marketing_Dataset.arff') # Path to the input ARFF dataset.
    raw_data_path: str=os.path.join('artifacts','Bank_Marketing_Dataset.csv') # Path where the raw data will be saved in CSV format.
    os.makedirs(save_dir,exist_ok=True) # Create the save directory if it doesn't already exist.

@dataclass
class DataTransformationConfig:
    input_path: str=os.path.join('artifacts','Bank_Marketing_Dataset.csv') # Path to the input CSV file used for transformation.
    train_path: str=os.path.join('artifacts','train_data.csv') # Path to save the transformed training data.
    test_path: str=os.path.join('artifacts','test_data.csv')# Path to save the transformed testing data.

@dataclass
class ModelTrainingConfig:
    MLFLOW_TRACKING_URI: str= 'https://dagshub.com/OluomaOji/Bank_LongTerm_Investment_Detection_with_MLFLOW.mlflow' # MLflow tracking server URI.
    #MLFLOW_TRACKING_USERNAME: str = os.environ.get("MLFLOW_TRACKING_USERNAME")# MLflow tracking username.
    #MLFLOW_TRACKING_PASSWORD: str = os.environ.get("MLFLOW_TRACKING_PASSWORD")  # MLflow tracking password.
    MLFLOW_TRACKING_USERNAME: str = "OluomaOji"
    MLFLOW_TRACKING_PASSWORD: str = "760c90844a305272f1981be1879bceef551c3385"
    train_path: str=os.path.join('artifacts','train_data.csv')# Path to the training data file.
    test_path: str=os.path.join('artifacts','test_data.csv') # Path to the testing data file.
    best_model: str=os.path.join('artifacts','model.pkl') # Path to save the best model after training.

@dataclass
class ShapExplanationConfig:
    shap_png: str=os.path.join("artifacts", "shap_summary.png") # Path to Shap Explanation png.