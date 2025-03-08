import sys
import os
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    save_dir: str=os.path.join('artifacts')
    data_path: str=os.path.join('Notebook', 'dataset', 'Bank_Marketing_Dataset.arff')
    raw_data_path: str=os.path.join('artifacts','Bank_Marketing_Dataset.csv')
    os.makedirs(save_dir,exist_ok=True)

@dataclass
class DataTransformationConfig:
    input_path: str=os.path.join('artifacts','Bank_Marketing_Dataset.csv')
    save_path: str=os.path.join('artifacts','transformed_data.csv')