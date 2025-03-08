import os
import sys
import pandas as pd
import arff

from src.logging import get_logger
from src.exception import CustomException
from src.utils.config import DataIngestionConfig

logging = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiiate_data_ingestion(self):
        """
        Initiating Data Ingestion Process
        1. Read the arff data file using 'data_path' from the config
        2. Convert the dataset to pandas Dataframe
        3. Save the DataFrame as a CSV file at 'raw_data_path'
        """
        logging.info("Data Ingestion Process Commences")
        try:
            # Loading the Dataset
            with open(self.data_ingestion_config.data_path,"r") as f:
                 dataset=arff.load(f)

            df = pd.DataFrame(dataset['data'],columns=[attr[0] for attr in dataset['attributes']])
            logging.info("Reading Dataset as a DataFrame")

            # Saving the Dataset in the Raw Data Path as a CSV file
            to_csv = df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info("Saving Dataset as a CSV file")

            return to_csv

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    ingestion = DataIngestion()
    csv_path = ingestion.initiiate_data_ingestion()

