import os
import sys

from src.exception import CustomException
from src.logging import get_logger

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.explain.shap_explain import ShapExplanation

logging = get_logger(__name__)

class RunDataPipeline:
    def __init__(self):
        pass

    def initiate_data_pipleine(self):
        try:
            logging.info("====== Starting the Pipeline ========")

            # 1) Data Ingestion
            logging.info("Starting Data Ingestion")
            ingestion = DataIngestion()
            csv_path = ingestion.initiiate_data_ingestion()
            logging.info(f"Data Ingestion completed. Raw data path: {csv_path}")

            # 2) Data Transformation
            logging.info("Starting Data Transformation...")
            transformation = DataTransformation()
            transformed_file = transformation.initialising_data_transformation()
            logging.info(f"Data Transformation completed. Transformed data path: {transformed_file}")
            
            # 3) Model Training
            logging.info("Starting Model Training...")
            trainer = ModelTraining()
            trainer.initialising_model_training()
            logging.info("Model Training completed.")

            # 4) Shap Explanation
            logging.info("Starting Shap Explanation")
            explanation = ShapExplanation()
            explanation_png = explanation.generate_shap_explanation()

        except Exception as e:
            CustomException(sys,e)

if __name__ == "__main__":
    pipeline = RunDataPipeline()
    pipeline.initiate_data_pipleine()
