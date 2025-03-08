import os
import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logging import get_logger
from src.exception import CustomException
from src.utils.config import DataTransformationConfig

logging = get_logger(__name__)

class DataTransformation:
    def __init__(self):
        """
        Initiating the Data Transformation Configuration
        # 1) Load dataset
        # 2) Identify target column and convert yes/no to 1/0
        # 3) Define numeric & categorical columns
        # 4) Build numeric pipeline
        # 5) Build categorical pipeline
        # 6) Combine using ColumnTransformer
        # 7) Fit and transform X
        # 9) Save final DataFrame
        """
        self.data_transformation_config = DataTransformationConfig()

    def initialising_data_transformation(self):
        try:
            df = pd.read_csv(self.data_transformation_config.input_path)
            logging.info('Loading the Business Marketing CSV dataset')

            target_column = 'deposit'
            # Convert target to binary (yes -> 1, no -> 0)
            df[target_column] = df[target_column].map({"yes": 1, "no": 0})
            
            # Separate X (features) and y (target)
            X = df.drop(columns=[target_column])
            y = df[target_column]

            numerical_columns = [
                'age',
                'balance',
                'day',
                'duration',
                'campaign',
                'pdays',
                'previous'
                ]
            categorical_columns =[
                'job',
                'marital',
                'education',
                'default',
                'housing',
                'loan',
                'contact',
                'month',
                'poutcome',
                ]
            
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")), ## Handling Missing Values
                    ("scaler",StandardScaler()),# Handling Standard Scaler
                ]
            )
            # 5) Build categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder())
                ]
            )

            # 6) Combine using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numerical_pipeline, numerical_columns),
                    ("cat_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            # 7) Fit and transform X
            X_transformed = preprocessor.fit_transform(X)

            # Retrieve column names for OneHotEncoder
            ohe = preprocessor.named_transformers_["cat_pipeline"].named_steps["one_hot"]
            ohe_feature_names = list(ohe.get_feature_names_out(categorical_columns))

            # Final column names = numeric_columns + expanded one-hot columns
            final_columns = numerical_columns + ohe_feature_names

            # 8) Construct a new DataFrame with columns
            X_transformed_df = pd.DataFrame(X_transformed, columns=final_columns)

            # Reattach the target
            X_transformed_df[target_column] = y.values
            logging.info("Feature Engineering Completed")

            
            
            X_transformed_df.to_csv(self.data_transformation_config.save_path, index=False)
            logging.info('Data Preprocessing Completed')
            return self.data_transformation_config.save_path
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    transformation = DataTransformation()
    transformed_file = transformation.initialising_data_transformation()



