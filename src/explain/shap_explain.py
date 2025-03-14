import os
import sys
import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import load

from src.logging import get_logger
from src.exception import CustomException
from src.utils.config import ModelTrainingConfig
from src.utils.config import ShapExplanationConfig

# Initialize logger for tracking execution
logging = get_logger(__name__)

class ShapExplanation:
    def __init__(self):
        self.shap_explanation_config = ShapExplanationConfig()
        self.model_training_config = ModelTrainingConfig()

    def generate_shap_explanation(self):
        """
        Generates a SHAP explanation summary plot for the best trained model.

        Process:
        1. Load the test dataset from the file specified in the configuration.
        2. If available, remove the target column (e.g., 'deposit') to obtain the feature set.
        3. Load the best model from disk.
        4. Initialize a SHAP explainer:
            - Use TreeExplainer for tree-based models.
            - Fall back to KernelExplainer for other model types.
        5. Compute SHAP values for the test data.
            - For binary classification, if the explainer returns a list of arrays,
            select the values corresponding to the positive class (index 1).
        6. Create and save a SHAP summary plot as a PNG file.
        """
        try:
            # Load the test dataset
            test_df = pd.read_csv(self.model_training_config.test_path)
            logging.info(f"Loaded test dataset from: {self.model_training_config.test_path}")

            # Separate features (X_test) from the target
            target_column = "deposit"
            X_test = test_df.drop(columns=[target_column])

            # Load the best trained model from disk
            best_model = load(self.model_training_config.best_model)
            logging.info(f"Loaded best model from: {self.model_training_config.best_model}")

            # Initialize the SHAP explainer.
            explainer = shap.TreeExplainer(best_model)
            logging.info("Using SHAP TreeExplainer.")
            
            # Compute SHAP values for the test data.
            shap_values = explainer.shap_values(X_test)
            # For binary classification, SHAP returns a list of arrays; choose the one for the positive class.
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[1]
            else:
                shap_values_to_plot = shap_values

            # Generate a SHAP summary plot (using a bar plot for feature importance)
            shap.summary_plot(shap_values_to_plot, X_test, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance")
            
            # Define the path to save the SHAP summary plot.
            shap_plot_path = self.shap_explanation_config
            plt.savefig(shap_plot_path)
            plt.close()
            logging.info(f"SHAP summary plot saved to: {shap_plot_path}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    explanation = ShapExplanation()
    explanation_png = explanation.generate_shap_explanation()
