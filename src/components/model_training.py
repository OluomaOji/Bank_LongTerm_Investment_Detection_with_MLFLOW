import os
import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from src.logging import get_logger
from src.exception import CustomException
from src.utils.config import ModelTrainingConfig

logging = get_logger(__name__)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from joblib import dump

class ModelTraining:
    def __init__(self):
        self.model_training_configuration = ModelTrainingConfig()

    def initialising_model_training(self):
        """
        Initialising Model Training
        1. Load train and test datasets from the CSV files specified in the configuration.
        2. Define the models in a dictionary.
        3. Separate the feature set (X) from the target (y) for both training and testing.
        4. Set up MLflow for tracking experiments:
             - Define the tracking URI, username, and password.
             - Set the experiment name.
        5. Loop through each model:
             a. Train the model on the training data.
             b. Predict on the test data and calculate metrics such as accuracy, F1 score, ROC-AUC, and confusion matrix.
             c. Log parameters, metrics, and model artifacts (including the model signature) with MLflow.
        6. Determine the best model based on the ROC-AUC score.
        7. Save the best model to .pkl
        """
        try:
            #1. Load train and test datasets from the CSV files specified in the configuration.
            train_df = pd.read_csv(self.model_training_configuration.train_path)
            test_df = pd.read_csv(self.model_training_configuration.test_path)
            logging.info("Train/Test datasets loaded successfully.")

            #2. Define the models in a dictionary.
            models = {
            "Logistic Regression": LogisticRegression(solver="liblinear", max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
            }
            
            #3. Separate the feature set (X) from the target (y) for both training and testing.
            target_column = 'deposit'

            X_train = train_df.drop(columns=target_column)
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=target_column)
            y_test = test_df[target_column]

            # Setting the metric for the best model
            best_model = None
            best_model_name = None
            best_roc_auc = float("-inf")

            #4. Set up MLflow for tracking experiments:
            #   - Define the tracking URI, username, and password.
            #   - Set the experiment name.

            # Tracking URI, Username, Password
            os.environ["MLFLOW_TRACKING_URI"] = self.model_training_configuration.MLFLOW_TRACKING_URI
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.model_training_configuration.MLFLOW_TRACKING_USERNAME
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.model_training_configuration.MLFLOW_TRACKING_PASSWORD

            # Set up MLFLOW
            mlflow_tracking_uri = self.model_training_configuration.MLFLOW_TRACKING_URI
            mlflow.set_tracking_uri(mlflow_tracking_uri)

            # Setting the Experiment Name
            experiment_name = "Bank_Loan_Detector_Experiment"
            mlflow.set_experiment(experiment_name)

            # Train and Evaluate Each Model, Log Metrics and Save Artifacts.
            for model_name, model in models.items():
                logging.info(f"\n======= Training Modle: {model_name} ======")

                with mlflow.start_run(run_name = model_name):
                    #5. Loop through each model:
                        #a. Train the model on the training data.
                        #b. Predict on the test data and calculate metrics such as accuracy, F1 score, ROC-AUC, and confusion matrix.
                        #c. Log parameters, metrics, and model artifacts (including the model signature) with MLflow.
                    
                    # Train the Model
                    model.fit(X_train,y_train)
                    # Predictions
                    y_pred = model.predict(X_test)
                    # Metrics
                    acc = accuracy_score(y_test,y_pred)
                    f1 = f1_score(y_test,y_pred)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_proba)
                    cls_report = classification_report(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)

                    # Log metrics
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_metric("roc_auc", roc_auc)

                    # Log confusion matrix details (optional)
                    mlflow.log_metric("tn", cm[0,0])
                    mlflow.log_metric("fp", cm[0,1])
                    mlflow.log_metric("fn", cm[1,0])
                    mlflow.log_metric("tp", cm[1,1])

                    logging.info(f"Model: {model_name} | "
                                 f"Accuracy: {acc:.4f} | "
                                 f"F1: {f1:.4f} | "
                                 f"ROC-AUC: {roc_auc}")
                    logging.info(f"Confusion Matrix:\n{cm}")
                    logging.info(f"Classification Report:\n{cls_report}")

                    # Infer the model Signature
                    signature = infer_signature(X_train,model.predict(X_train))
                    # Log the model artifact
                    mlflow.sklearn.log_model(
                        model, 
                        artifact_path="artifacts",
                        signature=signature,
                        input_example=X_train,
                        )   

                    # End run (this can also happen automatically upon exit of the `with` block)
                    mlflow.end_run()

                # 6. Determine the best model based on the ROC-AUC score.
                current_roc = roc_auc if roc_auc else 0.0
                if current_roc > best_roc_auc:
                    best_roc_auc = current_roc
                    best_model = model
                    best_model_name = model_name

            logging.info(f"\nBest Model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")

            #7. Save the best model to .pkl
            
            if best_model is not None:
                from joblib import dump
                model_path = self.model_training_configuration.best_model
                dump(best_model, model_path)
                logging.info(f"Best model saved to: {model_path}")


        except Exception as e:
            raise CustomException(e,sys)