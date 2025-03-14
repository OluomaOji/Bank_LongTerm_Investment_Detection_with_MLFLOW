# Bank Longterm Investment Detector

https://banklongterminvestor.streamlit.app/
An end-to-end machine learning project that predicts whether a bank client will invest longterm. The project covers data ingestion, transformation, model training, evaluation with explainability (using SHAP), and deployment through a Streamlit app. Experiment tracking is performed using MLflow, and CI/CD is set up via GitHub Actions.

## Table of Contents

- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
  - [Running the Pipeline](#running-the-pipeline)
  - [Running the Streamlit App](#running-the-streamlit-app)
- [CI/CD Pipeline](#cicd-pipeline)
- [Model Explainability](#model-explainability)

## Project Overview

This project builds a predictive model for determining if a bank customer is likely to subscribe to a deposit. The project is organized into several steps:
- **Data Ingestion**: Reads raw ARFF data and converts it to CSV.
- **Data Transformation**: Cleans the data, handles missing values, performs feature engineering (e.g., one-hot encoding).
- **Model Training**: Trains multiple candidate models (e.g., Logistic Regression, Random Forest, SVM, Decision Tree) and logs metrics to MLflow.
- **Model Evaluation & Explainability**: Evaluates model performance using metrics like accuracy, F1, and ROC-AUC; explains model predictions with SHAP.
- **Deployment**: Provides a Streamlit web interface for interactive predictions.
- **CI/CD**: Automated testing and deployment using GitHub Actions.

## File Structure
Bank_Loan_Detector/ ├── artifacts/ │ ├── Bank_Marketing_Dataset.csv # Raw data output from ingestion │ ├── transformed_data.csv # Cleaned & transformed data │ ├── train_data.csv, test_data.csv # Train/Test splits │ └── model.pkl # The final saved best model ├── Logs/ │ └── bank_longterm_investment.logs # Log file capturing pipeline execution logs ├── Notebook/ │ └── bank_longterm_investment.ipynb # Jupyter Notebook for EDA and prototyping ├── src/ │ ├── components/ │ │ ├── data_ingestion.py # Data ingestion code (e.g., ARFF to CSV) │ │ ├── data_transformation.py # Data cleaning and feature engineering code │ │ └── model_training.py # Model training and MLflow logging │ ├── pipeline/ │ │ └── pipeline.py # Orchestrates the end-to-end pipeline │ ├── explain/ │ │ └── shap-explain.py # Generates SHAP plots for model explainability │ ├── utils/ │ │ └── config.py # Configuration dataclasses for file paths and credentials │ ├── logging.py # Custom logger configuration │ ├── exception.py # Custom exception handling │ └── app.py # Streamlit app for real-time predictions ├── .github/ │ └── workflows/ │ └── ci.yml # GitHub Actions workflow file for CI/CD ├── requirements.txt # Python dependencies list └── README.md

## Usage
Running the Pipeline
The full ML pipeline (data ingestion, transformation, and model training) is orchestrated by the pipeline script.

### Run the entire pipeline with:
python -m src.pipeline.pipeline
This will:
Ingest raw data and save it to the artifacts folder.
Transform and clean the data.
Split data into train/test sets.
Train and evaluate multiple models while logging metrics to MLflow.
Save the best model as artifacts/model.pkl.

### Running the Streamlit App
To run the interactive prediction interface:
streamlit run app.py

## CI/CD Pipeline
The project uses GitHub Actions for continuous integration and deployment. The CI/CD configuration is located in .github/workflows/ci.yml and performs the following:
Checks out the repository.
Sets up Python 3.10.
Installs dependencies from requirements.txt.
Runs linting (using flake8) and unit tests (using pytest).
Executes the full pipeline.

## Model Explainability
Model explainability is performed using SHAP. The module src/explain/shap-explain.py:
Loads the best trained model and test dataset.
Uses SHAP’s TreeExplainer to compute SHAP values.
Generates a summary plot of feature importance, saved as artifacts/shap_summary.png.

