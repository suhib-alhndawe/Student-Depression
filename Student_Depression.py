import pandas as pd
import random
import numpy as np
import mlflow
import mlflow.keras
import mlflow.sklearn
from sklearn.metrics import accuracy_score

data_path="Student Depression Dataset.csv"
my_data=pd.read_csv(data_path)
print(my_data.head())
print("-------------------------------------------------------------------------------------------"
      "-------------------------------------------------------------------------------------------"
      "\n------------------------------------------------------------------------------------------"
      "--------------------------------------------------------------------------------------------")
print(my_data.info())
print("-------------------------------------------------------------------------------------------"
      "-------------------------------------------------------------------------------------------"
      "\n------------------------------------------------------------------------------------------"
      "--------------------------------------------------------------------------------------------")
print(my_data.isnull().sum())
print("-------------------------------------------------------------------------------------------"
      "-------------------------------------------------------------------------------------------"
      "\n------------------------------------------------------------------------------------------"
      "--------------------------------------------------------------------------------------------")
print(my_data.duplicated().sum())
print("-------------------------------------------------------------------------------------------"
      "-------------------------------------------------------------------------------------------"
      "\n------------------------------------------------------------------------------------------"
      "--------------------------------------------------------------------------------------------")
print(my_data.describe())
print("-------------------------------------------------------------------------------------------"
      "-------------------------------------------------------------------------------------------"
      "\n------------------------------------------------------------------------------------------"
      "--------------------------------------------------------------------------------------------")

my_data.dropna(inplace=True)
print(my_data.isnull().sum())
print("-------------------------------------------------------------------------------------------"
      "-------------------------------------------------------------------------------------------"
      "\n------------------------------------------------------------------------------------------"
      "--------------------------------------------------------------------------------------------")


from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for column in my_data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    my_data[column] = le.fit_transform(my_data[column])
    label_encoders[column] = le
print(my_data.info())
print("-------------------------------------------------------------------------------------------"
      "-------------------------------------------------------------------------------------------"
      "\n------------------------------------------------------------------------------------------"
      "--------------------------------------------------------------------------------------------")

from sklearn.model_selection import train_test_split
my_data.drop(columns=["id", "City", "Profession"], inplace=True)
X = my_data.drop("Depression", axis=1)
y = my_data["Depression"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X.info())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numeric_cols = ["CGPA", "Work/Study Hours", "Financial Stress"]
my_data[numeric_cols] = scaler.fit_transform(my_data[numeric_cols])
print("-------------------------------------------------------------------------------------------"
      "-------------------------------------------------------------------------------------------"
      "\n------------------------------------------------------------------------------------------"
      "--------------------------------------------------------------------------------------------")
print(my_data.head())
print("-------------------------------------------------------------------------------------------"
      "-------------------------------------------------------------------------------------------"
      "\n------------------------------------------------------------------------------------------"
      "--------------------------------------------------------------------------------------------")
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(32, 16, 8),activation='relu',
                    solver='adam',
                    max_iter=1500,
                    random_state=42)

mlp.fit(X_train, y_train.values.ravel())
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
report=classification_report(y_test,predictions,output_dict=True)
print(report)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
experiment_name = "Student Depression Classifier"

# Create experiment if it does not exist
try:
    mlflow.create_experiment(experiment_name)
except:
    print(f"Experiment '{experiment_name}' already exists.")

mlflow.set_experiment(experiment_name)

# Start the MLflow run
with mlflow.start_run(run_name="MLP Classifier Run") as mlflow_run:
    # Train your model
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)

    # Log the model
    mlflow.sklearn.log_model(mlp, "model")

    # Log hyperparameters
    mlflow.log_param("hidden_layer_sizes", (32, 16, 8))
    mlflow.log_param("activation", "relu")
    mlflow.log_param("solver", "adam")
    mlflow.log_param("max_iter", 1500)

    # Log metrics
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_metric("accuracy", mlp.score(X_test, y_test))
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("test_size", len(X_test))
    # Random Baseline
    random_predictions = [random.choice([0, 1]) for _ in range(len(y_test))]
    random_accuracy = accuracy_score(y_test, random_predictions)
    mlflow.log_metric("random_accuracy", random_accuracy)

    # Simple Heuristic
    most_frequent_class = y_train.value_counts().idxmax()
    heuristic_predictions = [most_frequent_class] * len(y_test)
    heuristic_accuracy = accuracy_score(y_test, heuristic_predictions)
    mlflow.log_metric("heuristic_accuracy", heuristic_accuracy)

    # Zero Rule Baseline
    zero_rule_predictions = [y_train.value_counts().idxmax()] * len(y_test)
    zero_rule_accuracy = accuracy_score(y_test, zero_rule_predictions)
    mlflow.log_metric("zero_rule_accuracy", zero_rule_accuracy)

    # Human Baseline (if available)
    human_baseline_predictions = [0] * len(y_test)  # Example: Classifying all as 0
    human_baseline_accuracy = accuracy_score(y_test, human_baseline_predictions)
    mlflow.log_metric("human_baseline_accuracy", human_baseline_accuracy)
    print(f"Human Baseline Accuracy: {human_baseline_accuracy}")

    print(f"MLP Model Accuracy: {accuracy}")
    print(f"Random Baseline Accuracy: {random_accuracy}")
    print(f"Simple Heuristic Accuracy: {heuristic_accuracy}")
    print(f"Zero Rule Baseline Accuracy: {zero_rule_accuracy}")
    print(f"Human Baseline Accuracy: {human_baseline_accuracy}")
    # Log the classification report
    report = classification_report(y_test, predictions, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Avoid logging accuracy scores (which is a float)
            for metric, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric}", value)

    # Print the MLflow run ID
    mlflow_run_id = mlflow_run.info.run_id
    print("MLFlow Run ID: ", mlflow_run_id)