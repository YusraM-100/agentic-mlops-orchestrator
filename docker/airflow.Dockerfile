# Airflow 3 + the libraries the training DAG needs.
FROM apache/airflow:3.3.2-python3.12
RUN pip install --no-cache-dir \
    "mlflow-skinny==3.15.2" \
    "scikit-learn==1.8.0" \
    "pandas==2.3.3" \
    "joblib==1.5.2" \
    "pyyaml"
