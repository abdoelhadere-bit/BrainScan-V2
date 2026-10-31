FROM python:3.11-slim

WORKDIR /mlflow

RUN pip install --upgrade pip && pip install mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlruns"]
