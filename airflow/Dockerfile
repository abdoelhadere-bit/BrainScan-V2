FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AIRFLOW_HOME=/opt/brainscan/airflow

WORKDIR /opt/brainscan

COPY requirements.txt /opt/brainscan/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /opt/brainscan/requirements.txt && pip install --no-cache-dir apache-airflow==2.9.3

COPY airflow /opt/brainscan/airflow
COPY src /opt/brainscan/src
COPY models /opt/brainscan/models

RUN mkdir -p /opt/brainscan/airflow/dags

EXPOSE 8080

CMD ["bash", "-lc", "airflow db init && airflow webserver --port 8080"]
