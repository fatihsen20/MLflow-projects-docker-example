FROM python:3.10
COPY train_copy.py .
COPY breast_cancer.csv .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt