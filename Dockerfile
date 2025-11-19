FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY test.py .
COPY app.py .      
COPY savedmodel.pth .


EXPOSE 5000

CMD ["python", "app.py"]
