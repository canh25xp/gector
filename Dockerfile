FROM python:3.10

WORKDIR /app

# copy and install requirements first to leverage Docker cache
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python","api.py"]

EXPOSE 3000
