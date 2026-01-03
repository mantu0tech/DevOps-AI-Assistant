FROM python:3.10-slim AS build

WORKDIR /app 

RUN rm -rf /var/lib/apt/lists/*
COPY requirements.txt .

RUN pip install --no-cache-dir  --prefix=/install -r requirements.txt


# stage ========== 2 ===========
FROM python:3.10-slim

WORKDIR /app

COPY --from=build /install /usr/local/
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY main.py .

EXPOSE 8501

CMD [ "streamlit", "run", "main.py"]
