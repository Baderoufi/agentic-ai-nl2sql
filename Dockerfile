FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# أدخل كل ما تحتاج لتثبيت PyTorch + sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# انسخ الملفات المطلوبة فقط
COPY src/ /app/src/
COPY schema.json /app/schema.json
COPY api.py /app/api.py
COPY nl2sql_core.py /app/nl2sql_core.py

# إذا كنت تحتاج vector DB يجب mount من خارج الحاوية وليس نسخ
# مثال:
# VOLUME /app/vector_sql_schema3

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
