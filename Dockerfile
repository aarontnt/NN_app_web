FROM python:3.9.13-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD gunicorn app:crear_app()