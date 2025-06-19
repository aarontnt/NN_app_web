FROM python:3.9.13-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para TensorFlow
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip primero
RUN pip install --upgrade pip

COPY . .

# Instalar dependencias
RUN pip install -r requirements.txt

# Configuración óptima para Gunicorn + TensorFlow
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "120", "--workers", "1", "app:app"]