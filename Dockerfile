FROM python:3.9.13-slim

WORKDIR /app

# Actualizar pip primero
RUN pip install --upgrade pip

COPY . .

# Instalar dependencias (asegúrate de que requirements.txt no tenga conflictos)
RUN pip install -r requirements.txt

# Forma correcta (usa esta):
CMD ["gunicorn", "app:crear_app"]