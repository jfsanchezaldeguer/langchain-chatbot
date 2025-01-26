# Usamos una imagen base de Python
FROM python:3.12-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos los archivos de la aplicación al contenedor
COPY requirements.txt /app/

# Instalamos las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código de la aplicación dentro del contenedor
COPY ./src /app/src
COPY ./docs /app/docs
COPY ./storage /app/storage
COPY .env /app/

# Exponemos el puerto por donde Streamlit servirá la aplicación (por defecto 8501)
EXPOSE 8501

# Comando para ejecutar la aplicación de Streamlit
CMD ["streamlit", "run", "src/app.py"]