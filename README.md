# LangChain Chatbot

## Instrucciones de instalación
Hay dos formas de ejecución: mediante contenedores Docker, o mediante ejecución del script en local.

### Ejecución mediante contenedores Docker
1. Compilación de la imagen Docker:
    docker build -t [tag] .
    Ejemplo: docker build -t langchain-chatbot .
2. Ejecución del container en base a la imagen Docker compilada:
    docker run -p 8501:8501 [tag]
    Ejemplo: docker run -p 8501:8501 langchain-chatbot
3. Acceso a la aplicación web:
    Network URL: http://172.17.0.2:8501

## Documentación de configuración
