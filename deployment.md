<!--deployment.md-->
Este fichero incluye indicaciones para el despliegue y ejecución del chatbot.


## Instrucciones de instalación

Hay que configurar las variables de entorno necesarias para las APIs. Para ello hay que crear fichero .env en base a .env.example, y especificar en él las variables de entorno que se vayan a utilizar:
   - En caso de utilizar OpenAI:
     - OPENAI_API_KEY
     - OPENAI_MODEL_NAME
   - En caso de utilizar HuggingFace:
     - HUGGINGFACEHUB_API_TOKEN
     - HUGGINGFACEHUB_REPO_ID

Una vez hecho esto, hay varias formas de instalación y ejecución:
- Mediante contenedor Docker
- Sin contenedores Docker

### Mediante contenedores Docker
1. Compilación de la imagen Docker:
   ```bash
    docker build -t [my-image:my-tag] .
   ```
    Ejemplo:
   ```bash
   docker build -t langchain-chatbot .
   ```
2. Ejecución del container en base a la imagen Docker compilada con _docker run_:
   ```bash
   docker run -v ./storage:/app/storage -p 8501:8501 [my-image:my-tag]
   ```
    Ejemplo:
   ```bash
   docker run -v ./storage:/app/storage -p 8501:8501 langchain-chatbot
   ```
3. O bien, mediante _docker-compose_:
   ```bash
   docker-compose -f docker-compose.yml up
   ```

### Ejecución sin contenedores Docker
1. Crear un entorno virtual de Python:
   ```bash
    python -m venv
   ```
2. Activar el entorno virtual:
    ```bash
    source venv/bin/activate
    ```
2. Instalar las dependencias necesarias:
    ```bash
    pip install -r requirements.txt
    ```
3. Ejecutar script de inicio:
   ```bash
    streamlit run src/app.py
   ```

Con cualquiera de las 2 opciones se puede acceder a la interfaz del chatbot a través del navegador web:
    http://localhost:8501/


El fichero _docker-compose-dev.yml_ se utiliza solamente para la fase de desarrollo.