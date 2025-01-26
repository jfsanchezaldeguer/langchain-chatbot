# LangChain Chatbot

Este archivo incluye información sobre:
- Documentación de configuración (interfaz de usuario del chatbot)
- Documentación de uso (interfaz de usuario del chatbot)
- Explicación de decisiones de diseño (código)

Para indicaciones sobre el despliegue y ejecución del chatbot, ver el archivo _deployment.md_.

El código completo se encuentra en el siguiente repositorio Git:
https://github.com/jfsanchezaldeguer/langchain-chatbot

## Documentación de configuración
Hay varias opciones de configuración que se pueden ir modificando desde la sidebar de la interfaz de usuario del chatbot:
- Instrucciones / Personalidad
- Temperatura
- Integrar base de conocimientos

### Instrucciones / Personalidad
Campo de texto libre donde se puede indicar una serie de instrucciones de comportamiento del asistente, así como indicaciones sobre su personalidad. 

### Temperatura
Sirve para indicar el nivel de creatividad del asistente en sus respuestas, donde 0 indica nada de creatividad, y 2 es el máximo nivel de creatividad.

### Integrar base de conocimientos
Como parte de una de las características avanzadas planteadas, marcando este checkbox se activa la integración con una base de conocimientos.
La base de conocimientos de muestra escogida ha sido un documento txt, cuya temática trata sobre la historia de España. No se ha escogido por nada en especial, sino simplemente por ser un documento que sirve para integrar base de conocmientos mediante base de datos vectorizada y codificación a embeddings de la base de conocimientos.


## Documentación de uso

El uso del chatbot es simple, el usuario escribe su consulta en la caja de texto, y hay que pulsar la tecla _intro_ para que el mensaje se envíe.

Mientras se está enviando el mensaje, la pantalla se pone en modo "RUNNING", como se indicará en la esquina superior derecha. Se recomienda no enviar nuevos mensajes hasta que la pantalla no vuelva al estado normal, es decir, hasta obtener la respuesta del chatbot.

Es posible que el primer mensaje de la conversación tarde un poco en enviarse, debido a que es ahí cuando se limpia la memoria almacenada de conversaciones anteriores.

También tardará en el primer mensaje que se envíe con el checkbox marcado _Integrar base de conocimientos_, ya que es ahí cuando el documento de texto se convierte a embeddings y se genera la base de datos vectorizada para que el asistente pueda utilizar esta información.

Los ajustes de configuración de la sidebar se pueden modificar en cualquier momento y aplican a la conversación actual.

Se recomienda probar primero la funcionalidad básica del chatbot sin marcar la opción _Integrar base de conocimientos_, ya que permite conversaciones más fluidas, al no tener que interactuar con la base de conocimientos.

## Explicación de decisiones de diseño

### Fichero _app.py_
Este fichero incluye principalmente la implementación de los elementos de frontend de la interfaz de usuario mediante Streamlit.

### Fichero _chatbot.py_
Este fichero incluye la parte backend que obtiene la respuesta a la consulta realizada por el usuario, interaccionando con los modelos LLM.
Consta principalmente de dos funciones:
- _obtener_respuesta_chatbot_informacion_interna_modelo()_: Esta función implementa la funcionalidad básica del chatbot. Incluye sistema de memoria de conversación, para que el asistente tenga un contexto con los últimos mensajes intercambiados, y así pueda relacionar preguntas escuetas con preguntas más completas realizadas previamente.
Se ha configurado un buffer de memoria con ventana de 3 iteraciones, es decir, 3 parejas de mensajes usuario-asistente.
Se utiliza una chain con un prompt para guiar al asistente en la conversación.
- _obtener_respuesta_agente_base_conocimientos_externa()_: Esta función implementa la característica avanzada de integración con una base de conocimientos. Para ello, he definido un agente que hace uso de una Tool, _consulta_base_conocimiento_, la cual invoca la consulta en la base de datos de vectores sobre la información vectorizada que trata de la historia de España.

### Fichero _utils.py_
Este fichero incluye:
- Funciones de apoyo
- Funciones que sirven para encapsular accesos a infraestructura, como la gestión de la persistencia de memoria de conversación y la creación y acceso a la base de datos de vectores.
- Funciones que sirven para encapsular los modelos LLM y de embedding específicos utilizados, de forma que en caso de querer implementar integración con otros modelos, solo haya que modificar estas pequeñas funciones de este fichero, y sea lo más transparente posible para las funciones principales de _chatbot.py_, donde está el flujo principal del funcionamiento del sistema, que se abstrae de los modelos LLM específicos.