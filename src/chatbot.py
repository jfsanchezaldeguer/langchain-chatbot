from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import load_tools,initialize_agent,AgentType,create_react_agent,AgentExecutor,tool,Tool
from huggingface_hub.utils import HfHubHTTPError
from utils import (persistir_memoria_conversacion, cargar_memoria_conversacion, obtener_tool_consulta_base_conocimiento,
                   inicializacion_modelo_llm, inicializacion_chat_model, SYSTEM_PROMPT, CONVERSATION_HISTORY_PROMPT)
from pydantic import ValidationError


# Obtiene la respuesta a partir de la información con la que el modelo fue entrenado, sin acceso a base de conocimiento externa.
def obtener_respuesta_chatbot_informacion_interna_modelo(user_input: str, custom_prompt: str, temperatura: float):

    # Inicialización del modelo
    try:
        llm = inicializacion_modelo_llm(temperatura)
        chat_model = inicializacion_chat_model(llm)
    except OSError as e:
        return f'<span style="color:red">Ocurrió al validar el modelo LLM: {e}</span>'
    except Exception as e:
        return f'<span style="color:red">Ocurrió un error al inicializar el modelo LLM: {e}</span>'

    prompt = SYSTEM_PROMPT + custom_prompt + CONVERSATION_HISTORY_PROMPT

    system_message_prompt = SystemMessagePromptTemplate.from_template(prompt)
    PROMPT = ChatPromptTemplate.from_messages([
        system_message_prompt,
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content=user_input)
    ])

    # Cargamos memoria de conversación previamente almacenada con historial de conversación
    memory = cargar_memoria_conversacion()

    try:
        conversation = ConversationChain(
            prompt=PROMPT,
            llm=chat_model,
            verbose=False,
            memory=memory
        )
    except ValidationError as e:
        return f'<span style="color:red">Ocurrió un error de validación: {e}</span>'
    except Exception as e:
        return f'<span style="color:red">Ocurrió un error: {e}</span>'

    try:
        result = conversation.predict(input=user_input)

        # Persistimos memoria de conversación para poder cargarla en futuras consultas al chat
        persistir_memoria_conversacion(conversation.memory)

    except HfHubHTTPError as e:
        return f'<span style="color:red">Ocurrió un error HTTP al intentar acceder al Hub: {e}</span>'
    except Exception as e:
        return f'<span style="color:red">Ocurrió un error: {e}</span>'

    response = result
    return response


# Obtiene la respuesta mediante un agente que consulta la base de datos vectorizada
def obtener_respuesta_agente_base_conocimientos_externa(user_input: str, system_message: str, temperatura: float):

    # Inicialización del modelo
    try:
        llm = inicializacion_modelo_llm(temperatura if temperatura > 0.0 else 0.01)
        chat_model = inicializacion_chat_model(llm)
    except OSError as e:
        return f'<span style="color:red">Ocurrió al validar el modelo LLM: {e}</span>'
    except Exception as e:
        return f'<span style="color:red">Ocurrió un error al inicializar el modelo LLM: {e}</span>'

    # memory = ConversationBufferMemory(memory_key="chat_history") #ponemos una denominada clave a la memoria "chat_history"
    memory = cargar_memoria_conversacion()

    tools = [obtener_tool_consulta_base_conocimiento(chat_model)]

    # Crear el agente
    try:
        agent = initialize_agent(
            tools, llm, agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=False, handle_parsing_errors=True
        )
    except Exception as e:
        return f'<span style="color:red">Ocurrió un error al inicializar el agente: {e}</span>'

    # Ejecutar el agente con la entrada del usuario
    try:
        response = agent.invoke(user_input)

        # Persistimos memoria de conversación para poder cargarla en futuras consultas al chat
        memory.save_context({"input": user_input}, {"output": response['output']})
        persistir_memoria_conversacion(memory)

    except HfHubHTTPError as e:
        return f'<span style="color:red">Ocurrió un error HTTP al intentar acceder al Hub: {e}</span>'
    except ValueError as e:
        return f'<span style="color:red">Ocurrió un error al intentar procesar la entrada del usuario: {user_input} {e}</span>'
    except Exception as e:
        return f'<span style="color:red">Ocurrió un error: {e}</span>'

    return response['output']


def execute(user_input: str, system_message: str, temperatura: float, usar_base_conocimiento: bool):
    if usar_base_conocimiento:
        return obtener_respuesta_agente_base_conocimientos_externa(user_input, system_message, temperatura)
    else:
        return obtener_respuesta_chatbot_informacion_interna_modelo(user_input, system_message, temperatura)
