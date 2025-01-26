from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import load_tools,initialize_agent,AgentType,create_react_agent,AgentExecutor,tool,Tool
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from huggingface_hub import login
from huggingface_hub.utils import HfHubHTTPError
from langchain_huggingface import HuggingFaceEmbeddings
from utils import (persistir_memoria_conversacion, cargar_memoria_conversacion, obtener_tools_consulta_base_conocimiento,
                   SYSTEM_PROMPT, CONVERSATION_HISTORY_PROMPT)
from pydantic import ValidationError

from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])


def huggingface_chatbot_response(user_input: str, custom_prompt: str, temperatura: float):

    # Inicialización del modelo
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/starchat2-15b-v0.1",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        top_p=0.7,
        temperature=temperatura,
        repetition_penalty=1.03
    )

    chat_model = ChatHuggingFace(llm=llm)

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

    try:
        result = conversation.predict(input=user_input)

        # Persistimos memoria de conversación para poder cargarla en futuras consultas al chat
        persistir_memoria_conversacion(conversation.memory)

    except HfHubHTTPError as e:
        return f'<span style="color:red">Ocurrió un error HTTP al intentar acceder al Hub: {e}</span>'

    response = result
    return response

def huggingface_agent_base_conocimiento_response(user_input: str, system_message: str, temperatura: float):
    # Inicialización del modelo
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/starchat2-15b-v0.1",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        top_p=0.7,
        temperature=temperatura,
        repetition_penalty=1.03
    )
    chat_model = ChatHuggingFace(llm=llm)

    # memory = ConversationBufferMemory(memory_key="chat_history") #ponemos una denominada clave a la memoria "chat_history"
    memory = cargar_memoria_conversacion()

    tools = obtener_tools_consulta_base_conocimiento(chat_model)

    # Crear el agente
    agent = initialize_agent(
        tools, llm, agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True, handle_parsing_errors=True
    )

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

    return response['output']

# def openai_chatbot_response(user_input: str, system_message: str, temperature: float):
    # Crear el LLM (Modelo de Lenguaje) usando OpenAI
    # llm = OpenAI(model="text-davinci-003")  # O usa "gpt-3.5-turbo" o "gpt-4" si tienes acceso
    # llm = ChatOpenAI(model="gpt-4o-mini")  # O usa "gpt-3.5-turbo" o "gpt-4" si tienes acceso

    # Definir el prompt y el agente
    # tools = [
    #     Tool(
    #         name="Chatbot",
    #         func=llm,
    #         description="Responde preguntas basadas en el modelo OpenAI GPT."
    #     ),
    # ]
    #
    # # Crear el agente
    # agent = initialize_agent(
    #     tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    # )
    #
    # # Ejecutar el agente con la entrada del usuario
    # response = agent.run(user_input)

    # response = llm.invoke([HumanMessage(content=user_input)])

    # response = 'respuesta test'
    # return response


def execute(user_input: str, system_message: str, temperatura: float, usar_base_conocimiento: bool):
    if usar_base_conocimiento:
        return huggingface_agent_base_conocimiento_response(user_input, system_message, temperatura)
    else:
        return huggingface_chatbot_response(user_input, system_message, temperatura)
