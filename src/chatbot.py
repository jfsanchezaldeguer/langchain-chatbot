from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
import pickle
from langchain.agents import load_tools,initialize_agent,AgentType,create_react_agent,AgentExecutor,Tool
# from langchain_community.llms import HuggingFaceHub
# from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import login
from huggingface_hub.utils import HfHubHTTPError

from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])


def huggingface_chatbot_response(user_input: str, custom_prompt: str, temperature: float):
    if os.path.exists('storage/memory.pkl'):
        memoria_cargada = open('storage/memory.pkl', 'rb').read()  # rb para indicar que leemos el objeto binario
        memory = pickle.loads(memoria_cargada)
    else:
        memory = ConversationBufferWindowMemory(return_messages=True, k=1)

    # Inicialización del modelo
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/starchat2-15b-v0.1",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        # top_k=10,
        top_p=0.7,
        temperature=temperature,
        repetition_penalty=1.03,
        # return_full_text=False,
        # huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )

    chat_model = ChatHuggingFace(llm=llm)
    # chat_model = llm

    prompt = ("""You're a helpful assistant. Answer all questions to the best of your ability.
              Answer to the user's greeting with a greeting.""" +
              custom_prompt + """

# Current conversation:
# {history}
# Human: {input}
# AI Assistant:""")

    system_message = ("""You're a helpful assistant. Answer all questions to the best of your ability.
              Answer to the user's greeting with a greeting.""" +
                      custom_prompt)

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_input)
    ]

    # PROMPT = PromptTemplate(input_variables=["history", "input"], template=prompt)
    system_message_prompt = SystemMessagePromptTemplate.from_template(prompt)
    PROMPT = ChatPromptTemplate.from_messages([
        system_message_prompt,
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content=user_input)
    ])

    conversation = ConversationChain(
        prompt=PROMPT,
        llm=chat_model,
        verbose=False,
        memory=memory
    )
    # chat_model._to_chat_prompt(messages)

    try:
        # result = chat_model.invoke(messages)
        result = conversation.predict(input=user_input)

        # Obtenemos el histórico
        # print(memory.load_memory_variables({}))
        # print(conversation.memory.buffer)
        # print('********************')
        pickled_str = pickle.dumps(conversation.memory)  # Crea un objeto binario con todo el objeto de la memoria
        with open('storage/memory.pkl',
                  'wb') as f:  # wb para indicar que escriba un objeto binario, en este caso en la misma ruta que el script
            f.write(pickled_str)
        # print(result)
    except HfHubHTTPError as e:
        return f"Ocurrió un error HTTP al intentar acceder al Hub: {e}"

    response = result
    return response

def huggingface_agent_chatbot_response(user_input: str, system_message: str, temperature: float):
    # Inicialización del modelo
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        # task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        # top_k=10,
        top_p=0.7,
        temperature=temperature,
        repetition_penalty=1.03,
        # return_full_text=False,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )
#     # Definir el prompt y el agente
#
#     @tool
#     def chatbot_response(text: str) -> str:
#         '''Responde preguntas basadas en el modelo OpenAI GPT'''
#
    tools = [
        Tool(
            name="Chatbot",
            func=llm,
            description="You're a helpful conversational ai assistant. Just answer the question only in the same language."
        ),
    ]
    # tools = []

    # Crear el agente
    agent = initialize_agent(
        tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
    )

    # Ejecutar el agente con la entrada del usuario
    response = agent.invoke(user_input)

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


def execute(user_input: str, system_message: str, temperature: float):
    return huggingface_chatbot_response(user_input, system_message, temperature)
    # return huggingface_agent_chatbot_response(user_input, system_message, temperature)
    # return "Respuesta"
