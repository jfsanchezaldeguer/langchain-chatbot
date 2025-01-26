import pickle
import os
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SKLearnVectorStore
# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.agents import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

# Configurar la clave API de HuggingFace
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Configurar la clave API de OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")

SYSTEM_PROMPT = """You're a helpful assistant. Answer all questions to the best of your ability. Answer to the user's greeting with a greeting. """
CONVERSATION_HISTORY_PROMPT = """

# Current conversation:
# {history}
# Human: {input}
# AI Assistant:"""


def persistir_memoria_conversacion(conversation_memory):
    pickled_str = pickle.dumps(conversation_memory)  # Crea un objeto binario con todo el objeto de la memoria
    with open('storage/memory.pkl',
              'wb') as f:  # wb para indicar que escriba un objeto binario, en este caso en la misma ruta que el script
        f.write(pickled_str)


def cargar_memoria_conversacion():
    if os.path.exists('storage/memory.pkl'):
        memoria_cargada = open('storage/memory.pkl', 'rb').read()  # rb para indicar que leemos el objeto binario
        return pickle.loads(memoria_cargada)
    else:
        return ConversationBufferWindowMemory(return_messages=True, k=3)


def limpiar_memoria_conversacion_previa():
    if os.path.exists('storage/memory.pkl'):
        os.remove('storage/memory.pkl')


def inicializacion_modelo_llm(temperatura: float):
    # Crear el LLM (Modelo de Lenguaje) usando OpenAI
    if os.environ["HUGGINGFACEHUB_API_TOKEN"] and os.environ["HUGGINGFACEHUB_REPO_ID"]:
        llm = HuggingFaceEndpoint(
            repo_id=os.environ["HUGGINGFACEHUB_REPO_ID"],
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            top_p=0.7,
            temperature=temperatura,
            repetition_penalty=1.03
        )
    else:
        # Crear el LLM (Modelo de Lenguaje) usando OpenAI
        llm = ChatOpenAI(
            model=os.environ["OPENAI_MODEL_NAME"],
            temperature=temperatura,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
    return llm


def inicializacion_chat_model(llm):
    if os.environ["HUGGINGFACEHUB_API_TOKEN"] and os.environ["HUGGINGFACEHUB_REPO_ID"]:
        return ChatHuggingFace(llm=llm)
    else:
        return llm


def obtener_funcion_embedding():
    if os.environ["HUGGINGFACEHUB_API_TOKEN"] and os.environ["HUGGINGFACEHUB_REPO_ID"]:
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    else:
        return OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

def inicializar_base_conocimiento_vectorizada():

    if not os.path.exists('storage/ejemplosk_embedding_db'):
        # Cargamos el documento
        loader = TextLoader('docs/Base conocimiento Historia España.txt', encoding="utf8")
        documents = loader.load()

        # Dividir en chunks
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500)  # Método de split basándose en tokens
        docs = text_splitter.split_documents(documents)

        persist_path = "storage/ejemplosk_embedding_db"  # ruta donde se guardará la BBDD vectorizada

        funcion_embedding = obtener_funcion_embedding()

        # Creamos la BBDD de vectores a partir de los documentos y la función embeddings
        vector_store = SKLearnVectorStore.from_documents(
            documents=docs,
            embedding=funcion_embedding,
            persist_path=persist_path,
            serializer="parquet",  # el serializador o formato de la BD lo definimos como parquet
        )

        vector_store.persist()


def obtener_tool_consulta_base_conocimiento(llm):

    funcion_embedding = obtener_funcion_embedding()

    persist_path = "storage/ejemplosk_embedding_db"
    vector_store_connection = SKLearnVectorStore(embedding=funcion_embedding, persist_path=persist_path,
                                                 serializer="parquet")
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                           base_retriever=vector_store_connection.as_retriever())

    @tool
    def consulta_base_conocimiento(text: str) -> str:
        '''Retorna respuestas sobre la historia de España. Se espera que la entrada sea una cadena de texto
        y retorna una cadena con el resultado más relevante. Si la respuesta con esta herramienta es relevante,
        no debes usar ninguna herramienta más ni tu propio conocimiento como LLM'''
        compressed_docs = compression_retriever.invoke(text)
        resultado = compressed_docs[0].page_content
        return resultado

    return consulta_base_conocimiento
