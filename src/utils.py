import pickle
import os
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SKLearnVectorStore
# from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.agents import tool

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
        return ConversationBufferWindowMemory(return_messages=True, k=1)


def limpiar_memoria_conversacion_previa():
    if os.path.exists('storage/memory.pkl'):
        os.remove('storage/memory.pkl')


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

        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        funcion_embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # Creamos la BBDD de vectores a partir de los documentos y la función embeddings
        vector_store = SKLearnVectorStore.from_documents(
            documents=docs,
            embedding=funcion_embedding,
            persist_path=persist_path,
            serializer="parquet",  # el serializador o formato de la BD lo definimos como parquet
        )

        vector_store.persist()


def obtener_tools_consulta_base_conocimiento(llm):

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    funcion_embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

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

    return [consulta_base_conocimiento]
