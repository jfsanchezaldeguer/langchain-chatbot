import unittest
from unittest.mock import patch
import chatbot as chatbot


class TestChatbot(unittest.TestCase):

    @patch('chatbot.obtener_respuesta_chatbot_informacion_interna_modelo')
    def test_llama_obtener_respuesta_chatbot_informacion_interna_modelo(self, mock_metodo):
        # Definir el valor que queremos que devuelva el mock
        mock_metodo.return_value = "Buenas, soy tu asistente."
        chatbot_response = chatbot.execute('Hola', 'Esto es un test', 0.0, False)
        mock_metodo.assert_called_with('Hola', 'Esto es un test', 0.0)
        self.assertEqual(chatbot_response, "Buenas, soy tu asistente.")

    @patch('chatbot.obtener_respuesta_agente_base_conocimientos_externa')
    def test_llama_obtener_respuesta_agente_base_conocimientos_externa(self, mock_metodo):
        # Definir el valor que queremos que devuelva el mock
        mock_metodo.return_value = "Buenas, pregunta lo que quieras sobre la base de conocimientos."
        chatbot_response = chatbot.execute('Hola', 'Esto es un test', 0.0, True)
        mock_metodo.assert_called_with('Hola', 'Esto es un test', 0.0)
        self.assertEqual(chatbot_response, "Buenas, pregunta lo que quieras sobre la base de conocimientos.")
