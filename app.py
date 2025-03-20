from openai import OpenAI
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.cloud import speech, texttospeech
import io

load_dotenv()

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SERVICE_ACCOUNT_FILE = 'credentials/service_account.json'
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']


# Inicializar las credenciales una vez
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Cliente para speech to text
speech_client = speech.SpeechClient(credentials=credentials)

# Cliente para text to speech
text_to_speech_client = texttospeech.TextToSpeechClient(credentials=credentials)

# Endpoint para obtener el token de acceso
@app.route('/get-credentials', methods=['GET'])
def get_credentials_route():
    try:
        # Si las credenciales están expiradas, renovar el token
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        
        # Devolver el token de acceso
        return jsonify({
            'access_token': credentials.token,
            'expires_in': credentials.expiry.timestamp()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/chat', methods=['POST'])
def chat():
    datos = request.json
    mensajes_usuario = datos.get("messages", [])  # Cambié "message" a "messages"

    if not mensajes_usuario:
        return jsonify({"error": "No se realizó una consulta"}), 400

    # Se asegura de que el primer mensaje sea el de tipo 'user'
    # y que se mantenga el contexto de los mensajes previos
    if mensajes_usuario[-1]["role"] != "user":
        return jsonify({"error": "El último mensaje debe ser del usuario"}), 400

    # Agregar el mensaje de sistema al historial para mantener el contexto
    historial_de_conversacion = [
        {"role": "system", "content": "Responder solo en español"}
    ] + mensajes_usuario

    # Solicitar la respuesta del modelo con el historial completo
    respuesta_chat = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=historial_de_conversacion
    )

    return jsonify({"response": respuesta_chat.choices[0].message.content})


# Endpoint para convertir voz a texto
@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    audio_file = request.files['audio']  # Asumimos que el archivo de audio se manda en un formulario

    # Convertir el archivo de audio a un formato adecuado para Google Speech-to-Text
    audio = audio_file.read()

    # Crear una solicitud para Google Speech-to-Text
    audio_content = speech.RecognitionAudio(content=audio)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="es-ES",  # Puedes cambiarlo según el idioma que necesites
    )

    # Realizar la transcripción
    response = speech_client.recognize(config=config, audio=audio_content)

    # Obtener el texto reconocido
    if response.results:
        recognized_text = response.results[0].alternatives[0].transcript
        return jsonify({"transcription": recognized_text})
    else:
        return jsonify({"error": "No se pudo transcribir el audio"}), 400


# Endpoint para convertir texto a voz (Text-to-Speech)
@app.route('/text_to_speech', methods=['POST'])
def text_to_speech_route():
    # Recibir el texto del cuerpo de la solicitud
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "No se proporcionó texto"}), 400

    try:
        # Solicitar la conversión de texto a audio
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="es-US",  
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL  
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3  
        )

        # Generar el audio a partir del texto
        response = text_to_speech_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Devolver el archivo de audio como respuesta
        return send_file(
            io.BytesIO(response.audio_content),
            mimetype="audio/mp3",
            as_attachment=True,
            download_name="response.mp3"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
    