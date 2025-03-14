from openai import OpenAI
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv


app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


@app.route('/chat', methods=['POST'])

def chat():
    datos = request.json
    mensaje_usuario = datos.get("message", "")

    if not mensaje_usuario:
        return jsonify({"error": "No se realizó una consulta"}), 400

    respuesta_chat = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role":"user",
            "content": mensaje_usuario}
        ]
    )

    return jsonify({"response": respuesta_chat.choices[0].message.content})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
