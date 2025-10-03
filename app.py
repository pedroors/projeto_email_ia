# app.py MODIFICADO E OTIMIZADO
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
from dotenv import load_dotenv
import fitz
import requests # <--- NOVO

load_dotenv()

# --- REMOVEMOS O CARREGAMENTO DO MODELO LOCAL ---

# --- CONFIGURAÇÃO DAS APIS ---
HF_API_KEY = os.getenv("HF_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# URL da API de inferência para o modelo de classificação
CLASSIFIER_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

# Configuração do Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    modelo_generativo = genai.GenerativeModel('gemini-1.5-pro-latest')
    print("Modelo generativo configurado!")
except Exception as e:
    print(f"Erro ao configurar o Gemini: {e}")
    modelo_generativo = None

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def classificar_texto_com_api(texto):
    """Função para chamar a API da Hugging Face."""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": texto,
        "parameters": {"candidate_labels": ["produtivo", "improdutivo", "urgente", "informativo"]},
    }
    response = requests.post(CLASSIFIER_API_URL, headers=headers, json=payload)
    return response.json()

@app.route('/analisar', methods=['POST'])
def analisar():
    # ... (A lógica para extrair o texto_email do arquivo ou formulário continua a mesma) ...
    texto_email = ""
    if 'arquivo_email' in request.files:
        # ... (código de extração de arquivo) ...
    elif 'texto_email' in request.form:
        texto_email = request.form.get('texto_email')

    if not texto_email:
        return jsonify({'erro': 'Nenhum texto ou arquivo de e-mail fornecido.'}), 400

    # --- ETAPA 1: CLASSIFICAR O E-MAIL (AGORA COM API) ---
    try:
        resultado_classificacao = classificar_texto_com_api(texto_email)
        if 'error' in resultado_classificacao:
            raise Exception(resultado_classificacao['error'])
        categoria = resultado_classificacao['labels'][0]
    except Exception as e:
        print(f"Erro na API de classificação: {e}")
        return jsonify({'erro': 'Não foi possível classificar o texto.'}), 500

    # --- ETAPA 2: GERAR RESPOSTA (continua igual) ---
    # ... (O resto da sua função com a chamada para o Gemini continua exatamente o mesmo) ...
    prompt = f"..."
    try:
        resposta_generativa = modelo_generativo.generate_content(prompt)
        sugestao_resposta = resposta_generativa.text.strip()
    except Exception as e:
        print(f"Erro na API do Gemini: {e}")
        sugestao_resposta = "Não foi possível gerar uma resposta dinâmica."

    return jsonify({
        'categoria': categoria.capitalize(),
        'sugestao_resposta': sugestao_resposta
    })

# ...