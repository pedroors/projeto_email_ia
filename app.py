from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import requests # <-- Importação que estava faltando

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- CONFIGURAÇÃO DAS APIS ---
HF_API_KEY = os.getenv("HF_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# URL da API de inferência para o modelo de classificação da Hugging Face
CLASSIFIER_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

# Configuração do Gemini
modelo_generativo = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        modelo_generativo = genai.GenerativeModel('gemini-2.5-flash')
        print("Modelo generativo do Gemini configurado com sucesso!")
    except Exception as e:
        print(f"Erro ao configurar o Gemini: {e}")
else:
    print("Aviso: Chave da API do Gemini não encontrada. A geração de resposta não funcionará.")

app = Flask(__name__)

@app.route('/')
def home():
    """Renderiza a página HTML principal."""
    return render_template('index.html')

def classificar_texto_com_api(texto):
    """
    Função para chamar a API da Hugging Face e classificar o texto.
    Retorna um dicionário JSON com o resultado.
    """
    if not HF_API_KEY:
        raise Exception("Chave da API da Hugging Face (HF_API_KEY) não foi configurada.")
        
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": texto,
        "parameters": {"candidate_labels": ["produtivo", "improdutivo", "urgente", "informativo"]},
    }
    response = requests.post(CLASSIFIER_API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Lança um erro para respostas ruins (4xx ou 5xx)
    return response.json()

@app.route('/analisar', methods=['POST'])
def analisar():
    """Recebe o texto, classifica e gera uma sugestão de resposta."""
    if not modelo_generativo:
        return jsonify({'erro': 'O modelo generativo do Gemini não está carregado.'}), 500

    texto_email = ""

    # Lógica para extrair texto do arquivo ou do campo de texto
    if 'arquivo_email' in request.files:
        arquivo = request.files['arquivo_email']
        if arquivo.filename != '':
            try:
                if arquivo.filename.lower().endswith('.pdf'):
                    doc = fitz.open(stream=arquivo.read(), filetype="pdf")
                    texto_pdf = "".join(page.get_text() for page in doc)
                    texto_email = texto_pdf
                    doc.close()
                else:
                    texto_email = arquivo.read().decode('utf-8')
            except Exception as e:
                return jsonify({'erro': f'Não foi possível ler o arquivo: {e}'}), 400
    
    elif 'texto_email' in request.form:
        texto_email = request.form.get('texto_email')

    if not texto_email:
        return jsonify({'erro': 'Nenhum texto ou arquivo de e-mail fornecido.'}), 400

    # ETAPA 1: CLASSIFICAR O E-MAIL USANDO A API
    try:
        resultado_classificacao = classificar_texto_com_api(texto_email)
        if 'error' in resultado_classificacao:
            raise Exception(resultado_classificacao['error'])
        categoria = resultado_classificacao['labels'][0]
    except Exception as e:
        print(f"Erro na API de classificação: {e}")
        return jsonify({'erro': f'Não foi possível classificar o texto: {e}'}), 500
    
    # ETAPA 2: GERAR UMA RESPOSTA COM O GEMINI
    prompt = f"""
    Você é um assistente de e-mail altamente eficiente.
    Um e-mail foi classificado como '{categoria}'. O conteúdo do e-mail é:
    ---
    {texto_email}
    ---
    Com base no conteúdo e na categoria, gere uma sugestão de resposta curta e profissional em português.
    - Se a categoria for 'improdutivo', sugira uma ação como "Arquivar" ou "Marcar como spam".
    - Se for 'produtivo' ou 'urgente', sugira uma resposta que confirme o recebimento e indique o próximo passo.
    - Seja direto e conciso.
    """
    
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

if __name__ == '__main__':
    app.run(debug=True)