from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import google.generativeai as genai
import os


print("Carregando o modelo de classificação...")
try:
    classificador = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("Modelo de classificação carregado!")
except Exception as e:
    print(f"Erro ao carregar o classificador: {e}")
    classificador = None


print("Configurando a API do Gemini...")
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Variável de ambiente GOOGLE_API_KEY não configurada.")
    genai.configure(api_key=api_key)
    modelo_generativo = genai.GenerativeModel('gemini-1.0-pro')
    print("Modelo generativo configurado!")
except Exception as e:
    print(f"Erro ao configurar o Gemini: {e}")
    modelo_generativo = None

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analisar', methods=['POST'])
def analisar():
    if not classificador or not modelo_generativo:
        return jsonify({'erro': 'Um ou mais modelos de IA não estão carregados.'}), 500

    dados = request.get_json()
    texto_email = dados.get('email', '')
    if not texto_email:
        return jsonify({'erro': 'Nenhum texto de e-mail fornecido.'}), 400


    candidate_labels = ["produtivo", "improdutivo", "urgente", "informativo"] 
    resultado_classificacao = classificador(texto_email, candidate_labels)
    categoria = resultado_classificacao['labels'][0]

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