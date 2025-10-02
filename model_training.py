import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

emails_treino = [
    ("Olá, gostaria de agendar uma reunião para discutir o projeto.", "produtivo"),
    ("Promoção imperdível! Descontos de até 70% só hoje.", "improdutivo"),
    ("Segue em anexo o relatório de vendas do último trimestre.", "produtivo"),
    ("Você ganhou! Clique aqui para resgatar seu prêmio.", "improdutivo"),
    ("Lembrete: nossa chamada de alinhamento é amanhã às 10h.", "produtivo"),
    ("Confira as notícias que estão bombando na internet!", "improdutivo"),
    ("Poderia, por favor, revisar o documento e me dar seu feedback?", "produtivo"),
    ("Spam: Aumente seus seguidores agora mesmo com nossa ferramenta.", "improdutivo"),
    ("Solicitação de cadastro", "produtivo"),
    ("Solicitação de integração", "produtivo"),
]

textos, categorias = zip(*emails_treino)

modelo = make_pipeline(TfidfVectorizer(), MultinomialNB())

print("Iniciando treinamento do modelo...")
modelo.fit(textos, categorias)
print("Modelo treinado com sucesso!")

joblib.dump(modelo, 'modelo_email.joblib')
print("Modelo salvo em 'modelo_email.joblib'")