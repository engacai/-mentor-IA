import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Carrega as variáveis de ambiente (sua API Key)
load_dotenv()

def configurar_base_conhecimento(caminho_pdf):
    """Lê o manual em PDF, divide em partes e salva no banco de dados vetorial."""
    print("📚 Lendo e processando o manual de regras...")
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()

    # Divide o texto em pedaços menores para não estourar o limite de tokens
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documentos)

    # Cria os embeddings (transforma texto em números) e salva no ChromaDB local
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 3}) # Retorna os 3 trechos mais relevantes

def criar_agente(retriever):
    """Cria a corrente (chain) que une a busca no manual com o LLM."""
    
    # Este é o "Cérebro" restritivo do seu agente. Aqui evitamos alucinações.
    template_prompt = """
    Você é um engenheiro de robótica atuando como mentor técnico.
    Use APENAS os trechos de contexto abaixo (extraídos das regras oficiais) para responder à pergunta.
    
    COMPETIÇÃO: {competicao}
    INVENTÁRIO DISPONÍVEL (peças que a equipe tem): {inventario}
    
    REGRAS GERAIS:
    1. Se a solução exigir peças que não estão no inventário, recuse e sugira uma alternativa com o que eles têm.
    2. Se a ação ferir alguma regra do contexto fornecido, alerte a equipe imediatamente.
    3. Para OBR/Arduino, forneça lógicas em C++. Para FLL, foque na montagem LEGO e blocos/Python.
    
    CONTEXTO DO MANUAL: {context}
    
    PERGUNTA DA EQUIPE: {pergunta}
    
    RESPOSTA:
    """
    
    prompt = PromptTemplate.from_template(template_prompt)
    
    # Define o LLM (Gemini 1.5 Flash é rápido e excelente para RAG)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2) # Temperatura baixa = menos alucinação
    
    def formatar_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Monta o fluxo de execução (Chain)
    rag_chain = (
        {"context": retriever | formatar_docs, 
         "competicao": RunnablePassthrough(), 
         "inventario": RunnablePassthrough(), 
         "pergunta": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# ==========================================
# EXECUÇÃO DO TESTE
# ==========================================
if __name__ == "__main__":
    # Certifique-se de ter um arquivo 'regras_obr.pdf' na mesma pasta
    caminho_arquivo = "regras_obr.pdf" 
    
    if not os.path.exists(caminho_arquivo):
        print(f"⚠️ Erro: Coloque um arquivo PDF chamado '{caminho_arquivo}' na pasta do projeto.")
    else:
        # 1. Configura o banco vetorial com o manual
        retriever_regras = configurar_base_conhecimento(caminho_arquivo)
        
        # 2. Inicializa o agente
        agente = criar_agente(retriever_regras)
        
        # 3. Simula as entradas do usuário (Isso virá da sua futura interface Web)
        print("\n🤖 Agente Pronto! Processando requisição da equipe...\n")
        
        resposta = agente.invoke({
            "competicao": "OBR - Modalidade Resgate",
            "inventario": "1 Arduino Uno, 2 Motores DC, Chassi de Acrílico, 3 Sensores de Linha TCRT5000",
            "pergunta": "Como podemos fazer nosso robô alinhar perfeitamente na fita preta usando esses componentes? O Genesis Robotic Team precisa de uma lógica em C++."
        })
        
        print("--------------------------------------------------")
        print("RESPOSTA DO AGENTE:")
        print("--------------------------------------------------")
        print(resposta)