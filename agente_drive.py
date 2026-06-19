import os
import io
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Alterado: Removemos o ChatGoogleGenerativeAI e adicionamos o ChatOllama
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ID DA TUA PASTA DO GOOGLE DRIVE
ID_PASTA_DRIVE = "1O3UEDbcTeEsZamCnzJ0ZakrtBFzRAUP1"

def descarregar_manuais_do_drive():
    """Liga-se ao Drive usando a Conta de Serviço e descarrega os PDFs da pasta."""
    print("🔄 A aceder à pasta do Google Drive...")
    
    # Define o escopo de leitura do Drive
    escopos = ['https://www.googleapis.com/auth/drive.readonly']
    credenciais = Credentials.from_service_account_file('credenciais_drive.json', scopes=escopos)
    servico = build('drive', 'v3', credentials=credenciais)

    # Lista os ficheiros dentro da pasta especificada
    query = f"'{ID_PASTA_DRIVE}' in parents and mimeType='application/pdf' and trashed=false"
    resultados = servico.files().list(q=query, fields="files(id, name)").execute()
    ficheiros = resultados.get('files', [])

    if not ficheiros:
        print("⚠️ Nenhum ficheiro PDF encontrado na pasta do Drive.")
        return []

    caminhos_locais = []
    
    # Cria uma pasta temporária local para guardar os ficheiros descarregados
    if not os.path.exists("manuais_temporarios"):
        os.makedirs("manuais_temporarios")

    for ficheiro in ficheiros:
        nome_ficheiro = ficheiro['name']
        id_ficheiro = ficheiro['id']
        print(f"📥 A descarregar: {nome_ficheiro}...")

        requisicao = servico.files().get_media(fileId=id_ficheiro)
        caminho_salvar = os.path.join("manuais_temporarios", nome_ficheiro)
        
        with open(caminho_salvar, 'wb') as f:
            downloader = MediaIoBaseDownload(f, requisicao)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                
        caminhos_locais.append(caminho_salvar)
        
    return caminhos_locais

def inicializar_base_conhecimento(caminhos_pdfs):
    """Processa todos os PDFs descarregados e cria o banco de dados vetorial."""
    print("📚 A processar a base de conhecimento...")
    todos_os_documentos = []

    for caminho in caminhos_pdfs:
        loader = PyPDFLoader(caminho)
        todos_os_documentos.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(todos_os_documentos)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def criar_agente_inteligente(retriever):
    """Monta a estrutura mestre do agente de IA."""
    template_prompt = """
    Você é um Mentor de Inteligência Artificial altamente avançado e especializado em Robótica Educacional de Competição. 
    Sua personalidade é amigável, direta, encorajadora e típica de um engenheiro sênior ajudando uma equipe de estudantes.

    COMPETIÇÃO ATUAL: {competicao}
    INVENTÁRIO DA EQUIPE: {inventario}

    REGRAS DE COMPORTAMENTO (SIGA ESTRITAMENTE):
    1. MODO CONVERSA: Se o usuário apenas disser "olá", "tudo bem?", "bom dia" ou fizer um comentário geral sem expor um problema do robô, APENAS responda de forma natural, amigável e pergunte como pode ajudar a equipe hoje. NÃO invente problemas, NÃO gere códigos e NÃO sugira soluções mecânicas não solicitadas.
    2. MODO ENGENHARIA: Se o usuário fizer uma pergunta técnica (ex: "como sigo a linha?", "meu robô não faz a curva", "posso usar essa peça?"), aí sim, forneça lógicas de programação, códigos estruturados e dicas de design mecânico.
    3. RESTRIÇÃO DE HARDWARE: No 'Modo Engenharia', você NUNCA deve sugerir a compra ou o uso de motores, sensores ou microcontroladores que não estejam listados no "Inventário da Equipe". Trabalhe apenas com o que eles têm.
    4. JUIZ DE ARENA: Use os trechos do regulamento abaixo como uma trava de segurança. Se a ideia do usuário for proibida pelo manual, avise-o.

    TEXTO EXTRAÍDO DO REGULAMENTO: 
    {context}
    
    MENSAGEM DO USUÁRIO: {pergunta}
    
    SUA RESPOSTA:
    """
    
    prompt = PromptTemplate.from_template(template_prompt)
    
    # ... (o resto da função continua igual) ...
    
    # Alterado: Agora instanciamos o Ollama local apontando para o teu llama3 instalado
    llm = ChatOllama(model="llama3", temperature=0.2)
    
    def formatar_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
         "context": lambda x: formatar_docs(retriever.invoke(x["pergunta"])), 
         "competicao": lambda x: x["competicao"], 
         "inventario": lambda x: x["inventario"], 
         "pergunta": lambda x: x["pergunta"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# ==========================================
# EXECUÇÃO DO FLUXO
# ==========================================
if __name__ == "__main__":
    if ID_PASTA_DRIVE == "COLE_AQUI_O_ID_DA_SUA_PASTA":
        print("⚠️ Configuração necessária: Substitua o ID_PASTA_DRIVE pelo ID real da sua pasta do Drive.")
    elif not os.path.exists("credenciais_drive.json"):
        print("⚠️ Erro: O ficheiro 'credenciais_drive.json' não foi encontrado na raiz do projeto.")
    else:
        # 1. Descarrega os PDFs dinamicamente da nuvem (se houver internet)
        pdfs_locais = descarregar_manuais_do_drive()
        
        if pdfs_locais:
            # 2. Transforma os manuais em vetores localmente
            retriever = inicializar_base_conhecimento(pdfs_locais)
            
            # 3. Liga o agente de IA Local
            agente = criar_agente_inteligente(retriever)
            
            print("\n🤖 Agente de IA Online e Conectado (Motor Local Llama3 ativo)!")
            
            # 4. Simulação de uma consulta de utilizador
            pergunta_teste = "Podemos utilizar sensores ultrassónicos para detetar as paredes na arena de resgate? O que diz a regra?"
            inventario_teste = "Kit Arduino Básico, 2 Sensores Ultrassónicos HC-SR04, Motores DC."
            
            resposta = agente.invoke({
                "competicao": "OBR - Olimpíada Brasileira de Robótica",
                "inventario": inventario_teste,
                "pergunta": pergunta_teste
            })
            
            print("\n--------------------------------------------------")
            print("RESPOSTA DO AGENTE:")
            print("--------------------------------------------------")
            print(resposta)