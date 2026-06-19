import streamlit as st
from agente_drive import descarregar_manuais_do_drive, inicializar_base_conhecimento, criar_agente_inteligente

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Mentor IA - Engenheiros do Açaí", page_icon="🤖", layout="wide")

# ==========================================
# 2. INICIALIZAÇÃO DO CÉREBRO (Executado apenas 1x)
# ==========================================
# O @st.cache_resource garante que o modelo de 4.7GB não seja recarregado a cada nova mensagem
@st.cache_resource(show_spinner="Iniciando o motor local Llama 3 e processando as regras...")
def ligar_motor_ia():
    pdfs_locais = descarregar_manuais_do_drive()
    retriever = inicializar_base_conhecimento(pdfs_locais)
    agente = criar_agente_inteligente(retriever)
    return agente

agente = ligar_motor_ia()

# ==========================================
# 3. INTERFACE VISUAL (SIDEBAR)
# ==========================================
with st.sidebar:
    st.title("⚙️ Setup - Engenheiros do Açaí")
    st.write("Defina as restrições físicas antes de consultar o mentor.")
    
    # Seleção de Torneio
    competicao_selecionada = st.selectbox(
        "Selecione o Torneio:",
        ("OBR - Modalidade Prática", "OBR - Modalidade Resgate", "FLL - FIRST LEGO League")
    )
    
    # Campo de Inventário Restritivo editável
    inventario_equipe = st.text_area(
        "Inventário Físico Disponível:",
        value="Kit Arduino Básico, 2 Sensores Ultrassónicos HC-SR04, Motores DC.",
        height=150
    )
    
    st.markdown("---")
    st.caption("🟢 Sistema de Inferência: 100% Offline (Local)")

# ==========================================
# 4. INTERFACE VISUAL (CHAT PRINCIPAL)
# ==========================================
st.title("🤖 Mentor Técnico Especializado")
st.markdown(f"*Contexto de regras carregado: **{competicao_selecionada}***")

# Cria a memória da conversa na sessão atual
if "mensagens" not in st.session_state:
    st.session_state.mensagens = [{"role": "assistant", "content": "Olá, engenheiro! Descreva o seu problema de montagem mecânica ou desafio de lógica de programação."}]

# Exibe o histórico de mensagens
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Caixa de input estilo WhatsApp/ChatGPT
pergunta_usuario = st.chat_input("Ex: Como o robô deve reagir ao redutor de velocidade?")

if pergunta_usuario:
    # 1. Mostra a pergunta do usuário na tela
    st.session_state.mensagens.append({"role": "user", "content": pergunta_usuario})
    with st.chat_message("user"):
        st.write(pergunta_usuario)

    # 2. Mostra o indicador de que a IA está a "pensar"
    with st.chat_message("assistant"):
        with st.spinner("Processando lógicas e regras localmente..."):
            # Invoca o agente de IA passando os parâmetros da interface gráfica
            resposta_ia = agente.invoke({
                "competicao": competicao_selecionada,
                "inventario": inventario_equipe,
                "pergunta": pergunta_usuario
            })
            
            # 3. Imprime a resposta e salva no histórico
            st.write(resposta_ia)
            st.session_state.mensagens.append({"role": "assistant", "content": resposta_ia})