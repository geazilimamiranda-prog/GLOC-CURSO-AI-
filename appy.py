import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# Configuração da página em modo wide
st.set_page_config(
    page_title="Análise de CSV com Gráficos e Chat RAG (Groq)",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Laboratório de Dados: CSV, Gráficos e RAG com Groq")
st.markdown("Faça o upload de uma planilha CSV, explore gráficos visuais interativos e converse diretamente com os dados usando modelos de IA da Groq.")

# Inicialização do histórico de mensagens no estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================================================================
# BARRA LATERAL: CONFIGURAÇÕES E UPLOAD DO CSV
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Chave da API da Groq
    user_api_key = st.text_input(
        "API Key (Groq):", 
        type="password", 
        placeholder="gsk_...",
        help="Obtenha sua chave gratuita em console.groq.com"
    )

    BASE_URL = "https://api.groq.com/openai/v1"
    modelo_selecionado = None
    client = None

    if user_api_key:
        try:
            client = OpenAI(base_url=BASE_URL, api_key=user_api_key.strip())
            
            # Busca dinâmica de modelos de chat disponíveis na Groq
            modelos_disponiveis = [
                m.id for m in client.models.list().data 
                if not any(termo in m.id.lower() for termo in ["whisper", "guard", "audio"])
            ]
            
            if modelos_disponiveis:
                padrao_idx = 0
                for idx, nome in enumerate(sorted(modelos_disponiveis)):
                    if "llama-3.3-70b" in nome or "llama3" in nome:
                        padrao_idx = idx
                        break
                        
                modelo_selecionado = st.selectbox(
                    "Modelo Ativo:", 
                    sorted(modelos_disponiveis),
                    index=padrao_idx
                )
                st.success("Conectado à Groq!", icon="✅")
            else:
                st.error("Nenhum modelo de chat liberado.")
        except Exception as e:
            st.error(f"Erro na conexão: {e}")
            client = None
    else:
        st.info("Insira sua chave da Groq para habilitar o chat.")

    st.markdown("---")
    st.header("📁 Fonte de Dados")
    
    arquivo_enviado = st.file_uploader("Envie seu arquivo CSV", type=["csv"])
    
    df = None
    csv_texto = ""
    
    if arquivo_enviado is not None:
        try:
            # Tenta ler o CSV (com tratamento automático básico de separadores comuns)
            df = pd.read_csv(arquivo_enviado)
            # Converte uma amostra/resumo do DataFrame para texto para uso no RAG da IA
            csv_texto = df.head(100).to_string(index=False)
            st.success(f"Arquivo carregado com sucesso! ({df.shape[0]} linhas, {df.shape[1]} colunas)")
        except Exception as e:
            st.error(f"Erro ao ler o CSV: {e}")

    st.markdown("---")
    if st.button("🧹 Limpar Histórico do Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# CORPO PRINCIPAL: ABAS DE VISUALIZAÇÃO E CHAT
# ==============================================================================
if df is not None:
    tab1, tab2, tab3 = st.tabs(["📋 Visualização e Resumo", "📈 Visualizador de Gráficos", "💬 Chat RAG com os Dados"])

    # --- ABA 1: TABELA E ESTATÍSTICAS ---
    with tab1:
        st.subheader("Visualização das Primeiras Linhas")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Estatísticas Descritivas")
            st.dataframe(df.describe(), use_container_width=True)
        with col2:
            st.subheader("Informações do DataFrame")
            # Captura o info do pandas em formato de texto para exibir no Streamlit
            import io
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

    # --- ABA 2: GRÁFICOS INTERATIVOS ---
    with tab2:
        st.subheader("Geração Dinâmica de Gráficos (Plotly)")
        
        col_g1, col_g2, col_g3 = st.columns(3)
        
        colunas_disponiveis = df.columns.tolist()
        colunas_numericas = df.select_dtypes(include=['number']).columns.tolist()
        
        with col_g1:
            tipo_grafico = st.selectbox(
                "Tipo de Gráfico", 
                ["Dispersão (Scatter)", "Linha", "Histograma", "Barra"]
            )
        
        with col_g2:
            eixo_x = st.selectbox("Eixo X", colunas_disponiveis)
            
        with col_g3:
            eixo_y = st.selectbox("Eixo Y (Numérico)", colunas_numericas if colunas_numericas else colunas_disponiveis)

        # Renderização do gráfico com base na seleção
        try:
            if tipo_grafico == "Dispersão (Scatter)":
                fig = px.scatter(df, x=eixo_x, y=eixo_y, title=f"Dispersão de {eixo_y} por {eixo_x}")
            elif tipo_grafico == "Linha":
                fig = px.line(df, x=eixo_x, y=eixo_y, title=f"Gráfico de Linha: {eixo_y} vs {eixo_x}")
            elif tipo_grafico == "Histograma":
                fig = px.histogram(df, x=eixo_x, title=f"Histograma de {eixo_x}")
            elif tipo_grafico == "Barra":
                fig = px.bar(df, x=eixo_x, y=eixo_y, title=f"Gráfico de Barras: {eixo_y} por {eixo_x}")
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Selecione colunas compatíveis para gerar o gráfico. Detalhe: {e}")

    # --- ABA 3: CHAT RAG COM OS DADOS ---
    with tab3:
        st.subheader("Converse com sua base de dados CSV")
        st.markdown("Faça perguntas sobre tendências, valores máximos, mínimos ou resumos do arquivo carregado.")
        
        # Caixa de exibição do histórico do chat
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Entrada de texto do chat
        if user_input := st.chat_input("Ex: Qual é a média dos valores da coluna X?"):
            if not client or not modelo_selecionado:
                st.warning("⚠️ Insira sua chave de API da Groq na barra lateral para conversar com os dados.")
                st.stop()

            # Adiciona mensagem do usuário
            st.session_state.messages.append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user"):
                    st.write(user_input)

            # Monta o Prompt do Sistema combinando contexto do CSV
            prompt_sistema = (
                "Você é um cientista de dados especialista em analisar planilhas CSV.\n"
                "Responda às perguntas do usuário baseando-se EXCLUSIVAMENTE nos dados da amostra fornecida abaixo.\n"
                "Se a resposta não puder ser encontrada nos dados, diga que não possui informações suficientes.\n\n"
                f"--- AMOSTRA DOS DADOS DO CSV ---\n{csv_texto}\n---------------------------------"
            )

            mensagens_api = [{"role": "system", "content": prompt_sistema}]
            for m in st.session_state.messages:
                mensagens_api.append({"role": m["role"], "content": m["content"]})

            # Chamada à API da Groq
            try:
                with chat_container:
                    with st.spinner("Analisando dados e gerando resposta..."):
                        response = client.chat.completions.create(
                            model=modelo_selecionado,
                            messages=mensagens_api,
                            temperature=0.1,
                            max_tokens=500
                        )
                        resposta = response.choices[0].message.content

                # Adiciona e exibe a resposta do assistente
                st.session_state.messages.append({"role": "assistant", "content": resposta})
                with chat_container:
                    with st.chat_message("assistant"):
                        st.write(resposta)
            except Exception as e:
                st.error(f"Erro ao comunicar com a API da Groq: {e}")

else:
    st.info("👈 Por favor, faça o upload de um arquivo CSV na barra lateral para iniciar a análise, visualizar gráficos e conversar com os dados.")
