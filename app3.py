# Importações necessárias
import streamlit as st
import pandas as pd
import requests
from openai import OpenAI
import json
import os
from io import StringIO
from groq import Groq
import zipfile
import tempfile
import shutil

# Inicialização do estado da sessão
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "groq"
if "example_question" not in st.session_state:
    st.session_state.example_question = ""
if "use_text_input" not in st.session_state:
    st.session_state.use_text_input = False
if "edited_prompt" not in st.session_state:
    st.session_state.edited_prompt = ""

# Configuração da página
st.set_page_config(
    page_title="Chatbot Planilha",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Chatbot da Planilha")
st.markdown("Faça perguntas sobre os dados da planilha e receba respostas inteligentes!")

# Seção de upload de arquivo ZIP
st.header("📁 Upload de Arquivos")
uploaded_file = st.file_uploader(
    "Faça upload de um arquivo ZIP contendo os CSVs:",
    type=['zip'],
    help="Arraste e solte ou clique para selecionar um arquivo ZIP contendo os arquivos CSV necessários."
)

# Processamento do arquivo ZIP
if uploaded_file is not None:
    try:
        # Criar diretório temporário para extração
        script_dir = os.path.dirname(__file__) if __file__ else os.getcwd()
        
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            # Extrair todos os arquivos para o diretório do script
            zip_ref.extractall(script_dir)
            
        st.success("✅ Arquivo ZIP extraído com sucesso!")
        st.info("Os arquivos foram extraídos para o diretório do aplicativo.")
        
        # Listar arquivos extraídos
        extracted_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
        if extracted_files:
            st.write("**Arquivos CSV encontrados:**")
            for file in extracted_files:
                st.write(f"- {file}")
        
    except Exception as e:
        st.error(f"Erro ao processar o arquivo ZIP: {e}")

# Função para carregar dados dos arquivos CSV locais
@st.cache_data
def load_data_from_local():
    """Carrega dados de arquivos CSV especificados do diretório local."""
    print("Iniciando carregamento de dados dos arquivos locais...")

    header_file_name = "202401_NFs_Cabecalho.csv"
    items_file_name = "202401_NFs_Itens.csv"
    header_df = None
    items_df = None

    try:
        # Obter diretório do script e construir caminhos dos arquivos
        script_dir = os.path.dirname(__file__) if __file__ else os.getcwd()
        header_file_path = os.path.join(script_dir, header_file_name)
        items_file_path = os.path.join(script_dir, items_file_name)

        # Carregar arquivo de cabeçalho
        print(f"Tentando carregar '{header_file_path}'...")
        if os.path.exists(header_file_path):
            header_df = pd.read_csv(header_file_path)
            print(f"{header_file_name} carregado. Shape: {header_df.shape}")
        else:
            print(f"Arquivo não encontrado: {header_file_path}")
            st.warning(f"Arquivo não encontrado: {header_file_path}")

        # Carregar arquivo de itens
        print(f"Tentando carregar '{items_file_path}'...")
        if os.path.exists(items_file_path):
            items_df = pd.read_csv(items_file_path)
            print(f"{items_file_name} carregado. Shape: {items_df.shape}")
        else:
            print(f"Arquivo não encontrado: {items_file_path}")
            st.warning(f"Arquivo não encontrado: {items_file_path}")

        # Verificar se ambos os arquivos foram carregados
        if header_df is None or items_df is None:
            print(f"Não foi possível encontrar um ou ambos os arquivos CSV esperados ({header_file_name}, {items_file_name}).")
            st.warning(f"Não foi possível encontrar um ou ambos os arquivos CSV esperados ({header_file_name}, {items_file_name}).")
            return None, None

        print("Ambos os dataframes carregados com sucesso.")
        return header_df, items_df

    except Exception as e:
        print(f"Erro geral ao carregar dados dos arquivos locais: {e}")
        st.error(f"Erro ao carregar dados dos arquivos locais: {e}")
        return None, None

# Função principal para consultar a IA
def query_ai(question, header_df, items_df):
    """Processa a pergunta do usuário, realiza análises de dados e consulta a IA para obter uma resposta."""
    try:
        # Configuração do modelo LLM baseado na seleção do usuário
        modelo_llm = st.session_state.llm_model
        api_key = ""
        model_name = ""
        model_prefix = ""
        
        if modelo_llm == "groq":
            api_key = st.secrets["groq"]["api_key"]
            client = Groq(api_key=api_key)
            model_name = "llama3-8b-8192"
            model_prefix = "Groq diz: "
        elif modelo_llm == "openai":
            api_key = st.secrets["openai"]["api_key"]
            client = OpenAI(api_key=api_key)
            model_name = "gpt-4o"
            model_prefix = "OpenAI diz: "
        else:
            st.error("Modelo de LLM inválido. Por favor, escolha 'groq' ou 'openai'.")
            return "Erro: Configuração de modelo inválida."

        # Funções auxiliares para identificar colunas relevantes
        def find_supplier_column(df):
            """Tenta identificar a coluna que contém nomes de fornecedores em um DataFrame."""
            possible_supplier_cols = ['fornecedor', 'emitente', 'nome fornecedor', 'razao social']
            for col in df.columns:
                for possible_name in possible_supplier_cols:
                    if possible_name in col.lower():
                        if 'chave' not in col.lower() and 'cnpj' not in col.lower() and 'cpf' not in col.lower():
                            return col
            return None

        def find_chave_acesso_column(df):
             """Tenta identificar a coluna que contém as chaves de acesso das notas fiscais em um DataFrame."""
             possible_chave_cols = ['chave de acesso', 'chaveacesso', 'chave_acesso', 'nfkey']
             for col in df.columns:
                 for possible_name in possible_chave_cols:
                    if possible_name in col.lower():
                        return col
             return None

        def find_value_column(df):
            """Tenta identificar a coluna que contém o valor total das notas fiscais em um DataFrame."""
            possible_value_cols = ['valor total', 'valortotal', 'total nf', 'valornotafiscal', 'valor nota fiscal', 'valor', 'total']
            for col in df.columns:
                for possible_name in possible_value_cols:
                    if possible_name in col.lower():
                        return col
            return None

        # Análise específica para perguntas pré-definidas
        prompt_lower = question.lower()
        formatted_result = None

        # Lógica para top 10 fornecedores por valor
        if "10 maiores fornecedores por valor de nota fiscal" in prompt_lower:
            supplier_col = find_supplier_column(header_df)
            value_col = find_value_column(header_df)

            if supplier_col and value_col:
                try:
                    fornecedores_valor = header_df.groupby(supplier_col)[value_col].sum()
                    top_10_fornecedores_valor = fornecedores_valor.sort_values(ascending=False).head(10)
                    print(f"DEBUG - Top 10 Fornecedores por Valor (antes do LLM):\n{top_10_fornecedores_valor}")

                    formatted_result = "Os 10 maiores fornecedores por valor de Nota Fiscal são:\n\n"
                    if not top_10_fornecedores_valor.empty:
                        for fornecedor, valor in top_10_fornecedores_valor.items():
                            formatted_result += f"- {fornecedor}: R$ {valor:,.2f}\n"
                    else:
                        formatted_result += "Não foram encontrados dados de fornecedores ou valores de notas fiscais para esta análise."

                except Exception as e:
                    formatted_result = f"Erro interno ao calcular os 10 maiores fornecedores por valor de nota fiscal: {e}"
                    st.error(formatted_result)
            else:
                formatted_result = "Não foi possível identificar as colunas de fornecedor ou valor da nota fiscal nos dados para calcular o top 10 por valor."
                st.warning(formatted_result)

        # Lógica para top 10 fornecedores por quantidade de notas
        elif "10 nomes de fornecedores com mais notas fiscais" in prompt_lower or "top 10 fornecedores" in prompt_lower:
            supplier_col = find_supplier_column(header_df)
            chave_acesso_col = find_chave_acesso_column(header_df)

            if supplier_col and chave_acesso_col:
                try:
                    fornecedores_count = header_df.groupby(supplier_col)[chave_acesso_col].nunique()
                    top_10_fornecedores = fornecedores_count.sort_values(ascending=False).head(10)

                    formatted_result = "Os 10 fornecedores com mais Notas Fiscais são:\n\n"
                    if not top_10_fornecedores.empty:
                        for fornecedor, count in top_10_fornecedores.items():
                            formatted_result += f"- {fornecedor}: {count} Notas Fiscais\n"
                    else:
                        formatted_result += "Não foram encontrados dados de fornecedores ou notas fiscais para esta análise."

                except Exception as e:
                    formatted_result = f"Erro interno ao calcular os 10 principais fornecedores: {e}"
                    st.error(formatted_result)
            else:
                 formatted_result = "Não foi possível identificar as colunas de fornecedor ou chave de acesso nos dados para calcular o top 10."
                 st.warning(formatted_result)

        # Função para criar resumo dos dados para perguntas gerais
        def create_data_summary(df, name="Dados"):
            """Cria um resumo estatístico simplificado de um DataFrame para ser enviado à IA."""
            if df is None:
                return f"Nenhum {name} disponível para resumo."
            try:
                summary = {
                    'total_rows': len(df),
                    'columns': list(df.columns),
                    'sample': df.head(5).to_dict('records'),
                    'numeric_summary': df.describe().to_dict() if df.select_dtypes(include=['float64', 'int64']).shape[1] > 0 else {}
                }
                return f"RESUMO DOS {name.upper()}:\n{json.dumps(summary, indent=2)}"
            except Exception as e:
                 return f"Erro ao criar resumo dos {name}: {e}"

        # Preparação do contexto para a IA
        if formatted_result:
            data_context_for_groq = f"RESULTADO DA ANÁLISE PRÉVIA:\n{formatted_result}"
        else:
            header_summary_str = create_data_summary(header_df, name="Dados de Cabeçalho")
            items_summary_str = create_data_summary(items_df, name="Dados de Itens")
            data_context_for_groq = f"{header_summary_str}\n\n{items_summary_str}"

        # Construção do prompt final para a IA
        prompt_to_groq = f"""
        Você é um assistente especializado em análise de dados. Responda à pergunta do usuário baseado nos dados ou resumos fornecidos.
        \n\n{data_context_for_groq}\n\nPERGUNTA DO USUÁRIO: {question}\n\nInstruções:
        \n- Responda de forma clara e objetiva.\n- Use os dados/resumos fornecidos nos arquivos 202401_NFs_Cabecalho.csv e 202401_NFs_Itens.csv para fundamentar sua resposta.
        \n- Se um resultado específico foi fornecido (como uma lista de top 10), apresente esse resultado de forma amigável e ignore a análise dos dados brutos para essa parte.
        \n- Se a pergunta não puder ser respondida com os dados/resumos disponíveis, informe isso.
        \n- O arquivo 202401_NFs_Cabecalho.csv contém os dados de cabeçalho das notas fiscais e o arquivo 202401_NFs_Itens.csv contém os dados de itens das notas fiscais.
        \n- Se precisar de mais detalhes específicos que não estão no resumo, peça ao usuário para refinar a pergunta ou fornecer mais dados.\n"""

        # Chamada para a API da IA
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_to_groq}],
            temperature=0.1,
            max_tokens=1000
        )

        return model_prefix + response.choices[0].message.content
    except Exception as e:
        return f"Erro ao consultar a IA: {e}"

# Carregamento dos dados
header_df, items_df = load_data_from_local()

# Interface principal - só exibe se os dados foram carregados
if header_df is not None and items_df is not None:
    # Sidebar otimizada - apenas seleção do modelo LLM e exemplos de perguntas
    st.sidebar.header("⚙️ Configurações")
    
    # Seleção do modelo LLM movida para a sidebar
    selected_llm_model = st.sidebar.selectbox(
        "Escolha o Modelo de LLM:",
        ("groq", "openai"),
        index=("groq", "openai").index(st.session_state.llm_model),
        help="Selecione o modelo de Linguagem Grande (LLM) para as respostas."
    )

    # Atualização do modelo selecionado
    if selected_llm_model != st.session_state.llm_model:
        st.session_state.llm_model = selected_llm_model
        st.info(f"Modelo alterado para: **{st.session_state.llm_model.upper()}**")
        st.rerun()

    # Botão para limpar chat na sidebar
    if st.sidebar.button("🗑️ Limpar Chat"):
        st.session_state.messages = []
        st.rerun()

    # Exemplos de perguntas na sidebar
    st.sidebar.header("💡 Exemplos de Perguntas")
    example_questions = [
        "Quantas notas fiscais temos?",
        "Qual o valor total dos itens da nota fiscal com CHAVE DE ACESSO X?",
        "Liste os itens da nota fiscal com CHAVE DE ACESSO Y",
        "Qual a quantidade total de um determinado produto (descreva o produto)?",
        "Qual nota fiscal tem mais itens?",
        "Me mostre os 10 maiores fornecedores por valor de nota fiscal."
    ]

    st.sidebar.markdown("_Para testar perguntas específicas sobre notas fiscais, substitua X ou Y por uma CHAVE DE ACESSO real dos seus dados._")

    # Botões para perguntas de exemplo - preenchem o campo de digitação
    for question in example_questions:
        if st.sidebar.button(question, key=f"example_{question}"):
            st.session_state.example_question = question
            st.session_state.use_text_input = True
            st.session_state.edited_prompt = question  # Inicializa com a pergunta de exemplo
            st.rerun()

    # Interface de chat otimizada - área principal focada apenas no chat
    st.header("💬 Chat")

    # Inicialização do histórico de mensagens
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibição do histórico de mensagens
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.markdown(message["content"])

            # Botão de recarregar para mensagens do usuário
            if message["role"] == "user":
                with col2:
                    if st.button("🔄", key=f"reload_btn_{i}", help="Refazer esta pergunta"):
                        st.session_state.re_ask_prompt = message["original_prompt"]
                        st.rerun()

    # Input para nova pergunta - usa text_input quando há exemplo selecionado
    prompt = None
    
    if st.session_state.use_text_input and st.session_state.example_question:
        # Usar text_input para permitir edição da pergunta de exemplo
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.session_state.edited_prompt = st.text_input(
                "Edite sua pergunta:",
                value=st.session_state.example_question,
                key="editable_question"
            )
        with col2:
            if st.button("Enviar", type="primary"):
                prompt = st.session_state.edited_prompt
                # Resetar estados
                st.session_state.use_text_input = False
                st.session_state.example_question = ""
                # Processar pergunta
                st.session_state.messages.append({"role": "user", "content": prompt, "original_prompt": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Obtenção e exibição da resposta da IA
                with st.chat_message("assistant"):
                    with st.spinner("Analisando os dados..."):
                        response = query_ai(prompt, header_df, items_df)
                        st.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

        # Botão para cancelar edição
        if st.button("❌ Cancelar"):
            st.session_state.use_text_input = False
            st.session_state.example_question = ""
            st.rerun()
    else:
        # Usar chat_input normal
        prompt = st.chat_input("Faça uma pergunta sobre as notas fiscais...")

    # Processa a pergunta se o usuário digitou algo (apenas para chat_input)
    if prompt and not st.session_state.use_text_input:
        st.session_state.messages.append({"role": "user", "content": prompt, "original_prompt": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Obtenção e exibição da resposta da IA
        with st.chat_message("assistant"):
            with st.spinner("Analisando os dados..."):
                response = query_ai(prompt, header_df, items_df)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    # Lógica para refazer perguntas
    if "re_ask_prompt" in st.session_state and st.session_state.re_ask_prompt:
        prompt_to_re_ask = st.session_state.re_ask_prompt
        st.session_state.re_ask_prompt = None

        st.session_state.messages.append({"role": "user", "content": prompt_to_re_ask, "original_prompt": prompt_to_re_ask})
        with st.chat_message("user"):
            st.markdown(prompt_to_re_ask)

        with st.chat_message("assistant"):
            with st.spinner("Analisando os dados (refazendo pergunta)..."):
                response = query_ai(prompt_to_re_ask, header_df, items_df)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    # Mensagem quando dados não são carregados
    st.warning("Não foi possível carregar os dados das Notas Fiscais. Verifique se os arquivos CSV estão no diretório correto.")
    if "messages" in st.session_state:
        st.session_state.messages = []