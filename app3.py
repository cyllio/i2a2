
import streamlit as st
import pandas as pd
import os
import zipfile
import base64
import requests
import json
from openai import OpenAI
from groq import Groq

# Função para upload de arquivo para o GitHub via API
def upload_file_to_github(local_file_path, repo, path_in_repo, branch, token):
    with open(local_file_path, "rb") as f:
        content = f.read()
    content_b64 = base64.b64encode(content).decode()

    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }

    # Verifica se o arquivo já existe para pegar o SHA
    r = requests.get(url + f"?ref={branch}", headers=headers)
    if r.status_code == 200:
        sha = r.json()["sha"]
    else:
        sha = None

    data = {
        "message": f"Upload automático de {os.path.basename(local_file_path)}",
        "content": content_b64,
        "branch": branch
    }
    if sha:
        data["sha"] = sha

    response = requests.put(url, headers=headers, json=data)
    if response.status_code in [200, 201]:
        return True, f"Arquivo {path_in_repo} carregado para o sistema com sucesso!"
    else:
        return False, f"Erro ao enviar {path_in_repo}: {response.text}"

# Inicialização do estado da sessão
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "groq"
if "example_question" not in st.session_state:
    st.session_state.example_question = ""
if "use_text_input" not in st.session_state:
    st.session_state.use_text_input = False
if "edited_prompt" not in st.session_state:
    st.session_state.edited_prompt = ""

st.set_page_config(
    page_title="Agente Smart",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Agente de Notas Fiscais Smart - Grupo Comunicação")
st.markdown("Faça perguntas sobre os dados da planilha e receba respostas inteligentes!")

# Seção de upload de arquivo ZIP
st.header("📁 Upload de Arquivos")
uploaded_file = st.file_uploader(
    "Faça upload de um arquivo ZIP contendo os CSVs:",
    type=['zip'],
    help="Arraste e solte ou clique para selecionar um arquivo ZIP contendo os arquivos CSV necessários. Os arquivos serão extraídos para o repositório GitHub e sobrescreverão arquivos existentes."
)

# Processamento do arquivo ZIP e upload para o GitHub
if uploaded_file is not None:
    try:
        github_repo_dir = os.getcwd()
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(github_repo_dir)
        st.success("✅ Arquivo ZIP extraído com sucesso para o diretório do sistema!")

        # Lista os arquivos extraídos
        extracted_files = [f for f in os.listdir(github_repo_dir) if f.endswith('.csv')]
        if extracted_files:
            st.write("**Arquivos CSV encontrados:**")
            for file in extracted_files:
                st.write(f"- {file}")

            # Upload para o GitHub
            github_token = st.secrets["github"]["token"]
            github_repo = st.secrets["github"]["repo"]
            github_branch = st.secrets["github"]["branch"]

            st.info("Enviando arquivos extraídos para o diretório do sistema...")
            for file in extracted_files:
                local_path = os.path.join(github_repo_dir, file)
                ok, msg = upload_file_to_github(
                    local_file_path=local_path,
                    repo=github_repo,
                    path_in_repo=file,
                    branch=github_branch,
                    token=github_token
                )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
    except Exception as e:
        st.error(f"Erro ao processar o arquivo ZIP: {e}")

# Função para carregar dados dos arquivos CSV locais
@st.cache_data
def load_data_from_local():
    header_file_name = "202401_NFs_Cabecalho.csv"
    items_file_name = "202401_NFs_Itens.csv"
    header_df = None
    items_df = None

    try:
        github_repo_dir = os.getcwd()
        header_file_path = os.path.join(github_repo_dir, header_file_name)
        items_file_path = os.path.join(github_repo_dir, items_file_name)

        if os.path.exists(header_file_path):
            header_df = pd.read_csv(header_file_path)
        else:
            st.warning(f"Arquivo não encontrado: {header_file_path}")

        if os.path.exists(items_file_path):
            items_df = pd.read_csv(items_file_path)
        else:
            st.warning(f"Arquivo não encontrado: {items_file_path}")

        if header_df is None or items_df is None:
            st.warning(f"Não foi possível encontrar um ou ambos os arquivos CSV esperados ({header_file_name}, {items_file_name}).")
            return None, None

        return header_df, items_df

    except Exception as e:
        st.error(f"Erro ao carregar dados dos arquivos locais: {e}")
        return None, None

# Função principal para consultar a IA (igual ao seu código original)
def query_ai(question, header_df, items_df):
    try:
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

        def find_supplier_column(df):
            possible_supplier_cols = ['fornecedor', 'emitente', 'nome fornecedor', 'razao social']
            for col in df.columns:
                for possible_name in possible_supplier_cols:
                    if possible_name in col.lower():
                        if 'chave' not in col.lower() and 'cnpj' not in col.lower() and 'cpf' not in col.lower():
                            return col
            return None

        def find_chave_acesso_column(df):
             possible_chave_cols = ['chave de acesso', 'chaveacesso', 'chave_acesso', 'nfkey']
             for col in df.columns:
                 for possible_name in possible_chave_cols:
                    if possible_name in col.lower():
                        return col
             return None

        def find_value_column(df):
            possible_value_cols = ['valor total', 'valortotal', 'total nf', 'valornotafiscal', 'valor nota fiscal', 'valor', 'total']
            for col in df.columns:
                for possible_name in possible_value_cols:
                    if possible_name in col.lower():
                        return col
            return None

        prompt_lower = question.lower()
        formatted_result = None

        if "10 maiores fornecedores por valor de nota fiscal" in prompt_lower:
            supplier_col = find_supplier_column(header_df)
            value_col = find_value_column(header_df)

            if supplier_col and value_col:
                try:
                    fornecedores_valor = header_df.groupby(supplier_col)[value_col].sum()
                    top_10_fornecedores_valor = fornecedores_valor.sort_values(ascending=False).head(10)
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

        def create_data_summary(df, name="Dados"):
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

        if formatted_result:
            data_context_for_groq = f"RESULTADO DA ANÁLISE PRÉVIA:\n{formatted_result}"
        else:
            header_summary_str = create_data_summary(header_df, name="Dados de Cabeçalho")
            items_summary_str = create_data_summary(items_df, name="Dados de Itens")
            data_context_for_groq = f"{header_summary_str}\n\n{items_summary_str}"

        prompt_to_groq = f"""
        Você é um assistente especializado em análise de dados. Responda à pergunta do usuário baseado nos dados ou resumos fornecidos.
        \n\n{data_context_for_groq}\n\nPERGUNTA DO USUÁRIO: {question}\n\nInstruções:
        \n- Responda de forma clara e objetiva.\n- Use os dados/resumos fornecidos nos arquivos 202401_NFs_Cabecalho.csv e 202401_NFs_Itens.csv para fundamentar sua resposta.
        \n- Se um resultado específico foi fornecido (como uma lista de top 10), apresente esse resultado de forma amigável e ignore a análise dos dados brutos para essa parte.
        \n- Se a pergunta não puder ser respondida com os dados/resumos disponíveis, informe isso.
        \n- O arquivo 202401_NFs_Cabecalho.csv contém os dados de cabeçalho das notas fiscais e o arquivo 202401_NFs_Itens.csv contém os dados de itens das notas fiscais.
        \n- Se precisar de mais detalhes específicos que não estão no resumo, peça ao usuário para refinar a pergunta ou fornecer mais dados.\n"""

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
    st.sidebar.header("⚙️ Configurações")
    selected_llm_model = st.sidebar.selectbox(
        "Escolha o Modelo de LLM:",
        ("groq", "openai"),
        index=("groq", "openai").index(st.session_state.llm_model),
        help="Selecione o modelo de Linguagem Grande (LLM) para as respostas."
    )

    if selected_llm_model != st.session_state.llm_model:
        st.session_state.llm_model = selected_llm_model
        st.info(f"Modelo alterado para: **{st.session_state.llm_model.upper()}**")
        st.rerun()

    if st.sidebar.button("🗑️ Limpar Chat"):
        st.session_state.messages = []
        st.rerun()

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

    for question in example_questions:
        if st.sidebar.button(question, key=f"example_{question}"):
            st.session_state.example_question = question
            st.session_state.use_text_input = True
            st.session_state.edited_prompt = question
            st.rerun()

    st.header("💬 Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.markdown(message["content"])
            if message["role"] == "user":
                with col2:
                    if st.button("🔄", key=f"reload_btn_{i}", help="Refazer esta pergunta"):
                        st.session_state.re_ask_prompt = message["original_prompt"]
                        st.rerun()

    prompt = None
    if st.session_state.use_text_input and st.session_state.example_question:
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
                st.session_state.use_text_input = False
                st.session_state.example_question = ""
                st.session_state.messages.append({"role": "user", "content": prompt, "original_prompt": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Analisando os dados..."):
                        response = query_ai(prompt, header_df, items_df)
                        st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        if st.button("❌ Cancelar"):
            st.session_state.use_text_input = False
            st.session_state.example_question = ""
            st.rerun()
    else:
        prompt = st.chat_input("Faça uma pergunta sobre as notas fiscais...")

    if prompt and not st.session_state.use_text_input:
        st.session_state.messages.append({"role": "user", "content": prompt, "original_prompt": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analisando os dados..."):
                response = query_ai(prompt, header_df, items_df)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

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
    st.warning("Não foi possível carregar os dados das Notas Fiscais. Verifique se os arquivos CSV estão no diretório correto ou faça upload de um arquivo ZIP contendo os dados.")
    if "messages" in st.session_state:
        st.session_state.messages = []