# Importa a biblioteca Streamlit para criar a interface web.
import streamlit as st
# Importa a biblioteca Pandas para manipulação e análise de dados (DataFrames).
import pandas as pd
# Importa a biblioteca requests para fazer requisições HTTP (embora não seja usada no fluxo atual de carregamento local).
import requests
# Importa a classe OpenAI para interagir com a API do OpenAI.
from openai import OpenAI
# Importa a biblioteca json para trabalhar com dados JSON.
import json
# Importa a biblioteca os para interagir com o sistema operacional, como caminhos de arquivo.
import os
# Importa StringIO para tratar strings como arquivos, útil para ler CSVs em memória.
from io import StringIO
# Importa a classe Groq para interagir com a API da Groq.
from groq import Groq

# Inicializar o estado do modelo LLM se não estiver definido
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "groq" # Modelo padrão

# Configuração da página do Streamlit.
st.set_page_config(
    page_title="Chatbot Planilha", # Define o título que aparece na aba do navegador.
    page_icon="📊", # Define o ícone da página na aba do navegador.
    layout="wide" # Define o layout da página como "wide" para ocupar mais espaço.
)

# Título principal exibido na aplicação Streamlit.
st.title("📊 Chatbot da Planilha")
# Subtítulo/descrição abaixo do título.
st.markdown("Faça perguntas sobre os dados da planilha e receba respostas inteligentes!")

# Adicionar o selectbox para seleção do modelo LLM
selected_llm_model = st.selectbox(
    "Escolha o Modelo de LLM:",
    ("groq", "openai"),
    index=("groq", "openai").index(st.session_state.llm_model),
    help="Selecione o modelo de Linguagem Grande (LLM) para as respostas."
)
# Atualizar o estado da sessão com a escolha do usuário
if selected_llm_model != st.session_state.llm_model:
    st.session_state.llm_model = selected_llm_model
    st.info(f"Modelo alterado para: **{st.session_state.llm_model.upper()}**")
    st.rerun() # Reruns para aplicar a mudança do modelo imediatamente.

# Função para carregar dados dos arquivos CSV locais.
# O decorador `@st.cache_data` faz com que o Streamlit "cacheie" o resultado desta função.
# Isso significa que, se os argumentos da função não mudarem, ela não será executada novamente,
# o que acelera a aplicação ao evitar o recarregamento de dados a cada interação do usuário.
@st.cache_data
def load_data_from_local():
    """Carrega dados de arquivos CSV especificados do diretório local."""
    print("Iniciando carregamento de dados dos arquivos locais...") # Mensagem de log para console.

    header_file_name = "202401_NFs_Cabecalho.csv" # Nome do arquivo CSV de cabeçalho.
    items_file_name = "202401_NFs_Itens.csv"     # Nome do arquivo CSV de itens.

    header_df = None # Inicializa o DataFrame para o cabeçalho como None.
    items_df = None  # Inicializa o DataFrame para os itens como None.

    try:
        # Obtém o diretório (pasta) onde o script Python atual (`app3.py`) está sendo executado.
        script_dir = os.path.dirname(__file__)
        # Constrói o caminho completo para o arquivo de cabeçalho,
        # unindo o diretório do script com o nome do arquivo.
        header_file_path = os.path.join(script_dir, header_file_name)
        # Constrói o caminho completo para o arquivo de itens.
        items_file_path = os.path.join(script_dir, items_file_name)

        print(f"Tentando carregar '{header_file_path}'...") # Log do caminho do arquivo de cabeçalho.
        # Verifica se o arquivo de cabeçalho existe no caminho especificado.
        if os.path.exists(header_file_path):
            # Se existir, lê o arquivo CSV para um DataFrame Pandas.
            header_df = pd.read_csv(header_file_path)
            print(f"{header_file_name} carregado. Shape: {header_df.shape}") # Log de sucesso e dimensões do DataFrame.
        else:
            # Se o arquivo não for encontrado, exibe um aviso no console e no Streamlit.
            print(f"Arquivo não encontrado: {header_file_path}")
            st.warning(f"Arquivo não encontrado: {header_file_path}")

        print(f"Tentando carregar '{items_file_path}'...") # Log do caminho do arquivo de itens.
        # Verifica se o arquivo de itens existe.
        if os.path.exists(items_file_path):
            # Se existir, lê o arquivo CSV para um DataFrame Pandas.
            items_df = pd.read_csv(items_file_path)
            print(f"{items_file_name} carregado. Shape: {items_df.shape}") # Log de sucesso e dimensões do DataFrame.
        else:
            # Se o arquivo não for encontrado, exibe um aviso no console e no Streamlit.
            print(f"Arquivo não encontrado: {items_file_path}")
            st.warning(f"Arquivo não encontrado: {items_file_path}")

        # Verifica se ambos os DataFrames foram carregados com sucesso (não são None).
        if header_df is None or items_df is None:
            # Se um ou ambos não foram encontrados, exibe um aviso e retorna None para ambos.
            print(f"Não foi possível encontrar um ou ambos os arquivos CSV esperados ({header_file_name}, {items_file_name}).")
            st.warning(f"Não foi possível encontrar um ou ambos os arquivos CSV esperados ({header_file_name}, {items_file_name}).")
            return None, None

        print("Ambos os dataframes carregados com sucesso.") # Log de sucesso no carregamento.
        return header_df, items_df # Retorna os dois DataFrames carregados.

    except Exception as e:
        # Captura qualquer erro que ocorra durante o processo de carregamento de arquivos.
        print(f"Erro geral ao carregar dados dos arquivos locais: {e}")
        st.error(f"Erro ao carregar dados dos arquivos locais: {e}") # Exibe o erro no Streamlit.
        return None, None # Em caso de erro, retorna None para ambos os DataFrames.

def query_ai(question, header_df, items_df):
    """Processa a pergunta do usuário, realiza análises de dados e consulta a IA para obter uma resposta."""
    try:
        # Carregar a chave da API dos segredos do Streamlit
        # openai_api_key = st.secrets["openai"]["api_key"] # A chave da API do OpenAI é carregada aqui dos segredos do Streamlit.
        # client = OpenAI(api_key=openai_api_key) # O cliente OpenAI é inicializado com a chave da API.

        # Define qual modelo de LLM será utilizado (pode ser 'groq' ou 'openai')
        modelo_llm = st.session_state.llm_model # Usar o modelo selecionado no selectbox

        # Configurações específicas para cada modelo
        api_key = ""
        model_name = ""
        model_prefix = "" # Novo: prefixo para a resposta
        if modelo_llm == "groq":
            api_key = st.secrets["groq"]["api_key"]
            client = Groq(api_key=api_key)
            model_name = "llama3-8b-8192" # Exemplo de modelo da Groq
            model_prefix = "Groq diz: "
        elif modelo_llm == "openai":
            api_key = st.secrets["openai"]["api_key"]
            client = OpenAI(api_key=api_key)
            model_name = "gpt-4o" # Exemplo de modelo da OpenAI
            model_prefix = "OpenAI diz: "
        else:
            st.error("Modelo de LLM inválido. Por favor, escolha 'groq' ou 'openai'.")
            return "Erro: Configuração de modelo inválida."

        # Função auxiliar para tentar encontrar a coluna de fornecedor
        def find_supplier_column(df):
            """Tenta identificar a coluna que contém nomes de fornecedores em um DataFrame."""
            # Nomes de colunas comuns para fornecedor
            possible_supplier_cols = ['fornecedor', 'emitente', 'nome fornecedor', 'razao social']
            for col in df.columns:
                for possible_name in possible_supplier_cols:
                    if possible_name in col.lower():
                        # Adicionar verificação adicional para evitar falsos positivos como 'chave do fornecedor'
                        if 'chave' not in col.lower() and 'cnpj' not in col.lower() and 'cpf' not in col.lower():
                            return col # Retorna o nome da coluna encontrada.
            return None # Retorna None se não encontrar uma coluna de fornecedor adequada.

        # Função auxiliar para tentar encontrar a coluna de chave de acesso
        def find_chave_acesso_column(df):
             """Tenta identificar a coluna que contém as chaves de acesso das notas fiscais em um DataFrame."""
             # Nomes de colunas comuns para chave de acesso
             possible_chave_cols = ['chave de acesso', 'chaveacesso', 'chave_acesso', 'nfkey']
             for col in df.columns:
                 for possible_name in possible_chave_cols:
                    if possible_name in col.lower():
                        return col # Retorna o nome da coluna encontrada.
             return None # Retorna None se não encontrar uma coluna de chave de acesso adequada.

        # Função auxiliar para tentar encontrar a coluna de valor total da nota fiscal
        def find_value_column(df):
            """Tenta identificar a coluna que contém o valor total das notas fiscais em um DataFrame."""
            # Nomes de colunas comuns para valor total
            possible_value_cols = ['valor total', 'valortotal', 'total nf', 'valornotafiscal', 'valor nota fiscal', 'valor', 'total']
            for col in df.columns:
                for possible_name in possible_value_cols:
                    if possible_name in col.lower():
                        return col
            return None

        prompt_lower = question.lower() # Converte a pergunta do usuário para minúsculas para facilitar a comparação.
        formatted_result = None # Variável para armazenar resultados pré-calculados de análises específicas.

        # --- Lógica de detecção e cálculo para perguntas específicas ---
        # Verifica se a pergunta do usuário solicita os 10 maiores fornecedores por valor de nota fiscal.
        if "10 maiores fornecedores por valor de nota fiscal" in prompt_lower:
            supplier_col = find_supplier_column(header_df)
            value_col = find_value_column(header_df)

            if supplier_col and value_col:
                try:
                    # Agrupa o DataFrame de cabeçalho pela coluna do fornecedor e soma o valor total das notas fiscais.
                    fornecedores_valor = header_df.groupby(supplier_col)[value_col].sum()

                    # Obtém os top 10 fornecedores com base no valor total, em ordem decrescente.
                    top_10_fornecedores_valor = fornecedores_valor.sort_values(ascending=False).head(10)

                    # DEBUG: Imprimir o resultado do cálculo antes de enviar para a IA
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
        # Verifica se a pergunta do usuário solicita os 10 principais fornecedores (por quantidade de notas).
        elif "10 nomes de fornecedores com mais notas fiscais" in prompt_lower or "top 10 fornecedores" in prompt_lower:
            # Tenta encontrar as colunas de fornecedor e chave de acesso no DataFrame de cabeçalho.
            supplier_col = find_supplier_column(header_df)
            chave_acesso_col = find_chave_acesso_column(header_df)

            # Se ambas as colunas forem encontradas, procede com o cálculo.
            if supplier_col and chave_acesso_col:
                try:
                    # Contar notas fiscais por fornecedor:
                    # Agrupa o DataFrame de cabeçalho pela coluna do fornecedor
                    # e conta o número de chaves de acesso únicas para cada fornecedor.
                    # Isso garante que cada nota fiscal seja contada uma única vez, mesmo que haja linhas duplicadas no cabeçalho.
                    fornecedores_count = header_df.groupby(supplier_col)[chave_acesso_col].nunique()

                    # Obtém os top 10 fornecedores com base na contagem de notas fiscais, em ordem decrescente.
                    top_10_fornecedores = fornecedores_count.sort_values(ascending=False).head(10)

                    # Formata o resultado para ser incluído no prompt da IA como uma string legível.
                    formatted_result = "Os 10 fornecedores com mais Notas Fiscais são:\n\n"
                    if not top_10_fornecedores.empty:
                        # Itera sobre os top 10 fornecedores e adiciona cada um ao resultado formatado.
                        for fornecedor, count in top_10_fornecedores.items():
                            formatted_result += f"- {fornecedor}: {count} Notas Fiscais\n"
                    else:
                        formatted_result += "Não foram encontrados dados de fornecedores ou notas fiscais para esta análise." # Mensagem se não houver dados.


                except Exception as e:
                    # Em caso de erro durante o cálculo, informa a IA e exibe um erro no Streamlit.
                    formatted_result = f"Erro interno ao calcular os 10 principais fornecedores: {e}"
                    st.error(formatted_result) # Exibe o erro no Streamlit.

            else:
                 # Se as colunas necessárias não forem encontradas, informa a IA e exibe um aviso no Streamlit.
                 formatted_result = "Não foi possível identificar as colunas de fornecedor ou chave de acesso nos dados para calcular o top 10."
                 st.warning(formatted_result) # Exibe o warning no Streamlit.
        # --- Fim da lógica de detecção e cálculo ---

        # Função para criar um resumo estatístico dos dados (usada para perguntas gerais)
        def create_data_summary(df, name="Dados"): # Adicionado nome para identificar no prompt
            """Cria um resumo estatístico simplificado de um DataFrame para ser enviado à IA."""
            if df is None: # Lida com o caso onde o DataFrame não está disponível.
                return f"Nenhum {name} disponível para resumo."
            try:
                # Limita o resumo para economizar tokens que seriam enviados à IA.
                summary = {
                    'total_rows': len(df), # Número total de linhas no DataFrame.
                    'columns': list(df.columns), # Lista de todas as colunas.
                    'sample': df.head(5).to_dict('records'), # Amostra das 5 primeiras linhas como lista de dicionários.
                    # Resumo estatístico para colunas numéricas, se existirem.
                    'numeric_summary': df.describe().to_dict() if df.select_dtypes(include=['float64', 'int64']).shape[1] > 0 else {}
                }
                # Retorna o resumo formatado como uma string JSON legível.
                return f"RESUMO DOS {name.upper()}:\n{json.dumps(summary, indent=2)}"
            except Exception as e:
                 return f"Erro ao criar resumo dos {name}: {e}"


        # Preparar o contexto para a IA
        # Decide qual contexto enviar à IA: um resultado pré-calculado ou um resumo dos DataFrames.
        if formatted_result:
            # Se um resultado específico (ex: top 10 fornecedores) foi pré-calculado,
            # o prompt incluirá esse resultado já processado.
            data_context_for_groq = f"RESULTADO DA ANÁLISE PRÉVIA:\n{formatted_result}"
        else:
            # Para perguntas gerais, cria resumos dos DataFrames de cabeçalho e itens.
            header_summary_str = create_data_summary(header_df, name="Dados de Cabeçalho")
            items_summary_str = create_data_summary(items_df, name="Dados de Itens")
            data_context_for_groq = f"{header_summary_str}\n\n{items_summary_str}" # Concatena os resumos.

        # Construir o prompt final para a Groq (modelo de IA)
        # O prompt é ajustado para lidar com resultados pré-calculados ou resumos de dados.
        prompt_to_groq = f"""
        Você é um assistente especializado em análise de dados. Responda à pergunta do usuário baseado nos dados ou resumos fornecidos.
        \n\n{data_context_for_groq}\n\nPERGUNTA DO USUÁRIO: {question}\n\nInstruções:
        \n- Responda de forma clara e objetiva.\n- Use os dados/resumos fornecidos nos arquivos 202401_NFs_Cabecalho.csv e 202401_NFs_Itens.csv para fundamentar sua resposta.
        \n- Se um resultado específico foi fornecido (como uma lista de top 10), apresente esse resultado de forma amigável e ignore a análise dos dados brutos para essa parte.
        \n- Se a pergunta não puder ser respondida com os dados/resumos disponíveis, informe isso.
        \n- O arquivo 202401_NFs_Cabecalho.csv contém os dados de cabeçalho das notas fiscais e o arquivo 202401_NFs_Itens.csv contém os dados de itens das notas fiscais.
        \n- Se precisar de mais detalhes específicos que não estão no resumo, peça ao usuário para refinar a pergunta ou fornecer mais dados.\n"""

        # Chama a API do OpenAI com o modelo "gpt-4o", a mensagem do usuário (que inclui o prompt construído)
        # Define a temperatura (criatividade da resposta) e o máximo de tokens.
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_to_groq}],
            temperature=0.1,
            max_tokens=1000
        )

        return model_prefix + response.choices[0].message.content # Retorna o conteúdo da resposta da IA com o prefixo.
    except Exception as e:
        # Captura e retorna qualquer erro que ocorra durante a consulta à IA.
        return f"Erro ao consultar a IA: {e}"

# Carregar dados
# Chamando a nova função para carregar dados dos arquivos locais
header_df, items_df = load_data_from_local()

# Verificar se os dados foram carregados com sucesso antes de continuar
if header_df is not None and items_df is not None:
    st.sidebar.header("📋 Dados das Notas Fiscais")
    st.sidebar.write(f"**Total de Notas Fiscais (Cabeçalho):** {len(header_df)}")
    st.sidebar.write(f"**Total de Itens de Notas Fiscais:** {len(items_df)}")
    st.sidebar.write(f"**Colunas Cabeçalho:** {', '.join(header_df.columns)}")
    st.sidebar.write(f"**Colunas Itens:** { ', '.join(items_df.columns)}")

    # Mostrar preview dos dados
    with st.expander("👀 Visualizar dados do Cabeçalho"):
        st.dataframe(header_df)
    with st.expander("👀 Visualizar dados dos Itens"):
        st.dataframe(items_df)

    # Chat interface
    st.header("💬 Chat")

    # Inicializar histórico do chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar histórico do chat
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Use um container para o conteúdo e o botão de recarregar
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.markdown(message["content"])

            # Adicionar botão de recarregar apenas para mensagens do usuário
            if message["role"] == "user":
                with col2:
                    if st.button("🔄", key=f"reload_btn_{i}", help="Refazer esta pergunta"):
                        st.session_state.re_ask_prompt = message["original_prompt"]
                        st.rerun()

    # Input do usuário para digitar a pergunta.
    # `st.chat_input` fornece uma caixa de texto na parte inferior da interface de chat.
    if prompt := st.chat_input("Faça uma pergunta sobre as notas fiscais..."):
        # Adiciona a mensagem do usuário ao histórico do chat.
        st.session_state.messages.append({"role": "user", "content": prompt, "original_prompt": prompt}) # Armazena o prompt original
        with st.chat_message("user"):
            st.markdown(prompt) # Exibe a pergunta do usuário na interface.

        # Obter resposta da IA.
        with st.chat_message("assistant"):
            with st.spinner("Analisando os dados..."):
                # Chama a função `query_ai` para obter a resposta da LLM.
                response = query_ai(prompt, header_df, items_df)
                st.markdown(response) # Exibe a resposta da IA na interface.

        # Adiciona a resposta da IA ao histórico do chat.
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Lógica para refazer uma pergunta se o botão de recarregar foi clicado
    if "re_ask_prompt" in st.session_state and st.session_state.re_ask_prompt:
        prompt_to_re_ask = st.session_state.re_ask_prompt
        st.session_state.re_ask_prompt = None # Limpa a flag

        st.session_state.messages.append({"role": "user", "content": prompt_to_re_ask, "original_prompt": prompt_to_re_ask})
        with st.chat_message("user"):
            st.markdown(prompt_to_re_ask)

        with st.chat_message("assistant"):
            with st.spinner("Analisando os dados (refazendo pergunta)..."):
                response = query_ai(prompt_to_re_ask, header_df, items_df)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Botão na barra lateral para limpar o histórico do chat.
    if st.sidebar.button("🗑️ Limpar Chat"):
        st.session_state.messages = [] # Limpa todas as mensagens do histórico.
        st.rerun() # Reinicia a aplicação para refletir a limpeza do chat.

    # Seção na barra lateral para exemplos de perguntas.
    st.sidebar.header("💡 Exemplos de Perguntas")
    example_questions = [
        "Quantas notas fiscais temos?",
        "Qual o valor total dos itens da nota fiscal com CHAVE DE ACESSO X?", # Substituir X por uma chave real nos exemplos
        "Liste os itens da nota fiscal com CHAVE DE ACESSO Y", # Substituir Y por uma chave real
        "Qual a quantidade total de um determinado produto (descreva o produto)?",
        "Qual nota fiscal tem mais itens?",
        "Me mostre os 10 maiores fornecedores por valor de nota fiscal."
    ]

    # Adicionar um aviso sobre os exemplos de perguntas.
    st.sidebar.markdown("_Para testar perguntas específicas sobre notas fiscais, substitua X ou Y por uma CHAVE DE ACESSO real dos seus dados._")

    # Cria botões para cada exemplo de pergunta.
    for question in example_questions:
        if st.sidebar.button(question, key=f"example_{question}"):
            # Quando um botão de exemplo é clicado, adiciona a pergunta ao histórico do chat
            # e força um "rerun" para que a pergunta apareça no `st.chat_input`.
            # A resposta da IA só será gerada quando o usuário "enviar" a pergunta no chat input.
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

else:
    # Mensagem mostrada se os dados não forem carregados (ex: arquivos CSV não encontrados).
    st.warning("Não foi possível carregar os dados das Notas Fiscais. Verifique se os arquivos CSV estão no diretório correto.")
    # Limpa o histórico do chat se os dados não carregarem para evitar erros em um estado inválido.
    if "messages" in st.session_state:
        st.session_state.messages = []
