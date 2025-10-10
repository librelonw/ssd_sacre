import mysql.connector
import streamlit as st
import os
import base64
from PyPDF2 import PdfReader

def show():
    # Configurações do banco de dados
    MYSQL_HOST = "localhost"
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "dbssd@#"
    MYSQL_DB = "db_ssd"
    PASTA_PDFS = "artigos"  

    # Garantir que a pasta de PDFs exista
    if not os.path.exists(PASTA_PDFS):
        os.makedirs(PASTA_PDFS)


    # Função para conectar ao banco de dados
    def conectar_banco():
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        return conn

    # Função para inserir um artigo no banco
    def inserir_artigo(titulo, resumo, abstract, doi, pasta_pdf):
        conn = conectar_banco()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO artigos (titulo, resumo, abstract, doi, pasta_pdf) VALUES (%s, %s, %s, %s, %s)", (titulo, resumo, abstract, doi, pasta_pdf))
        conn.commit()
        conn.close()

    # Função para buscar artigos
    def buscar_artigos():
        conn = conectar_banco()
        cursor = conn.cursor(dictionary=True)  # Retorna os resultados como dicionários
        cursor.execute("SELECT id_artigo, titulo, resumo, abstract, doi,  pasta_pdf FROM artigos")
        artigos = cursor.fetchall()
        conn.close()
        return artigos

    # Função para exibir PDF no app
    def exibir_pdf_no_app(caminho_pdf, altura=1000):
        """Exibe o PDF renderizado na tela, usando a largura máxima do navegador."""
        with open(caminho_pdf, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
            pdf_display = f"""
            <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="{altura}" type="application/pdf"></iframe>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)

    # Tabela de artigos com resumo e botão para exibir PDF
    def exibir_tabela_com_resumo_e_pdf():
        artigos = buscar_artigos()  # Buscar do banco os dados dos artigos

        if not artigos:
            st.warning("Nenhum artigo encontrado no banco de dados.")
            return

        # Dividir a interface em duas colunas
        margem, col1, col2, col3 = st.columns([0.05, 1, 1, 1])  # 1/3 para a tabela, 2/3 para o resumo

        with col1:
            st.subheader("Artigos Publicados")
            for idx, artigo in enumerate(artigos):
                st.write(f"**{artigo['id_artigo']} - {artigo['titulo']}**")

                col_res, col_doi = st.columns([1,4])

                with col_res:
                    # Botão para visualizar o resumo
                    if st.button("Resumo", key=f"resumo_{artigo['id_artigo']}"):
                        st.session_state['artigo_selecionado'] = artigo
                with col_doi:
                    # Link clicável para o DOI
                    if artigo.get("doi"):
                        st.markdown(
                            f"""<a href="{artigo['doi']}" target="_blank" style="color: #1f77b4; text-decoration: none;">
                            🔗 Acessar DOI</a>""",
                            unsafe_allow_html=True
                        )
                if idx < len(artigos) - 1:
                    st.markdown("""<hr style="margin-top: 3px; margin-bottom: 2px;">""",unsafe_allow_html=True)

            # st.info("Clique em 'Resumo' para ver mais detalhes ou 'Acessar DOI' para abrir o artigo online.")

        # Coluna 2: Exibição do Resumo inglÊs
        with col2:
            st.subheader("Abstract")
            artigo_selecionado = st.session_state.get('artigo_selecionado')  # Resgatar o artigo selecionado

            if artigo_selecionado:
                # Exibição do Resumo
                st.text_area("", artigo_selecionado["abstract"], height=400, disabled=True)

        # Coluna 3: Exibição do Resumo portuguÊs
        with col3:
            st.subheader("Resumo")
            artigo_selecionado = st.session_state.get('artigo_selecionado')  # Resgatar o artigo selecionado

            if artigo_selecionado:
                # Exibição do Resumo
                st.text_area("", artigo_selecionado["resumo"], height=400, disabled=True)

        st.markdown("""<hr style="margin-top: 3px; margin-bottom: 2px;">""",unsafe_allow_html=True)

        # Formulário de Edição abaixo das colunas
        if st.session_state.get('modo_edicao', False) and artigo_selecionado:
            st.markdown("---")
            st.subheader("Editar Artigo Selecionado")

            novo_titulo = st.text_input("Título do Artigo", value=artigo_selecionado['titulo'])
            novo_resumo = st.text_area("Resumo", value=artigo_selecionado['resumo'])
            novo_abstract = st.text_area("Abstract", value=artigo_selecionado['abstract'])
            novo_doi = st.text_input("DOI", value=artigo_selecionado['doi'])

            if st.button("Salvar Alterações"):
                conn = conectar_banco()
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE artigos 
                    SET titulo = %s, resumo = %s, abstract = %s, doi = %s 
                    WHERE id_artigo = %s
                """, (novo_titulo, novo_resumo, novo_abstract, novo_doi, artigo_selecionado['id_artigo']))
                conn.commit()
                conn.close()
                st.success("Artigo atualizado com sucesso!")
                st.session_state['modo_edicao'] = False


    # Streamlit - Interface
    st.sidebar.title("Gerenciamento de Artigos e Autores")
    # Define "Artigos Publicados" como a seção inicial do menu
    # menu = st.sidebar.selectbox("", ["Artigos Publicados", "Inserir Artigos", "Inserir Autores", "Relacionar Artigos e Autores"], index=0)

    # Inicializar estado da sessão para controle do menu
    if 'menu' not in st.session_state:
        st.session_state['menu'] = 'Artigos Publicados'
    if 'modo_edicao' not in st.session_state:
        st.session_state['modo_edicao'] = False
    if 'artigo_selecionado' not in st.session_state:
        st.session_state['artigo_selecionado'] = None


    if st.sidebar.button("Artigos Publicados"):
        st.session_state['menu'] = "Artigos Publicados"
        st.session_state['modo_edicao'] = False  # resetar edição

    if st.sidebar.button("Inserir Artigos"):
        st.session_state['menu'] = "Inserir Artigos"
        st.session_state['modo_edicao'] = False

    if st.sidebar.button("Inserir Autores"):
        st.session_state['menu'] = "Inserir Autores"
        st.session_state['modo_edicao'] = False

    if st.sidebar.button("Relacionar Artigos e Autores"):
        st.session_state['menu'] = "Relacionar Artigos e Autores"
        st.session_state['modo_edicao'] = False

    # Se um artigo estiver selecionado, mostrar opções adicionais
    if st.session_state['artigo_selecionado']:
        st.sidebar.markdown("""<hr style="margin-top: 3px; margin-bottom: 2px;">""",unsafe_allow_html=True)
        # st.sidebar.write(f"**Selecionado:** {st.session_state['artigo_selecionado']['titulo']}")
        
        if st.sidebar.button("Exibir Artigo"):
            artigo_selecionado = st.session_state.get('artigo_selecionado')  # Resgatar o artigo selecionado
            st.session_state['menu'] = "Exibir Artigo"
            caminho_pdf = artigo_selecionado["pasta_pdf"]
            st.subheader(f"{artigo_selecionado['titulo']}")
            exibir_pdf_no_app(caminho_pdf)
        
        if st.sidebar.button("Editar Artigo"):
            st.session_state['modo_edicao'] = True
            st.session_state['menu'] = "Artigos Publicados"

    menu = st.session_state['menu']


################################
    # Artigos Publicados (exibe por padrão ao abrir o app)
    if menu == "Artigos Publicados":
        # st.header("Visualização de Artigos")
        exibir_tabela_com_resumo_e_pdf()

    # Inserir Artigos
    elif menu == "Inserir Artigos":
        st.header("Inserir um Novo Artigo")
        titulo = st.text_input("Título do Artigo")
        resumo = st.text_area("Resumo")
        abstract = st.text_area("abstract")
        doi = st.text("doi")
        arquivo_pdf = st.file_uploader("Upload do Arquivo PDF", type=["pdf"])

        if st.button("Salvar Artigo"):
            if titulo and arquivo_pdf:
                # Salvar o arquivo PDF na pasta 'artigos'
                caminho_pdf = os.path.join(PASTA_PDFS, arquivo_pdf.name)
                with open(caminho_pdf, "wb") as f:
                    f.write(arquivo_pdf.read())
                
                # Inserir no banco o título, resumo, abstract e o caminho do PDF
                inserir_artigo(titulo, resumo, abstract, doi, caminho_pdf)
                st.success("Artigo inserido com sucesso!")
            else:
                st.error("Título e o arquivo PDF são obrigatórios.")

    # Inserir Autores
    elif menu == "Inserir Autores":
        st.header("Inserir um Novo Autor")
        nome = st.text_input("Nome do Autor")
        link_internet = st.text_input("Link (ex: LinkedIn, Lattes, etc.)")

        if st.button("Salvar Autor"):
            if nome:
                conn = conectar_banco()
                cursor = conn.cursor()
                cursor.execute("INSERT INTO autores (nome, link_internet) VALUES (%s, %s)", (nome, link_internet))
                conn.commit()
                conn.close()
                st.success("Autor inserido com sucesso!")
            else:
                st.error("O nome do autor é obrigatório.")

    # Relacionar Artigos e Autores
    elif menu == "Relacionar Artigos e Autores":
        st.header("Relacionar Artigo a Autor")

        # Selecionar Artigo
        conn = conectar_banco()
        cursor = conn.cursor()
        cursor.execute("SELECT id_artigo, titulo FROM artigos")
        artigos = cursor.fetchall()

        # Selecionar Autor
        cursor.execute("SELECT id_autor, nome FROM autores")
        autores = cursor.fetchall()
        conn.close()

        artigo_escolhido = st.selectbox("Selecione o Artigo", artigos, format_func=lambda x: f"{x[0]} - {x[1]}")
        autor_escolhido = st.selectbox("Selecione o Autor", autores, format_func=lambda x: f"{x[0]} - {x[1]}")

        if st.button("Relacionar"):
            if artigo_escolhido and autor_escolhido:
                conn = conectar_banco()
                cursor = conn.cursor()
                cursor.execute("INSERT INTO artigos_autores (id_artigo, id_autor) VALUES (%s, %s)", (artigo_escolhido[0], autor_escolhido[0]))
                conn.commit()
                conn.close()
                st.success("Relacionamento criado com sucesso!")