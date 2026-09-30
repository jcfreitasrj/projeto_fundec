# =============================================================================
# INTERFACE INTERATIVA - ANÁLISE DE DADOS FUNDEC
# =============================================================================
# Autores: José Carlos & Victor Machado
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
from pathlib import Path
import re
import unicodedata

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuração da página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FUNDEC - Análise de Dados",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Constantes do projeto
# -----------------------------------------------------------------------------
AUTORES = "José Carlos & Victor Machado"  # NOVO

# -----------------------------------------------------------------------------
# CSS personalizado
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a237e;
        text-align: center;
        padding: 1rem 0;
    }
    /* NOVO: estilo do subtítulo com autores */
    .autores-header {
        text-align: center;
        font-size: 1rem;
        color: #455a64;
        font-style: italic;
        margin-top: -0.5rem;
        margin-bottom: 0.5rem;
    }
    /* NOVO: estilo do rodapé */
    .footer {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        margin-top: 2rem;
        border-top: 1px solid #e0e0e0;
        color: #607d8b;
        font-size: 0.9rem;
    }
    .footer-autores {
        font-weight: bold;
        color: #1a237e;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #0d47a1;
        padding: 0.5rem 0;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1a237e;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1a237e;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff6f00;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2e7d32;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1565c0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNÇÕES UTILITÁRIAS
# ============================================================================

def remover_acentos(texto):
    """Remove acentos de uma string."""
    if pd.isna(texto):
        return texto
    texto = str(texto)
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )


def normalizar_nome_coluna(nome):
    """Normaliza o nome de uma coluna: remove espaços, acentos e deixa maiúsculo."""
    if pd.isna(nome):
        return nome
    nome = str(nome).strip().upper()
    nome = remover_acentos(nome)
    nome = re.sub(r'\s+', ' ', nome)
    return nome


# ============================================================================
# CARREGAMENTO E PROCESSAMENTO DE DADOS
# ============================================================================

@st.cache_data(show_spinner=False)
def carregar_dados(caminho_arquivo: str):
    """Carrega e processa os dados do arquivo Excel."""
    try:
        arquivo = Path(caminho_arquivo)
        if not arquivo.exists():
            return None, f"Arquivo não encontrado em: {arquivo}"

        df_bruto = pd.read_excel(arquivo, header=None)

        if df_bruto.empty:
            return None, "O arquivo Excel está vazio."

        linha_cabecalho = None
        for i, linha in df_bruto.iterrows():
            valores = [normalizar_nome_coluna(v) for v in linha.tolist()]
            if 'NOME' in valores and 'CPF' in valores:
                linha_cabecalho = i
                break

        if linha_cabecalho is None:
            return None, (
                "Não foi possível localizar a linha de cabeçalho. "
                "Esperava encontrar as colunas 'NOME' e 'CPF'."
            )

        df = pd.read_excel(arquivo, header=linha_cabecalho)
        df.columns = [normalizar_nome_coluna(c) for c in df.columns]
        df = df.dropna(axis=1, how='all')
        df = df.dropna(how='all')

        if df.empty:
            return None, "O arquivo foi lido, mas não há dados."

        df = limpar_dados_basico(df)
        return df, "Sucesso"

    except Exception as e:
        return None, f"Erro ao carregar o arquivo: {str(e)}"


def limpar_dados_basico(df):
    """Limpeza e enriquecimento dos dados."""
    df_limpo = df.copy()

    mapeamento = {
        'NOME': 'nome',
        'CPF': 'cpf',
        'IDENTIDADE': 'identidade',
        'DATA DE NASCIMENTO': 'data_nascimento',
        'TELEFONE': 'telefone',
        'ENDERECO': 'endereco',
        'BAIRRO': 'bairro',
        'MUNICIPIO': 'municipio',
        'CURSO': 'curso',
        'MODULO': 'modulo',
        'PERIODO': 'periodo',
        'CARGA HORARIA': 'carga_horaria',
        'SITUACAO': 'situacao',
        'PENDENCIA': 'pendencia'
    }

    colunas_existentes = {k: v for k, v in mapeamento.items() if k in df_limpo.columns}
    df_limpo.rename(columns=colunas_existentes, inplace=True)

    if 'situacao' in df_limpo.columns:
        def padronizar_situacao(s):
            if pd.isna(s):
                return 'NAO_INFORMADO'
            s = remover_acentos(str(s)).upper().strip()
            if 'APROV' in s:
                return 'APROVADO'
            elif 'REPROV' in s:
                return 'REPROVADO'
            elif 'DESIST' in s:
                return 'DESISTENTE'
            elif 'EVAD' in s or 'TRANSF' in s:
                return 'EVADIDO'
            elif 'RET' in s:
                return 'RETIDO'
            elif 'ATIVO' in s:
                return 'ATIVO'
            else:
                return s

        df_limpo['situacao'] = df_limpo['situacao'].apply(padronizar_situacao)
        df_limpo['evadido'] = df_limpo['situacao'].apply(
            lambda x: 1 if x in ['EVADIDO', 'DESISTENTE'] else 0
        )

    if 'pendencia' in df_limpo.columns:
        df_limpo['pendencia'] = df_limpo['pendencia'].fillna('SEM_PENDENCIA').astype(str)
        df_limpo['tem_pendencia'] = df_limpo['pendencia'].apply(
            lambda x: False if remover_acentos(x).upper().strip() in ['NAO', 'SEM_PENDENCIA', 'NAN', ''] else True
        )

    if 'data_nascimento' in df_limpo.columns:
        df_limpo['data_nascimento'] = pd.to_datetime(
            df_limpo['data_nascimento'], errors='coerce'
        )
        hoje = pd.Timestamp(datetime.now())

        def calcular_idade(data):
            if pd.isna(data):
                return np.nan
            if data.year < 1900 or data > hoje:
                return np.nan
            return hoje.year - data.year - ((hoje.month, hoje.day) < (data.month, data.day))

        df_limpo['idade'] = df_limpo['data_nascimento'].apply(calcular_idade)

        bins = [0, 17, 25, 40, 60, 120]
        labels = ['0-17', '18-25', '26-40', '41-60', '60+']
        df_limpo['faixa_etaria'] = pd.cut(
            df_limpo['idade'], bins=bins, labels=labels, right=False
        )

    if 'periodo' in df_limpo.columns:
        def extrair_ano(periodo):
            if pd.isna(periodo):
                return np.nan
            match = re.search(r'\b(19|20)\d{2}\b', str(periodo))
            return int(match.group(0)) if match else np.nan

        df_limpo['ano'] = df_limpo['periodo'].apply(extrair_ano)

    if 'carga_horaria' in df_limpo.columns:
        def extrair_carga(valor):
            if pd.isna(valor):
                return np.nan
            match = re.search(r'\d+', str(valor))
            return int(match.group(0)) if match else np.nan

        df_limpo['carga_horaria_num'] = df_limpo['carga_horaria'].apply(extrair_carga)

    for col in ['municipio', 'bairro', 'curso']:
        if col in df_limpo.columns:
            df_limpo[col] = df_limpo[col].astype(str).str.strip().str.upper()

    return df_limpo


# ============================================================================
# COMPONENTES DE VISUALIZAÇÃO
# ============================================================================

def criar_metricas_principais(df):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total de Alunos", f"{len(df):,}".replace(",", "."))

    with col2:
        cursos = df['curso'].nunique() if 'curso' in df.columns else 0
        st.metric("📚 Total de Cursos", cursos)

    with col3:
        if 'evadido' in df.columns and len(df) > 0:
            taxa = (df['evadido'].sum() / len(df)) * 100
            st.metric("⚠️ Taxa de Evasão", f"{taxa:.1f}%")

    with col4:
        if 'idade' in df.columns:
            media = df['idade'].mean()
            if pd.notna(media):
                st.metric("👤 Idade Média", f"{media:.1f} anos")


def grafico_distribuicao_cursos(df):
    if 'curso' not in df.columns or df.empty:
        st.info("Sem dados de cursos para exibir.")
        return
    dados = df['curso'].value_counts().reset_index()
    dados.columns = ['Curso', 'Quantidade']
    fig = px.bar(
        dados.head(15), x='Curso', y='Quantidade',
        title='Top 15 Cursos com Mais Alunos',
        color='Quantidade', color_continuous_scale='Blues', height=450
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def grafico_distribuicao_situacao(df):
    if 'situacao' not in df.columns or df.empty:
        st.info("Sem dados de situação para exibir.")
        return
    dados = df['situacao'].value_counts().reset_index()
    dados.columns = ['Situação', 'Quantidade']
    fig = px.pie(
        dados, values='Quantidade', names='Situação',
        title='Distribuição da Situação dos Alunos', height=450
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)


def grafico_faixa_etaria(df):
    if 'faixa_etaria' not in df.columns or df.empty:
        st.info("Sem dados de faixa etária para exibir.")
        return
    dados = df['faixa_etaria'].value_counts().sort_index().reset_index()
    dados.columns = ['Faixa Etária', 'Quantidade']
    fig = px.bar(
        dados, x='Faixa Etária', y='Quantidade',
        title='Distribuição por Faixa Etária',
        color='Quantidade', color_continuous_scale='Greens', height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def grafico_evasao_por_curso(df, minimo_alunos=5):
    if 'curso' not in df.columns or 'evadido' not in df.columns or df.empty:
        st.info("Sem dados suficientes para análise de evasão.")
        return
    evasao = df.groupby('curso').agg(
        total=('evadido', 'count'),
        evadidos=('evadido', 'sum'),
        taxa_evasao=('evadido', 'mean')
    ).reset_index()
    evasao = evasao[evasao['total'] >= minimo_alunos].sort_values(
        'taxa_evasao', ascending=False
    )
    if evasao.empty:
        st.info(f"Nenhum curso com pelo menos {minimo_alunos} alunos.")
        return
    fig = px.bar(
        evasao.head(15), x='curso', y='taxa_evasao',
        title=f'Taxa de Evasão por Curso (mín. {minimo_alunos} alunos)',
        color='taxa_evasao', color_continuous_scale='Reds', height=450,
        labels={'taxa_evasao': 'Taxa de Evasão', 'curso': 'Curso'}
    )
    fig.update_layout(xaxis_tickangle=-45, yaxis_tickformat='.1%')
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Ver tabela detalhada de evasão por curso"):
        evasao_display = evasao.copy()
        evasao_display['taxa_evasao'] = evasao_display['taxa_evasao'].apply(
            lambda x: f"{x:.2%}"
        )
        st.dataframe(evasao_display.head(30), use_container_width=True)


def grafico_evasao_por_bairro(df, minimo_alunos=5):
    if 'bairro' not in df.columns or 'evadido' not in df.columns or df.empty:
        return
    evasao_bairro = df.groupby('bairro').agg(
        total=('evadido', 'count'),
        evadidos=('evadido', 'sum'),
        taxa_evasao=('evadido', 'mean')
    ).reset_index()
    evasao_bairro = evasao_bairro[evasao_bairro['total'] >= minimo_alunos]
    evasao_bairro = evasao_bairro.sort_values('taxa_evasao', ascending=False)
    if evasao_bairro.empty:
        return
    with st.expander(f"📊 Evasão por Bairro (mín. {minimo_alunos} alunos)"):
        evasao_display = evasao_bairro.copy()
        evasao_display['taxa_evasao'] = evasao_display['taxa_evasao'].apply(
            lambda x: f"{x:.2%}"
        )
        st.dataframe(evasao_display.head(30), use_container_width=True)


def grafico_top_bairros(df):
    if 'bairro' not in df.columns or df.empty:
        st.info("Sem dados de bairro para exibir.")
        return
    dados = df['bairro'].value_counts().head(15).reset_index()
    dados.columns = ['Bairro', 'Quantidade']
    fig = px.bar(
        dados, x='Quantidade', y='Bairro', orientation='h',
        title='Top 15 Bairros com Mais Alunos',
        color='Quantidade', color_continuous_scale='Viridis', height=500
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def grafico_evolucao_temporal(df):
    if 'ano' not in df.columns or df.empty:
        st.info("Sem dados de período/ano para exibir.")
        return
    anos = df['ano'].dropna()
    if anos.empty:
        st.info("Não foi possível extrair anos da coluna PERÍODO.")
        return
    dados = anos.value_counts().sort_index().reset_index()
    dados.columns = ['Ano', 'Matrículas']
    fig = px.line(
        dados, x='Ano', y='Matrículas',
        title='Evolução de Matrículas por Ano',
        markers=True, height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    if 'evadido' in df.columns:
        evasao_ano = df.groupby('ano').agg(
            total=('evadido', 'count'),
            taxa_evasao=('evadido', 'mean')
        ).reset_index()
        evasao_ano = evasao_ano.dropna().sort_values('ano')
        if not evasao_ano.empty:
            fig2 = px.line(
                evasao_ano, x='ano', y='taxa_evasao',
                title='Evolução da Taxa de Evasão por Ano',
                markers=True, height=400,
                labels={'taxa_evasao': 'Taxa de Evasão', 'ano': 'Ano'}
            )
            fig2.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig2, use_container_width=True)


def grafico_top_cursos_por_ano(df):
    if 'ano' not in df.columns or 'curso' not in df.columns or df.empty:
        return
    dados = df.dropna(subset=['ano', 'curso'])
    if dados.empty:
        return
    top_cursos = dados['curso'].value_counts().head(10).index.tolist()
    dados = dados[dados['curso'].isin(top_cursos)]
    pivot = dados.groupby(['curso', 'ano']).size().unstack(fill_value=0)
    if pivot.empty:
        return
    fig = px.imshow(
        pivot, aspect='auto', color_continuous_scale='YlOrRd',
        title='Top 10 Cursos - Matrículas por Ano', height=500
    )
    fig.update_xaxes(title='Ano')
    fig.update_yaxes(title='Curso')
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# APLICAÇÃO PRINCIPAL
# ============================================================================

def main():
    # ---- Header ----
    st.markdown(
        '<div class="main-header">📊 FUNDEC - Análise de Dados Acadêmicos</div>',
        unsafe_allow_html=True
    )
    # NOVO: Subtítulo com autores
    st.markdown(
        f'<div class="autores-header">Desenvolvido por {AUTORES}</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    BASE_DIR = Path(__file__).resolve().parent
    caminho_padrao = BASE_DIR / "dados" / "CADASTRO_ALUNOS_FUNDEC.xlsx"

    # ---- Sidebar ----
    with st.sidebar:
        # NOVO: Identificação dos autores na sidebar
        st.markdown(
            f"""
            <div style="background-color: #1a237e; padding: 0.8rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 0.75rem; opacity: 0.8;">Desenvolvido por</div>
                <div style="font-size: 1rem; font-weight: bold;">{AUTORES}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("## 📋 Menu de Navegação")
        st.markdown("### 📂 Dados")

        caminho_input = st.text_input(
            "Caminho do arquivo Excel",
            value=str(caminho_padrao),
            help="Caminho completo para o arquivo CADASTRO_ALUNOS_FUNDEC.xlsx"
        )

        if st.button("🔄 Carregar Dados", use_container_width=True):
            with st.spinner("Carregando dados..."):
                df, mensagem = carregar_dados(caminho_input)
                if df is not None:
                    st.session_state['df'] = df
                    st.session_state['dados_carregados'] = True
                    st.success("✅ Dados carregados com sucesso!")
                else:
                    st.session_state['dados_carregados'] = False
                    st.error(f"❌ {mensagem}")

        if st.session_state.get('dados_carregados', False):
            df = st.session_state['df']
            st.markdown("### ✅ Dados Carregados")
            st.info(f"📊 {len(df):,} registros".replace(",", "."))
            st.info(f"📋 {len(df.columns)} colunas")

        st.markdown("---")

        if st.session_state.get('dados_carregados', False):
            st.markdown("### 🔍 Filtros")
            df = st.session_state['df']
            df_filtrado = df.copy()

            if 'curso' in df.columns:
                cursos = ['Todos'] + sorted(
                    df['curso'].dropna().astype(str).unique().tolist()
                )
                curso_sel = st.selectbox("Curso", cursos)
                if curso_sel != 'Todos':
                    df_filtrado = df_filtrado[df_filtrado['curso'] == curso_sel]

            if 'situacao' in df.columns:
                situacoes = ['Todos'] + sorted(
                    df['situacao'].dropna().astype(str).unique().tolist()
                )
                situacao_sel = st.selectbox("Situação", situacoes)
                if situacao_sel != 'Todos':
                    df_filtrado = df_filtrado[df_filtrado['situacao'] == situacao_sel]

            if 'faixa_etaria' in df.columns:
                faixas_validas = df['faixa_etaria'].dropna().unique().tolist()
                faixas = ['Todas'] + sorted([str(f) for f in faixas_validas])
                faixa_sel = st.selectbox("Faixa Etária", faixas)
                if faixa_sel != 'Todas':
                    df_filtrado = df_filtrado[
                        df_filtrado['faixa_etaria'].astype(str) == faixa_sel
                    ]

            st.session_state['df_filtrado'] = df_filtrado

            st.markdown("---")
            st.markdown("### 📊 Resumo do Filtro")
            st.metric("Registros filtrados", f"{len(df_filtrado):,}".replace(",", "."))

            with st.expander("🔧 Debug (colunas do arquivo)"):
                st.write("**Colunas disponíveis:**")
                st.write(df.columns.tolist())
                st.write("**Amostra dos dados:**")
                st.dataframe(df.head(5))

        # NOVO: Rodapé na sidebar com créditos
        st.markdown("---")
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 0.75rem; color: #90a4ae; padding-top: 1rem;">
                © {datetime.now().year} FUNDEC<br>
                <span style="color: #1a237e; font-weight: bold;">{AUTORES}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---- Tela inicial ----
    if not st.session_state.get('dados_carregados', False):
        st.markdown("""
        <div class="info-box">
            <h3>👋 Bem-vindo à Análise de Dados da FUNDEC!</h3>
            <p>Para começar:</p>
            <ol>
                <li>Verifique se o arquivo <code>CADASTRO_ALUNOS_FUNDEC.xlsx</code> está em <code>dados/</code>.</li>
                <li>Confirme o caminho no campo acima.</li>
                <li>Clique em <strong>"🔄 Carregar Dados"</strong>.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📈 O que você poderá analisar:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">📊</div>
                <div class="metric-label">Distribuição de Cursos</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">🎯</div>
                <div class="metric-label">Análise de Evasão</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">📈</div>
                <div class="metric-label">Evolução Temporal</div>
            </div>
            """, unsafe_allow_html=True)

        # NOVO: Rodapé com autores na tela inicial também
        st.markdown(
            f"""
            <div class="footer">
                <div>📊 FUNDEC - Análise de Dados Acadêmicos</div>
                <div style="margin-top: 0.5rem;">Desenvolvido por <span class="footer-autores">{AUTORES}</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # ---- Dados carregados ----
    df = st.session_state['df']
    df_filtrado = st.session_state.get('df_filtrado', df)

    st.markdown("### 📊 Métricas Principais")
    criar_metricas_principais(df_filtrado)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visão Geral",
        "🎯 Análise de Evasão",
        "📍 Geografia",
        "📈 Evolução",
        "📋 Dados"
    ])

    with tab1:
        st.markdown("### 📊 Visão Geral dos Dados")
        st.info("Distribuição geral dos alunos, cursos e faixas etárias.")
        col_a, col_b = st.columns(2)
        with col_a:
            grafico_distribuicao_cursos(df_filtrado)
        with col_b:
            grafico_distribuicao_situacao(df_filtrado)
        st.markdown("---")
        grafico_faixa_etaria(df_filtrado)

    with tab2:
        st.markdown("### 🎯 Análise de Evasão")
        if 'evadido' in df_filtrado.columns and not df_filtrado.empty:
            total = len(df_filtrado)
            evadidos = int(df_filtrado['evadido'].sum())
            taxa = (evadidos / total) * 100 if total > 0 else 0
            st.markdown(f"""
            <div class="warning-box">
                <h3>⚠️ Taxa Geral de Evasão: {taxa:.1f}%</h3>
                <p>Total de evadidos/desistentes: <strong>{evadidos:,}</strong> de {total:,} alunos</p>
            </div>
            """.replace(",", "."), unsafe_allow_html=True)

        grafico_evasao_por_curso(df_filtrado)

        if 'faixa_etaria' in df_filtrado.columns and 'evadido' in df_filtrado.columns:
            st.markdown("### 🎯 Evasão por Faixa Etária")
            dados = df_filtrado.groupby('faixa_etaria').agg(
                total=('evadido', 'count'),
                taxa_evasao=('evadido', 'mean')
            ).reset_index().dropna()
            if not dados.empty:
                fig = px.bar(
                    dados, x='faixa_etaria', y='taxa_evasao',
                    title='Taxa de Evasão por Faixa Etária',
                    color='taxa_evasao', color_continuous_scale='OrRd', height=400,
                    labels={'faixa_etaria': 'Faixa Etária', 'taxa_evasao': 'Taxa de Evasão'}
                )
                fig.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 📍 Análise Geográfica")
        grafico_top_bairros(df_filtrado)
        grafico_evasao_por_bairro(df_filtrado)
        if 'municipio' in df_filtrado.columns and not df_filtrado.empty:
            st.markdown("### 🏙️ Alunos por Município")
            dados = df_filtrado['municipio'].value_counts().reset_index()
            dados.columns = ['Município', 'Quantidade']
            fig = px.bar(
                dados.head(15), x='Município', y='Quantidade',
                title='Top 15 Municípios',
                color='Quantidade', color_continuous_scale='Teal', height=400
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### 📈 Evolução Temporal")
        grafico_evolucao_temporal(df_filtrado)
        grafico_top_cursos_por_ano(df_filtrado)

    with tab5:
        st.markdown("### 📋 Dados Completos")
        search = st.text_input(
            "🔍 Buscar em todas as colunas",
            placeholder="Digite para filtrar..."
        )
        df_display = df_filtrado.copy()
        if search:
            mask = df_display.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            df_display = df_display[mask]

        st.write(
            f"Exibindo **{min(len(df_display), 100):,}** de **{len(df_display):,}** registros."
            .replace(",", ".")
        )
        st.dataframe(df_display.head(100), use_container_width=True)

        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar CSV (dados filtrados)",
            data=csv,
            file_name=f"fundec_dados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # NOVO: Rodapé com autores no final da página principal
    st.markdown(
        f"""
        <div class="footer">
            <div>📊 FUNDEC - Análise de Dados Acadêmicos</div>
            <div style="margin-top: 0.5rem;">Desenvolvido por <span class="footer-autores">{AUTORES}</span></div>
            <div style="font-size: 0.75rem; margin-top: 0.3rem;">© {datetime.now().year} - Todos os direitos reservados</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    main()
