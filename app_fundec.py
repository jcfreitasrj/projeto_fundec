# =============================================================================
# INTERFACE INTERATIVA - ANÁLISE DE DADOS FUNDEC
# VERSÃO COMPATÍVEL COM STREAMLIT 1.17.0
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="FUNDEC - Análise de Dados",
    page_icon="📊",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a237e;
        text-align: center;
        padding: 1rem 0;
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
# FUNÇÕES DE PROCESSAMENTO
# ============================================================================

@st.cache
def carregar_dados():
    """Carrega e processa os dados do arquivo Excel"""
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "dados"
    ARQUIVO_DADOS = DATA_DIR / "CADASTRO_ALUNOS_FUNDEC.xlsx"
    
    try:
        if not ARQUIVO_DADOS.exists():
            return None, "Arquivo não encontrado. Coloque o arquivo em: dados/CADASTRO_ALUNOS_FUNDEC.xlsx"
        
        df = pd.read_excel(ARQUIVO_DADOS, header=0)
        
        if df.empty:
            df = pd.read_excel(ARQUIVO_DADOS, header=1)
        
        df = limpar_dados_basico(df)
        
        return df, "Sucesso"
    except Exception as e:
        return None, f"Erro ao carregar: {str(e)}"

def limpar_dados_basico(df):
    """Limpeza básica dos dados"""
    df_limpo = df.copy()
    
    colunas = {
        'NOME': 'nome', 'CPF': 'cpf', 'IDENTIDADE': 'identidade',
        'DATA DE NASCIMENTO': 'data_nascimento', 'TELEFONE': 'telefone',
        'ENDEREÇO': 'endereco', 'BAIRRO': 'bairro', 'MUNICIPIO': 'municipio',
        'CURSO': 'curso', 'MODULO': 'modulo', 'PERÍODO': 'periodo',
        'CARGA HORÁRIA': 'carga_horaria', 'SITUAÇÃO': 'situacao',
        'PENDENCIA': 'pendencia'
    }
    
    colunas_existentes = {k: v for k, v in colunas.items() if k in df_limpo.columns}
    df_limpo.rename(columns=colunas_existentes, inplace=True)
    
    if 'situacao' in df_limpo.columns:
        def padronizar_situacao(s):
            if pd.isna(s):
                return 'NAO_INFORMADO'
            s = str(s).upper().strip()
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
            else:
                return s
        df_limpo['situacao'] = df_limpo['situacao'].apply(padronizar_situacao)
        df_limpo['evadido'] = df_limpo['situacao'].apply(
            lambda x: 1 if x in ['EVADIDO', 'DESISTENTE'] else 0
        )
    
    if 'pendencia' in df_limpo.columns:
        df_limpo['tem_pendencia'] = df_limpo['pendencia'].notna()
        df_limpo['pendencia'] = df_limpo['pendencia'].fillna('SEM_PENDENCIA')
    
    if 'data_nascimento' in df_limpo.columns:
        df_limpo['data_nascimento'] = pd.to_datetime(df_limpo['data_nascimento'], errors='coerce')
        hoje = datetime.now()
        df_limpo['idade'] = df_limpo['data_nascimento'].apply(
            lambda x: hoje.year - x.year - ((hoje.month, hoje.day) < (x.month, x.day))
            if pd.notna(x) else np.nan
        )
        
        bins = [0, 17, 25, 40, 60, 100]
        labels = ['0-17', '18-25', '26-40', '41-60', '60+']
        df_limpo['faixa_etaria'] = pd.cut(df_limpo['idade'], bins=bins, labels=labels, right=False)
    
    if 'periodo' in df_limpo.columns:
        import re
        def extrair_ano(periodo):
            if pd.isna(periodo):
                return np.nan
            anos = re.findall(r'\b(20\d{2})\b', str(periodo))
            return int(anos[0]) if anos else np.nan
        df_limpo['ano'] = df_limpo['periodo'].apply(extrair_ano)
    
    return df_limpo

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">📊 FUNDEC - Análise de Dados Acadêmicos</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Menu de Navegação")
        
        st.markdown("### 📂 Dados")
        if st.button("🔄 Carregar Dados"):
            with st.spinner("Carregando dados..."):
                df, mensagem = carregar_dados()
                if df is not None:
                    st.session_state['df'] = df
                    st.session_state['dados_carregados'] = True
                    st.success("✅ Dados carregados com sucesso!")
                else:
                    st.error(f"❌ {mensagem}")
        
        if st.session_state.get('dados_carregados', False):
            df = st.session_state['df']
            st.markdown("### ✅ Dados Carregados")
            st.info(f"📊 {len(df):,} registros")
            st.info(f"📋 {len(df.columns)} colunas")
        
        st.markdown("---")
        
        if st.session_state.get('dados_carregados', False):
            st.markdown("### 🔍 Filtros")
            
            df = st.session_state['df']
            df_filtrado = df.copy()
            
            if 'curso' in df.columns:
                cursos = ['Todos'] + sorted(df['curso'].dropna().unique().tolist())
                curso_selecionado = st.selectbox("Curso", cursos)
                if curso_selecionado != 'Todos':
                    df_filtrado = df_filtrado[df_filtrado['curso'] == curso_selecionado]
            
            if 'situacao' in df.columns:
                situacoes = ['Todos'] + sorted(df['situacao'].dropna().unique().tolist())
                situacao_selecionada = st.selectbox("Situação", situacoes)
                if situacao_selecionada != 'Todos':
                    df_filtrado = df_filtrado[df_filtrado['situacao'] == situacao_selecionada]
            
            if 'faixa_etaria' in df.columns:
                faixas = ['Todas'] + sorted(df['faixa_etaria'].dropna().unique().tolist())
                faixa_selecionada = st.selectbox("Faixa Etária", faixas)
                if faixa_selecionada != 'Todas':
                    df_filtrado = df_filtrado[df_filtrado['faixa_etaria'] == faixa_selecionada]
            
            st.session_state['df_filtrado'] = df_filtrado
            
            st.markdown("---")
            st.markdown("### 📊 Resumo do Filtro")
            st.metric("Registros", f"{len(df_filtrado):,}")
    
    # Conteúdo principal
    if not st.session_state.get('dados_carregados', False):
        st.markdown("""
        <div class="info-box">
            <h3>👋 Bem-vindo à Análise de Dados da FUNDEC!</h3>
            <p>Para começar, clique em <strong>"Carregar Dados"</strong> no menu lateral.</p>
            <p>Certifique-se de que o arquivo <code>CADASTRO_ALUNOS_FUNDEC.xlsx</code> esteja na pasta <code>dados/</code></p>
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
        return
    
    # Dados carregados
    df = st.session_state['df']
    df_filtrado = st.session_state.get('df_filtrado', df)
    
    # Métricas principais
    st.markdown("### 📊 Métricas Principais")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total de Alunos", f"{len(df_filtrado):,}")
    
    with col2:
        cursos = df_filtrado['curso'].nunique() if 'curso' in df_filtrado.columns else 0
        st.metric("📚 Total de Cursos", cursos)
    
    with col3:
        if 'evadido' in df_filtrado.columns:
            taxa = (df_filtrado['evadido'].sum() / len(df_filtrado)) * 100
            st.metric("⚠️ Taxa de Evasão", f"{taxa:.1f}%")
    
    with col4:
        if 'faixa_etaria' in df_filtrado.columns:
            media = df_filtrado['idade'].mean()
            st.metric("👤 Idade Média", f"{media:.1f} anos")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visão Geral",
        "🎯 Análise de Evasão",
        "📍 Geografia",
        "📈 Evolução",
        "📋 Dados"
    ])
    
    with tab1:
        st.markdown("### 📊 Visão Geral dos Dados")
        
        if 'curso' in df_filtrado.columns:
            dados = df_filtrado['curso'].value_counts().reset_index()
            dados.columns = ['Curso', 'Quantidade']
            
            fig = px.bar(
                dados.head(15),
                x='Curso',
                y='Quantidade',
                title='Top 15 Cursos com Mais Alunos',
                color='Quantidade',
                color_continuous_scale='Blues',
                height=400
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        if 'situacao' in df_filtrado.columns:
            dados = df_filtrado['situacao'].value_counts().reset_index()
            dados.columns = ['Situação', 'Quantidade']
            
            fig = px.pie(
                dados,
                values='Quantidade',
                names='Situação',
                title='Distribuição da Situação dos Alunos',
                height=400
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🎯 Análise de Evasão")
        
        if 'evadido' in df_filtrado.columns:
            taxa = (df_filtrado['evadido'].sum() / len(df_filtrado)) * 100
            st.markdown(f"""
            <div class="warning-box">
                <h3>⚠️ Taxa Geral de Evasão: {taxa:.1f}%</h3>
                <p>Total de evadidos: {df_filtrado['evadido'].sum():,} alunos</p>
            </div>
            """, unsafe_allow_html=True)
        
        if 'curso' in df_filtrado.columns and 'evadido' in df_filtrado.columns:
            evasao = df_filtrado.groupby('curso').agg({
                'evadido': ['count', 'sum', 'mean']
            }).round(2)
            evasao.columns = ['total', 'evadidos', 'taxa_evasao']
            evasao = evasao[evasao['total'] >= 5].sort_values('taxa_evasao', ascending=False)
            
            fig = px.bar(
                evasao.reset_index().head(15),
                x='curso',
                y='taxa_evasao',
                title='Taxa de Evasão por Curso',
                color='taxa_evasao',
                color_continuous_scale='Reds',
                height=400
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 Ver tabela detalhada"):
                st.dataframe(evasao.head(20))
    
    with tab3:
        st.markdown("### 📍 Análise Geográfica")
        
        if 'bairro' in df_filtrado.columns:
            dados = df_filtrado['bairro'].value_counts().head(15).reset_index()
            dados.columns = ['Bairro', 'Quantidade']
            
            fig = px.bar(
                dados,
                x='Quantidade',
                y='Bairro',
                orientation='h',
                title='Top 15 Bairros com Mais Alunos',
                color='Quantidade',
                color_continuous_scale='Viridis',
                height=500
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        if 'bairro' in df_filtrado.columns and 'evadido' in df_filtrado.columns:
            evasao_bairro = df_filtrado.groupby('bairro').agg({
                'evadido': ['count', 'sum', 'mean']
            }).round(2)
            evasao_bairro.columns = ['total', 'evadidos', 'taxa_evasao']
            evasao_bairro = evasao_bairro[evasao_bairro['total'] >= 5]
            evasao_bairro = evasao_bairro.sort_values('taxa_evasao', ascending=False)
            
            if not evasao_bairro.empty:
                with st.expander("📊 Evasão por Bairro"):
                    st.dataframe(evasao_bairro.head(20))
    
    with tab4:
        st.markdown("### 📈 Evolução Temporal")
        
        if 'ano' in df_filtrado.columns:
            anos = df_filtrado['ano'].dropna()
            if not anos.empty:
                dados = anos.value_counts().sort_index().reset_index()
                dados.columns = ['Ano', 'Matrículas']
                
                fig = px.line(
                    dados,
                    x='Ano',
                    y='Matrículas',
                    title='Evolução de Matrículas por Ano',
                    markers=True,
                    height=400
                )
                fig.update_layout(xaxis_tickangle=0)
                st.plotly_chart(fig, use_container_width=True)
                
                if 'evadido' in df_filtrado.columns:
                    evasao_ano = df_filtrado.groupby('ano').agg({
                        'evadido': ['count', 'mean']
                    }).round(2)
                    evasao_ano.columns = ['total', 'taxa_evasao']
                    evasao_ano = evasao_ano.dropna()
                    evasao_ano = evasao_ano.reset_index()
                    
                    if not evasao_ano.empty:
                        fig2 = px.line(
                            evasao_ano,
                            x='ano',
                            y='taxa_evasao',
                            title='Evolução da Taxa de Evasão por Ano',
                            markers=True,
                            height=400
                        )
                        fig2.update_layout(yaxis_title='Taxa de Evasão')
                        st.plotly_chart(fig2, use_container_width=True)
    
    with tab5:
        st.markdown("### 📋 Dados Completos")
        
        search = st.text_input("🔍 Buscar", placeholder="Digite para filtrar...")
        
        df_display = df_filtrado.copy()
        if search:
            mask = df_display.astype(str).apply(
                lambda x: x.str.contains(search, case=False, na=False)
            ).any(axis=1)
            df_display = df_display[mask]
        
        st.dataframe(df_display.head(100), use_container_width=True)
        
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar CSV",
            data=csv,
            file_name=f"fundec_dados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
