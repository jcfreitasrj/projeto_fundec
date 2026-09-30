# =============================================================================
# PROJETO DE EXTENSÃO EM CIÊNCIA DE DADOS
# ANÁLISE DE DADOS DE ATENDIMENTO - FUNDEC
# VERSÃO 2.1 - COMPLETAMENTE CORRIGIDA
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# ============================================================================
# CONFIGURAÇÃO INICIAL
# ============================================================================

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("="*80)
print("PROJETO DE EXTENSÃO - ANÁLISE DE DADOS FUNDEC")
print("VERSÃO 2.1 - ANÁLISE COMPLETA CORRIGIDA")
print("="*80)
print(f"Data da execução: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
print("="*80)

# ============================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dados"
OUTPUT_DIR = BASE_DIR / "outputs"
GRAFICOS_DIR = OUTPUT_DIR / "graficos"
RELATORIOS_DIR = OUTPUT_DIR / "relatorios"

for dir_path in [DATA_DIR, OUTPUT_DIR, GRAFICOS_DIR, RELATORIOS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

ARQUIVO_DADOS = DATA_DIR / "CADASTRO_ALUNOS_FUNDEC.xlsx"

print(f"\n📁 Diretório de dados: {DATA_DIR}")
print(f"📁 Diretório de saída: {OUTPUT_DIR}")

# ============================================================================
# FUNÇÃO 1: CARREGAR DADOS
# ============================================================================

def carregar_dados():
    print("\n" + "="*80)
    print("📂 CARREGANDO OS DADOS")
    print("="*80)
    
    try:
        if not ARQUIVO_DADOS.exists():
            print(f"❌ Arquivo não encontrado: {ARQUIVO_DADOS}")
            print("   Coloque o arquivo CADASTRO_ALUNOS_FUNDEC.xlsx na pasta dados/")
            return None
        
        df = pd.read_excel(ARQUIVO_DADOS, header=0)
        
        if df.empty:
            df = pd.read_excel(ARQUIVO_DADOS, header=1)
        
        print(f"✅ Dados carregados com sucesso!")
        print(f"   - Total de registros: {len(df)}")
        print(f"   - Total de colunas: {len(df.columns)}")
        
        print("\n📊 Primeiras 5 linhas:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"❌ Erro ao carregar os dados: {e}")
        return None

# ============================================================================
# FUNÇÃO 2: LIMPAR DADOS
# ============================================================================

def limpar_dados(df):
    print("\n" + "="*80)
    print("🧹 LIMPANDO OS DADOS")
    print("="*80)
    
    df_limpo = df.copy()
    
    # 2.1. Padronizar colunas
    print("\n📝 2.1. Padronizando colunas...")
    colunas = {
        'NOME': 'nome',
        'CPF': 'cpf',
        'IDENTIDADE': 'identidade',
        'DATA DE NASCIMENTO': 'data_nascimento',
        'TELEFONE': 'telefone',
        'ENDEREÇO': 'endereco',
        'BAIRRO': 'bairro',
        'MUNICIPIO': 'municipio',
        'CURSO': 'curso',
        'MODULO': 'modulo',
        'PERÍODO': 'periodo',
        'CARGA HORÁRIA': 'carga_horaria',
        'SITUAÇÃO': 'situacao',
        'PENDENCIA': 'pendencia'
    }
    
    colunas_existentes = {k: v for k, v in colunas.items() if k in df_limpo.columns}
    df_limpo.rename(columns=colunas_existentes, inplace=True)
    print(f"   ✅ {len(colunas_existentes)} colunas padronizadas")
    
    # 2.2. Tratar datas de nascimento
    print("\n📅 2.2. Processando datas...")
    
    def tratar_data(data):
        if pd.isna(data):
            return np.nan
        if isinstance(data, (datetime, pd.Timestamp)):
            return data
        if isinstance(data, str):
            try:
                for fmt in ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y', '%Y-%m-%d']:
                    try:
                        return pd.to_datetime(data.strip(), format=fmt)
                    except:
                        continue
            except:
                pass
        return np.nan
    
    if 'data_nascimento' in df_limpo.columns:
        df_limpo['data_nascimento'] = df_limpo['data_nascimento'].apply(tratar_data)
        
        hoje = datetime.now()
        df_limpo['idade'] = df_limpo['data_nascimento'].apply(
            lambda x: hoje.year - x.year - ((hoje.month, hoje.day) < (x.month, x.day))
            if pd.notna(x) else np.nan
        )
        
        bins = [0, 17, 25, 40, 60, 100]
        labels = ['0-17', '18-25', '26-40', '41-60', '60+']
        df_limpo['faixa_etaria'] = pd.cut(df_limpo['idade'], bins=bins, labels=labels, right=False)
        
        print(f"   ✅ Idade calculada")
        print(f"      - Média: {df_limpo['idade'].mean():.1f} anos")
        print(f"      - Mínima: {df_limpo['idade'].min():.0f} anos")
        print(f"      - Máxima: {df_limpo['idade'].max():.0f} anos")
    
    # 2.3. Processar período
    print("\n📆 2.3. Processando períodos...")
    
    def extrair_ano_periodo(periodo_str):
        if pd.isna(periodo_str):
            return np.nan
        import re
        anos = re.findall(r'\b(20\d{2})\b', str(periodo_str))
        if anos:
            return int(anos[0])
        return np.nan
    
    if 'periodo' in df_limpo.columns:
        df_limpo['ano'] = df_limpo['periodo'].apply(extrair_ano_periodo)
        print(f"   ✅ Anos extraídos")
    
    # 2.4. Padronizar situação
    print("\n📊 2.4. Padronizando situações...")
    
    def padronizar_situacao(situacao):
        if pd.isna(situacao):
            return 'NAO_INFORMADO'
        situacao = str(situacao).upper().strip()
        if 'APROV' in situacao:
            return 'APROVADO'
        elif 'REPROV' in situacao:
            return 'REPROVADO'
        elif 'DESIST' in situacao:
            return 'DESISTENTE'
        elif 'EVAD' in situacao or 'TRANSF' in situacao or 'TRASNF' in situacao:
            return 'EVADIDO'
        elif 'RET' in situacao:
            return 'RETIDO'
        else:
            return situacao
    
    if 'situacao' in df_limpo.columns:
        df_limpo['situacao'] = df_limpo['situacao'].apply(padronizar_situacao)
        df_limpo['evadido'] = df_limpo['situacao'].apply(
            lambda x: 1 if x in ['EVADIDO', 'DESISTENTE'] else 0
        )
        print(f"   ✅ Situações padronizadas")
    
    # 2.5. Tratar pendências
    print("\n⚠️ 2.5. Processando pendências...")
    
    if 'pendencia' in df_limpo.columns:
        df_limpo['tem_pendencia'] = df_limpo['pendencia'].notna()
        df_limpo['pendencia'] = df_limpo['pendencia'].fillna('SEM_PENDENCIA')
        print(f"   ✅ Pendências processadas")
    
    # 2.6. Processar carga horária
    print("\n⏱️ 2.6. Processando carga horária...")
    
    def extrair_carga_horaria(valor):
        if pd.isna(valor):
            return np.nan
        if isinstance(valor, (int, float)):
            return float(valor)
        if isinstance(valor, str):
            import re
            numeros = re.findall(r'\d+', valor)
            if numeros:
                return float(numeros[0])
        return np.nan
    
    if 'carga_horaria' in df_limpo.columns:
        df_limpo['carga_horaria_numerica'] = df_limpo['carga_horaria'].apply(extrair_carga_horaria)
        print(f"   ✅ Carga horária processada")
    
    print(f"\n✅ LIMPEZA CONCLUÍDA!")
    print(f"   - Total de registros: {len(df_limpo)}")
    
    return df_limpo

# ============================================================================
# FUNÇÃO 3: ANÁLISE EXPLORATÓRIA COMPLETA
# ============================================================================

def analise_exploratoria(df):
    print("\n" + "="*80)
    print("📊 ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("="*80)
    
    # 3.1. Distribuição dos cursos
    if 'curso' in df.columns:
        print("\n📚 3.1. Distribuição dos Cursos")
        print("-" * 40)
        cursos = df['curso'].value_counts()
        print(cursos)
        
        plt.figure(figsize=(14, 7))
        cursos.plot(kind='bar', color='steelblue')
        plt.title('Distribuição de Alunos por Curso', fontsize=14, fontweight='bold')
        plt.xlabel('Curso')
        plt.ylabel('Número de Alunos')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'distribuicao_cursos.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: distribuicao_cursos.png")
    
    # 3.2. Distribuição por faixa etária
    if 'faixa_etaria' in df.columns:
        print("\n👤 3.2. Distribuição por Faixa Etária")
        print("-" * 40)
        faixas = df['faixa_etaria'].value_counts().sort_index()
        print(faixas)
        
        plt.figure(figsize=(10, 6))
        cores_faixa = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        faixas.plot(kind='bar', color=cores_faixa)
        plt.title('Distribuição de Alunos por Faixa Etária', fontsize=14, fontweight='bold')
        plt.xlabel('Faixa Etária')
        plt.ylabel('Número de Alunos')
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'distribuicao_faixa_etaria.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: distribuicao_faixa_etaria.png")
    
    # 3.3. Top bairros
    if 'bairro' in df.columns:
        print("\n📍 3.3. Top 15 Bairros com mais alunos")
        print("-" * 40)
        bairros = df['bairro'].value_counts().head(15)
        print(bairros)
        
        plt.figure(figsize=(14, 7))
        bairros.plot(kind='bar', color='green')
        plt.title('Top 15 Bairros com Mais Alunos', fontsize=14, fontweight='bold')
        plt.xlabel('Bairro')
        plt.ylabel('Número de Alunos')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'top_bairros.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: top_bairros.png")
    
    # 3.4. Situação dos alunos
    if 'situacao' in df.columns:
        print("\n📈 3.4. Distribuição da Situação dos Alunos")
        print("-" * 40)
        situacao = df['situacao'].value_counts()
        print(situacao)
        
        # Calcular porcentagens
        situacao_pct = (situacao / len(df)) * 100
        print(f"\n📊 TAXAS GERAIS:")
        for status, pct in situacao_pct.items():
            print(f"   - {status}: {pct:.1f}%")
        
        # Gráfico
        plt.figure(figsize=(12, 6))
        cores_situacao = {
            'APROVADO': '#2ecc71',
            'REPROVADO': '#e74c3c',
            'EVADIDO': '#f39c12',
            'DESISTENTE': '#95a5a6',
            'PROMOVIDO': '#3498db',
            'NAO_INFORMADO': '#7f8c8d',
            'RETIDO': '#e67e22'
        }
        cores_plot = [cores_situacao.get(s, '#95a5a6') for s in situacao.index]
        situacao.plot(kind='bar', color=cores_plot)
        plt.title('Distribuição da Situação dos Alunos', fontsize=14, fontweight='bold')
        plt.xlabel('Situação')
        plt.ylabel('Número de Alunos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'distribuicao_situacao.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: distribuicao_situacao.png")
    
    # 3.5. Evolução temporal
    if 'ano' in df.columns:
        anos_validos = df['ano'].dropna()
        if not anos_validos.empty:
            print("\n📅 3.5. Evolução de Matrículas por Ano")
            print("-" * 40)
            anos = anos_validos.value_counts().sort_index()
            print(anos)
            
            plt.figure(figsize=(12, 6))
            anos.plot(kind='line', marker='o', color='purple', linewidth=2, markersize=8)
            plt.title('Evolução de Matrículas por Ano', fontsize=14, fontweight='bold')
            plt.xlabel('Ano')
            plt.ylabel('Número de Matrículas')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(GRAFICOS_DIR / 'evolucao_matriculas.png', dpi=300)
            plt.close()
            print(f"   ✅ Gráfico salvo: evolucao_matriculas.png")
    
    # 3.6. ANÁLISE DE EVASÃO POR CURSO
    print("\n" + "="*80)
    print("📊 ANÁLISE DE EVASÃO POR CURSO")
    print("="*80)
    
    if 'curso' in df.columns and 'evadido' in df.columns:
        evasao_curso = df.groupby('curso').agg({
            'evadido': ['count', 'sum', 'mean']
        }).round(2)
        evasao_curso.columns = ['total', 'evadidos', 'taxa_evasao']
        evasao_curso = evasao_curso.sort_values('taxa_evasao', ascending=False)
        
        print("\n📚 EVASÃO POR CURSO (TOP 10)")
        print("-" * 50)
        print(evasao_curso.head(10))
        
        print("\n📚 EVASÃO POR CURSO (BOTTOM 5 - MENOR EVASÃO)")
        print("-" * 50)
        print(evasao_curso.tail(5))
        
        # Gráfico de evasão por curso
        plt.figure(figsize=(14, 7))
        evasao_curso.head(15)['taxa_evasao'].plot(kind='bar', color='coral')
        plt.title('Top 15 Cursos com Maior Taxa de Evasão', fontsize=14, fontweight='bold')
        plt.xlabel('Curso')
        plt.ylabel('Taxa de Evasão')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'evasao_por_curso.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: evasao_por_curso.png")
    
    # 3.7. ANÁLISE DE EVASÃO POR FAIXA ETÁRIA
    print("\n" + "="*80)
    print("📊 ANÁLISE DE EVASÃO POR FAIXA ETÁRIA")
    print("="*80)
    
    if 'faixa_etaria' in df.columns and 'evadido' in df.columns:
        evasao_idade = df.groupby('faixa_etaria').agg({
            'evadido': ['count', 'sum', 'mean']
        }).round(2)
        evasao_idade.columns = ['total', 'evadidos', 'taxa_evasao']
        evasao_idade = evasao_idade.sort_index()
        
        print("\n👤 EVASÃO POR FAIXA ETÁRIA")
        print("-" * 50)
        print(evasao_idade)
        
        # Gráfico
        plt.figure(figsize=(10, 6))
        cores_idade = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        evasao_idade['taxa_evasao'].plot(kind='bar', color=cores_idade)
        plt.title('Taxa de Evasão por Faixa Etária', fontsize=14, fontweight='bold')
        plt.xlabel('Faixa Etária')
        plt.ylabel('Taxa de Evasão')
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'evasao_por_faixa_etaria.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: evasao_por_faixa_etaria.png")
    
    # 3.8. ANÁLISE DE EVASÃO POR BAIRRO
    print("\n" + "="*80)
    print("📊 ANÁLISE DE EVASÃO POR BAIRRO")
    print("="*80)
    
    if 'bairro' in df.columns and 'evadido' in df.columns:
        # Filtrar bairros com pelo menos 10 alunos
        evasao_bairro = df.groupby('bairro').agg({
            'evadido': ['count', 'sum', 'mean']
        }).round(2)
        evasao_bairro.columns = ['total', 'evadidos', 'taxa_evasao']
        evasao_bairro = evasao_bairro[evasao_bairro['total'] >= 10]
        evasao_bairro = evasao_bairro.sort_values('taxa_evasao', ascending=False)
        
        print("\n📍 EVASÃO POR BAIRRO (TOP 10)")
        print("-" * 50)
        print(evasao_bairro.head(10))
        
        # Gráfico
        plt.figure(figsize=(14, 7))
        evasao_bairro.head(15)['taxa_evasao'].plot(kind='bar', color='orange')
        plt.title('Top 15 Bairros com Maior Taxa de Evasão', fontsize=14, fontweight='bold')
        plt.xlabel('Bairro')
        plt.ylabel('Taxa de Evasão')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'evasao_por_bairro.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: evasao_por_bairro.png")
    
    # 3.9. ANÁLISE DE PENDÊNCIAS
    print("\n" + "="*80)
    print("📊 ANÁLISE DE PENDÊNCIAS")
    print("="*80)
    
    if 'pendencia' in df.columns:
        pendencias = df['pendencia'].value_counts().head(15)
        print("\n📋 PRINCIPAIS PENDÊNCIAS")
        print("-" * 50)
        print(pendencias)
        
        # Gráfico
        plt.figure(figsize=(14, 7))
        pendencias.plot(kind='bar', color='#F18F01')
        plt.title('Principais Pendências dos Alunos', fontsize=14, fontweight='bold')
        plt.xlabel('Pendência')
        plt.ylabel('Número de Alunos')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'analise_pendencias.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: analise_pendencias.png")
        
        # Análise de evasão por pendência
        if 'evadido' in df.columns:
            evasao_pendencia = df.groupby('pendencia')['evadido'].mean().sort_values(ascending=False)
            print("\n📊 EVASÃO POR PENDÊNCIA (TOP 5)")
            print("-" * 50)
            print(evasao_pendencia.head(5))
    
    # 3.10. ANÁLISE DE CARGA HORÁRIA
    print("\n" + "="*80)
    print("📊 ANÁLISE DE CARGA HORÁRIA")
    print("="*80)
    
    if 'carga_horaria_numerica' in df.columns:
        print("\n⏱️ ESTATÍSTICAS DE CARGA HORÁRIA")
        print("-" * 50)
        print(f"   Média: {df['carga_horaria_numerica'].mean():.1f} horas")
        print(f"   Mínima: {df['carga_horaria_numerica'].min():.0f} horas")
        print(f"   Máxima: {df['carga_horaria_numerica'].max():.0f} horas")
        print(f"   Mediana: {df['carga_horaria_numerica'].median():.0f} horas")
        
        # Análise de evasão por carga horária
        if 'evadido' in df.columns:
            # Criar faixas de carga horária
            bins = [0, 40, 80, 120, 200, 500]
            labels = ['0-40', '41-80', '81-120', '121-200', '200+']
            df['faixa_carga'] = pd.cut(df['carga_horaria_numerica'], bins=bins, labels=labels, right=False)
            
            evasao_carga = df.groupby('faixa_carga')['evadido'].mean().round(2)
            print("\n📊 EVASÃO POR FAIXA DE CARGA HORÁRIA")
            print("-" * 50)
            print(evasao_carga)
            
            plt.figure(figsize=(10, 6))
            evasao_carga.plot(kind='bar', color='teal')
            plt.title('Taxa de Evasão por Faixa de Carga Horária', fontsize=14, fontweight='bold')
            plt.xlabel('Carga Horária (horas)')
            plt.ylabel('Taxa de Evasão')
            plt.tight_layout()
            plt.savefig(GRAFICOS_DIR / 'evasao_por_carga.png', dpi=300)
            plt.close()
            print(f"   ✅ Gráfico salvo: evasao_por_carga.png")
    
    # 3.11. MATRIZ DE CORRELAÇÃO
    print("\n" + "="*80)
    print("📊 MATRIZ DE CORRELAÇÃO")
    print("="*80)
    
    # Selecionar colunas numéricas
    colunas_numericas = ['idade', 'carga_horaria_numerica', 'evadido']
    colunas_existentes = [c for c in colunas_numericas if c in df.columns]
    
    if colunas_existentes:
        # Adicionar colunas codificadas
        df_temp = df.copy()
        for col in ['curso', 'modulo', 'bairro']:
            if col in df.columns:
                df_temp[f'{col}_cod'] = LabelEncoder().fit_transform(df[col].astype(str))
                colunas_existentes.append(f'{col}_cod')
        
        plt.figure(figsize=(12, 10))
        correlacao = df_temp[colunas_existentes].corr()
        mascara = np.triu(np.ones_like(correlacao, dtype=bool))
        sns.heatmap(correlacao, mask=mascara, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
        plt.title('Matriz de Correlação das Variáveis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'matriz_correlacao.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: matriz_correlacao.png")

# ============================================================================
# FUNÇÃO 4: MODELAGEM PREDITIVA
# ============================================================================

def modelagem_preditiva(df):
    print("\n" + "="*80)
    print("🤖 MODELAGEM PREDITIVA - PREVISÃO DE EVASÃO")
    print("="*80)
    
    if 'evadido' not in df.columns:
        print("❌ Coluna 'evadido' não encontrada. Pulando modelagem.")
        return None, None
    
    # 4.1. Preparar features
    print("\n📊 4.1. Preparando os dados...")
    
    features = []
    for f in ['idade', 'curso', 'modulo', 'bairro', 'tem_pendencia', 'carga_horaria_numerica']:
        if f in df.columns:
            features.append(f)
    
    if not features:
        print("❌ Nenhuma feature disponível para modelagem.")
        return None, None
    
    X = df[features].copy()
    
    # Tratar valores nulos
    for col in features:
        if col in ['idade', 'carga_horaria_numerica']:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna('DESCONHECIDO')
    
    # Codificar variáveis categóricas
    le = LabelEncoder()
    for col in features:
        if col not in ['idade', 'carga_horaria_numerica']:
            X[col] = le.fit_transform(X[col].astype(str))
    
    y = df['evadido'].fillna(0)
    
    print(f"   - Amostras: {len(X)}")
    print(f"   - Features: {list(X.columns)}")
    print(f"   - Evadidos: {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
    print(f"   - Não evadidos: {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
    
    # 4.2. Divisão treino/teste
    print("\n📊 4.2. Dividindo dados...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   - Treino: {len(X_train)} amostras")
    print(f"   - Teste: {len(X_test)} amostras")
    
    # 4.3. Treinar modelos
    print("\n🤖 4.3. Treinando modelos...")
    
    modelos = {
        'Regressão Logística': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    resultados = {}
    
    for nome, modelo in modelos.items():
        print(f"\n   Treinando {nome}...")
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
        y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
        
        acuracia = (y_pred == y_test).mean()
        auc = roc_auc_score(y_test, y_prob)
        
        resultados[nome] = {
            'modelo': modelo,
            'acuracia': acuracia,
            'auc': auc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        print(f"      ✅ Acurácia: {acuracia:.3f}")
        print(f"      ✅ AUC-ROC: {auc:.3f}")
    
    # 4.4. Comparação de modelos
    print("\n📊 4.4. Comparação de Modelos")
    print("-" * 50)
    comparacao = pd.DataFrame({
        'Modelo': list(resultados.keys()),
        'Acurácia': [r['acuracia'] for r in resultados.values()],
        'AUC-ROC': [r['auc'] for r in resultados.values()]
    })
    print(comparacao.to_string(index=False))
    
    # 4.5. Melhor modelo
    melhor_nome = max(resultados, key=lambda x: resultados[x]['auc'])
    melhor_modelo = resultados[melhor_nome]
    
    print(f"\n✅ MELHOR MODELO: {melhor_nome}")
    print(f"   - Acurácia: {melhor_modelo['acuracia']:.3f}")
    print(f"   - AUC-ROC: {melhor_modelo['auc']:.3f}")
    
    # 4.6. Matriz de confusão
    print("\n📊 4.5. Matriz de Confusão")
    print("-" * 50)
    
    cm = confusion_matrix(y_test, melhor_modelo['y_pred'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {melhor_nome}', fontsize=14, fontweight='bold')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / 'matriz_confusao.png', dpi=300)
    plt.close()
    print(f"   ✅ Gráfico salvo: matriz_confusao.png")
    
    print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_test, melhor_modelo['y_pred']))
    
    # 4.7. Importância das Features
    if 'Random Forest' in resultados:
        rf = resultados['Random Forest']['modelo']
        importancias = pd.DataFrame({
            'Feature': X.columns,
            'Importância': rf.feature_importances_
        }).sort_values('Importância', ascending=False)
        
        print("\n📊 4.6. Importância das Features")
        print("-" * 50)
        print(importancias.to_string(index=False))
        
        plt.figure(figsize=(10, 6))
        plt.barh(importancias['Feature'], importancias['Importância'], color='teal')
        plt.xlabel('Importância')
        plt.title('Importância das Features para Previsão de Evasão', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'importancia_features.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: importancia_features.png")
    
    # 4.8. Curva ROC
    print("\n📊 4.7. Curva ROC")
    print("-" * 50)
    
    plt.figure(figsize=(10, 8))
    
    for nome, resultado in resultados.items():
        fpr, tpr, _ = roc_curve(y_test, resultado['y_prob'])
        plt.plot(fpr, tpr, label=f'{nome} (AUC = {resultado["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC - Comparação de Modelos', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / 'curva_roc.png', dpi=300)
    plt.close()
    print(f"   ✅ Gráfico salvo: curva_roc.png")
    
    return resultados, melhor_nome

# ============================================================================
# FUNÇÃO 5: RECOMENDAÇÕES
# ============================================================================

def gerar_recomendacoes(df, resultados, melhor_nome):
    print("\n" + "="*80)
    print("💡 RECOMENDAÇÕES ESTRATÉGICAS")
    print("="*80)
    
    recomendacoes = []
    
    # 5.1. Análise de cursos com maior evasão
    if 'curso' in df.columns and 'evadido' in df.columns:
        evasao_curso = df.groupby('curso')['evadido'].mean().sort_values(ascending=False)
        cursos_criticos = evasao_curso[evasao_curso > 0.3].head(5)
        
        if not cursos_criticos.empty:
            print("\n🎯 CURSOS COM MAIOR TAXA DE EVASÃO (>30%)")
            print("-" * 50)
            for curso, taxa in cursos_criticos.items():
                print(f"   - {curso}: {taxa:.1%}")
            
            recomendacoes.append({
                'categoria': 'Cursos',
                'recomendacao': 'Realizar diagnóstico aprofundado nos cursos com maior evasão',
                'detalhes': list(cursos_criticos.index)
            })
    
    # 5.2. Análise por faixa etária
    if 'faixa_etaria' in df.columns and 'evadido' in df.columns:
        evasao_idade = df.groupby('faixa_etaria')['evadido'].mean()
        faixas_criticas = evasao_idade[evasao_idade > 0.3]
        
        if not faixas_criticas.empty:
            print("\n👤 FAIXAS ETÁRIAS CRÍTICAS")
            print("-" * 50)
            for faixa, taxa in faixas_criticas.items():
                print(f"   - {faixa}: {taxa:.1%}")
            
            recomendacoes.append({
                'categoria': 'Faixa Etária',
                'recomendacao': 'Desenvolver programas de apoio específicos para as faixas etárias com maior evasão',
                'detalhes': list(faixas_criticas.index)
            })
    
    # 5.3. Análise por bairro
    if 'bairro' in df.columns and 'evadido' in df.columns:
        evasao_bairro = df.groupby('bairro')['evadido'].mean().sort_values(ascending=False)
        evasao_bairro = evasao_bairro[df.groupby('bairro').size() >= 10]
        bairros_criticos = evasao_bairro[evasao_bairro > 0.4].head(5)
        
        if not bairros_criticos.empty:
            print("\n📍 BAIRROS CRÍTICOS")
            print("-" * 50)
            for bairro, taxa in bairros_criticos.items():
                print(f"   - {bairro}: {taxa:.1%}")
            
            recomendacoes.append({
                'categoria': 'Bairros',
                'recomendacao': 'Implementar ações de mobilidade e transporte para bairros com alta evasão',
                'detalhes': list(bairros_criticos.index)
            })
    
    # 5.4. Análise de pendências
    if 'pendencia' in df.columns and 'evadido' in df.columns:
        evasao_pendencia = df.groupby('pendencia')['evadido'].mean().sort_values(ascending=False)
        pendencias_criticas = evasao_pendencia[evasao_pendencia > 0.3].head(5)
        
        if not pendencias_criticas.empty:
            print("\n⚠️ PENDÊNCIAS CRÍTICAS")
            print("-" * 50)
            for pendencia, taxa in pendencias_criticas.items():
                if pendencia != 'SEM_PENDENCIA':
                    print(f"   - {pendencia}: {taxa:.1%}")
            
            recomendacoes.append({
                'categoria': 'Pendências',
                'recomendacao': 'Criar programa de regularização de pendências com maior impacto na evasão',
                'detalhes': list(pendencias_criticas.index)
            })
    
    # 5.5. Recomendações baseadas no modelo
    if resultados and melhor_nome:
        print("\n🤖 INSIGHTS DO MODELO PREDITIVO")
        print("-" * 50)
        print(f"   Melhor modelo: {melhor_nome}")
        print(f"   Acurácia: {resultados[melhor_nome]['acuracia']:.3f}")
        print(f"   AUC-ROC: {resultados[melhor_nome]['auc']:.3f}")
        
        if 'Random Forest' in resultados:
            rf = resultados['Random Forest']['modelo']
            if hasattr(rf, 'feature_importances_'):
                importancias = pd.DataFrame({
                    'Feature': ['idade', 'curso', 'modulo', 'bairro', 'tem_pendencia', 'carga_horaria_numerica'][:len(rf.feature_importances_)],
                    'Importância': rf.feature_importances_
                }).sort_values('Importância', ascending=False)
                
                print("\n   Fatores mais influentes para evasão:")
                for _, row in importancias.head(3).iterrows():
                    print(f"   - {row['Feature']}: {row['Importância']:.2%}")
                
                recomendacoes.append({
                    'categoria': 'Modelo Preditivo',
                    'recomendacao': 'Utilizar o modelo para identificar alunos em risco de evasão',
                    'detalhes': importancias.head(3)['Feature'].tolist()
                })
    
    # 5.6. Recomendações gerais
    print("\n📋 RECOMENDAÇÕES ESTRATÉGICAS GERAIS")
    print("-" * 50)
    
    recomendacoes_gerais = [
        "1. Implementar sistema de monitoramento contínuo de evasão",
        "2. Criar programa de mentoria para alunos em risco",
        "3. Desenvolver ações de acolhimento para novos alunos",
        "4. Melhorar comunicação com alunos de bairros distantes",
        "5. Oferecer flexibilidade de horários para alunos trabalhadores",
        "6. Criar campanhas de regularização de pendências",
        "7. Implementar pesquisa de satisfação para identificar problemas",
        "8. Desenvolver parcerias com empresas para estágios e empregabilidade"
    ]
    
    for rec in recomendacoes_gerais:
        print(f"   {rec}")
    
    return recomendacoes

# ============================================================================
# FUNÇÃO 6: GERAR RELATÓRIO COMPLETO
# ============================================================================

def gerar_relatorio_completo(df, recomendacoes):
    print("\n" + "="*80)
    print("📄 GERANDO RELATÓRIO COMPLETO")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = RELATORIOS_DIR / f"relatorio_completo_{timestamp}.txt"
    
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO COMPLETO - ANÁLISE DE DADOS FUNDEC\n")
        f.write("="*80 + "\n")
        f.write(f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
        f.write(f"Total de registros: {len(df)}\n")
        f.write("="*80 + "\n\n")
        
        # Estatísticas gerais
        f.write("📊 ESTATÍSTICAS GERAIS\n")
        f.write("-"*50 + "\n")
        
        if 'curso' in df.columns:
            f.write(f"Total de cursos: {df['curso'].nunique()}\n")
            f.write(f"Cursos mais ofertados:\n")
            for curso, count in df['curso'].value_counts().head(5).items():
                f.write(f"  - {curso}: {count}\n")
        
        if 'situacao' in df.columns:
            f.write(f"\n📈 SITUAÇÃO DOS ALUNOS\n")
            for status, count in df['situacao'].value_counts().items():
                pct = count / len(df) * 100
                f.write(f"  - {status}: {count} ({pct:.1f}%)\n")
        
        if 'evadido' in df.columns:
            evadidos = df['evadido'].sum()
            taxa_evasao = evadidos / len(df) * 100
            f.write(f"\n⚠️ TAXA DE EVASÃO: {taxa_evasao:.1f}%\n")
            f.write(f"  - Evadidos: {evadidos}\n")
            f.write(f"  - Não evadidos: {len(df) - evadidos}\n")
        
        # Recomendações
        f.write("\n" + "="*80 + "\n")
        f.write("💡 RECOMENDAÇÕES ESTRATÉGICAS\n")
        f.write("="*80 + "\n\n")
        
        if recomendacoes:
            for rec in recomendacoes:
                f.write(f"📌 {rec['categoria']}\n")
                f.write(f"   Recomendação: {rec['recomendacao']}\n")
                f.write(f"   Detalhes: {', '.join(str(d) for d in rec['detalhes'])}\n\n")
        else:
            f.write("Nenhuma recomendação específica gerada.\n")
    
    print(f"✅ Relatório salvo em: {nome_arquivo}")
    return nome_arquivo

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    print("\n🚀 INICIANDO ANÁLISE COMPLETA DA FUNDEC")
    print("="*80)
    
    try:
        # 1. Carregar dados
        df = carregar_dados()
        if df is None:
            return
        
        # 2. Limpar dados
        df_limpo = limpar_dados(df)
        if df_limpo is None:
            return
        
        # 3. Análise exploratória
        analise_exploratoria(df_limpo)
        
        # 4. Modelagem preditiva
        resultados, melhor_nome = modelagem_preditiva(df_limpo)
        
        # 5. Recomendações
        recomendacoes = gerar_recomendacoes(df_limpo, resultados, melhor_nome)
        
        # 6. Relatório completo
        relatorio = gerar_relatorio_completo(df_limpo, recomendacoes)
        
        print("\n" + "="*80)
        print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("="*80)
        print(f"\n📂 Resultados salvos em: {OUTPUT_DIR}")
        print(f"   - Gráficos: {GRAFICOS_DIR}")
        print(f"   - Relatório: {relatorio}")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
