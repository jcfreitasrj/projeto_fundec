# =============================================================================
# PROJETO DE EXTENSÃO EM CIÊNCIA DE DADOS
# ANÁLISE DE DADOS DE ATENDIMENTO - FUNDEC
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# CONFIGURAÇÃO DE ESTILO - CORRIGIDA
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
print("="*80)
print(f"Data da execução: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
print("="*80)

# CONFIGURAÇÃO DE DIRETÓRIOS
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

# =============================================================================
# CARREGAMENTO DOS DADOS
# =============================================================================

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

# =============================================================================
# LIMPEZA DOS DADOS
# =============================================================================

def limpar_dados(df):
    print("\n" + "="*80)
    print("🧹 LIMPANDO OS DADOS")
    print("="*80)
    
    df_limpo = df.copy()
    
    # Padronizar colunas
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
    
    print(f"   ✅ Colunas padronizadas: {list(df_limpo.columns)}")
    
    # Tratar datas
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
        
        print(f"   ✅ Idade calculada - Média: {df_limpo['idade'].mean():.1f} anos")
    
    # Processar período
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
    
    # Padronizar situação
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
        elif 'EVAD' in situacao or 'TRANSF' in situacao:
            return 'EVADIDO'
        else:
            return situacao
    
    if 'situacao' in df_limpo.columns:
        df_limpo['situacao'] = df_limpo['situacao'].apply(padronizar_situacao)
        df_limpo['evadido'] = df_limpo['situacao'].apply(
            lambda x: 1 if x in ['EVADIDO', 'DESISTENTE'] else 0
        )
        print(f"   ✅ Situações padronizadas")
    
    # Tratar pendências
    if 'pendencia' in df_limpo.columns:
        df_limpo['tem_pendencia'] = df_limpo['pendencia'].notna()
        df_limpo['pendencia'] = df_limpo['pendencia'].fillna('SEM_PENDENCIA')
        print(f"   ✅ Pendências processadas")
    
    print(f"\n✅ Limpeza concluída! Total: {len(df_limpo)} registros")
    return df_limpo

# =============================================================================
# ANÁLISE EXPLORATÓRIA
# =============================================================================

def analise_exploratoria(df):
    print("\n" + "="*80)
    print("📊 ANÁLISE EXPLORATÓRIA")
    print("="*80)
    
    if 'curso' in df.columns:
        print("\n📚 Distribuição dos Cursos:")
        cursos = df['curso'].value_counts()
        print(cursos)
        
        plt.figure(figsize=(12, 6))
        cursos.plot(kind='bar', color='steelblue')
        plt.title('Distribuição de Alunos por Curso', fontsize=14)
        plt.xlabel('Curso')
        plt.ylabel('Número de Alunos')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'distribuicao_cursos.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: distribuicao_cursos.png")
    
    if 'faixa_etaria' in df.columns:
        print("\n📊 Distribuição por Faixa Etária:")
        faixas = df['faixa_etaria'].value_counts().sort_index()
        print(faixas)
        
        plt.figure(figsize=(10, 6))
        faixas.plot(kind='bar', color='coral')
        plt.title('Distribuição por Faixa Etária', fontsize=14)
        plt.xlabel('Faixa Etária')
        plt.ylabel('Número de Alunos')
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'distribuicao_faixa_etaria.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: distribuicao_faixa_etaria.png")
    
    if 'bairro' in df.columns:
        print("\n📍 Top 10 Bairros:")
        bairros = df['bairro'].value_counts().head(10)
        print(bairros)
        
        plt.figure(figsize=(12, 6))
        bairros.plot(kind='bar', color='green')
        plt.title('Top 10 Bairros com Mais Alunos', fontsize=14)
        plt.xlabel('Bairro')
        plt.ylabel('Número de Alunos')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'top_bairros.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: top_bairros.png")
    
    if 'situacao' in df.columns:
        print("\n📈 Distribuição da Situação:")
        situacao = df['situacao'].value_counts()
        print(situacao)
        
        situacao_pct = (situacao / len(df)) * 100
        print(f"\n   Taxa de Aprovação: {situacao_pct.get('APROVADO', 0):.1f}%")
        print(f"   Taxa de Evasão: {situacao_pct.get('EVADIDO', 0):.1f}%")
        print(f"   Taxa de Desistência: {situacao_pct.get('DESISTENTE', 0):.1f}%")
        
        plt.figure(figsize=(10, 6))
        cores = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
        situacao.plot(kind='bar', color=cores[:len(situacao)])
        plt.title('Distribuição da Situação dos Alunos', fontsize=14)
        plt.xlabel('Situação')
        plt.ylabel('Número de Alunos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(GRAFICOS_DIR / 'distribuicao_situacao.png', dpi=300)
        plt.close()
        print(f"   ✅ Gráfico salvo: distribuicao_situacao.png")
    
    if 'ano' in df.columns:
        anos_validos = df['ano'].dropna()
        if not anos_validos.empty:
            print("\n📅 Evolução de Matrículas por Ano:")
            anos = anos_validos.value_counts().sort_index()
            print(anos)
            
            plt.figure(figsize=(12, 6))
            anos.plot(kind='line', marker='o', color='purple', linewidth=2)
            plt.title('Evolução de Matrículas por Ano', fontsize=14)
            plt.xlabel('Ano')
            plt.ylabel('Número de Matrículas')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(GRAFICOS_DIR / 'evolucao_matriculas.png', dpi=300)
            plt.close()
            print(f"   ✅ Gráfico salvo: evolucao_matriculas.png")

# =============================================================================
# MODELAGEM PREDITIVA
# =============================================================================

def modelagem_preditiva(df):
    print("\n" + "="*80)
    print("🤖 MODELAGEM PREDITIVA - PREVISÃO DE EVASÃO")
    print("="*80)
    
    if 'evadido' not in df.columns:
        print("❌ Coluna 'evadido' não encontrada. Pulando modelagem.")
        return None, None
    
    # Selecionar features disponíveis
    features = []
    for f in ['idade', 'curso', 'modulo', 'bairro', 'tem_pendencia']:
        if f in df.columns:
            features.append(f)
    
    if not features:
        print("❌ Nenhuma feature disponível para modelagem.")
        return None, None
    
    X = df[features].copy()
    
    # Tratar valores nulos
    for col in features:
        if col == 'idade':
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna('DESCONHECIDO')
    
    # Codificar variáveis categóricas
    le = LabelEncoder()
    for col in features:
        if col != 'idade':
            X[col] = le.fit_transform(X[col].astype(str))
    
    y = df['evadido'].fillna(0)
    
    print(f"\n📊 Preparação dos dados:")
    print(f"   - Amostras: {len(X)}")
    print(f"   - Features: {list(X.columns)}")
    print(f"   - Evadidos: {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
    print(f"   - Não evadidos: {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Padronização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos
    modelos = {
        'Regressão Logística': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
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
    
    # Melhor modelo
    melhor_nome = max(resultados, key=lambda x: resultados[x]['auc'])
    
    print(f"\n✅ Melhor modelo: {melhor_nome}")
    print(f"   - Acurácia: {resultados[melhor_nome]['acuracia']:.3f}")
    print(f"   - AUC-ROC: {resultados[melhor_nome]['auc']:.3f}")
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, resultados[melhor_nome]['y_pred'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {melhor_nome}', fontsize=14)
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / 'matriz_confusao.png', dpi=300)
    plt.close()
    print(f"   ✅ Matriz de confusão salva")
    
    return resultados, melhor_nome

# =============================================================================
# RECOMENDAÇÕES
# =============================================================================

def gerar_recomendacoes(df, resultados, melhor_nome):
    print("\n" + "="*80)
    print("💡 RECOMENDAÇÕES E INSIGHTS")
    print("="*80)
    
    if 'situacao' in df.columns:
        total = len(df)
        aprovados = len(df[df['situacao'] == 'APROVADO'])
        evadidos = len(df[df['situacao'] == 'EVADIDO'])
        desistentes = len(df[df['situacao'] == 'DESISTENTE'])
        reprovados = len(df[df['situacao'] == 'REPROVADO'])
        
        print(f"\n📊 Panorama Geral:")
        print(f"   Total de alunos: {total}")
        print(f"   ✅ Aprovados: {aprovados} ({aprovados/total*100:.1f}%)")
        print(f"   ❌ Evasão: {evadidos} ({evadidos/total*100:.1f}%)")
        print(f"   ⚠️ Desistentes: {desistentes} ({desistentes/total*100:.1f}%)")
        print(f"   📝 Reprovados: {reprovados} ({reprovados/total*100:.1f}%)")
    
    if 'curso' in df.columns and 'evadido' in df.columns:
        evasao_curso = df.groupby('curso')['evadido'].mean().sort_values(ascending=False)
        if not evasao_curso.empty:
            print(f"\n🔴 Curso com MAIOR evasão: {evasao_curso.index[0]} ({evasao_curso.iloc[0]*100:.1f}%)")
            if len(evasao_curso) > 1:
                print(f"🟢 Curso com MENOR evasão: {evasao_curso.index[-1]} ({evasao_curso.iloc[-1]*100:.1f}%)")
    
    if 'faixa_etaria' in df.columns and 'evadido' in df.columns:
        evasao_idade = df.groupby('faixa_etaria')['evadido'].mean().sort_values(ascending=False)
        if not evasao_idade.empty:
            print(f"\n👤 Faixa etária com MAIOR evasão: {evasao_idade.index[0]} ({evasao_idade.iloc[0]*100:.1f}%)")
    
    if resultados and melhor_nome:
        print(f"\n🤖 Modelo preditivo: {melhor_nome}")
        print(f"   AUC-ROC: {resultados[melhor_nome]['auc']:.3f}")
        if resultados[melhor_nome]['auc'] > 0.8:
            print("   ✅ Modelo com boa capacidade preditiva")
    
    print("\n📌 RECOMENDAÇÕES PRÁTICAS:")
    recomendacoes = [
        "1. 🎯 Implementar programa de mentoria para alunos com maior risco de evasão",
        "2. 🔔 Criar sistema de alerta precoce usando o modelo preditivo",
        "3. 📢 Desenvolver campanhas de engajamento para alunos no início do curso",
        "4. 📝 Coletar feedback dos alunos sobre as causas de evasão",
        "5. 📊 Monitorar mensalmente as taxas de evasão por curso e faixa etária"
    ]
    for rec in recomendacoes:
        print(f"   {rec}")

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    print("\n🚀 INICIANDO ANÁLISE...")
    
    df = carregar_dados()
    if df is None:
        return
    
    df_limpo = limpar_dados(df)
    analise_exploratoria(df_limpo)
    resultados, melhor_nome = modelagem_preditiva(df_limpo)
    gerar_recomendacoes(df_limpo, resultados, melhor_nome)
    
    print("\n" + "="*80)
    print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
    print(f"   📁 Gráficos salvos em: {GRAFICOS_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
