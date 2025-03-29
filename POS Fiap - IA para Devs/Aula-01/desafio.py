import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    try:
        # Carregar dados
        dados = importar_dados()
        if dados is None:
            return
        
        # Explorar os dados
        print("\nPrimeiras 5 linhas dos dados:")
        print(dados.head())
        
        print("\nInformações dos dados:")
        print(dados.info())
        
        print("\nEstatísticas descritivas:")
        print(dados.describe())
        
        dados = remover_dados_duplicados_e_null(dados)
        
        # Pré-processar os dados
        X_train, X_test, y_train, y_test = divisao_caracteristica(dados, test_size=0.3, random_state=42)        
        
        # análise exploratória: crie uma pergunta sobre a base de dados e responda através de um gráfico
        analise_exploratoria(dados)
        
        modelo_rf = treinar_modelo_random_forest(x=X_train, y=y_train, n_estimators=100, random_state=42)
        y_pred_rf = modelo_rf.predict(X_test)
        accuracy_modelo, recall_modelo = executar_modelo(y_test, y_pred_rf, 'RandomForest','Dropout')        
        avaliar_modelo(accuracy_modelo, recall_modelo, 'RandomForest')
        
        modelo_lr = treinar_modelo_logistic_regression(x=X_train, y=y_train,max_iter=1000)
        y_pred_lr = modelo_lr.predict(X_test)
        accuracy_modelo, recall_modelo = executar_modelo(y_test, y_pred_lr, 'LogisticRegression','Dropout')        
        avaliar_modelo(accuracy_modelo, recall_modelo,'LogisticRegression')
        
        return None
    except Exception as ex:
        print(f"Erro ao processar rotina: {ex}")
        return None


def importar_dados():
    try:
        dados = pd.read_csv('dropout-inaugural.csv')
        print(f"Dados importados com sucesso.")
        return dados
    except Exception as ex:
        print(f"Erro ao importar dados: {ex}")
        return None

def analise_exploratoria(dados):
    dados_grouped = dados.groupby(['Curricular units 1st sem (approved)', 'Target']).size().unstack()
    
    plt.figure(figsize=(12,6))
    dados_grouped.plot(kind='bar', stacked=True, colormap='viridis', alpha=0.85)
    
    #personalizar o grafico
    plt.title('Relação entre Disciplinas Aprovadas e Dropout/Graduate')
    plt.xlabel('Disciplinas Aprovadas no 1º Semestre')
    plt.ylabel('Quantidade de Alunos')
    plt.legend(title='Status do Aluno')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    #exibir o grafico
    plt.show()
    
def divisao_caracteristica(dados, test_size, random_state):
    X = dados.drop(columns=['Target'])
    y = dados['Target']
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # print(f"Tamanho do conjunto de treino: {X_train.shape[0]} linhas")
    # print(f"Tamanho do conjunto de teste: {X_test.shape[0]} linhas")
    
def remover_dados_duplicados_e_null(dados):
    dados.drop_duplicates(inplace=True)
    dados.dropna(inplace=True)
    return dados
    
def divisao_entre_treino_test(dados, test_size, random_state):
    X = dados.drop(columns=['Target', 'Dropout'])
    y = dados['Target']
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def treinar_modelo_random_forest(x, y, n_estimators, random_state):
    rf_modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_modelo.fit(x, y)
    return rf_modelo

def treinar_modelo_logistic_regression(x, y, max_iter):
    lr_modelo = LogisticRegression(max_iter=max_iter)
    lr_modelo.fit(x, y)
    return lr_modelo

def executar_modelo(y_test, y_pred, modelo_name, pos_label):
    accuracy_modelo = accuracy_score(y_test, y_pred)
    recall_modelo = recall_score(y_test, y_pred, pos_label=pos_label)
    print(f'Acurácia do {modelo_name} Logística: {accuracy_modelo * 100:.2f}%')
    print(f'Recall ({pos_label}) do {modelo_name} Logística: {recall_modelo * 100:.2f}%')
    
    return accuracy_modelo, recall_modelo
    
def avaliar_modelo(accuracy_modelo, recall_modelo, pos_label):
    if accuracy_modelo >= 0.90 and recall_modelo >= 0.80:
        print(f'O modelo {pos_label} atendeu aos critérios de desempenho!')
    else:
        print(f'O modelo {pos_label} NÃO atendeu aos critérios de desempenho.')

# Executar o código principal
if __name__ == "__main__":
    
    # Executar o processo completo
    # modelo, caracteristicas, classes = main()
    main()