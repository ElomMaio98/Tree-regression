# 💳 Análise de Risco de Crédito com Machine Learning

Este projeto desenvolve e avalia modelos de classificação para prever a probabilidade de um cliente de cartão de crédito não conseguir pagar seu empréstimo. O principal desafio deste problema é o grande desbalanceamento de classes no dataset, onde o número de "bons pagadores" é muito superior ao de "maus pagadores".

Para resolver isso, o projeto utiliza técnicas robustas de pré-processamento e a técnica de oversampling **SMOTE** (Synthetic Minority Over-sampling Technique), tudo encapsulado em pipelines do Scikit-learn para garantir um fluxo de trabalho limpo e sem vazamento de dados (data leakage).

## 📊 Fonte dos Dados

O projeto utiliza o dataset "Credit Card", obtido através do Kaggle Hub.

* **Dataset:** [Credit Card on Kaggle](https://www.kaggle.com/datasets/mishra5001/credit-card)
* **Arquivo Principal:** `application_data.csv`

## 🛠️ Tecnologias Utilizadas

* **Python 3.x**
* **Pandas & NumPy**: Para manipulação e análise de dados.
* **Scikit-learn**: Para criação de pipelines de pré-processamento e modelagem (`LogisticRegression`, `DecisionTreeClassifier`).
* **Imbalanced-learn**: Para balanceamento de classes com `SMOTE`.
* **KaggleHub**: Para download do dataset.
* **Joblib**: Para salvar os artefatos do modelo (demonstrado na primeira parte do script).

## 🔬 Metodologia

O fluxo de trabalho foi construído utilizando `Pipelines` para automatizar e organizar as etapas:

1.  **Divisão dos Dados**: Os dados são divididos em conjuntos de treino e teste **antes** de qualquer processamento, utilizando a estratificação para manter a proporção original das classes em ambos os conjuntos.

2.  **Pré-processamento**: Um `ColumnTransformer` é usado para aplicar transformações diferentes para cada tipo de dado:
    * **Features Numéricas**: Os valores ausentes são preenchidos com a **mediana** e, em seguida, os dados são padronizados com `StandardScaler`.
    * **Features Categóricas**: Os valores ausentes são preenchidos com a **moda** (valor mais frequente) e, em seguida, as variáveis são convertidas em formato numérico usando `OneHotEncoder`.

3.  **Balanceamento de Classes (SMOTE)**: A técnica SMOTE é integrada diretamente ao pipeline de treinamento. Ela cria exemplos sintéticos da classe minoritária (clientes com dificuldade de pagamento) **apenas nos dados de treino**, evitando que o modelo "veja" dados sintéticos durante a avaliação.

4.  **Modelagem**: Dois modelos de classificação são treinados e avaliados:
    * Regressão Logística
    * Árvore de Decisão

## 🚀 Como Executar o Projeto

### 1. Instalação

Instale as bibliotecas necessárias com o pip:
```bash
pip install pandas scikit-learn imbalanced-learn kagglehub numpy joblib matplotlib
