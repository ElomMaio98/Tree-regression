
---

### **Código Organizado**

Esta é a versão refatorada e mais eficiente do seu código, usando pipelines.

```python
# -*- coding: utf-8 -*-

# ===================================================================
# SEÇÃO 1: IMPORTAÇÃO DE BIBLIOTECAS E CARREGAMENTO DOS DADOS
# ===================================================================
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Download e carregamento dos dados
print("Baixando dataset do Kaggle Hub...")
path = kagglehub.dataset_download("mishra5001/credit-card")
DATA_PATH = f"{path}/application_data.csv"
df = pd.read_csv(DATA_PATH)
print("Dataset carregado com sucesso.")


# ===================================================================
# SEÇÃO 2: DIVISÃO DOS DADOS E DEFINIÇÃO DAS FEATURES
# ===================================================================
# Separação da variável alvo (TARGET) das features
X = df.drop('TARGET', axis=1)
y = df['TARGET']

# Divisão em treino e teste com estratificação para manter a proporção das classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Identificação automática das colunas numéricas e categóricas
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include='object').columns.tolist()


# ===================================================================
# SEÇÃO 3: CRIAÇÃO DOS PIPELINES DE PRÉ-PROCESSAMENTO
# ===================================================================
# Pipeline para tratar dados numéricos: imputação com mediana e padronização
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para tratar dados categóricos: imputação com moda e OneHotEncoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combina os pipelines de pré-processamento em um único objeto
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Mantém colunas não especificadas (se houver)
)


# ===================================================================
# SEÇÃO 4: CONSTRUÇÃO, TREINAMENTO E AVALIAÇÃO DOS MODELOS
# ===================================================================
# --- Modelo 1: Regressão Logística com SMOTE ---
pipeline_lr = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
])

print("\nTreinando o modelo de Regressão Logística...")
pipeline_lr.fit(X_train, y_train)
predictions_lr = pipeline_lr.predict(X_test)
print("Treinamento concluído.")

# --- Modelo 2: Árvore de Decisão com SMOTE ---
pipeline_dt = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', DecisionTreeClassifier(max_depth=10, random_state=42))
])

print("\nTreinando o modelo de Árvore de Decisão...")
pipeline_dt.fit(X_train, y_train)
predictions_dt = pipeline_dt.predict(X_test)
print("Treinamento concluído.")


# ===================================================================
# SEÇÃO 5: EXIBIÇÃO DOS RESULTADOS
# ===================================================================
print("\n--- Relatório de Avaliação: Regressão Logística ---")
print(classification_report(y_test, predictions_lr))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, predictions_lr))

print("\n--- Relatório de Avaliação: Árvore de Decisão ---")
print(classification_report(y_test, predictions_dt))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, predictions_dt))
