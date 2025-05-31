# Importa biblioteca para medir tempo de execução
import time

# Importa biblioteca NumPy para manipulação numérica
import numpy as np

# Importa pandas para manipulação de dados em formato tabular
import pandas as pd

# Importa datasets integrados do scikit-learn
from sklearn import datasets

# Função para dividir dataset em treino e teste
from sklearn.model_selection import train_test_split

# Função para normalização dos dados
from sklearn.preprocessing import StandardScaler

# Importa classificador Decision Tree (usado para ID3 com criterion="entropy")
from sklearn.tree import DecisionTreeClassifier

# Importa classificador Support Vector Machine
from sklearn.svm import SVC

# Importa métricas de avaliação para classificadores
from sklearn.metrics import (
    accuracy_score,          # Acurácia: % de classificações corretas
    confusion_matrix,        # Matriz de confusão: reais vs previstos
    classification_report,   # Relatório completo por classe
    precision_score,         # Precisão: TP / (TP + FP)
    recall_score,            # Recall: TP / (TP + FN)
    f1_score                 # F1: harmónica entre precisão e recall
)

# Importa modelo de Regressão Linear
from sklearn.linear_model import LinearRegression

# Importa modelo SVR (Support Vector Regression)
from sklearn.svm import SVR

# Importa métricas para avaliação de regressão
from sklearn.metrics import (
    mean_squared_error,      # Erro Quadrático Médio (MSE)
    mean_absolute_error,     # Erro Absoluto Médio (MAE)
    r2_score                 # Coeficiente de Determinação (R²)
)

# Função que imprime estatísticas descritivas de um DataFrame
def estatisticas_basicas(df: pd.DataFrame, nome: str):
    print(f"\n=== Estatísticas básicas: {nome} ===")
    print(df.describe().transpose())  # Mostra média, std, min, quartis e max
    print("\n")

# Função para treinar e avaliar modelos de classificação no dataset Iris
def executar_iris():
    # Carrega o dataset Iris (flores)
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)  # Dados (características)
    y = pd.Series(iris.target, name="target")                # Rótulos (classes)

    # Mostra estatísticas básicas do conjunto completo
    estatisticas_basicas(pd.concat([X, y], axis=1), "Iris")

    # Divide em treino/teste (70% treino, 30% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Normaliza os dados (necessário para SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === ID3 (DecisionTreeClassifier com entropia) ===
    dt_id3 = DecisionTreeClassifier(criterion="entropy", random_state=42)

    # Mede tempo de treino
    t0_train_dt = time.perf_counter()
    dt_id3.fit(X_train, y_train)
    t1_train_dt = time.perf_counter()

    # Mede tempo de predição
    t0_pred_dt = time.perf_counter()
    y_pred_dt = dt_id3.predict(X_test)
    t1_pred_dt = time.perf_counter()

    # Calcula tempos
    tempo_treino_dt = t1_train_dt - t0_train_dt
    tempo_pred_dt = t1_pred_dt - t0_pred_dt

    # Avalia o modelo com métricas padrão
    acc_dt = accuracy_score(y_test, y_pred_dt)
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    precision_dt = precision_score(y_test, y_pred_dt, average="macro")
    recall_dt = recall_score(y_test, y_pred_dt, average="macro")
    f1_dt = f1_score(y_test, y_pred_dt, average="macro")

    # Exibe os resultados
    print(">>> ID3 (Decision Tree com critério Entropy) - Iris <<<")
    print(f"Tempo de Treino (fit): {tempo_treino_dt:.6f} seg")
    print(f"Tempo de Predição (predict): {tempo_pred_dt:.6f} seg")
    print(f"Acurency: {acc_dt:.4f}")
    print(f"Precision (média macro): {precision_dt:.4f}")
    print(f"Recall (média macro):    {recall_dt:.4f}")
    print(f"F1‐Score (média macro): {f1_dt:.4f}")
    print("Matriz de Confusão:")
    print(cm_dt)
    print("Relatório de Classificação (por classe):")
    print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))
    print("\n")

    # === SVM com kernel RBF ===
    svm_clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)

    # Mede tempo de treino
    t0_train_svm = time.perf_counter()
    svm_clf.fit(X_train_scaled, y_train)
    t1_train_svm = time.perf_counter()

    # Mede tempo de predição
    t0_pred_svm = time.perf_counter()
    y_pred_svm = svm_clf.predict(X_test_scaled)
    t1_pred_svm = time.perf_counter()

    # Calcula tempos
    tempo_treino_svm = t1_train_svm - t0_train_svm
    tempo_pred_svm = t1_pred_svm - t0_pred_svm

    # Avalia o modelo
    acc_svm = accuracy_score(y_test, y_pred_svm)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm, average="macro")
    recall_svm = recall_score(y_test, y_pred_svm, average="macro")
    f1_svm = f1_score(y_test, y_pred_svm, average="macro")

    # Exibe os resultados
    print(">>> SVM (SVC, kernel RBF) - Iris <<<")
    print(f"Tempo de Treino (fit): {tempo_treino_svm:.6f} seg")
    print(f"Tempo de Predição (predict): {tempo_pred_svm:.6f} seg")
    print(f"Acurácia: {acc_svm:.4f}")
    print(f"Precision (média macro): {precision_svm:.4f}")
    print(f"Recall (média macro):    {recall_svm:.4f}")
    print(f"F1‐Score (média macro): {f1_svm:.4f}")
    print("Matriz de Confusão:")
    print(cm_svm)
    print("Relatório de Classificação (por classe):")
    print(classification_report(y_test, y_pred_svm, target_names=iris.target_names))
    print("\n")

# Função para treinar e avaliar modelos de regressão no dataset Diabetes
def executar_diabetes():
    # Carrega dataset Diabetes
    diabetes = datasets.load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name="target")

    # Mostra estatísticas básicas do dataset
    estatisticas_basicas(pd.concat([X, y], axis=1), "Diabetes")

    # Divide os dados em treino/teste (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # Normaliza os dados (essencial para SVR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Regressão Linear ===
    reg_lin = LinearRegression()

    # Mede tempo de treino
    t0_train_rl = time.perf_counter()
    reg_lin.fit(X_train, y_train)
    t1_train_rl = time.perf_counter()

    # Mede tempo de predição
    t0_pred_rl = time.perf_counter()
    y_pred_rl = reg_lin.predict(X_test)
    t1_pred_rl = time.perf_counter()

    # Calcula tempos
    tempo_treino_rl = t1_train_rl - t0_train_rl
    tempo_pred_rl = t1_pred_rl - t0_pred_rl

    # Avalia modelo
    mse_rl = mean_squared_error(y_test, y_pred_rl)
    rmse_rl = np.sqrt(mse_rl)
    mae_rl = mean_absolute_error(y_test, y_pred_rl)
    r2_rl = r2_score(y_test, y_pred_rl)

    # Exibe resultados
    print(">>> Regressão Linear - Diabetes <<<")
    print(f"Tempo de Treino (fit): {tempo_treino_rl:.6f} seg")
    print(f"Tempo de Predição (predict): {tempo_pred_rl:.6f} seg")
    print(f"MSE (Linear Regression): {mse_rl:.4f}")
    print(f"RMSE (Linear Regression): {rmse_rl:.4f}")
    print(f"MAE (Linear Regression): {mae_rl:.4f}")
    print(f"R²  (Linear Regression): {r2_rl:.4f}")
    print("\n")

    # === Support Vector Regressor ===
    svr_reg = SVR(kernel="rbf", C=1.0, epsilon=0.1)

    # Tempo de treino
    t0_train_svr = time.perf_counter()
    svr_reg.fit(X_train_scaled, y_train)
    t1_train_svr = time.perf_counter()

    # Tempo de predição
    t0_pred_svr = time.perf_counter()
    y_pred_svr = svr_reg.predict(X_test_scaled)
    t1_pred_svr = time.perf_counter()

    # Calcula tempos
    tempo_treino_svr = t1_train_svr - t0_train_svr
    tempo_pred_svr = t1_pred_svr - t0_pred_svr

    # Avaliação
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    rmse_svr = np.sqrt(mse_svr)
    mae_svr = mean_absolute_error(y_test, y_pred_svr)
    r2_svr = r2_score(y_test, y_pred_svr)

    # Exibe resultados
    print(">>> SVR (Support Vector Regression) - Diabetes <<<")
    print(f"Tempo de Treino (fit): {tempo_treino_svr:.6f} seg")
    print(f"Tempo de Predição (predict): {tempo_pred_svr:.6f} seg")
    print(f"MSE (SVR): {mse_svr:.4f}")
    print(f"RMSE (SVR): {rmse_svr:.4f}")
    print(f"MAE (SVR): {mae_svr:.4f}")
    print(f"R²  (SVR): {r2_svr:.4f}")
    print("\n")

# Bloco principal — executa os dois processos quando o ficheiro é corrido diretamente
if __name__ == "__main__":
    print("\n############################")
    print("=== PROCESSO: IRIS (CLASSIFICAÇÃO) ===")
    print("############################")
    executar_iris()

    print("\n############################")
    print("=== PROCESSO: DIABETES (REGRESSÃO) ===")
    print("############################")
    executar_diabetes()
