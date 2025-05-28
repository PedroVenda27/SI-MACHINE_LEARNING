# SI-MACHINE_LEARNING

# 1. INTRODUÇÃO

Nos últimos anos, o **Machine Learning** (Aprendizagem Automática) tem se destacado como uma das áreas mais promissoras da inteligência artificial, impulsionando avanços em diversas áreas como saúde, finanças, reconhecimento de padrões e automação. 

Esta Trata-se de um campo que permite aos sistemas aprenderem automaticamente a partir de dados, identificando padrões e tomando decisões sem serem explicitamente programados para cada tarefa específica.

Neste relatório, exploraremos os conceitos fundamentais de Machine Learning por meio do estudo de dois conjuntos de dados clássicos: o dataset **diabetes** e o dataset **iris**. 

Esses datasets são amplamente utilizados para ilustrar os princípios básicos de classificação, análise de dados e modelagem preditiva.

Serão abordados tópicos essenciais como:

- Conceitos base de Machine Learning
- Técnicas de pré-processamento e análise exploratória dos dados
- Algoritmos de classificação supervisionada
- Avaliação e validação de modelos

Utilizando os datasets diabetes e iris, será possível entender a aplicação prática dos algoritmos, desde a preparação dos dados até a interpretação dos resultados, evidenciando a importância da escolha correta do modelo e dos parâmetros para alcançar alta precisão preditiva.

Além disso, discutiremos os desafios comuns em Machine Learning, como o balanceamento entre viés e variância, overfitting e underfitting, e estratégias para otimizar o desempenho dos modelos.

Este relatório visa proporcionar uma visão geral clara e aplicada de Machine Learning, destacando sua relevância e potencial em resolver problemas complexos baseados em dados reais.


## 1.1 Conceitos Base

### 1.1.1 O que é o conceito de Machine Learning?

Machine Learning (Aprendizagem Automática) consiste numa área da inteligência artificial que permite que os agentes neste caso "computadores" aprendam a realizar tarefas sem serem explicitamente programados para isso. 
Em vez de seguir instruções rígidas definidas por um programador, um sistema de machine learning analisa dados, identifica padrões e constrói modelos que conseguem fazer previsões ou tomar decisões com base em novas informações.

Um agente inteligente pode ser considerado em aprendizagem quando melhora seu comportamento ao observar e interpretar o mundo ao seu redor. No caso dos computadores, essa aprendizagem ocorre através da análise de grandes volumes de dados, a partir dos quais o sistema cria uma “hipótese” sobre como o mundo funciona.

### 1.1.1 Por que deixar a máquina aprender em vez de programá-la diretamente?

Existem duas razões principais para deixar-mos a maquina aprender por si propria:

1. **Complexidade e variabilidade do problema:** Uma vez que em muitos casos, o problema apresenta tanta variabilidade e casos possíveis que seria impossível antecipar todas as situações com regras fixas.

2. **Falta de conhecimento explícito:** Pois Muitas vezes, não sabemos exatamente como resolver um problema, mas a máquina pode encontrar padrões e soluções que ainda não conseguimos identificar.

Assim, o machine learning permite que sistemas computacionais se adaptem, aprendam com experiências e atuem de forma mais eficaz em contextos dinâmicos e complexos.


### 1.1.3 Tipos de Machine Learning

Dentro ainda do Conceito de Machine Learning Existem ainda três principais abordagens, cada uma adequada a diferentes tipos de problemas e dados:

- **Aprendizagem Supervisionada:**  
  Onde o agente recebe um conjunto de dados composto por pares de entrada (input) e saída desejada (output), chamados de *labels* , e através desses exemplos, o sistema aprende a função que relaciona entradas a saídas corretas.
  
  **Por Exemplo:** identificar se uma imagem contém um gato ou não, com base em imagens previamente classificadas.

- **Aprendizagem Não Supervisionada:**  
  Onde o agente recebe apenas os dados de entrada, sem qualquer informação sobre o resultado esperado tendo como objetivo descobrir padrões, estruturas ou agrupamentos nos dados.
  
  **Por Exemplo:** agrupar clientes com comportamentos de compra semelhantes (clustering).

- **Aprendizagem por Reforço (Reinforcement Learning):**  
  Onde o agente aprende a tomar decisões ao interagir com um ambiente, recebendo feedback na forma de recompensas ou penalidades, que o ajudam a melhorar suas ações ao longo do tempo.
  
  **Por Exemplo:** um programa que aprende a jogar xadrez ganhando ou perdendo partidas e ajustando sua estratégia.

Cada tipo de aprendizagem serve a propósitos distintos e pode ser usado isoladamente ou em conjunto para resolver problemas complexos.

## 3. Aprendizagem Supervisionada Formal

- Dados de treino: pares (x_i, y_i) gerados por função desconhecida y = f(x).
- Objetivo: descobrir função h (hipótese) que aproxime f.
- h faz parte do espaço de hipóteses H (conjunto de possíveis modelos).
- Avaliação do modelo considera desempenho em dados não vistos (conjunto de teste), para verificar a capacidade de generalização.

## 4. Conceitos Importantes: Bias e Variância

- **Bias:** tendência do modelo de não capturar padrões reais devido à simplicidade do modelo (underfitting).
- **Variância:** sensibilidade do modelo a variações nos dados de treino, que pode causar overfitting.
- O tradeoff entre bias e variância é fundamental para um bom modelo (bias-variance tradeoff).
- Princípio de Ockham's Razor: escolher o modelo mais simples que explica bem os dados.

## 5. Árvores de Decisão (Decision Trees)

- Estrutura que representa uma função mapeando atributos de entrada para uma saída.
- Cada nó interno faz um teste sobre um atributo; as folhas definem a saída.
- Algoritmo ID3: constrói árvore escolhendo atributos que maximizam o ganho de informação.
- Ganho de informação é baseado na entropia, que mede a incerteza dos dados.
- Entropia baixa = menos incerteza; Entropia alta = mais incerteza.

## 6. Regressão Linear

- Modelo simples que representa uma relação linear entre input x e output y.
- Função linear: y = w1*x + w0, onde w são pesos que se ajustam para minimizar erro (loss function).
- Minimização feita usando soma dos erros quadráticos (squared error).
- Fórmulas fechadas para encontrar os melhores pesos.
- Pode ser estendido para múltiplas variáveis.

## 7. Support Vector Machines (SVM)

- Algoritmo poderoso para classificação e regressão.
- Busca a linha ou hiperplano que separa classes com a maior margem possível.
- Uso de kernels para mapear dados para espaços de maior dimensão e permitir separação não linear.
- Parâmetro C controla rigidez da margem.
- Vantagens: compacto, rápido na predição, bom para dados de alta dimensão.
- Desvantagens: treino pode ser lento em datasets muito grandes, sensível à escolha de C, modelo pouco interpretável.

## 8. Random Forests

- Método ensemble baseado em bagging de várias decision trees treinadas em subconjuntos aleatórios dos dados.
- Melhora a generalização diminuindo overfitting.
- Aplicável para classificação e regressão.
- Vantagens: treino e predição rápidos, classificações probabilísticas, flexível para evitar underfitting.
- Desvantagem: perde interpretabilidade das árvores individuais.

## 9. Principal Component Analysis (PCA)

- Algoritmo não supervisionado para redução de dimensionalidade.
- Identifica os principais eixos de variação nos dados, preservando a maior variância possível.
- Útil para redução de ruído e extração de características.
- Pode ser afetado por outliers; versões robustas (RandomizedPCA, SparsePCA) podem ser usadas.

## 10. Clustering: k-means

- Algoritmo não supervisionado para agrupamento de dados em k clusters.
- Cada cluster é definido pelo centro (média dos pontos).
- Pontos são atribuídos ao cluster mais próximo.
- Limitado por:
  - Necessidade de definir k previamente.
  - Pode convergir para máximos locais (não ótimo global).
  - Assume clusters linearmente separáveis.
- Variantes para grandes datasets e clusters não lineares existem (MiniBatchKMeans, SpectralClustering).

## 11. Métricas de Avaliação para Classificadores

- **Accuracy:** proporção de classificações corretas, mas pode ser enganadora em dados desbalanceados.
- **Matriz de Confusão:** compara valores previstos vs reais, base para outras métricas.
- **Precision:** proporção de verdadeiros positivos entre as predições positivas.
- **Recall:** proporção de verdadeiros positivos entre os casos reais positivos.
- **F1 Score:** média harmônica entre Precision e Recall, balanceando os dois.

## 12. Métricas de Avaliação para Regressão

- **Mean Squared Error (MSE):** média dos quadrados dos erros, penaliza erros grandes.
- **Root Mean Squared Error (RMSE):** raiz quadrada do MSE.
- **Mean Absolute Error (MAE):** média dos valores absolutos dos erros, penaliza igualmente todos os erros.

---

# Conclusão

Este conjunto de slides apresenta uma introdução sólida e prática aos conceitos fundamentais de Machine Learning, abordando os tipos de aprendizagem, modelos clássicos como árvores de decisão, regressão linear, SVM e Random Forests, bem como métodos para redução de dimensionalidade (PCA) e clustering (k-means). O conteúdo também enfatiza a importância do equilíbrio bias-variância e apresenta as principais métricas para avaliação de modelos de classificação e regressão.

---

Se desejar, posso também ajudar a criar exemplos práticos em código Python para estes conceitos.
