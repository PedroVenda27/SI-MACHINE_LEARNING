# 1. INTRODUÇÃO

Nos últimos anos, o **Machine Learning** (Aprendizagem Automática) tem se destacado como uma das áreas mais promissoras da inteligência artificial, impulsionando avanços em diversas áreas como saúde, finanças, reconhecimento de padrões e automação. 

Machine Learning consiste numa área que permite que sistemas computacionais aprendam e se adaptem automaticamente, extraindo conhecimento a partir de grandes volumes de dados. Diferentemente dos métodos tradicionais, onde todas as regras e exceções são programadas manualmente, os algoritmos de aprendizagem automática conseguem identificar padrões subjacentes, fazer previsões e tomar decisões sem serem explicitamente instruídos para cada cenário específico.

Neste relatório, exploraremos os conceitos fundamentais de Machine Learning por meio do estudo de dois conjuntos de dados clássicos: o dataset **diabetes** e o dataset **iris** . 

Serão abordados tópicos essenciais como:

- Conceitos base de Machine Learning
- Técnicas de pré-processamento e análise exploratória dos dados
- Algoritmos de classificação supervisionada
- Avaliação e validação de modelos

Utilizando os datasets diabetes e iris, será possível entender a aplicação prática dos algoritmos, desde a preparação dos dados até a interpretação dos resultados, evidenciando a importância da escolha correta do modelo e dos parâmetros para alcançar alta precisão preditiva.

Além disso, discutiremos os desafios comuns em Machine Learning, como o balanceamento entre bias e variância, overfitting e underfitting, e estratégias para otimizar o desempenho dos modelos.

Este relatório visa proporcionar uma visão geral clara e aplicada de Machine Learning, destacando sua relevância e potencial em resolver problemas complexos baseados em dados reais.

## 1.1 Conceitos Base

### 1.1.1 O que é o conceito de Machine Learning?

Machine Learning (Aprendizagem Automática) consiste numa área da inteligência artificial que permite que os agentes neste caso "computadores" aprendam a realizar tarefas sem serem explicitamente programados para isso. 
Em vez de seguir instruções rígidas definidas por um programador, um sistema de machine learning analisa dados, identifica padrões e constrói modelos que conseguem fazer previsões ou tomar decisões com base em novas informações.

Um agente inteligente pode ser considerado em aprendizagem quando melhora seu comportamento ao observar e interpretar o mundo ao seu redor. No caso dos computadores, essa aprendizagem ocorre através da análise de grandes volumes de dados, a partir dos quais o sistema cria uma “hipótese” sobre como o mundo funciona.

### 1.1.2 Por que deixar a máquina aprender em vez de programá-la diretamente?

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


### 1.1.4 Aprendizagem Supervisionada - Detalhada

Neste trabalho, vamos incidir apenas em algoritmos e modelos baseados em aprendizagem supervisionada (Decision Trees, Regressão Linear, SVM, Random Forests). Esta tarefa pode ser definida da seguinte forma:

Dado um conjunto de treino com **N** pares input-output  
**(X<sub>1</sub>, Y<sub>1</sub>), (X<sub>2</sub>, Y<sub>2</sub>), ..., (X<sub>N</sub>, Y<sub>N</sub>)**  
onde cada par foi gerado por uma função desconhecida **y = f(x)**,  
o objetivo é descobrir **uma função h** que aproxime a verdadeira função **f**.

Essa **função h** é chamada **hipótese** e pertence a um espaço de hipóteses **H**, que é o conjunto de todas as possíveis funções candidatas.

Também designamos **h** como o **modelo** dos dados, obtido a partir da classe de modelos **H**.

O output **Y<sub>i</sub>** é chamado de **ground truth**, pois representa a resposta verdadeira que queremos que o modelo aprenda a prever.


---


# 2 PROJETO DE ESTUDO / PROBLEMA

## 2.1 Objetivo trabalho em que consiste?

Passando agora objetivo principal deste projeto este consite na aplicaçao de técnicas de Machine Learning para analisar os datasets iris e diabetes, treinando modelos capazes de identificar padrões relevantes e avaliando a performance desses modelos.

Para a realização deste trabalho, serão seguidas três fases principais:

- 1. Preparação e Compreensão dos Dados  
Nesta fase, será feita a descrição dos datasets utilizados — **diabetes** e **iris** — explicando a sua composição e importância para o estudo.

- 2. Compreensão e Aplicação dos Algoritmos  
Serão apresentados os algoritmos de aprendizagem supervisionada escolhidos, nomeadamente as **Decision Trees** e as **SVM**, incluindo a explicação do seu funcionamento e a aplicação prática aos datasets.

- 3. Avaliação e Interpretação dos Resultados  
Por fim, serão analisados o desempenho dos modelos, considerando o tempo de processamento, utilização de memória e qualidade dos resultados, seguidos das conclusões do estudo.

## 2.2 Descriçao dos Datasets (DIABETES E IRIS)

### 2.2.1 Dataset Diabetes

O **Dataset Diabetes** é um conjunto de dados amplamente utilizado para tarefas de aprendizagem supervisionada, especialmente para problemas de **regressão**, como a previsão de valores contínuos relacionados à diabetes em pacientes. Este dataset contém informações clínicas e biométricas de pacientes que podem ser usadas para prever variáveis numéricas associadas ao estado de saúde.

#### 2.2.1.1 Características principais:

- **Número de instâncias:** 768  
- **Número de atributos:** 8 variáveis de entrada (features) + 1 variável alvo (target)  
- **Atributos de entrada:** incluem dados clínicos como:  
  - Número de gestações  
  - Índice de massa corporal (IMC)  
  - Pressão arterial diastólica  
  - Nível de insulina no sangue  
  - Idade do paciente  
  - Entre outros indicadores fisiológicos relevantes  
- **Variável alvo:** Valor numérico contínuo relacionado ao diagnóstico ou progresso da diabetes (exemplo: medida de glicose)  
- **Origem:** Base de dados do National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)

#### 2.2.1.2 Aplicação:

O dataset é utilizado para treinar modelos que realizem previsões numéricas (regressão) relacionadas à diabetes com base nas características do paciente, sendo uma referência clássica para avaliação de algoritmos de regressão em Machine Learning.

---

### 2.2.2 Dataset Iris

O **Dataset Iris** é um dos datasets mais famosos para tarefas de aprendizagem supervisionada de **classificação multiclasse**, utilizado para identificar espécies de flores da planta **Iris** com base em medidas morfológicas.

#### 2.2.2.1 Características principais:

- **Número de instâncias:** 150  
- **Número de atributos:** 4 variáveis de entrada + 1 variável alvo  
- **Atributos de entrada:**  
  - Comprimento da sépala (em cm)  
  - Largura da sépala (em cm)  
  - Comprimento da pétala (em cm)  
  - Largura da pétala (em cm)  
- **Variável alvo:** Espécie da flor, com três classes possíveis:  
  - *Iris setosa*  
  - *Iris versicolor*  
  - *Iris virginica*  
- **Origem:** Coletado por Ronald Fisher em 1936 para estudos estatísticos

#### 2.2.2.2 Aplicação:

Este dataset é utilizado para demonstrar e testar algoritmos de classificação, destacando-se pela simplicidade e clareza dos dados, sendo ideal para exemplos educacionais e provas de conceito em aprendizagem supervisionada.

## 2.3 Decision Trees (ID3 Algorithm)

O algoritmo **ID3** (Iterative Dichotomiser 3) é uma técnica popular de aprendizagem supervisionada utilizada para construir árvores de decisão a partir de conjuntos de dados. Este algoritmo é amplamente aplicado em problemas de classificação, onde o objetivo é predizer a classe ou categoria de um dado exemplo com base em atributos observados.

Este algoritmo baseia-se no conceitos de Entropia e Ganho de informação para decidir qual atributo deve ser usado para dividir os dados em cada passo, criando uma árvore que tenta representar da melhor forma possível as regras de decisão presentes no conjunto de dados.

### 2.3.1 Funcionamento do Algoritmo

O funcionamento do algoritmo ID3 assenta na construção recursiva de uma árvore de decisão, selecionando em cada passo o atributo que melhor separa os dados segundo uma métrica estatística chamada **ganho de informação**. O processo é o seguinte:

### 2.3.2 Cálculo da Entropia

O conceito de Entropia consiste numa medida de incerteza relativamente a uma variável aleatória: quanto mais informação, menos entropia.

Por Exemplo: 

Se uma variavel aleatoria tem apenas um valor possível, a entropia é 0.
Uma moeda equilibrada com 2 faces tem 1 de entropia (2 resultados igaulmente Possiveis)

Por outro lado se uma moeda está viciada e 99% das vezes que a atiramos, recebemos
”cara”, então a entropia será muito menor que 1, uma vez que a incerteza é também ela muito menor.

Podemos Defenir Entropia Como:

**H(V) = - &sum;<sub>k</sub> P(v<sub>k</sub>) log<sub>2</sub> P(v<sub>k</sub>)**

Exemplo 1: <p><i>H</i>(equilibrada) = - (0.5 log<sub>2</sub> 0.5 + 0.5 log<sub>2</sub> 0.5) = 1</p>
Exemplo 2: <p><i>H</i>(viciada) = - (0.99 log<sub>2</sub> 0.99 + 0.01 log<sub>2</sub> 0.01) = 0.08</p>

Para cada conjunto de exemplos, calcula-se a entropia, que mede a impureza ou incerteza dos dados em relação às classes.

### 2.3.3 Cálculo do Ganho de Informação

Para cada atributo disponível, calcula-se o ganho de informação, que representa a redução da entropia ao segmentar os dados por esse atributo. O atributo com maior ganho de informação é escolhido para fazer a divisão naquele nó da árvore.

A ideia é escolher um atributo **A** de tal forma que a entropia do conjunto de dados desça. Medimos esta redução calculando a entropia que resta depois de efectuado o teste ao atributo.

Um atributo **A** com *d* valores diferentes divide o conjunto de treino **E** em subconjuntos **(E<sub>1</sub>,...,E<sub>d</sub>).** Cada subconjunto **(E<sub>k</sub>)**  tem **(P<sub>k</sub>)**  exemplos positivos e **(N<sub>k</sub>)**  exemplos negativos, pelo que precisaremos de B**(<sup>p<sub>k</sub></sup>/<sub>p<sub>k</sub> + n<sub>k</sub></sub>)** bits de informação para responder à questão. 

Um exemplo escolhido aleatoriamente tem probabilidade  
**(p<sub>k</sub> + n<sub>k</sub>)/(p + n)** de pertencer a **(E<sub>k</sub>)**, pelo que a restante entropia depois de escolhido o atributo pode ser calculada da seguinte forma:

**Resto(A) = ∑<sub>k=1</sub><sup>d</sup> ((p<sub>k</sub> + n<sub>k</sub>) / (p + n)) × B(p<sub>k</sub> / (p<sub>k</sub> + n<sub>k</sub>))**

O ganho de informação é então calculado da seguinte forma para um atributo

**Ganho(A) = B(p / (p + n)) - Resto(A)**

### 2.3.4 Divisão dos Dados Criaçao da Arvore

Passamos entao a divisão dos dados estes são divididos em subconjuntos com base nos valores do atributo selecionado.

Neste caso, podemos ver que o atributo mais importante dos dois é o `Patrons`.

Com `Type`, mantemos exactamente a mesma distribui¸ção após a separaçãao
dos exemplos pelos valores de atributos.
Com `Patrons`, conseguimos logo dar uma resposta relativamente a bastantes exemplos, ficando apenas por resolver o caso em que o valor de Patrons é `Full`

![image](https://github.com/user-attachments/assets/3a619240-0ee4-44c6-ab4a-c229ea2247fd)

Após escolhermos o atributo mais importante, existem quatro casos possíveis:

- Se todos os exemplos que restam são todos positivos ou negativos, então já podemos dar uma resposta: Sim ou Não. Por exemplo, *Patrons = None*.
- Se existem alguns exemplos positivos ou negativos então voltamos a escolher o melhor atributo para os separar. Por exemplo em *Patrons = Full* é escolhido o atributo *Hungry*.
- Se já não existem exemplos, então quer dizer que ainda não foi visto nenhum caso com aquela combinação de atributos. Assim sendo, retornamos o output mais comum do conjunto de exemplos que foi usado na construção do nó pai.
- Se já não existem atributos para usar, mas ainda temos exemplos positivos e negativos, então quer dizer que estes exemplos têm a mesma descrição, mas diferentes classificações. Neste caso, retornamos o valor de output mais comum neste conjunto de exemplos.

Desta forma, o ID3 cria uma árvore onde cada nó interno corresponde a um teste num atributo, cada ramo corresponde a um resultado possível desse teste, e cada folha representa uma classe final.

O objetivo final é gerar uma árvore de decisão que generalize bem os dados, permitindo classificar novos exemplos de forma eficiente e precisa.


### 2.3.5 Código





### 2.3.6 Exemplo Prático (VIDEO)

De forma A entender tudo isto de uma Maneira mais vizual deixo aqui um video de toda a explicação do Funcionamento do Algoritmo ID3
[VIDEO](https://www.youtube.com/watch?v=aLsReomQ7AA)


## 2.4 Algoritmo de Regressão Linear 

O algoritmo **Regressão Linear** é uma técnica estatística e de aprendizagem supervisionada usada para modelar a relação entre uma variável dependente contínua e uma ou mais variáveis independentes (features). O objetivo é encontrar uma função linear que melhor ajuste os dados e permita prever valores futuros. Este é um dos algoritmos mais simples e amplamente utilizados para problemas de regressão.

### 2.4.1 Funcionamento do Algoritmo

O funcionamento da regressão linear baseia-se na modelação da relação entre uma variável dependente e uma ou mais variáveis independentes, ajustando uma função linear que minimiza a soma dos erros quadráticos entre os valores previstos e os valores observados, de forma a capturar a tendência dos dados para realizar previsões.O processo é o seguinte

### 2.4.2 Modelo Linear

Uma função linear univariada com input `x` e output `y` tem a forma `y = w<sub>1</sub>x + w<sub>0</sub>` onde `w<sub>0</sub>` e `w<sub>1</sub>` sâo coeficientes que temos de determinar.
Estes coeficientes funcionam como pesos: o valor de `y` varia consoante opeso relativo de um termo ou outro.
Vamos assumir que `w`e o vector `⟨w<sub>0</sub>,w<sub>1</sub>⟩` e a função linear com esses pesos é:

`h<sub>w</sub>(x) = w<sub>1</sub> x + w<sub>0</sub>`

O objectivo passa por encontrar a função `h<sub>w</sub>(X)` que melhor se ajusta aos dados. A esta tarefa chamamos regressao linear

### 2.4.3 Ajuste dos Coeficientes

O algoritmo busca encontrar os valores para os pesos `⟨w<sub>0</sub>,w<sub>1</sub>⟩` que minimizem um loss function 

Uma loss function clássica em casos de regressão linear é a Squared-Error:

`Loss(h<sub>w</sub>) = ∑<sub>j=1</sub><sup>N</sup> (y<sub>j</sub> − (w<sub>1</sub>x<sub>j</sub> + w<sub>0</sub>))²`

O objectivo é minimizar a funçãoo Loss⟨h<sub>w</sub>⟩. A função é minima quando assuas derivadas parciais são zero:

`∂/∂w<sub>0</sub> ∑<sub>j=1</sub><sup>N</sup> (y<sub>j</sub> − (w<sub>1</sub>x<sub>j</sub> + w<sub>0</sub>))² = 0`

`∂/∂w<sub>1</sub> ∑<sub>j=1</sub><sup>N</sup> (y<sub>j</sub> − (w<sub>1</sub>x<sub>j</sub> + w<sub>0</sub>))² = 0`

Estas equações tem uma solução única:

`w<sub>1</sub> = [N ∑ x<sub>j</sub> y<sub>j</sub> − (∑ x<sub>j</sub>) (∑ y<sub>j</sub>)] / [N ∑ x<sub>j</sub><sup>2</sup> − (∑ x<sub>j</sub>)<sup>2</sup>]`

`w<sub>0</sub> = (∑ y<sub>j</sub> − w<sub>1</sub> (∑ x<sub>j</sub>)) / N`

Desta forma, a regressão linear cria um modelo onde cada coeficiente representa o peso atribuído a uma variável explicativa, e a combinação linear desses pesos com os valores dos atributos gera a previsão do valor da variável dependente.

O objetivo final é encontrar os coeficientes que melhor ajustem os dados observados, permitindo prever novos exemplos de forma eficiente e precisa, minimizando o erro entre as predições e os valores reais. 
Aqui temos um exemplo:
![image](https://github.com/user-attachments/assets/f705e151-429f-4d09-9fe5-d43da34e54d1)

### 2.4.4 Código

### 2.4.5 Exemplo Prático (VIDEO)

De forma a entender tudo isto de uma maneira mais visual, deixo aqui um vídeo que explica detalhadamente o funcionamento do algoritmo de Regressão Linear:
[VIDEO](https://www.youtube.com/watch?v=CtsRRUddV2s)







## 2.3 Algoritmo SVM (Support Vector Machine) (EXTRA)

O algoritmo **Support Vector Machine (SVM)** é uma técnica poderosa de aprendizagem supervisionada utilizada principalmente para problemas de classificação e regressão. O seu principal objetivo é encontrar o hiperplano que melhor separa as classes num espaço de características, maximizando a margem entre os dados de diferentes classes.

O SVM é especialmente eficaz em espaços de alta dimensão e pode ser adaptado para problemas não lineares através do uso de funções kernel, que transformam os dados para um espaço onde a separação linear seja possível.

### 2.3.1 Funcionamento do Algoritmo

O funcionamento do SVM baseia-se nos seguintes conceitos:

- **Hiperplano Ótimo:** O SVM procura um hiperplano que divide os dados em classes diferentes, maximizando a distância (margem) entre o hiperplano e os pontos de dados mais próximos de cada classe, conhecidos como vetores de suporte.

- **Vetores de Suporte:** São os exemplos do conjunto de treino que ficam mais próximos do hiperplano e que influenciam diretamente a posição e orientação do mesmo.

- **Margem Máxima:** O objetivo é encontrar o hiperplano que maximiza a margem entre as classes, pois isso tende a melhorar a generalização do modelo.

- **Kernels:** Quando os dados não são linearmente separáveis no espaço original, o SVM utiliza funções kernel (como o kernel linear, polinomial, radial basis function - RBF) para mapear os dados para um espaço dimensional superior onde a separação linear é possível.

- **Restrições e Otimização:** O problema de encontrar o hiperplano ótimo é formulado como um problema de otimização convexa, que pode ser resolvido eficientemente por métodos matemáticos como o método dos multiplicadores de Lagrange.

Desta forma, o SVM constrói um modelo robusto para classificar novos dados, mesmo quando as classes não são linearmente separáveis no espaço original, apresentando bom desempenho em muitos cenários práticos.



























## 4. Conceitos Importantes: Bias e Variância

- **Bias:** tendência do modelo de não capturar padrões reais devido à simplicidade do modelo (underfitting).
- **Variância:** sensibilidade do modelo a variações nos dados de treino, que pode causar overfitting.
- O tradeoff entre bias e variância é fundamental para um bom modelo (bias-variance tradeoff).
- Princípio de Ockham's Razor: escolher o modelo mais simples que explica bem os dados.




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
