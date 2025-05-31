# 1. INTRODUÇÃO

Nos últimos anos, o **Machine Learning** (Aprendizagem Automática) tem se destacado como uma das áreas mais promissoras da inteligência artificial, impulsionando avanços em diversas áreas como saúde, finanças, reconhecimento de padrões e automação. 

Machine Learning consiste numa área que permite que sistemas computacionais aprendam e se adaptem automaticamente, extraindo conhecimento a partir de grandes volumes de dados. Diferentemente dos métodos tradicionais, onde todas as regras e exceções são programadas manualmente, os algoritmos de aprendizagem automática conseguem identificar padrões subjacentes, fazer previsões e tomar decisões sem serem explicitamente instruídos para cada cenário específico.

Neste relatório, exploraremos os conceitos fundamentais de Machine Learning por meio do estudo de dois conjuntos de dados clássicos: o dataset  `diabetes` e o dataset `iris` . 

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

**Complexidade e variabilidade do problema:** Uma vez que em muitos casos, o problema apresenta tanta variabilidade e casos possíveis que seria impossível antecipar todas as situações com regras fixas.

**Falta de conhecimento explícito:** Pois Muitas vezes, não sabemos exatamente como resolver um problema, mas a máquina pode encontrar padrões e soluções que ainda não conseguimos identificar.

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

Neste trabalho, vamos incidir apenas em algoritmos e modelos baseados em aprendizagem supervisionada (Decision Trees, Regressão Linear, SVM,). Esta tarefa pode ser definida da seguinte forma:

Dado um conjunto de treino com `N` pares input-output  **(X<sub>1</sub>, Y<sub>1</sub>), (X<sub>2</sub>, Y<sub>2</sub>), ..., (X<sub>N</sub>, Y<sub>N</sub>)**  onde cada par foi gerado por uma função desconhecida `y = f(x)`,  o objetivo é descobrir `uma função h` que aproxime a verdadeira `função f `.

Essa `função h` é chamada `hipótese` e pertence a um espaço de hipóteses `H`, que é o conjunto de todas as possíveis funções candidatas.

Também designamos `h` como o `modelo` dos dados, obtido a partir da classe de modelos `H`.

O output **Y<sub>i</sub>** é chamado de `ground truth`, pois representa a resposta verdadeira que queremos que o modelo aprenda a prever.

### 1.1.5 Conceitos Importantes: Bias e Variância

tendo em conta isto e antes de avançar-mos para o Projeto em si é ainda necessario perceber alguns conceitos chaves como:

**Bias** é a tendência de um modelo se desviar de um valor esperado quando exposto a diferentes conjuntos de treino.  O bias normalmente resulta de restrições impostas pelo espaço de possíveis modelos.  

Por exemplo, o espaço de hipóteses das funções lineares impõe um bias aos modelos na medida em que o próprio modelo apenas pode ser constituído por uma linha reta. Caso existam padrões nos dados não capturados por uma linha reta, o modelo não será capaz de os representar e aprender.  

Chamamos de **underfitting** à incapacidade do modelo de encontrar os devidos padrões nos dados.

**Variância** consiste na quantidade de mudanças nos dados devido a flutuações nos dados de treino.  

Chamamos de **overfitting** à excessiva capacidade de adaptação do modelo a um conjunto de dados em particular no qual foi treinado.




---

# 2 PROJETO DE ESTUDO / PROBLEMA

## 2.1 Objetivo trabalho em que consiste?

Passando agora objetivo principal deste projeto este consite na aplicaçao de técnicas de Machine Learning para analisar os datasets iris e diabetes, treinando modelos capazes de identificar padrões relevantes e avaliando a performance desses modelos.

Para a realização deste trabalho, serão seguidas três fases principais:

**Preparação e Compreensão dos Dados**  
Nesta fase, será feita a descrição dos datasets utilizados `diabetes` e `iris` — explicando a sua composição e importância para o estudo.

**Compreensão e Aplicação dos Algoritmos**  
Serão apresentados os algoritmos de aprendizagem supervisionada escolhidos, nomeadamente as **Decision Trees** e as **SVM**, incluindo a explicação do seu funcionamento e a aplicação prática aos datasets.

**Avaliação e Interpretação dos Resultados**  
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

---

## 2.3 ID3 Algorithm - Decision Trees - (Classificação)

O algoritmo **ID3** (Iterative Dichotomiser 3) é uma técnica popular de aprendizagem supervisionada utilizada para construir árvores de decisão a partir de conjuntos de dados. Este algoritmo é amplamente aplicado em problemas de classificação, onde o objetivo é predizer a classe ou categoria de um dado exemplo com base em atributos observados.

Este algoritmo baseia-se no conceitos de Entropia e Ganho de informação para decidir qual atributo deve ser usado para dividir os dados em cada passo, criando uma árvore que tenta representar da melhor forma possível as regras de decisão presentes no conjunto de dados.

### 2.3.1 Funcionamento do Algoritmo

O funcionamento do algoritmo ID3 assenta na construção recursiva de uma árvore de decisão, selecionando em cada passo o atributo que melhor separa os dados segundo uma métrica estatística chamada **ganho de informação**. O processo é o seguinte:

### 2.3.2 Cálculo da Entropia

O conceito de Entropia consiste numa medida de incerteza relativamente a uma variável aleatória: quanto mais informação, menos entropia.

Por Exemplo: 

Uma moeda equilibrada com 2 faces tem 1 de entropia (2 resultados igaulmente Possiveis)

Por outro lado se uma moeda está viciada e 99% das vezes que a atiramos, recebemos
”cara”, então a entropia será muito menor que 1, uma vez que a incerteza é também ela muito menor.

Podemos Defenir Entropia Como:

**H(V) = - &sum;<sub>k</sub> P(v<sub>k</sub>) log<sub>2</sub> P(v<sub>k</sub>)**

**Exemplo 1:** <p><i>H</i>(equilibrada) = - (0.5 log<sub>2</sub> 0.5 + 0.5 log<sub>2</sub> 0.5) = 1</p>

**Exemplo 2:** <p><i>H</i>(viciada) = - (0.99 log<sub>2</sub> 0.99 + 0.01 log<sub>2</sub> 0.01) = 0.08</p>

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

---

## 2.4 Linear Regression Alghorithm - (Regressão)

O algoritmo **Regressão Linear** é uma técnica estatística e de aprendizagem supervisionada usada para modelar a relação entre uma variável dependente contínua e uma ou mais variáveis independentes (features). O objetivo é encontrar uma função linear que melhor ajuste os dados e permita prever valores futuros. Este é um dos algoritmos mais simples e amplamente utilizados para problemas de regressão.

### 2.4.1 Funcionamento do Algoritmo

O funcionamento da regressão linear baseia-se na modelação da relação entre uma variável dependente e uma ou mais variáveis independentes, ajustando uma função linear que minimiza a soma dos erros quadráticos entre os valores previstos e os valores observados, de forma a capturar a tendência dos dados para realizar previsões.O processo é o seguinte

### 2.4.2 Modelo Linear

Uma função linear univariada com input `x` e output `y` tem a forma **y = w<sub>1</sub>x + w<sub>0</sub>** onde **w<sub>0</sub>** e **w<sub>1</sub>** sâo coeficientes que temos de determinar.
Estes coeficientes funcionam como pesos: o valor de `y` varia consoante opeso relativo de um termo ou outro.
Vamos assumir que `w`e o vector **⟨w<sub>0</sub>,w<sub>1</sub>⟩** e a função linear com esses pesos é:

**h<sub>w</sub>(x) = w<sub>1</sub> x + w<sub>0</sub**

O objectivo passa por encontrar a função **h<sub>w</sub>(X)** que melhor se ajusta aos dados. A esta tarefa chamamos regressao linear

### 2.4.3 Ajuste dos Coeficientes

O algoritmo busca encontrar os valores para os pesos **⟨w<sub>0</sub>,w<sub>1</sub>⟩** que minimizem um loss function 

Uma loss function clássica em casos de regressão linear é a Squared-Error:

**Loss(h<sub>w</sub>) = ∑<sub>j=1</sub><sup>N</sup> (y<sub>j</sub> − (w<sub>1</sub>x<sub>j</sub> + w<sub>0</sub>))²**

O objectivo é minimizar a funçãoo Loss⟨h<sub>w</sub>⟩. A função é minima quando assuas derivadas parciais são zero:

**∂/∂w<sub>0</sub> ∑<sub>j=1</sub><sup>N</sup> (y<sub>j</sub> − (w<sub>1</sub>x<sub>j</sub> + w<sub>0</sub>))² = 0**

**∂/∂w<sub>1</sub> ∑<sub>j=1</sub><sup>N</sup> (y<sub>j</sub> − (w<sub>1</sub>x<sub>j</sub> + w<sub>0</sub>))² = 0**

Estas equações tem uma solução única:

**w<sub>1</sub> = [N ∑ x<sub>j</sub> y<sub>j</sub> − (∑ x<sub>j</sub>) (∑ y<sub>j</sub>)] / [N ∑ x<sub>j</sub><sup>2</sup> − (∑ x<sub>j</sub>)<sup>2</sup>]**

**w<sub>0</sub> = (∑ y<sub>j</sub> − w<sub>1</sub> (∑ x<sub>j</sub>)) / N**

Desta forma, a regressão linear cria um modelo onde cada coeficiente representa o peso atribuído a uma variável explicativa, e a combinação linear desses pesos com os valores dos atributos gera a previsão do valor da variável dependente.

O objetivo final é encontrar os coeficientes que melhor ajustem os dados observados, permitindo prever novos exemplos de forma eficiente e precisa, minimizando o erro entre as predições e os valores reais. 
Aqui temos um exemplo:
![image](https://github.com/user-attachments/assets/f705e151-429f-4d09-9fe5-d43da34e54d1)

### 2.4.4 Código

### 2.4.5 Exemplo Prático (VIDEO)

De forma a entender tudo isto de uma maneira mais visual, deixo aqui um vídeo que explica detalhadamente o funcionamento do algoritmo de Regressão Linear:
[VIDEO](https://www.youtube.com/watch?v=CtsRRUddV2s)

---

## 2.5 Support Vector Machines SVM Algorithm - (Classificação ou Regressão)

O algoritmo **Support Vector Machines (SVM)** é uma técnica poderosa de aprendizagem supervisionada, utilizada tanto para problemas de classificação como de regressão. O objetivo principal da SVM é encontrar uma fronteira (hiperplano) que melhor separe os dados em diferentes classes, maximizando a margem entre as classes, ou que modele a relação entre variáveis para regressão.

### 2.5.1 Funcionamento do Algoritmo

O funcionamento das SVMs baseia-se na construção de um hiperplano ótimo que divide o espaço dos dados de forma a maximizar a margem entre as diferentes classes. A margem corresponde à distância entre o hiperplano e os pontos mais próximos de cada classe, chamados de **vetores de suporte**. 

No caso de dados linearmente separáveis, a SVM encontra uma linha (em 2D) ou um hiperplano (em dimensões superiores) que separa as classes com a maior margem possível. Para dados não linearmente separáveis, a SVM pode usar **kernels** para projetar os dados num espaço dimensional superior onde a separação linear seja possível.

### 2.5.2 Modelo SVM

O modelo SVM pode ser formalizado como:

<Encontrar `w` e `b` tais que

**<p>y<sub>i</sub> ( <b>w</b> &middot; x<sub>i</sub> + b ) &ge; 1 &nbsp;&nbsp; para todo i = 1, ..., N</p>**
onde:

<ul>
  <li><b>w</b> é o vetor normal ao hiperplano,</li>
  <li><b>b</b> é o termo de bias (intercepto),</li>
  <li><code>y<sub>i</sub> ∈ {-1, +1}</code> é a classe do ponto <b>x<sub>i</sub></b>.</li>
</ul>

O objetivo é minimizar |<b>w</b>|<sup>2</sup> sujeito a estas restrições, o que equivale a maximizar a margem entre as classes.

### 2.5.3 Margem e Vetores de Suporte

- **Margem**: Distância mínima entre o hiperplano e os dados mais próximos (os vetores de suporte).
- **Vetores de Suporte**: Pontos do conjunto de treino que estão mais próximos do hiperplano e que determinam sua posição.

![image](https://github.com/user-attachments/assets/f95f8a5b-38af-4972-bf52-b65fa495e052)


Maximizar a margem ajuda a garantir melhor generalização do modelo para novos dados.

### 2.5.4 Kernel Trick

Quando os dados não são linearmente separáveis no espaço original, a SVM pode usar funções kernel para transformar os dados para um espaço de maior dimensão, onde se torna possível encontrar um hiperplano linear. Exemplos de kernels comuns:

- Linear
- Polinomial
- RBF (Radial Basis Function)
- Sigmoidal

O **kernel trick** permite calcular produtos escalares nesse espaço elevado sem computar explicitamente a transformação.

### 2.5.5 Código

### 2.5.6 Exemplo Prático (VIDEO)

De forma A entender tudo isto de uma Maneira mais vizual deixo aqui um video de toda a explicação do Funcionamento do Algoritmo SVM
[VIDEO](https://www.youtube.com/watch?v=_YPScrckx28)

# 3. METRICAS DE AVALIAÇÃO

## 3.1 Métricas de Avaliação para Classificadores

Os classificadores atribuem labels a cada uma das observações que lhes forem fornecidas. No entanto, precisamos de arranjar técnicas para medir quão bem estas atribuições estão a ser feitas

## 3.1.1 Accuracy

Uma das medidas mais fáceis de entender é a **Accuracy**. Para efeitos de simplicidade vamos assumir que temos duas classes: Verdade e Falso.

<p>
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
</p>

Onde:
**TP** significa *True Positives* (valores que o modelo diz que são verdade e são de facto),  
**TN** significa *True Negatives* (valores que o modelo descreve como falso e são de facto),  
**FP** significa *False Positives* (valores que o modelo diz serem verdade, mas são falsos)  
**FN** significa *False Negatives* (valores previstos como falsos, mas que são verdadeiros).

Um valor de 99% de **Accuracy** significa que o nosso modelo acertou em 99% dos casos que lhe foram dados a classificar.

No entanto, a Accuracy pode ser bastante enganadora, dependendo das características do conjunto de dados.


## 3.1.2 Matriz de Confusão

Outra forma muito comum de avaliarmos um classificador é através da análise de uma **matriz de confusão**. A matriz de confusão combina os valores previstos e os valores verdadeiros das observações numa tabela.

![image](https://github.com/user-attachments/assets/cc76a0eb-5b56-41ad-93d2-03017fa34cfa)


Através da análise desta tabela, é possível extrair várias métricas como Precision, Recall, Accuracy e AUC-ROC, entre outras.

```python
from sklearn.metrics import confusion_matrix

trues = [2, 0, 2, 2, 0, 1]
preds = [0, 0, 2, 2, 0, 2]

confusion_matrix(trues, preds)
```

## 3.1.3 Precision

A Precision Calcula quantos dos casos em que o modelo acertou são positivos sendo útil quando é preferível existirem Falsos Negativos a existirem Falsos Positivos.

<p>
  Precision = TP / (TP + FP)
</p>

```python
from sklearn.metrics import precision_score

trues = [2, 0, 2, 2, 0, 1]
preds = [0, 0, 2, 2, 0, 2]

precision_score(trues, preds)
```

## 3.1.2 Recall

Por sua vez o Recall Calcula quantos positivos fomos capaz de prever correctamente sendo útil quando é preferível existirem Falsos Positivos a existirem Falsos Negativos.

<p>Recall = TP / (TP + FN)</p>

```python
from sklearn.metrics import recall_score

trues = [2, 0, 2, 2, 0, 1]
preds = [0, 0, 2, 2, 0, 2]

recall_score(trues, preds)

```

## 3.1.2 F1 Score

Por Fim F1 Score traduz a Média harmónica de Precision e Recall.  Máxima quando ambas as métricas são iguais esta é Bastante útil quando FP e FN são igualmente maus.

<p>
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
</p>

```python
from sklearn.metrics import f1_score

trues = [2, 0, 2, 2, 0, 1]
preds = [0, 0, 2, 2, 0, 2]

f1_score(trues, preds)
```

## 3.2 Métricas de Avaliação para Regressores

Tal como acontece na classificação, é útil saber quão boas estão a ser as previsões dos nossos regressores.  
Para tal, existem algumas métricas básicas e intuitivas que nos podem ajudar a perceber melhor quão boa é a performance do nosso modelo.

### 3.2.1 Mean Squared Error (MSE)

Uma das mais básicas de compreender é o **Mean Squared Error (MSE)**, esta é também muitas vezes usada como **loss function** em alguns algoritmos de ML e representa a média do quadrado dos erros do nosso regressor:

<p>
  MSE = (1 / N) × Σ<sub>i=1</sub><sup>n</sup> (Y<sub>i</sub> − TrueY<sub>i</sub>)²
</p>

O facto de considerarmos o quadrado do erro, irá inflacionar erros muito grosseiros.

```python
from sklearn.metrics import mean_squared_error

trues = [1, 1, 0, 0, 1, 0]
preds = [0.95, 0.85, 0.9, 0.8, 0.7, 0.3]

error = mean_squared_error(trues, preds)
- **Root Mean Squared Error (RMSE):** raiz quadrada do MSE.
- **Mean Absolute Error (MAE):** média dos valores absolutos dos erros, penaliza igualmente todos os erros.

```
#### 3.2.1 Root Mean Squared Error (RMSE)

Uma extensão desta métrica é a **RMSE (Root Mean Squared Error)**, esta corresponde à raiz quadrada do MSE.  

Podemos calcular esta função passando um argumento adicional à MSE do Scikit:

```python
error = mean_squared_error(trues, preds, squared=False)
```

### 3.2.2 Mean Absolute Error (MAE)

Outra métrica bastante comum é o **Mean Absolute Error (MAE)**, que ao contrário das suas métricas anteriores não penaliza com magnitudes diferentes erros de ordem diferentes (pois não faz o quadrado dos erros).

<p>
  MAE = (1 / N) × Σ<sub>i=1</sub><sup>n</sup> |Y<sub>i</sub> − TrueY<sub>i</sub>|
</p>

```python
from sklearn.metrics import mean_absolute_error

trues = [1, 1, 0, 0, 1, 0]
preds = [0.95, 0.85, 0.9, 0.8, 0.7, 0.3]

error = mean_absolute_error(trues, preds)
```

---


# 4 RESULTADOS/TESTES ESTUDO DE TEMPOS E MEMORIA

## 4.1 IRIS (CLASSIFICAÇÃO)

### 4.1.1 Estatísticas básicas: Dataset Iris

| Feature            | Count | Mean     | Std Dev  | Min | 25% | 50% (Mediana) | 75% | Max |
|--------------------|-------|----------|----------|-----|-----|----------------|-----|-----|
| Sepal Length (cm)  | 150.0 | 5.843    | 0.828    | 4.3 | 5.1 | 5.80           | 6.4 | 7.9 |
| Sepal Width (cm)   | 150.0 | 3.057    | 0.436    | 2.0 | 2.8 | 3.00           | 3.3 | 4.4 |
| Petal Length (cm)  | 150.0 | 3.758    | 1.765    | 1.0 | 1.6 | 4.35           | 5.1 | 6.9 |
| Petal Width (cm)   | 150.0 | 1.199    | 0.762    | 0.1 | 0.3 | 1.30           | 1.8 | 2.5 |
| Target             | 150.0 | 1.000    | 0.819    | 0.0 | 0.0 | 1.00           | 2.0 | 2.0 |

### Interpretação dos dados

- ``count``: número de amostras por coluna. Todas as colunas têm 150 valores (Iris dataset completo).
- ``mean`` : média 
  - Sepal Length ≈ 5.843
  - Sepal Width ≈ 3.057
  - Petal Length ≈ 3.758
  - Petal Width ≈ 1.199
  - Target = 1.000 → indica distribuição balanceada entre as classes (0, 1, 2).

- ``std``: desvio padrão
  - Sepal Length ≈ 0.828
  - Sepal Width ≈ 0.436
  - Petal Length ≈ 1.765 → maior variação
  - Petal Width ≈ 0.762
  - Target ≈ 0.819

- ``min / 25% / 50% (Mediana) / 75% / max``:
  - Sepal Length varia entre 4.3 e 7.9 cm (mediana: 5.80)
  - Sepal Width varia entre 2.0 e 4.4 cm (mediana: 3.00)
  - Petal Length varia entre 1.0 e 6.9 cm (mediana: 4.35)
  - Petal Width varia entre 0.1 e 2.5 cm (mediana: 1.30)
  - Target varia de 0 a 2 → 0 = *setosa*, 1 = *versicolor*, 2 = *virginica*

---
## 4.2 Modelo ID3 (Criterio Entorpy)

### 4.2.1 Resultados Modelo ID3 (Criterio Entorpy)

- **Tempo de Treino (fit)**: 0.003621 segundos     
- **Tempo de Predição (predict)**: 0.001658 segundos  
- **Acurácia**: 0.8889  
- **Precision (macro média)**: 0.8899  
- **Recall (macro média)**: 0.8889  
- **F1‐Score (macro média)**: 0.8888  

**Matriz de Confusão:**

|               | Previsto: Setosa | Previsto: Versicolor | Previsto: Virginica |
|---------------|------------------|-----------------------|---------------------|
| Real: Setosa  |        15        |          0            |         0           |
| Real: Versicolor |     0         |         13            |         2           |
| Real: Virginica  |     0         |          3            |        12           |

**Relatório de Classificação por Classe:**

| Classe       | Precision | Recall | F1-Score | Suporte |
|--------------|-----------|--------|----------|---------|
| Setosa       | 1.00      | 1.00   | 1.00     | 15      |
| Versicolor   | 0.81      | 0.87   | 0.84     | 15      |
| Virginica    | 0.86      | 0.80   | 0.83     | 15      |
| **Média**    | **0.89**  | **0.89** | **0.89** | **45**  |


### Interpretação dos dados

**Tempos de Treino e Predição**

- ``Treino (fit)``: ~0.003621 segundos  
- ``Predição (predict)``: ~0.001658 segundos  

> O modelo ID3 foi extremamente rápido tanto no treino quanto na predição das 45 amostras de teste (30% de 150).

**Acurácia (Accuracy)**

<p>
Accuracy = <span style="font-style: italic;">nº de acertos</span> / <span style="font-style: italic;">total de amostras de teste</span> = 
(15 + 13 + 12) / 45 = 40 / 45 ≈ 0.8889
</p>

> O modelo classificou corretamente **40 de 45 exemplos**.
> Errando **5 exemplos**.

**Precision, Recall e F1‐Score (Média Macro)**

- ``Precision (macro)`` = 0.8899
- ``Recall (macro)`` = 0.8889
- ``F1‐Score (macro)`` = 0.8888

**Nota**: Estas métricas são calculadas separadamente para cada classe e depois faz a média sem ponderação, ou seja, **cada classe tem o mesmo peso**.

> Precision (VP / (VP + FP)): o modelo acerta em ≈ 88.99% das vezes que prevê uma classe.

> Recall (VP / (VP + FN)): ≈ 88.89% dos exemplos reais de cada classe foram corretamente identificados.

> F1-Score: média harmónica entre precision e recall; equilíbrio geral entre os dois.


**Análise da Matriz de Confusão**

| Verdadeiro → Previsto | Setosa | Versicolor | Virginica |
|------------------------|--------|------------|-----------|
| **Setosa** (0)         | 15     | 0          | 0         |
| **Versicolor** (1)     | 0      | 13         | 2         |
| **Virginica** (2)      | 0      | 3          | 12        |

> Classe Setosa: 100% correta (nenhuma confusão).

> Classe Versicolor: 2 amostras foram confundidas com Virginica → Recall ≈ 86.7%.

> Classe Virginica: 3 amostras foram confundidas com Versicolor → Recall = 80.0%.

> Com base nisso conseguimos observar que as confusões ocorreram entre versicolor e virginica, o que é comum, dada a semelhança nas características das pétalas entre essas classes.

---

## 4.3 Modelo SVM (Kernel RBF)

### 4.3.1 Resultados Modelo SVM (Kernel RBF)

- **Tempo de Treino (fit)**: 0.002740 segundos  
- **Tempo de Predição (predict)**: 0.000679 segundos  
- **Acurácia**: 0.9333  
- **Precision (macro média)**: 0.9345  
- **Recall (macro média)**: 0.9333  
- **F1‐Score (macro média)**: 0.9333  

**Matriz de Confusão:**

|               | Previsto: Setosa | Previsto: Versicolor | Previsto: Virginica |
|---------------|------------------|-----------------------|---------------------|
| Real: Setosa  |        15        |          0            |         0           |
| Real: Versicolor |     0         |         14            |         1           |
| Real: Virginica  |     0         |          2            |        13           |

**Relatório de Classificação por Classe:**

| Classe       | Precision | Recall | F1-Score | Suporte |
|--------------|-----------|--------|----------|---------|
| Setosa       | 1.00      | 1.00   | 1.00     | 15      |
| Versicolor   | 0.88      | 0.93   | 0.90     | 15      |
| Virginica    | 0.93      | 0.87   | 0.90     | 15      |
| **Média**    | **0.93**  | **0.93** | **0.93** | **45**  |


### Interpretação dos dados

**Tempos de Treino e Predição**

- ``Treino (fit)``: ~0.002740 segundos  
- ``Predição (predict)``: ~0.000679 segundos  

> O SVM foi ligeiramente mais rápido a treinar e significativamente mais rápido a predizer comparado ao ID3.  
> O pequeno número de amostras (45) e a eficiência do kernel RBF contribuíram para esses tempos reduzidos.

**Acurácia (Accuracy)**

<p>
Accuracy = <span style="font-style: italic;">nº de acertos</span> / <span style="font-style: italic;">total de amostras de teste</span> =
(15 + 14 + 13) / 45 = 42 / 45 ≈ 0.9333
</p>

> O modelo classificou corretamente **42 de 45 exemplos**, errando apenas **3 exemplos** (melhor que o ID3).

**Precision, Recall e F1‐Score (Média Macro)**

- ``Precision (macro)`` = 0.9345  
- ``Recall (macro)`` = 0.9333  
- ``F1‐Score (macro)`` = 0.9333  

> O SVM apresenta **melhor desempenho geral** em comparação com o ID3.  

> Todas as classes apresentam scores altos e próximos entre si (≈ 0.90–1.00), com ótimo equilíbrio.  

> Isso indica um modelo bem generalizado para esse conjunto de teste.

**Análise da Matriz de Confusão**

| Verdadeiro → Previsto | Setosa | Versicolor | Virginica |
|------------------------|--------|------------|-----------|
| **Setosa** (0)         | 15     | 0          | 0         |
| **Versicolor** (1)     | 0      | 14         | 1         |
| **Virginica** (2)      | 0      | 2          | 13        |

> **Classe Setosa**: continua perfeitamente classificada (100% de precisão e recall).  

> **Versicolor**: 1 erro, confundida como Virginica → Recall ≈ 93.3%.  

> **Virginica**: 2 erros, confundidas como Versicolor → Recall ≈ 86.7%.  

> Comparando com o ID3, o SVM **errou menos** (3 erros contra 5).

---

## 4.4 Conclusão Comparativa Final (ID3 vs. SVM)

- Ambos os modelos se mostraram eficazes na **classificação do dataset Iris**, especialmente na classe **Setosa**, que foi sempre corretamente identificada.
- O **ID3**, apesar de simples e interpretável, com tempos de execução baixos, teve **mais dificuldades em distinguir entre Versicolor e Virginica**, cometendo 5 erros no total.
- O **SVM**, utilizando o **kernel RBF**, foi **mais preciso e confiável**, com apenas 3 erros e **métricas superiores** em todas as dimensões.
- Em termos de **tempo de execução**, **ambos foram rápidos**, mas o SVM foi ligeiramente **mais eficiente na predição**.
- Assim podemos concluir que, para este cenário e particionamento, o **SVM foi o modelo que melhor generalizou e apresentou melhor desempenho**, sendo preferível quando se pretende maior precisão, mesmo que à custa de menor interpretabilidade.

---

## 4.5 DIABETES (REGRESSÃO)

### 4.5.1 Estatísticas básicas: Dataset Diabetes

| Feature | Count | Mean | Std Dev | Min | 25% | 50% (Mediana) | 75% | Max |
|---------|-------|----------------|----------|---------|---------|------------------|---------|---------|
| age     | 442.0 | ≈ 0            | 0.0476   | -0.1072 | -0.0373 | 0.0054           | 0.0381 | 0.1107 |
| sex     | 442.0 | ≈ 0            | 0.0476   | -0.0446 | -0.0446 | -0.0446          | 0.0507 | 0.0507 |
| bmi     | 442.0 | ≈ 0            | 0.0476   | -0.0903 | -0.0342 | -0.0073          | 0.0312 | 0.1706 |
| bp      | 442.0 | ≈ 0            | 0.0476   | -0.1124 | -0.0367 | -0.0057          | 0.0356 | 0.1320 |
| s1–s6   | 442.0 | ≈ 0            | 0.0476   | ≈ -0.13 | ≈ -0.03 | ≈ -0.003         | ≈ 0.03 | ≈ 0.20 |
| target  | 442.0 | 152.13         | 77.09    | 25.0    | 87.0    | 140.5            | 211.5   | 346.0  |

**Nota**: Todas as features estão normalizadas com `mean ≈ 0` e `std = 0.047619`, exceto o `target` (não normalizado), que representa a **progressão da doença**.

### Interpretação dos dados

- `count`: número de amostras por coluna. Todas as colunas têm 442 valores (Diabetes dataset completo).

- `mean`: média  
  - Age ≈ 0  
  - Sex ≈ 0  
  - BMI ≈ 0  
  - BP ≈ 0  
  - S1–S6 ≈ 0  
  - Target = 152.13 → indica média do progresso da doença após 1 ano.

- `std`: desvio padrão  
  - Age ≈ 0.0476  
  - Sex ≈ 0.0476  
  - BMI ≈ 0.0476  
  - BP ≈ 0.0476  
  - S1–S6 ≈ 0.0476  
  - Target ≈ 77.09 → alta variação nas respostas de progressão da doença.

- `min / 25% / 50% (Mediana) / 75% / max`:  
  - Age varia entre -0.1072 e 0.1107 (mediana: 0.0054)  
  - Sex varia entre -0.0446 e 0.0507 (mediana: -0.0446)  
  - BMI varia entre -0.0903 e 0.1706 (mediana: -0.0073)  
  - BP varia entre -0.1124 e 0.1320 (mediana: -0.0057)  
  - S1–S6 variam aproximadamente entre -0.13 e 0.20 (mediana: ≈ -0.003)  
  - Target varia de 25.0 a 346.0 (mediana: 140.5)


## 4.6 Modelo de Regressão Linear

### 4.6.1 Resultados Regressão Linear

- **Tempo de Treino (fit)**: 0.007937 segundos  
- **Tempo de Predição (predict)**: 0.001121 segundos  
- **MSE**: 2821.7510  
- **RMSE**: 53.1202  
- **MAE**: 41.9194  
- **R²**: 0.4773  

### Interpretação dos Resultados

**Tempos de Treino e Predição**

- ``Treino (fit)``: ~0.007937 segundos  
- ``Predição (predict)``: ~0.0001121 segundos  

> A Regressão Linear levou cerca de 8 ms para ajustar o modelo nos 309 exemplos de treino (70% de 442) e 1.1 ms para predizer as 133 amostras de teste.

**MSE (Erro Quadrático Médio)**:  
<p>
  MSE = (1 / N) × Σ<sub>i=1</sub><sup>N</sup> (y<sub>i</sub> − ŷ<sub>i</sub>)² ≈ 2821.75
</p>

onde N=133 (tamanho do teste).

>Fixando que os valores de target variam de 25 a 346, um MSE de aprox: 2821.75 concluimos que, em média, o quadrado do erro está nesse patamar. Como a escala do target é alta ou seja (centenas), esse valor faz certo sentido num contexto de erro absoluto médio.

**RMSE (Raiz do Erro Quadrático Médio)**:  

<p>
  RMSE = √MSE ≈ 53.12
</p>

> Interpretando agora a raiz do erro medio esta fica em torno das 53 unidades da escala do target (por exemplo, se target mede algum índice de progressão de doença, o erro médio fica em ±53). Quanto menor for este valor melhor melhor.

**MAE (Erro Absoluto Médio)**:  

<p>
  MAE = (1 / N) × Σ<sub>i=1</sub><sup>N</sup> |y<sub>i</sub> − ŷ<sub>i</sub>| ≈ 41.92
</p>

>Isso quer dizer que, em média, o modelo erra em ≈ 41.9 unidades no target. Note que MAE tende a ser menor que RMSE quando há poucos outliers grandes uma vez que (RMSE penaliza erros grandes mais fortemente).

**Coeficiente de Determinação (R²)**

<p>
  R<sup>2</sup> = 0.4773
</p>

> Um <p>R<sup>2</sup> = 0.4773</p> significa que o modelo de regressão linear explica cerca de 47.73 % da variância total dos dados de teste. Em outras palavras, pouco menos da metade da variação em “target” é capturada pelo modelo linear com essas 10 features.
---

## 4.7 Modelo SVR (Support Vector Regression – Kernel RBF)

### 4.7.1 Resultados SVR

- **Tempo de Treino (fit)**: 0.003169 segundos  
- **Tempo de Predição (predict)**: 0.003118 segundos  
- **MSE**: 4528.1795  
- **RMSE**: 67.2917  
- **MAE**: 56.4342  
- **R²**: 0.1612  

### Interpretação dos Resultados

**Tempos de Execução**

- ``Treino:`` ~ 0.003169 s

- ``Predição:`` ~ 0.003118 s

> O SVR foi um pouco mais rápido para treinar do que a Regressão Linear (aprox. 3.2 ms vs 7.9 ms), mas mais lento para predizer (3.1 ms vs 1.1 ms). Em datasets pequenos, a diferença de tempos é desprezível, mas reflete a sobrecarga computacional do kernel RBF.

**MSE (Erro Quadrático Médio)**:

MSE = 4528.1795

Maior que o da regressão linear (2821.75). Isso indica que, no conjunto de teste, os quadrados dos erros são, em média, mais altos do que para a regressão linear. Em outras palavras, o SVR “erra” mais, ao quadrado.

**RMSE (Raiz do Erro Quadrático Médio)**:   
 <p>
  RMSE<sub>SVR</sub> ≈ 67.29 &gt; RMSE<sub>Linear</sub> ≈ 53.12
</p>

> Como RMSE = √4528.1795 ≈ 67.29, o erro médio em valor absoluto (considerando que RMSE penaliza erros maiores) é de ≈ 67 unidades do target. Lembrando que, para Regressão Linear, esse valor foi ≈ 53.12. Portanto, o SVR está pior neste particionamento..

**MAE (Erro Absoluto Médio)**:   
<p>
  MAE<sub>SVR</sub> ≈ 56.43 &gt; MAE<sub>Linear</sub> ≈ 41.92
</p>

> Em média, o SVR erra ≈ 56.43 unidades no target. Novamente, maior que o MAE da regressão linear (41.92). Isso confirma que, apesar de o SVR modelar relações não‐lineares, ele não encontrou um padrão preditivo suficiente para melhorar sobre o modelo linear (com os parâmetros default de C=1.0 e ε=0.1).

**Coeficiente de Determinação (R²)**

<p>R<sup>2</sup> = 0.1612</p> 

> Um valor de R² de apenas ≈ 0.1612 indica que o SVR explica 16.12 % da variância no conjunto de teste. Em contraste, a regressão linear explicava quase 48 %. Logo, o SVR com esses hiperparâmetros default está se saindo pior do que a regressão linear simples. Possíveis razões:

>Hiperparâmetros não ajustados (C, ε, gamma, etc.) — normalmente é preciso fazer validação cruzada e busca de grade para obter SVR satisfatório.

> A Relação primordialmente linear entre features e target → modelo linear já captura bem o sinal; SVR pode introduzir overfitting ou não se adaptar bem sem ajuste fino.

## 4.8 Conclusão Comparativa Final (Regressão Linear vs SVM)

 A **Regressão Linear** apresentou **menores erros** (MAE e RMSE) e **melhor capacidade explicativa** (R² quase três vezes maior) em comparação com o SVR.
- Embora o SVR tenha treinado um pouco mais rápido, ele foi significativamente mais lento na fase de predição e não conseguiu reduzir o erro médio nos valores preditos.
- Por fim, considerando o cenário específico (features já padronizadas e relação predominantemente linear), a Regressão Linear é o modelo **mais adequado**:
  - Oferece predições mais confiáveis, com menor erro absoluto.
  - Possui custo de predição inferior, o que pode ser relevante em aplicações em tempo real.
  - O SVR, sem ajuste fino de hiperparâmetros, não trouxe vantagem significativa e apresentou pior desempenho.


## 4.9 RESULTADO/OUTPUT (IMAGENS)


![image](https://github.com/user-attachments/assets/baad4f47-1deb-40ab-8e1d-49913c2606f2)
![image](https://github.com/user-attachments/assets/a1a7f267-0e73-4427-8c44-2fc0ee013ac5)


# 5 CONCLUSÃO FINAL

Neste trabalho, analisámos e comparámos os algoritmos Minimax, Alfa-Beta  e Monte Carlo , aplicados ao Jogo do Galo. Com isso, conseguimos perceber bem as diferenças entre estratégias que analisam tudo ao detalhe e outras que usam “atalhos” para decidir mais rápido. Também vimos como essas abordagens afetam a rapidez do algoritmo e a qualidade das jogadas.

O Minimax mostrou-se super preciso e é perfeito para jogos pequenos como o Jogo do Galo. Como analisa todas as jogadas possíveis, consegue sempre escolher a melhor opção — desde que os dois jogadores joguem bem. O problema é que, em jogos mais complicados, torna-se lento porque tem de ver demasiadas possibilidades.

A Alfa-Beta é uma melhoria do Minimax que ajuda o algoritmo a decidir mais depressa, porque corta as partes da análise que não vão influenciar na decisão final. Isso faz com que funcione muito melhor em tempo real e sem gastar tantos recursos. No Galo, funciona lindamente.

Já o Monte Carlo (MCTS) usa simulações aleatórias para decidir o que fazer. Em vez de calcular tudo como o Minimax, ele joga várias vezes “na sorte” para ver o que pode resultar melhor. Pode não ser perfeito, mas é bem mais leve e adapta-se bem quando o jogo é mais complexo ou quando queremos decisões rápidas e com jogadas diferentes do habitual.
Resumindo: se for para jogar bem e rápido no Jogo do Galo, o Minimax com Alfa-Beta é a melhor escolha. Mas se estivermos a lidar com jogos maiores ou quisermos uma IA que não jogue sempre igual, o MCTS é uma boa opção por ser mais flexível e escalável.


Neste trabalho, comparamos o ID3 e o SVM no dataset Iris e a Regressão Linear e o SVR no dataset Diabetes, e vimos como cada estratégia se comporta em termos de precisão e rapidez.

No Iris, o ID3, embora bem interpretável, acabou confundindo versicolor e virginica com mais frequência, enquanto o SVM (kernel RBF) conseguiu decidir melhor e foi ligeiramente mais veloz na predição, entregando acurácia superior.

Já no Diabetes, a Regressão Linear provou ser eficiente e simples, gerando erros médios muito menores e explicando quase metade da variância, enquanto o SVR padrão, sem ajustes, não conseguiu melhorar o resultado e ainda deixou a inferência mais lenta. 

Resumindo, para classificações pequenas e precisas no Iris, o SVM com Alpha-Beta (ou melhor, SVM puro, neste caso) é a melhor escolha; para regressões lineares nos dados de Diabetes, a Regressão Linear supera o SVR desconfigurado. Models mais sofisticados só valeriam a pena se fossem ajustados cuidadosamente ou em cenários maiores.

## FONTES

Slides Teoricos da Cadeira de Sistemas Inteligentes

https://scikit-learn.org/stable/modules/svm.html
https://scikit-learn.org/stable/modules/tree.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
https://www.simplilearn.com/10-algorithms-machine-learning-engineers-need-to-know-article

Videos:

https://www.youtube.com/watch?v=aLsReomQ7AA
https://www.youtube.com/watch?v=CtsRRUddV2s
https://www.youtube.com/watch?v=_YPScrckx28


