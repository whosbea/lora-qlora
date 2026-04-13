# Laboratório 07 — Especialização de LLMs com LoRA e QLoRA

## Objetivo do laboratório

Este laboratório teve como objetivo construir um pipeline completo de fine-tuning de um modelo de linguagem utilizando **LoRA** e **QLoRA**, com foco em eficiência de parâmetros e redução do consumo de memória. A proposta foi adaptar um modelo base para um domínio específico, usando um dataset sintético no formato `.jsonl`, treinamento supervisionado com `SFTTrainer` e salvamento do adaptador final treinado.

## Domínio escolhido

O domínio escolhido para o experimento foi **pinguins**. O dataset foi composto por instruções e respostas em português sobre habitat, alimentação, reprodução, adaptação ao frio, espécies e curiosidades relacionadas a pinguins.

## Modelos utilizados

### Modelo base para fine-tuning com QLoRA

Foi utilizado o modelo:

- **Qwen/Qwen2.5-0.5B-Instruct**

Esse modelo foi escolhido por ser relativamente leve para experimentos acadêmicos e por ser compatível com o pipeline de fine-tuning com quantização em 4 bits e adaptação por LoRA.

### Modelo utilizado para geração do dataset sintético

Na configuração declarada para a entrega, o dataset sintético foi gerado com:

- **GPT-3.5**, acessado via **OpenRouter**

> **Observação importante:** ao longo do desenvolvimento do laboratório, houve tentativas com outros provedores por limitações de quota e ambiente.

## Estrutura do projeto

```text
lora-qlora/
├── data/
│   ├── train.jsonl
│   └── test.jsonl
├── outputs/
│   └── qwen2_5_penguins_qlora/
├── src/
│   ├── generate_dataset.py
│   ├── train_qlora.py
│   └── test_model.py
├── requirements.txt
└── README.md
```

## Como gerar o dataset

O script de geração do dataset cria exemplos sintéticos no domínio escolhido, organiza os registros em JSONL e divide os dados em treino e teste.

### Passos gerais

1. Configurar a chave de API do provedor utilizado para geração.
2. Executar o script de geração.
3. Conferir se os arquivos `train.jsonl` e `test.jsonl` foram criados na pasta `data/`.

### Exemplo de execução

```bash
python src/generate_dataset.py
```

### Saída esperada

- `data/train.jsonl`
- `data/test.jsonl`

## Como treinar o modelo

O treinamento foi realizado com:

- quantização em **4 bits**
- tipo de quantização **nf4**
- `bnb_4bit_compute_dtype=torch.float16`
- **LoRA** com:
  - `r = 64`
  - `alpha = 16`
  - `dropout = 0.1`
- otimizador **paged_adamw_32bit**
- scheduler **cosine**
- warmup equivalente a **3%** dos passos de treino

## Justificativa dos parâmetros e nomenclaturas do treinamento

Nesta seção, são explicados os principais parâmetros utilizados no treinamento com QLoRA, bem como o significado de cada nomenclatura e a justificativa para sua escolha no contexto deste laboratório.

### Quantização em 4 bits

A quantização em 4 bits reduz a quantidade de memória necessária para armazenar os pesos do modelo. Em vez de manter os parâmetros em formatos mais pesados, como `float32`, o modelo passa a operar com uma representação muito mais compacta. A principal vantagem dessa escolha é permitir o treinamento de modelos relativamente grandes em hardware com recursos limitados.

Neste laboratório, a quantização em 4 bits foi utilizada para viabilizar o fine-tuning do modelo sem exigir o treinamento completo de todos os parâmetros, o que seria inviável no ambiente disponível.

### `bnb_4bit_quant_type="nf4"`

A sigla `bnb` vem de **bitsandbytes**, biblioteca responsável pelas rotinas de quantização utilizadas no projeto. O parâmetro `4bit_quant_type` define qual estratégia de quantização será aplicada aos pesos do modelo.

O valor `nf4` significa **NormalFloat 4-bit**. Esse formato foi projetado para representar pesos de redes neurais de forma mais adequada do que uma quantização genérica. Em termos práticos, ele busca comprimir os pesos preservando melhor a informação útil para o modelo.

A escolha de `nf4` foi feita porque esse é o formato recomendado para QLoRA e também foi o formato exigido no enunciado do laboratório.

### `bnb_4bit_compute_dtype=torch.float16`

Esse parâmetro define o tipo numérico usado nos cálculos internos durante o treinamento. Mesmo com os pesos armazenados em 4 bits, as operações matemáticas não são feitas diretamente nesse formato simplificado. Elas precisam de um tipo numérico intermediário para garantir estabilidade e desempenho.

A nomenclatura `torch.float16` indica o uso de números em ponto flutuante de 16 bits. Esse formato é menor e mais leve que `float32`, exigindo menos memória e acelerando o processamento em GPU compatível.

A escolha por `float16` foi feita para manter o treinamento mais leve e viável no ambiente utilizado, reduzindo custo computacional sem abandonar totalmente a precisão numérica.

### LoRA

LoRA significa **Low-Rank Adaptation**. Em vez de atualizar todos os pesos do modelo original, essa técnica adiciona pequenas matrizes treináveis a partes específicas da rede. Assim, o modelo base permanece praticamente congelado, e apenas uma fração dos parâmetros é ajustada.

A vantagem dessa abordagem é reduzir o custo do treinamento, tanto em memória quanto em tempo, ao mesmo tempo em que permite adaptar o modelo a um novo domínio.

### `r = 64`

O parâmetro `r` representa o **rank** da decomposição de baixa dimensão usada no LoRA. Em termos simples, ele define o tamanho das matrizes adicionais treináveis inseridas no modelo.

Quanto maior o valor de `r`, maior a capacidade de adaptação do modelo, pois ele ganha mais liberdade para aprender novos padrões. Por outro lado, valores maiores também aumentam custo computacional.

Neste laboratório, o valor `64` foi utilizado por ser o valor explicitamente exigido no enunciado. Além disso, é um valor relativamente alto para LoRA, o que dá ao modelo uma boa capacidade de ajuste ao novo domínio.

### `alpha = 16`

O parâmetro `alpha` é o **fator de escala** aplicado à atualização aprendida pelo LoRA. Ele controla o peso efetivo da adaptação no comportamento final do modelo.

Em outras palavras, o LoRA aprende uma modificação, e o `alpha` determina o quanto essa modificação influencia o modelo durante a inferência e o treinamento.

A escolha de `16` segue o valor solicitado no laboratório e mantém um equilíbrio razoável entre estabilidade e capacidade de ajuste.

### `dropout = 0.1`

O `dropout` é uma técnica de regularização. Durante o treinamento, ele desativa aleatoriamente parte das conexões temporariamente, forçando o modelo a não depender demais de padrões muito específicos.

A nomenclatura `0.1` significa que há uma taxa de 10% de dropout. Isso ajuda a reduzir o risco de **overfitting**, especialmente porque o dataset do laboratório é pequeno.

A escolha desse valor também veio diretamente do enunciado, além de ser um valor bastante comum para regularização leve.

### `task_type="CAUSAL_LM"`

Esse parâmetro informa ao PEFT/LoRA qual é o tipo de tarefa que o modelo executa.

A expressão `CAUSAL_LM` significa **Causal Language Modeling**, que é o modo tradicional de modelos autoregressivos como GPT e Qwen. Nesse tipo de tarefa, o modelo prevê o próximo token a partir dos anteriores.

Esse valor foi usado porque o modelo treinado é um modelo gerador de texto, e esse é exatamente o tipo de tarefa esperada para ele.

### Otimizador `paged_adamw_32bit`

O otimizador é o algoritmo que atualiza os pesos treináveis com base no erro calculado durante o treinamento.

A nomenclatura `AdamW` se refere a uma variante do Adam com regularização de pesos desacoplada. Já o termo `paged` indica uma estratégia voltada para gerenciar melhor memória, reduzindo picos de uso ao mover partes do estado do otimizador de forma mais eficiente. O sufixo `32bit` indica que esse estado interno é mantido em 32 bits.

A escolha de `paged_adamw_32bit` foi feita porque ele é amplamente utilizado em cenários de QLoRA e foi especificado no laboratório justamente por ser mais adequado a ambientes com limitação de memória.

### Scheduler `cosine`

O scheduler controla como a **learning rate** muda ao longo do treinamento.

A nomenclatura `cosine` significa que a taxa de aprendizado decresce seguindo uma curva cosseno. Na prática, isso faz com que o treinamento comece com atualizações mais intensas e vá reduzindo gradualmente a agressividade dos ajustes.

A escolha desse scheduler foi feita porque ele ajuda a estabilizar o treinamento ao longo do tempo e também foi explicitamente exigido pelo laboratório.

### Warmup de 3%

O warmup é um período inicial em que a taxa de aprendizado cresce gradualmente, em vez de começar já no valor máximo.

Neste laboratório, foi utilizado um warmup equivalente a 3% do total de passos de treino. Isso significa que, no começo, o modelo recebe atualizações mais suaves, o que reduz risco de instabilidade logo nos primeiros passos.

Essa escolha foi feita porque o enunciado exigia `warmup_ratio = 0.03`, ou seja, 3% do treinamento reservado para aquecimento da learning rate.

### Learning rate

A **learning rate** é a taxa de aprendizado, isto é, o tamanho do passo dado pelo otimizador a cada atualização.

No treinamento, foi utilizado o valor `2e-4`, que equivale a `0.0002`. Esse valor é comum em fine-tuning com LoRA/QLoRA, pois permite ajustar o modelo com velocidade razoável sem tornar o treinamento instável.

### `per_device_train_batch_size = 1`

Esse parâmetro define quantos exemplos são processados por vez em cada dispositivo durante o treino.

O valor `1` foi utilizado para reduzir o consumo de memória da GPU. Como o treinamento de LLMs é pesado, batches pequenos ajudam a manter o processo viável em ambiente limitado.

### `gradient_accumulation_steps = 4`

Esse parâmetro serve para acumular gradientes por vários passos antes de atualizar os pesos. Na prática, isso simula um batch maior sem precisar carregar todos os exemplos de uma vez na memória.

Com `batch_size = 1` e `gradient_accumulation_steps = 4`, o efeito aproximado é semelhante ao de um batch efetivo de 4 exemplos.

A escolha foi feita para equilibrar limitação de memória e estabilidade de treinamento.

### `num_train_epochs = 3`

Uma **época** representa uma passagem completa pelo conjunto de treino.

O valor `3` indica que o modelo percorreu o dataset inteiro três vezes. Como o conjunto de dados é pequeno, mais de uma época é útil para reforçar o aprendizado. Ao mesmo tempo, não se escolheu um valor muito alto para evitar overfitting.

### `max_seq_length = 512`

Esse parâmetro define o tamanho máximo da sequência de tokens processada pelo modelo.

A escolha de `512` foi suficiente para acomodar os prompts e respostas do dataset sobre pinguins sem desperdiçar memória com sequências muito longas. É um valor intermediário, comum em experimentos de fine-tuning supervisionado.

### Resumo final

Os parâmetros escolhidos buscaram equilibrar três objetivos principais:

1. **Viabilizar o treinamento em hardware limitado**, por meio de quantização em 4 bits e LoRA.
2. **Manter estabilidade durante o ajuste fino**, usando `float16`, scheduler `cosine`, warmup e regularização com dropout.
3. **Garantir aderência ao enunciado do laboratório**, respeitando os valores exigidos para LoRA, quantização e otimização.

Em conjunto, essas escolhas permitiram executar um pipeline de especialização de modelo com custo computacional reduzido, sem recorrer ao fine-tuning completo de todos os parâmetros do modelo base.

### Executar o treino

```bash
python src/train_qlora.py
```

### Saída esperada

Ao final do processo, o adaptador treinado é salvo em:

```text
outputs/qwen2_5_penguins_qlora
```

## Como testar o modelo treinado

Após o fine-tuning, o modelo pode ser testado com o script de inferência, que carrega o modelo base, aplica o adaptador salvo e executa uma geração a partir de um prompt de teste.

### Executar o teste

```bash
python src/test_model.py
```

### O que o script faz

- carrega o tokenizer
- carrega o modelo base quantizado
- carrega o adaptador treinado
- executa uma inferência com um prompt sobre pinguins
- imprime a saída bruta e a resposta extraída

## Ambiente local e migração para o Google Colab

### Ambiente local utilizado inicialmente

O desenvolvimento foi iniciado localmente em um **MacBook M4**, com as seguintes especificações informadas:

- **Chip:** Apple M4
- **Memória RAM:** 16 GB
- **Armazenamento:** 256 GB SSD

### Momento em que houve a migração para o Colab

A mudança para o **Google Colab** ocorreu **na etapa de treinamento do arquivo `src/train_qlora.py`**, quando o pipeline já havia conseguido carregar o modelo, preparar o dataset e iniciar a configuração do treino, mas falhou por limitação de backend do `bitsandbytes` no ambiente local.

O problema identificado foi, em essência, a impossibilidade de usar corretamente o fluxo com **QLoRA + bitsandbytes + paged_adamw_32bit** no ambiente do Mac para a execução completa do treino. Por isso, a conclusão do laboratório passou a ser feita no Colab com GPU CUDA.

### Como o Colab foi configurado

Para executar o laboratório corretamente no Colab, foi feita a seguinte configuração:

1. Criação de uma sessão no **Google Colab**.
2. Alteração do **Ambiente de execução** para **GPU**.
3. Instalação das dependências compatíveis com o treinamento.
4. Clonagem do repositório do GitHub para dentro de `/content/lora-qlora`.
5. Execução do treino com `src/train_qlora.py`.
6. Execução do teste com `src/test_model.py`.

### Bibliotecas principais utilizadas no Colab

- `torch==2.5.1`
- `torchvision==0.20.1`
- `torchaudio==2.5.1`
- `transformers==4.46.3`
- `accelerate==1.0.1`
- `peft==0.13.2`
- `trl==0.11.4`
- `bitsandbytes==0.44.1`
- `datasets==3.1.0`

## Resultado obtido

Ao final, o laboratório conseguiu:

- gerar o dataset sintético do domínio escolhido
- treinar um adaptador LoRA/QLoRA sobre o modelo base
- salvar o adaptador treinado
- carregar o adaptador posteriormente
- realizar inferência com o modelo ajustado

Isso confirma que o pipeline completo do laboratório foi implementado e executado com sucesso.

## Observação sobre uso de IA

**Este projeto contou com apoio de IA (ChatGPT 5.4 Thinking) na geração e organização do código. Todo o conteúdo foi revisado, ajustado e estudado por Beatriz Barreto.**

## Observação final

Como parte do processo de desenvolvimento, alguns ajustes de compatibilidade entre versões de bibliotecas e ambiente de execução foram necessários até a estabilização do treino no Colab. Isso não altera o objetivo do experimento, mas explica as adaptações técnicas realizadas para concluir o laboratório com sucesso.
