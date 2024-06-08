# Projeto de Previsão de Séries Temporais usando Modelos de Machine Learning

[![Badge de Licença](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Badge de Status do Projeto](https://img.shields.io/badge/Status-Em%20Desenvolvimento-blue.svg)](https://github.com/seu-usuario/seu-projeto)

Este projeto foi desenvolvido para realizar previsões de séries temporais utilizando três modelos de machine learning: MLP, LSTM e BiLSTM. Através de um processo de Grid Search, o projeto explora diferentes combinações de hiperparâmetros para encontrar a melhor configuração para cada modelo e dataset.

## Índice

1. [Estrutura do Projeto](#estrutura-do-projeto)
2. [Instalação](#instalação)
    - [Pré-requisitos](#pré-requisitos)
    - [Passos para Instalação](#passos-para-instalação)
3. [Estrutura do Arquivo settings.txt](#estrutura-do-arquivo-settingstxt)
    - [Descrição das Configurações](#descrição-das-configurações)
4. [Execução](#execução)
5. [Resultados](#resultados)
6. [Descrição das Funções](#descrição-das-funções)
7. [Licença](#licença)

## Estrutura do Projeto

- **Pastas**:
  - `.idea/`
  - `datasets/` (contém os datasets para treino e teste)
  - `results/` (contém os resultados das execuções do grid search)

- **Arquivos**:
  - `LICENSE`
  - `main.py` (arquivo principal do projeto)
  - `MANIFEST.in`
  - `pyproject.toml`
  - `README.md`
  - `requirements.txt` (lista de bibliotecas necessárias)
  - `settings.txt` (arquivo de configuração do grid search)

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)

### Passos para Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/Grogenski/DLgrid_search_timeseriesforecasting/
   cd DLgrid_search_timeseriesforecasting
   ```

2. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   ```

3. Ative o ambiente virtual:

   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. Instale as bibliotecas necessárias:
   ```bash
   pip install -r requirements.txt
   ```

## Estrutura do Arquivo settings.txt

O arquivo `settings.txt` contém as configurações para o grid search em formato de dicionário. Um exemplo de configuração é mostrado abaixo:

```python
{
    'n_in': [33],
    'n_out': [7],
    'n_split': [0.8],
    'n_nodes': [64],
    'stacked_layers': [1],
    'n_features': [1],
    'dropout': [0.2],
    'callback': [[True, 10, True]],
    'epochs': [100],
    'batch_size': [40],
    'learning_rate': [0.001],
    'predict_model': ["MLP", "LSTM", "BiLSTM"],
    'dataset_in': ["CEPEA_SOJA.csv"]
}
```

### Descrição das Configurações

- **'n_in'**: Quantidade de dias anteriores usados para prever valores futuros.
- **'n_out'**: Quantidade de dias futuros a serem preditos.
- **'n_split'**: Fração dos dados usados para treinamento (padrão: 0.8).
- **'n_nodes'**: Número de neurônios nas camadas intermediárias.
- **'stacked_layers'**: Número de camadas intermediárias.
- **'n_features'**: Número de atributos usados. (1 para univariado, >1 para multivariado).
- **'dropout'**: Taxa de dropout para evitar overfitting (0.0 para desativar, >0.0 para ativar).
- **'callback'**: Lista contendo [callback, patience, restore_best_weights]. Callback para early stopping.
- **'epochs'**: Número de épocas de treinamento.
- **'batch_size'**: Tamanho do batch.
- **'learning_rate'**: Taxa de aprendizado (padrão: 0.001).
- **'predict_model'**: Modelos a serem usados: "MLP", "LSTM", "BiLSTM".
- **'dataset_in'**: Datasets a serem utilizados. Opções: "CEPEA_SOJA.csv", "CEPEA_MILHO.csv", "CEPEA_ACUCAR.csv", "CEPEA_BOI.csv", "CEPEA_SOJA_100.csv".

## Execução

1. Configure os hiperparâmetros e selecione os datasets desejados no arquivo `settings.txt`.
2. Execute o script principal:
   ```bash
   python main.py
   ```

## Resultados

Os resultados serão armazenados em uma pasta na diretório `results/` com o nome da seed (formato da seed é dado pela data e horário de início do programa: `%d%m%Y_%H%M%S`). Cada pasta conterá:
- Plots das previsões vs. valores reais.
- Modelos salvos em formato `.keras`.
- Arquivo `results.txt` com os resultados e o tempo gasto em cada configuração.

## Descrição das Funções

- **model_configs()**: Importa as configurações do arquivo `settings.txt` e gera uma lista de listas de configurações.
- **build_model_MLP(n_in, n_nodes, dropout, stacked_layers)**:  Constrói um modelo Perceptron Multicamadas (MLP) simples.
- **build_model_LSTM(n_in, input_shape, n_nodes, dropout, stacked_layers)**: Constrói um modelo LSTM (Long Short-Term Memory) simples.
- **build_model_Bidirecional(n_in, input_shape, n_nodes, dropout, stacked_layers)**: Constrói um modelo BiLSTM (Bidirectional LSTM) simples.
- **fit_model(model, learning_rate, callback, x_train, y_train, epochs, batch_size)**: Treina o modelo selecionado com os dados de treinamento.
- **grid_search(df, cfg, key)**: Realiza um grid search para encontrar os melhores hiperparâmetros, constrói e treina modelos com base nas configurações fornecidas, e avalia os resultados com métricas de desempenho, como o erro quadrático médio (MSE) e a raiz do erro quadrático médio (RMSE), além de gerar plots com os valores preditos contra os reais.

## Licença

Este projeto está licenciado sob os termos da licença MIT. Para mais detalhes, veja o arquivo `LICENSE`.

---

Para quaisquer dúvidas ou problemas, por favor, entre em contato com lucas.grogenskimeloca@gmail.com.
