# Importar as bibliotecas necessárias
import pandas as pd
from pandas import DataFrame, read_csv, concat, merge
from matplotlib import pyplot as plt
from sklearn import metrics
#from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import time
import pytz
import os
import numpy as np
from numpy import array
import tensorflow as tf
#from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, Dense, Dropout

# Hiperparâmetros do modelo de predição
def model_configs():
    # Ler o arquivo settings.txt
    with open('settings.txt', 'r') as file:
        grid_search_configs = file.read()
    # Transforma o conteúdo no arquivo .txt em algo rodável em python
    gs = eval(grid_search_configs)
    # Dias passados utilizados para predição
    n_in = gs['n_in']
    # Dias futuros a serem previstos
    n_out = gs['n_out']
    # Parcela dos dados utilizados para o treinamento
    n_split  = gs['n_split']
    # Números de neurônios/nós
    n_nodes = gs['n_nodes']
    # Camadas LSTM
    stacked_layers = gs['stacked_layers']
    # Número de atributos (n_features = 1 univariada, n_features > 1 multivariada)
    n_features = gs['n_features']
    # Dropout dos neurônios
    dropout = gs['dropout']
    # [callbacks, patience (default=10), restore_best_weights (default=True)]
    callback = gs['callback']
    # Épocas
    epochs = gs['epochs']
    # Tamanho de lote
    batch_size = gs['batch_size']
    # Taxa de aprendizado (0.001 = Default learning rate)
    learning_rate = gs['learning_rate']
    # Qual modelo utilizar (opções: "LSTM", "BiLSTM", "MLP")
    predict_model = gs['predict_model']
    # Qual dataset utilizar (opções: "CEPEA_SOJA.csv", "CEPEA_MILHO.csv", "CEPEA_BOI.csv", "CEPEA_ACUCAR.csv", "CEPEA_SOJA_100.csv")
    dataset_in = gs['dataset_in']
    # Criar lista de hiperparâmetros
    configs = list()
    for a in n_in:
      for b in n_out:
        for c in n_split:
          for d in n_nodes:
            for e in stacked_layers:
              for f in n_features:
                for g in dropout:
                  for h in callback:
                    for i in epochs:
                      for j in batch_size:
                        for k in learning_rate:
                          for l in predict_model:
                            for m in dataset_in:
                              cfg = [a, b, c, d, e, f, g, h, i, j, k, l, m]
                              configs.append(cfg)
    print('Total configs: %d' % len(configs))
    rotulos = ["Dias anteriores:", "Dias previstos:", "Treino:", "Neurônios:", "Camadas:", "Atributos:",
               "Dropout:", "Callback/Patience/Melhores pesos:", "Épocas:", "Batch size:", "Taxa de aprendizado:",
               "Modelo:", "Dataset:"]
    return configs, rotulos


# Arquitetura MLP
def build_model_MLP(n_in, n_nodes, dropout, stacked_layers):
    model = Sequential()
    model.add(Dense(n_in, input_dim=n_in, activation='relu'))
    for i in range(stacked_layers):
        model.add(Dense(n_nodes, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    return model


# Arquitetura LSTM
def build_model_LSTM(n_in, input_shape, n_nodes, dropout, stacked_layers):
    model = Sequential()
    model.add(Dense(n_in, input_shape=(input_shape), activation='relu'))
    for i in range(stacked_layers):
      model.add(LSTM(n_nodes, activation='tanh', return_sequences=True if i != stacked_layers - 1 else False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    return model


# Arquitetura BiLSTM
def build_model_Bidirecional(n_in, input_shape, n_nodes, dropout, stacked_layers):
    model = Sequential()
    model.add(Dense(n_in, input_shape=(input_shape), activation='relu'))
    for i in range(stacked_layers):
      model.add(Bidirectional(LSTM(n_nodes, activation='tanh', return_sequences=True if i != stacked_layers - 1 else False)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    return model


# Treinar modelos
def fit_model(model, learning_rate, callback, x_train, y_train, epochs, batch_size):
    # Compilar modelo
    model.compile(
      loss='mse',
      optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
      )

    # Arquitetura do modelo
    model.summary()

    # Definir callback
    if callback[0]:
        _, patience, restore_best_weights = callback
        callbacks = [keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=restore_best_weights)]

    # Treinamento do modelo
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks if callback[0] else None,
        )

    return model


# Aqui é onde a mágica acontece. Apesar do nome, não necessariamente precisa ser um grid search
# Caso as listas de hiperparâmetros tenham apenas um elemento, não há Grid Search
def grid_search(df, cfg, key):
    print(f'===========================================================================================\nConfig nº{key+1}')
    print(f'{cfg}')

    # Descompactando configurações
    n_in, n_out, n_split, n_nodes, stacked_layers, n_features, dropout, callback, epochs, batch_size, learning_rate, predict_model, _ = cfg

    cols = list()
    # Sequência de entrada (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # Sequência de saída (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))

    agg = concat(cols, axis=1)
    # Eliminar linhas com valores NaN
    agg.dropna(inplace=True)
    xx = agg.iloc[: , :n_in]
    yy = agg.iloc[:,-1]
    # Unir os dados
    aggs = merge(xx, yy, left_index = True, right_index = True, how = "outer")
    aggs = aggs.to_numpy()
    # Separar os dados de treinamento e teste
    X, y = aggs[:, :-1], aggs[:, -1]

    # Remodelar dados de entrada: 3D shape para LSTM
    if predict_model == "LSTM" or "BiLSTM":
        X = X.reshape((X.shape[0], X.shape[1], 1))
    # Dividir os dados em x_train, x_test, y_train, y_test
    train_size = int(n_split * len(df['Price']))
    x_train, x_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    input_shape = x_train.shape[1:]

    # Definir modelo
    if predict_model == "MLP":

        x_train, y_train = x_train.reshape(x_train.shape[0],-1), y_train.reshape(y_train.shape[0],1)
        x_test, y_test = x_test.reshape(x_test.shape[0],-1), y_test.reshape(y_test.shape[0],1)

        # Construir a arquitetura do modelo
        model = build_model_MLP(
            n_in,
            n_nodes,
            dropout,
            stacked_layers,
        )
        # Treinar o modelo
        model = fit_model(model, learning_rate, callback, x_train, y_train, epochs, batch_size)
        # Predizer valores
        predict = model.predict(x_test)
    elif predict_model == "LSTM":
        # Construir a arquitetura do modelo
        model = build_model_LSTM(
            n_in,
            input_shape,
            n_nodes,
            dropout,
            stacked_layers,
        )
        # Treinar o modelo
        model = fit_model(model, learning_rate, callback, x_train, y_train, epochs, batch_size)
        # Predizer valores
        pred = list()
        for i in range(len(x_test)):
            x_input = array(x_test[i]).reshape((1, n_in, 1))
            yhat = model.predict(x_input, verbose=0)
            # Armazenar predição em uma lista de resultados
            pred.append(yhat[0])
        predict = array([y[0] for y in pred])
    elif predict_model == "BiLSTM":
        # Construir a arquitetura do modelo
        model = build_model_Bidirecional(
            n_in,
            input_shape,
            n_nodes,
            dropout,
            stacked_layers,
        )
        # Treinar o modelo
        model = fit_model(model, learning_rate, callback, x_train, y_train, epochs, batch_size)
        # Predizer valores
        pred = list()
        for i in range(len(x_test)):
            x_input = array(x_test[i]).reshape((1, n_in, 1))
            yhat = model.predict(x_input, verbose=0)
            # Armazenar predição em uma lista de resultados
            pred.append(yhat[0])
        predict = array([y[0] for y in pred])

    # Salvar o modelo
    model.save(f"results/{seed}/results{key + 1}_{seed}.keras")

    # Avaliar o modelo utilizando MSE e RMSE
    mse_score = metrics.mean_squared_error(y_test, predict)
    rmse_score = np.sqrt(metrics.mean_squared_error(y_test, predict))
    error = [mse_score, rmse_score]
    print(f"Score (MSE): {mse_score}")
    print(f"Score (RMSE): {rmse_score}")

    # Plot predict x real
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(predict)), np.array(predict), label='Prediction', linewidth=1, color="red")
    plt.plot(np.arange(len(y_test)), np.array(y_test), label='Test data', linewidth=1, color="blue")
    plt.title(f"LSTM - Cfg nº{key+1}: {cfg}")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(f"results/{seed}/results{key + 1}_{seed}.png", bbox_inches='tight', dpi=600)

    return key, error


# Semente exclusiva desta execução
seed = datetime.now(pytz.timezone('America/Sao_Paulo')).strftime('%d%m%Y_%H%M%S')
tf.random.set_seed(seed)
print(f"Runtime Seed: {seed}")

# Início da contagem de tempo de execução (em segundos)
start_time = time.time()

# Definir configurações do Grid Search
cfg_list, rotulos = model_configs()

# Verificar se há GPU e se ela está disponível para treinar o modelo
gpu_device = tf.config.experimental.list_physical_devices("GPU")
print("Número de GPUs disponíveis: ", len(gpu_device))
if len(gpu_device) > 0:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Verificar e criar pasta para armazenar os resultados
path = f"results/{seed}"
os.mkdir(path)

# Grid Search
scores = list()
for key, cfg in enumerate(cfg_list):
    # Carregar dados de amostra
    df = pd.read_csv(f'datasets/{cfg[-1]}', index_col="datetime", parse_dates=True)
    start = time.time()
    score = list()
    key, error = grid_search(df, cfg, key)
    score.append(key)
    # Cria e/ou escreve em um txt. Escreve o número da rodada.
    with open(f'results/{seed}/resultados.txt', 'a') as file:
        file.write(f'Config nº: {key+1} ')
    for i, j in enumerate(cfg):
        score.append(i)
        # Cria e/ou escreve em um txt. Escreve as configurações da rodada.
        with open(f'results/{seed}/resultados.txt', 'a') as file:
            file.write(f'{rotulos[i]} {j}, ')
    score.append(error)
    score = tuple(score)
    scores.append(score)
    checkpoint = (time.time() - start)
    print(f"Time (seconds): {checkpoint:.6f}")
    # Cria e/ou escreve em um txt. Escreve as métricas da rodada.
    with open(f'results/{seed}/resultados.txt', 'a') as file:
        file.write(f'--> MSE: {error[0]:.6f}, RMSE: {error[1]:.6f}, Time: {checkpoint:.3f} seconds\n')

# Resultados ordenados do melhor para o pior
scores.sort(key=lambda tup: tup[-1])

# Resumo dos scores
print("\n-----------------------------Resultados Finais-----------------------------")
for cfg,_,_,_,_,_,_,_,_,_,_,_,_,_,error in scores[:]:
    print(f'Config nº{cfg+1} - MSE: {error[0]:.6f}, RMSE: {error[1]:.6f}')

# Fim da execução
print(f"--- End: {(time.time() - start_time):.3f} seconds ---")
print('Done')
