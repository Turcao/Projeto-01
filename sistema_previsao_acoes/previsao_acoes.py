# bibliotecas
import yfinance as yf # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import matplotlib.pyplot as plt # type: ignore

# 1. Baixar dados históricos
dados = yf.download('AAPL', start='2015-01-01', end='2023-12-31')
precos = dados['Close']

# 2. Pré-processar os dados
scaler = MinMaxScaler(feature_range=(0, 1))
precos_normalizados = scaler.fit_transform(precos.values.reshape(-1, 1))

# Criar janelas de tempo
def criar_janelas(dados, janela):
    X, y = [], []
    for i in range(len(dados) - janela):
        X.append(dados[i:i+janela])
        y.append(dados[i+janela])
    return np.array(X), np.array(y)

janela = 60  # Usar 60 dias para prever o próximo
X, y = criar_janelas(precos_normalizados, janela)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 3. Dividir em treino e teste
tamanho_treino = int(len(X) * 0.8)
X_treino, X_teste = X[:tamanho_treino], X[tamanho_treino:]
y_treino, y_teste = y[:tamanho_treino], y[tamanho_treino:]

# 4. Construir e treinar o modelo
modelo = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_treino.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
modelo.compile(optimizer='adam', loss='mean_squared_error')
modelo.fit(X_treino, y_treino, epochs=10, batch_size=32)

# 5. Fazer previsões
previsoes = modelo.predict(X_teste)
previsoes = scaler.inverse_transform(previsoes)

# 6. Visualizar os resultados
plt.plot(precos.index[-len(y_teste):], scaler.inverse_transform(y_teste.reshape(-1, 1)), label='Real')
plt.plot(precos.index[-len(previsoes):], previsoes, label='Previsto')
plt.legend()
plt.show()
