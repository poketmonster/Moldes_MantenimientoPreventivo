import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime 
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt2



def get_data(path, filename, normalized=0):
    estados = pd.read_csv(path+'/data/'+filename, header=0) 
    print(estados.head())
    df = pd.DataFrame(estados)
    df = df.sort_values(by=['MoldeID','hora'])
    df.drop(df.columns[[0,1,3,4,5,6,7,8]], axis=1, inplace=True) 
    df.columns = ['molde', 'hora', 'piezas']
    return df



def load_data(cambios, seq_len = 15):
    amount_of_features = len(cambios.columns)
    data = cambios.as_matrix()
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]


def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

def build_model2(layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16,init='uniform',activation='relu'))        
    model.add(Dense(1,init='uniform',activation='relu'))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    return model



def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(
    X_train,
    y_train,
    batch_size=8,
    epochs=50,
    validation_split=0.2,
    verbose=1)

    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))


def predict(model, X_test, y_test):

    print(X_test[-1])
    diff=[]
    ratio=[]
    p = model.predict(X_test)
    for u in range(len(y_test)):
        pr = p[u][0]
        ratio.append((y_test[u]/pr)-1)
        diff.append(abs(y_test[u]- pr))
        print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))

    return p

def plotResults(p, y_test):
    plt2.plot(p,color='red', label='prediction')
    plt2.plot(y_test,color='blue', label='y_test')
    plt2.legend(loc='upper left')
    plt2.show()


#Cargar dataframe
path ="~/datarus/www/master/practicas-arf/Moldes_MantenimientoPreventivo"
filename="limpiezas.csv"
df = get_data(path, filename)
df = df.drop(df[df.piezas == 0].index)
#df = df.drop(df[df.hora > 500].index)
df

#Separar corpus
sequence_length = 15
X_train, y_train, X_test, y_test = load_data(df[::-1], sequence_length)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

#Construir el modelo
model = build_model([3,sequence_length,1])

#Entrenar el modelo
run_model(model, X_train, y_train, X_test, y_test)

#Predecir para X_test
prediccion = predict(model, X_test, y_test)

#Gráfico resultados
plotResults(prediccion, y_test)






