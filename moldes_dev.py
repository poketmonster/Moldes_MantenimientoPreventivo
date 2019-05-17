import io
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
from keras import optimizers
import keras.backend as K
import matplotlib.pyplot as plt2



def get_data(path, filename, objetivo = "horas", normalized=0):
    estados = pd.read_csv(path+'/data/'+filename, header=0) 
    print(estados.head())
    df = pd.DataFrame(estados)
    df = df.sort_values(by=['MoldeID','hora'])
    df.drop(df.columns[[0,1,3,4,5,6,7,8,10]], axis=1, inplace=True) 
    if objetivo == "piezas":
        df.columns = ['molde', 'piezas']
    else:
        df.columns = ['molde', 'horas']

    print("Horas inicial "+str(df[1:30]))

    if normalized and objetivo == "horas":
        # MinMax normalization (from 0 to 1)
        df['horas_norm'] = df['horas']
        cols_normalize = df.columns.difference(['molde','horas'])
        min_max_scaler = preprocessing.MinMaxScaler()
        norm_df = pd.DataFrame(min_max_scaler.fit_transform(df[cols_normalize]), 
                                    columns=cols_normalize, 
                                    index=df.index)
        join_df = df[df.columns.difference(cols_normalize)].join(norm_df)
        df = join_df.reindex(columns = df.columns)
    

    return df



def load_data(cambios, porcent_train, seq_len = 15):
    amount_of_features = len(cambios.columns)
    data = cambios.as_matrix()
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        #print('Primero '+str(data[index][0])+" ultimo "+str(data[index + sequence_length][0]))
        if(data[index][0] == data[index + sequence_length][0]):
            result.append(data[index: index + sequence_length])
            #print("Append",data[index: index + sequence_length])

    #print(result)
    result = np.array(result)
    row = round(porcent_train * result.shape[0])
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

    print(model.summary())
    return model


def se_met(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square(y_true - y_pred ))
    return (1 - SS_res)
    #SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    #return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def build_model2(layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(32, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(64,init='uniform',activation='linear'))        
    model.add(Dropout(d))
    model.add(Dense(32,init='uniform',activation='linear'))        
    model.add(Dropout(d))
    model.add(Dense(16,init='uniform',activation='linear'))   
    model.add(Dense(1,init='uniform',activation='linear'))
    
    #model.compile(loss='mse',optimizer='adam',metrics=[r2_keras])
    #model.compile(loss='mse', optimizer='rmsprop',metrics=['accuracy'])


    #sgd = optimizers.SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)
    rmsprop = optimizers.RMSprop()

    model.compile(loss='mse',optimizer=rmsprop, metrics=['accuracy', se_met])

    print(model.summary())
    return model


def build_model2piezas(layers):
    d = 0.1
    model = Sequential()
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(32, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16,init='uniform',activation='linear'))   
    model.add(Dense(1,init='uniform',activation='linear'))
    
    #model.compile(loss='mse',optimizer='adam',metrics=[r2_keras])
    #model.compile(loss='mse', optimizer='rmsprop',metrics=['accuracy'])


    #optimizer = optimizers.SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)
    optimizer = optimizers.RMSprop()

    model.compile(loss='mse',optimizer=optimizer, metrics=['accuracy', se_met])

    print(model.summary())
    return model

def run_model(model, epochs, X_train, y_train, X_test, y_test):
    model.fit(
    X_train,
    y_train,
    batch_size=8,
    epochs=epochs,
    validation_split=0.2,
    verbose=1)

    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.10f MSE (%.10f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.10f MSE (%.10f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

    return model


def predict(model, X_test, y_test):

    print(X_test[-1])
    diff=[]
    ratio=[]
    p = model.predict(X_test)
    for u in range(len(y_test)):
        pr = p[u][0]
        ratio.append((y_test[u]/pr)-1)
        diff.append(abs(y_test[u]- pr))
        #print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))

    return p

def plotResults(p, y_test):
    initplots
    plt2.plot(y_test,color='blue', label='y_test')
    plt2.plot(p,color='red', label='prediction')
    plt2.legend(loc='upper left')
    plt2.show()

def initplots():
    %matplotlib inline
    import mpld3
    mpld3.enable_notebook()
    plt.rcParams['figure.figsize'] = [20, 10]

def calcularError():

def main():
    #Cargar dataframe
    path ="~/datarus/www/master/practicas-arf/Moldes_MantenimientoPreventivo"
    filename="limpiezas.csv"
    objetivo = "piezas" #"horas" o "piezas" - default "horas"
    df = get_data(path, filename, objetivo, 0)
    df = df.sort_values(by=['molde'])
    if objetivo == "piezas":
        df = df.drop(df[df.piezas <= 0].index)
        df = df.drop(df[df.piezas > 1000].index)
    else:
        df = df.drop(df[df.horas <= 5].index)
        df = df.drop(df[df.horas > 500].index)
    print(df.head())
    print("Corpus", df.shape)


    #Separar corpus
    sequence_length = 5
    porcent_train = 0.6
    X_train, y_train, X_test, y_test = load_data(df[::-1], porcent_train, sequence_length)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)

    #Construir el modelo
    model = build_model2([X_train.shape[2],sequence_length])

    #Entrenar el modelo
    epochs = 500
    model = run_model(model, epochs, X_train, y_train, X_test, y_test)

    #Predecir para X_test
    prediccion = predict(model, X_test, y_test)

    #Gráfico resultados
    plotResults(prediccion, y_test)


    #Gráfico resultados
    fAmpliacion = 2
    plotResults((prediccion-np.mean(prediccion))*fAmpliacion+np.mean(prediccion), y_test)

    #Gráfico resultados
    fAmpliacion = 3
    plotResults((prediccion-np.mean(prediccion))*fAmpliacion+np.mean(prediccion), y_test)

    #Gráfico resultados
    fAmpliacion = 4
    plotResults((prediccion-np.mean(prediccion))*fAmpliacion+np.mean(prediccion), y_test)


  
if __name__== "__main__":
    main()


'''
#Cargar dataframe
path ="~/datarus/www/master/practicas-arf/Moldes_MantenimientoPreventivo"
filename="limpiezas.csv"
objetivo = "piezas" #"horas" o "piezas" - default "horas"
df = get_data(path, filename, objetivo, 0)
df = df.sort_values(by=['molde'])
if objetivo == "piezas":
    df = df.drop(df[df.piezas <= 0].index)
    df = df.drop(df[df.piezas > 1000].index)
else:
    df = df.drop(df[df.horas <= 5].index)
    df = df.drop(df[df.horas > 500].index)
print(df.head())
print("Corpus", df.shape)



#Separar corpus
sequence_length = 5
porcent_train = 0.6
X_train, y_train, X_test, y_test = load_data(df[::-1], porcent_train, sequence_length)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

#Construir el modelo
if objetivo == "piezas":
    model = build_model2piezas([X_train.shape[2],sequence_length])
else:
    model = build_model2([X_train.shape[2],sequence_length])

#Entrenar el modelo
epochs = 20
run_model(model, epochs, X_train, y_train, X_test, y_test)

#Predecir para X_test
prediccion = predict(model, X_test, y_test)


#Gráfico resultados
plotResults(prediccion, y_test)

prediccion
y_test

#Gráfico resultados
fAmpliacion = 3
plotResults((prediccion-np.mean(prediccion))*fAmpliacion+(fAmpliacion/2)*np.mean(prediccion), y_test)


%matplotlib inline
import mpld3
mpld3.enable_notebook()
plt.rcParams['figure.figsize'] = [20, 10]
'''