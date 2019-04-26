import io
import os
import numpy as np
import pandas as pd
from pandas import datetime

import time
from datetime import date
from datetime import timedelta

def loadfiles():
    final_df = pd.DataFrame()

    for file in os.listdir(inputdir):
        #print (file)
        if str(file).find("EstadoMolde")>=0:
            #print(file)
            new = pd.read_csv(inputdir+'/'+file, sep=',', engine='python')
            final_df = final_df.append(new)
    
    final_df = final_df.reset_index(drop=True)
    return final_df

def CalcularLapso(df):
    date_str = '01/01/2016' # The date - 29 Dec 2017
    format_str = '%d/%m/%Y' # The format
    initialDate = datetime.strptime(date_str, format_str)
     	
    df['created_at'] = pd.to_datetime(df['created_at']) 

    df['hora'] = df['created_at']-initialDate
    df['hora'] = df['hora'] / np.timedelta64(1, 's') / 3600
    df['hora'] = df['hora'].astype(int)

    return df

def eliminaEstadosCortos(df, min):
    #Eliminamos todos los estados en que ha pasado menos de "min" minutos
    deleted = 0
    aeliminar = []
    minimum_Time = timedelta(minutes=min)
    for index, row in df.iterrows():
        if index < len(df)-1:
            if row['MoldeID'] == df.at[index+1, 'MoldeID'] and df.at[index+1, 'created_at']-row['created_at'] < minimum_Time:
                #print("elimina: "+str(index)+" Hora next: "+str(df.at[index+1, 'created_at'])+" Hora act "+str(row['created_at']))
                aeliminar.append(index)
                deleted += 1

    #df = df.drop(df.index[aeliminar])
    #print('Eliminaods: '+str(aeliminar))
    return df.drop(aeliminar)

def cuentaPiezas(df):
    piezas = 0 
    df['piezas'] = 0
    #print(df)
    prev = 0
    moldeAct = 0
    for index, row in df.iterrows():
        if row['MoldeID']!=moldeAct:
            piezas = 0
            
        moldeAct = row['MoldeID']
        if prev == 1:
            prev = 0
        elif row['EstadoID'] == 12:
            piezas += 1
            df.at[index, 'piezas'] = piezas
            prev = 1
        elif row['EstadoID'] == 1 and piezas > 0:
            #print("Guardamos: "+str(piezas))
            df.at[index, 'piezas'] = piezas
            piezas = 0
    df = df.sort_values(by=['id'])
    return df


def exporta(df):
    export = df.loc[df['EstadoID']==1]
    export = export.reset_index(drop=True)
    export.to_csv(inputdir+'limpiezas.csv')

    export

inputdir = "/Users/rsanchis/datarus/www/master/practicas-arf/Moldes_MantenimientoPreventivo/data/"

#Carrega dels CSV
df = loadfiles()
#Calcula temps amb un delta des del inici (01/01/16)
df2 = CalcularLapso(df)
#Elimina estados de duración demasiado corta
df3 = df2.sort_values(by=['MoldeID', 'id'])
min = 30
df4 = eliminaEstadosCortos(df3, min)
#Reordena
df5 = cuentaPiezas(df4)
df5 = df5.sort_values(by=['id'])
df5 = df5.reset_index(drop=True)
exporta(df5)

df5