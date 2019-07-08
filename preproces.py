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

def horasRelativas(df):
    #Convierte las horas absolutas (desdeel inicio de los tiempos)
    #A relativas, horas desde la anterior limpieza
    df = df.sort_values(by=['MoldeID', 'id'])
    df['hora_diff'] = 0
    moldeAct = 0
    previo = 0

    for index, row in df.iterrows():
        if row['EstadoID'] == 1:
            if row['MoldeID']!=moldeAct:
                previo = row['hora']
                df.at[index, 'hora_diff'] = 0
                moldeAct = row['MoldeID']
            else:
                df.at[index, 'hora_diff'] = row['hora'] - previo
                previo = row['hora']

    df = df.sort_values(by=['id'])
    return df

def restaHorasFindeSemana(df):
    #Comprueba si entre la limpieza anterior y esta ha pasado algun fin de semana
    #De haberlo hecho resta 24h por fin de semana a la hora_diff
    df = df.sort_values(by=['MoldeID', 'id'])
    moldeAct = -1
    previo = 0

    for index, row in df.iterrows():
        if row['EstadoID'] == 1:
            if row['MoldeID'] != moldeAct:
                previo = row['created_at']
                moldeAct = row['MoldeID']
            else:
                daygenerator = (previo + timedelta(x + 1) for x in range((row['created_at'] - previo).days))
                fin_semana = sum(1 for day in daygenerator if day.weekday() >= 5)
                #print("A restar: "+str(fin_semana)+" dias en la limpieza han pasado "+str(row['hora_diff'])+" horas y hoy es: "+str(row['created_at']))
                #Restamos las horas trascurridas en fin de semana
                df.at[index, 'hora_diff'] = df.at[index, 'hora_diff'] - 24 * fin_semana
                previo = row['created_at']

    df = df.sort_values(by=['id'])
    return df

def cuentaPiezas(df):
    df = df.sort_values(by=['MoldeID', 'id'])
    piezas = 0 
    df['piezas'] = 0
    #print(df)
    prev = 0
    moldeAct = 0
    for index, row in df.iterrows():
        if row['MoldeID']!=moldeAct:
            piezas = 0
            prev = 0
            
        moldeAct = row['MoldeID']
        
        if row['EstadoID'] == 12:
            if prev == 1:
                prev = 0
            else:
                piezas += 1
                df.at[index, 'piezas'] = piezas
                prev = 1
        
        if row['EstadoID'] == 1:
            #print("Guardamos: "+str(piezas))
            df.at[index, 'piezas'] = piezas
            piezas = 0
    df = df.sort_values(by=['id'])
    return df


def loadInfoMoldes():
    moldes_info = pd.DataFrame()
    moldes_info = pd.read_csv(inputdir+'/moldes2.csv', sep=',', engine='python')
    return moldes_info


def addInfoMolde(df):
    m_info = loadInfoMoldes()

    df["extermo"] = np.nan
    df["demanda"] = np.nan
    df["reparaciones"] = np.nan

    for index, row in df.iterrows():
        molde = m_info.loc[m_info['id'] == row['MoldeID']]
        if molde.shape[0] == 1:
            df.at[index, 'extermo'] = molde['Externo']
            df.at[index, 'demanda'] = molde['MaxFabDia']
            df.at[index, 'reparaciones'] = molde['numReparaciones']
        else:
            print("Error al buscar el molde"+str(row['MoldeID']))
    return df


def exporta(df):
    export = df.loc[df['EstadoID']==1]
    export = export.reset_index(drop=True)
    export.to_csv(inputdir+'limpiezas.csv')

    export


inputdir = "/Users/rsanchis/datarus/www/master/practicas-arf/Moldes_MantenimientoPreventivo/data/"

#Carrega dels CSV
df = loadfiles()

#Carrega test
df = pd.read_csv(inputdir+'/test_set.csv', sep=',', engine='python')
df2 = pd.read_csv(inputdir+'/test_set2.csv', sep=',', engine='python')
df = df.append(df2)
df = df.reset_index(drop=True)

df
#Calcula temps amb un delta des del inici (01/01/16)
df2 = CalcularLapso(df)
#Elimina estados de duración demasiado corta
df3 = df2.sort_values(by=['MoldeID', 'id'])
min = 30
df4 = eliminaEstadosCortos(df3, min)
#Reordena
df5 = horasRelativas(df4)

df6 = restaHorasFindeSemana(df5)

df7 = cuentaPiezas(df6)

df8 = addInfoMolde(df7)

export = df8.loc[df8['EstadoID']==1]
export = export.reset_index(drop=True)

export
export.to_csv(inputdir+'test.csv')

df8
exporta(df8)


#Test

dftest = df7
df7.head()

df7["extermo"] = np.nan
df7["demanda"] = np.nan
df7["reparaciones"] = np.nan

dftest.at[349998, 'extermo'] = 1


m_info = loadInfoMoldes()

molde = m_info.loc[m_info['id'] == 666]
molde.shape[0]
