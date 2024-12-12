import pandas as pd 
import matplotlib.pyplot as plt 


df = pd.read_csv('indices_transp.csv', sep=";")
lista = df['Índice'].unique()
lista_unica = df[df['Índice'] == 'MXLA Index']
plt.figure(figsize=(10,8))
plt.plot(lista['Data'], lista['Valor'])
plt.savefig('teste.png')