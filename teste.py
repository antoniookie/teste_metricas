import pandas as pd

# Abrir o arquivo Excel
sheets = pd.read_excel('index_bbg.xlsx', sheet_name=None)

# Lista para armazenar os dataframes consolidados
dataframes = []

# Iterar por cada aba (sheet) do Excel
for sheet_name, sheet_data in sheets.items():
    # Identificar as tabelas dentro da aba
    # A estrutura é de colunas consecutivas: "Data", "Valor" (para cada índice)

    for i in range(0, len(sheet_data.columns), 2):
        # Extrair as colunas da Data e Valor do índice
        col_date = sheet_data.columns[i]
        col_value = sheet_data.columns[i + 1]

        # Criar um dataframe para a tabela atual
        df = sheet_data[[col_date, col_value]].dropna()
        df.columns = ['Data', 'Valor']  # Renomear colunas para consistência
        # Adicionar a coluna "Índice" com o nome da coluna de valores
        df['Índice'] = col_value

        # Adicionar o dataframe à lista
        dataframes.append(df)

# Concatenar todos os dataframes em um único dataframe consolidado
df_consolidado = pd.concat(dataframes, ignore_index=True)

# Exportar o dataframe consolidado para um novo arquivo Excel
output_path = 'consolidated_indices3.xlsx'
df_consolidado.to_excel(output_path, index=False)

print(f"Consolidação concluída! Os dados foram salvos em '{output_path}'.")