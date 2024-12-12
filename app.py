import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging

# Configurar o layout para ser amplo
st.set_page_config(layout="wide")

# Configuração para suprimir logs de depuração do 'websockets'
logging.getLogger('websockets').setLevel(logging.WARNING)

# Função para calcular a tabela de rentabilidade mensal
def generate_rentability_table(data, fund_name):
    data = data[data['Fundo'] == fund_name].copy()
    data['Data'] = pd.to_datetime(data['Data'], format='%d/%m/%Y', errors='coerce')
    data.dropna(subset=['Data'], inplace=True)
    data.set_index('Data', inplace=True)

    monthly_data = data.resample('M').last()
    monthly_data['Rentability'] = monthly_data['Cota'].pct_change() * 100

    month_abbr_pt = {
        1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
    }


    monthly_data.reset_index(inplace=True)
    monthly_data['Year'] = monthly_data['Data'].dt.year
    monthly_data['Month'] = monthly_data['Data'].dt.month.map(month_abbr_pt)

    pivot_table = monthly_data.pivot(index='Year', columns='Month', values='Rentability')

    ordered_months = list(month_abbr_pt.values())
    pivot_table = pivot_table.reindex(columns=ordered_months)

    return pivot_table

# Função para calcular a volatilidade
def calculate_volatility(data, column='Cota', window=126):
    data['Daily_Return'] = data[column].pct_change()  
    data['Volatility_12m'] = data['Daily_Return'].rolling(window).std() * np.sqrt(252) 
    return data.dropna()

# Função para calcular rentabilidade por períodos
# Função para calcular a rentabilidade
def calculate_rentability(data, fund_name):
    fund_data = data[data['Fundo'] == fund_name].copy()

    if fund_data.empty:
        logging.warning(f"Nenhum dado encontrado para o fundo: {fund_name}")
        return pd.DataFrame()

    fund_data['Data'] = pd.to_datetime(fund_data['Data'], format='%d/%m/%Y', errors='coerce')
    fund_data.dropna(subset=['Data'], inplace=True)
    fund_data.set_index('Data', inplace=True)

    # Resample para dados mensais
    fund_data = fund_data.resample('M').last()
    last_date = fund_data.index.max()
    end_cota = fund_data.loc[last_date, 'Cota']
    end_benchmark = fund_data.loc[last_date, 'Benchmark']

    periods = {'1 Mês': 1, '3 Meses': 3, '6 Meses': 6, '12 Meses': 12}
    results = {'Período': [], 'Fundo': [], 'Benchmark': [], 'Alpha': []}

    for period_name, months in periods.items():
        if len(fund_data) > months:
            start_cota = fund_data['Cota'].iloc[-(months + 1)]
            start_benchmark = fund_data['Benchmark'].iloc[-(months + 1)]

            fund_return = ((end_cota / start_cota) - 1) * 100
            benchmark_return = ((end_benchmark / start_benchmark) - 1) * 100
            alpha = fund_return - benchmark_return

            results['Período'].append(period_name)
            results['Fundo'].append(f'{fund_return:.2f}%')
            results['Benchmark'].append(f'{benchmark_return:.2f}%')
            results['Alpha'].append(f'{alpha:.2f}%')
        else:
            results['Período'].append(period_name)
            results['Fundo'].append('N/A')
            results['Benchmark'].append('N/A')
            results['Alpha'].append('N/A')

    return pd.DataFrame(results)
# Função para calcular os retornos móveis
def calculate_rolling_returns(data, column='Cota'):
    """
    Calcula os retornos móveis de 12m, 24m e 36m, ajustando para fundos com menos de 36 meses de dados.
    """
    data['Retorno 12m'] = ((data[column] / data[column].shift(252)) - 1) * 100  # 12 meses
    data['Retorno 24m'] = ((data[column] / data[column].shift(504)) - 1) * 100  # 24 meses
    data['Retorno 24m'] = ((1 + data['Retorno 24m'] / 100) ** (252 / 504) - 1) * 100
    data['Retorno 36m'] = ((data[column] / data[column].shift(756)) - 1) * 100  # 36 meses
    data['Retorno 36m'] = ((1 + data['Retorno 36m'] / 100) ** (252 / 756) - 1) * 100

    # Filtrar para períodos disponíveis
    if data['Retorno 12m'].isna().all():
        data['Retorno 12m'] = None  # Caso não haja dados para 12 meses
    if data['Retorno 24m'].isna().all():
        data['Retorno 24m'] = None  # Caso não haja dados para 24 meses
    if data['Retorno 36m'].isna().all():
        data['Retorno 36m'] = None  # Caso não haja dados para 36 meses

    return data


def highlight_values(val):
    """Aplica cores aos valores: verde para positivo, vermelho para negativo, cinza para zero."""
    if isinstance(val, str):
        try:
            val = float(val.replace('%', '')) if '%' in val else val
        except:
            return 'color: black'
    if isinstance(val, float) or isinstance(val, int):
        if val > 0:
            color = 'green'
        elif val < 0:
            color = 'red'
        else:
            color = 'gray'
    else:
        color = 'black'
    return f'color: {color}'

def calculate_correlation_matrix(fund_cota, indices_wide):
    """
    Calcula a correlação entre a Cota do fundo e cada índice.

    Parameters:
    - fund_cota: Series com a Cota do fundo, indexada por Data.
    - indices_wide: DataFrame com os índices, cada coluna sendo um índice, indexado por Data.

    Returns:
    - Series com a correlação entre Cota e cada índice.
    """
    combined_data = pd.concat([fund_cota, indices_wide], axis=1).dropna()

    # st.write("### Amostra dos Dados Combinados para Correlação:")
    # st.write(combined_data.head())
    # st.write(f"**Quantidade de Linhas Após Junção:** {combined_data.shape[0]}")

    if combined_data.empty:
        st.warning("Não há dados sobrepostos entre o fundo selecionado e os índices. Verifique os intervalos de datas.")

    correlations = combined_data.corr().loc['Cota'].drop('Cota')

    return correlations


file_path = 'HistCotas_29-nov-2024.csv'
try:
    data = pd.read_csv(file_path, sep=';')
except FileNotFoundError:
    st.error(f"O arquivo '{file_path}' não foi encontrado. Por favor, verifique o caminho e tente novamente.")
    st.stop()

data_cleaned = data[['Data', 'Fundo', 'Cota', 'Benchmark', 'PL']].copy()
data_cleaned['Data'] = pd.to_datetime(data_cleaned['Data'], format='%d/%m/%Y', errors='coerce')
data_cleaned.dropna(subset=['Data'], inplace=True) 

for col in ['Cota', 'Benchmark', 'PL']:
    data_cleaned[col] = data_cleaned[col].str.replace(',', '.')
    data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')
data_cleaned.dropna(subset=['Cota', 'Benchmark', 'PL'], inplace=True)  

index_file_path = 'indices_transp.csv'
try:
    index_data = pd.read_csv(index_file_path, sep=';')
except FileNotFoundError:
    st.error(f"O arquivo '{index_file_path}' não foi encontrado. Por favor, verifique o caminho e tente novamente.")
    st.stop()

# Verificar se há as colunas necessárias
required_columns = ['Data', 'Valor', 'Índice']
if all(col in index_data.columns for col in required_columns):
    # Converter 'Data' para datetime
    index_data['Data'] = pd.to_datetime(index_data['Data'], format='%d/%m/%Y', errors='coerce')
    index_data.dropna(subset=['Data'], inplace=True)  # Remove linhas com datas inválidas

    # Limpar nomes dos índices removendo espaços extras
    index_data['Índice'] = index_data['Índice'].str.strip()

    # Substituir vírgulas por pontos e converter 'Valor' para float
    index_data['Valor'] = index_data['Valor'].str.replace(',', '.')
    index_data['Valor'] = pd.to_numeric(index_data['Valor'], errors='coerce')
    index_data.dropna(subset=['Valor'], inplace=True)  # Remove linhas com valores inválidos

    # Pivotar o dataframe para formato largo
    indices_wide = index_data.pivot_table(index='Data', columns='Índice', values='Valor', aggfunc='last')

    # Remover possíveis índices duplicados, mantendo a última ocorrência
    indices_wide = indices_wide[~indices_wide.index.duplicated(keep='last')]

    # Remover colunas com apenas NaN
    indices_wide.dropna(axis=1, how='all', inplace=True)

    # Selecionar apenas colunas numéricas (já são numéricas, mas garante)
    indices_wide = indices_wide.select_dtypes(include=[np.number])

    # Adicionar informações de depuração
    # st.write("### Amostra dos Índices Pivotados:")
    # st.write(indices_wide.head())

    # st.write("### Tipos de Dados das Colunas dos Índices:")
    # st.write(indices_wide.dtypes)


    min_date_fund = data_cleaned['Data'].min()
    max_date_fund = data_cleaned['Data'].max()
    # st.write(f"**Intervalo de Datas do Fundo:** {min_date_fund.date()} até {max_date_fund.date()}")

    min_date_indices = indices_wide.index.min()
    max_date_indices = indices_wide.index.max()
    # st.write(f"**Intervalo de Datas dos Índices:** {min_date_indices.date()} até {max_date_indices.date()}")


    index_numeric = indices_wide.select_dtypes(include=[np.number])
    non_numeric_cols = indices_wide.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        st.warning(f"As seguintes colunas no 'indices_transp.csv' não são numéricas e serão ignoradas na matriz de correlação: {non_numeric_cols}")
        indices_wide = index_numeric
else:
    st.error("O arquivo 'indices_transp.csv' deve conter as colunas 'Data', 'Valor' e 'Índice'.")
    st.stop()


try:
    st.image("logo.svg", width=200)  
except FileNotFoundError:
    st.warning("Arquivo de logo 'logo.svg' não encontrado. Continuando sem exibir a logo.")

st.title("Gama Investimentos")
st.subheader("Dashboard de Fundos")

fundos_disponiveis = data_cleaned['Fundo'].unique()
fundo_selecionado = st.selectbox('Selecione o fundo:', fundos_disponiveis)

dados_fundo = data_cleaned[data_cleaned['Fundo'] == fundo_selecionado].copy()

if dados_fundo.empty:
    st.write("Nenhum dado disponível para o fundo selecionado.")
else:

    dados_fundo['Data'] = pd.to_datetime(dados_fundo['Data'], format='%d/%m/%Y', errors='coerce')
    dados_fundo.dropna(subset=['Data'], inplace=True)  
    dados_fundo.set_index('Data', inplace=True)

    dados_fundo = dados_fundo[~dados_fundo.index.duplicated(keep='last')]

    # st.write("### Amostra dos Dados do Fundo Selecionado:")
    # st.write(dados_fundo.head())
    # st.write("### Tipos de Dados das Colunas do Fundo:")
    # st.write(dados_fundo.dtypes)


    # min_date_selected_fund = dados_fundo.index.min()
    # max_date_selected_fund = dados_fundo.index.max()
    # st.write(f"**Intervalo de Datas do Fundo Selecionado:** {min_date_selected_fund.date()} até {max_date_selected_fund.date()}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Gráfico de Cota e Benchmark - {fundo_selecionado}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dados_fundo.index,
            y=dados_fundo['Cota'],
            mode='lines',
            name='Cota',
            line=dict(color='#b7d733')  
        ))
        fig.add_trace(go.Scatter(
            x=dados_fundo.index,
            y=dados_fundo['Benchmark'],
            mode='lines',
            name='Benchmark',
            line=dict(color='#1f77b4') 
        ))
        fig.update_layout(
            title=f'Cota e Benchmark - {fundo_selecionado}',
            xaxis_title='Data',
            yaxis_title='Valor',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Gráfico de Retornos Anuais - {fundo_selecionado}")

        dados_fundo['Year'] = dados_fundo.index.year
        annual_returns_fundo = dados_fundo.groupby('Year')['Cota'].last().pct_change() * 100
        annual_returns_benchmark = dados_fundo.groupby('Year')['Benchmark'].last().pct_change() * 100
        annual_returns_fundo = annual_returns_fundo.dropna()
        annual_returns_benchmark = annual_returns_benchmark.dropna()
        annual_returns_combined = pd.DataFrame({
            'Ano': annual_returns_fundo.index,
            'Fundo': annual_returns_fundo.values,
            'Benchmark': annual_returns_benchmark.reindex(annual_returns_fundo.index).values
        })
        fig_annual_returns = go.Figure()

        fig_annual_returns.add_trace(go.Bar(
            x=annual_returns_combined['Ano'],
            y=annual_returns_combined['Fundo'],
            name='Fundo',
            text=[f"{val:.2f}%" for val in annual_returns_combined['Fundo']],
            textposition='auto',
            marker_color='green'
        ))

        fig_annual_returns.add_trace(go.Bar(
            x=annual_returns_combined['Ano'],
            y=annual_returns_combined['Benchmark'],
            name='Benchmark',
            text=[f"{val:.2f}%" for val in annual_returns_combined['Benchmark']],
            textposition='auto',
            marker_color='#1f77b4' 
        ))

        fig_annual_returns.update_layout(
            title="Retornos Anuais (%) - Fundo vs Benchmark",
            xaxis_title="Ano",
            yaxis_title="Retorno (%)",
            template="plotly_white",
            xaxis=dict(
                tickmode='linear',
                tick0=annual_returns_combined['Ano'].min(),
                dtick=1  
            ),
            yaxis=dict(tickformat=".2f"),
            barmode='group',   
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5)  
        )

        st.plotly_chart(fig_annual_returns, use_container_width=True)



        st.subheader(f"Gráfico de Volatilidade - {fundo_selecionado}")
        dados_vol = calculate_volatility(dados_fundo.copy(), column='Cota')  
        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=dados_vol.index,  
                y=dados_vol['Volatility_12m'],  
                mode='lines',
                name='Volatilidade - Rolling 6 meses',
                line=dict(color='#BED754')  
            )
        )
        fig_vol.update_layout(
            xaxis_title='Data',
            yaxis_title='Volatilidade 6 meses (%)',
            font=dict(size=10),
            yaxis_tickformat='.2f%',
            yaxis=dict(range=[0, None]),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.10,
                xanchor='center',
                x=0.5,
                font=dict(size=10)
            ),
            template='plotly_white',
            autosize=True
        )
        st.plotly_chart(fig_vol, use_container_width=True)




    with col2:
        st.subheader(f"Gráfico de Drawdown - {fundo_selecionado}")
        dados_fundo['Cota_Maxima'] = dados_fundo['Cota'].cummax()
        dados_fundo['Drawdown (%)'] = (dados_fundo['Cota'] - dados_fundo['Cota_Maxima']) / dados_fundo['Cota_Maxima'] * 100
        fig_ddwn = go.Figure()
        fig_ddwn.add_trace(go.Scatter(
            x=dados_fundo.index,
            y=dados_fundo['Drawdown (%)'],
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            fillcolor='rgba(183, 215, 51, 0.3)',  
            line=dict(color='#b7d733')  
        ))
        fig_ddwn.update_layout(
            title=f'Drawdown - {fundo_selecionado}',
            xaxis_title='Data',
            yaxis_title='Drawdown (%)',
            template='plotly_white'
        )
        st.plotly_chart(fig_ddwn, use_container_width=True)

        st.subheader("Tabelas de Rentabilidade")
        st.write("Tabela de Rentabilidade Mensal (%)")
        tabela_mensal = generate_rentability_table(data_cleaned, fundo_selecionado)
        styled_tabela_mensal = tabela_mensal.style.format("{:.2f}%").applymap(highlight_values)
        st.dataframe(styled_tabela_mensal, use_container_width=True)

        st.write("Tabela de Rentabilidade por Períodos")
        tabela_periodos = calculate_rentability(data_cleaned, fundo_selecionado)

        if tabela_periodos.empty:
            st.write("Nenhum dado disponível para calcular a rentabilidade por períodos.")
        else:
    
            styled_tabela_periodos = tabela_periodos.style.format(precision=2).applymap(
                highlight_values, subset=["Fundo", "Benchmark", "Alpha"]
            )
            st.dataframe(styled_tabela_periodos, use_container_width=True)
    st.subheader(f"Gráfico de Retornos Móveis - {fundo_selecionado}")
    rolling_data = calculate_rolling_returns(dados_fundo.copy(), column='Cota')

    fig_rolling = go.Figure()

    if not rolling_data['Retorno 12m'].isna().all():
        fig_rolling.add_trace(go.Scatter(
            x=rolling_data.index,
            y=rolling_data['Retorno 12m'],
            mode='lines',
            name='Retorno 12m',
            line=dict(color='blue')
        ))

    if not rolling_data['Retorno 24m'].isna().all():
        fig_rolling.add_trace(go.Scatter(
            x=rolling_data.index,
            y=rolling_data['Retorno 24m'],
            mode='lines',
            name='Retorno 24m',
            line=dict(color='green')
        ))

    if not rolling_data['Retorno 36m'].isna().all():
        fig_rolling.add_trace(go.Scatter(
            x=rolling_data.index,
            y=rolling_data['Retorno 36m'],
            mode='lines',
            name='Retorno 36m',
            line=dict(color='red')
        ))

    fig_rolling.update_layout(
        title=f'Retorno Móvel - {fundo_selecionado}',
        xaxis_title='Data',
        yaxis_title='Retorno (%)',
        template='plotly_white'
    )
    st.plotly_chart(fig_rolling, use_container_width=True)


    try:
        # Extrair a série de Cota do fundo
        fund_cota = dados_fundo['Cota']

        # Calcular a correlação
        correlation_series = calculate_correlation_matrix(fund_cota, indices_wide)

        # Garantir que a série é numérica
        correlation_series = pd.to_numeric(correlation_series, errors='coerce').dropna()

    except Exception as e:
        st.error(f"Erro ao calcular a matriz de correlação: {e}")
        st.stop()

    # Verificar se a correlação está vazia
    if correlation_series.empty:
        st.warning("A matriz de correlação está vazia. Verifique se há sobreposição nas datas entre o fundo e os índices.")
    else:
        # Transformar a série de correlação em DataFrame para melhor exibição
        correlation_df = correlation_series.reset_index()
        correlation_df.columns = ['Índice', 'Correlação com Cota']

        # # Verificar tipos de dados na correlação_df
        # st.write("### Tipos de Dados na Correlação:")
        # st.write(correlation_df.dtypes)

        # Garantir que a coluna 'Correlação com Cota' é float
        correlation_df['Correlação com Cota'] = pd.to_numeric(correlation_df['Correlação com Cota'], errors='coerce')
        correlation_df.dropna(subset=['Correlação com Cota'], inplace=True)

        # # Verificar se ainda há dados após a conversão
        # st.write("### Valores Únicos em 'Correlação com Cota' Após Conversão:")
        # st.write(correlation_df['Correlação com Cota'].unique())

        if correlation_df.empty:
            st.warning("A matriz de correlação está vazia após a conversão para numérico.")
        else:
            # # Exibir a tabela sem estilização para verificar
            # st.write("### Tabela de Correlação Sem Estilização:")
            # st.dataframe(correlation_df, use_container_width=True)

            # Estilizar a tabela com formatação específica para a coluna 'Correlação com Cota'
            styled_corr_df = correlation_df.style.background_gradient(cmap="coolwarm").format({"Correlação com Cota": "{:.2f}"})
            st.dataframe(styled_corr_df, use_container_width=True)

            # Opcional: Gráfico de barras das correlações
            st.subheader("Gráfico de Correlação")
            fig_corr = go.Figure(data=[
                go.Bar(
                    x=correlation_df['Índice'],
                    y=correlation_df['Correlação com Cota'],
                    marker_color=correlation_df['Correlação com Cota'].apply(
                        lambda x: 'green' if x > 0 else 'red'
                    )
                )
            ])
            fig_corr.update_layout(
                title="Correlação entre Cota e Índices",
                xaxis_title="Índice",
                yaxis_title="Correlação",
                template='plotly_white',
                yaxis=dict(range=[-1, 1])
            )
            st.plotly_chart(fig_corr, use_container_width=True)


            # st.subheader("Gráficos dos Índices e Fundo Selecionado")

            combined_data = dados_fundo.join(indices_wide, how='inner').dropna()

            # Selecionar apenas colunas numéricas para os gráficos
            combined_numeric = combined_data.select_dtypes(include=[np.number])

            # # Verificar se há colunas para plotar
            # if combined_numeric.empty:
            #     st.warning("Não há colunas numéricas disponíveis para plotar gráficos de índices.")
            # else:
            #     for column in combined_numeric.columns:
            #         st.write(f"Gráfico para: {column}")
            #         fig = go.Figure()
            #         fig.add_trace(go.Scatter(
            #             x=combined_numeric.index,
            #             y=combined_numeric[column],
            #             mode='lines',
            #             name=column
            #         ))
            #         fig.update_layout(
            #             title=f"Evolução de {column}",
            #             xaxis_title="Data",
            #             yaxis_title="Valor",
            #             template="plotly_white"
            #         )
            #         st.plotly_chart(fig, use_container_width=True)
