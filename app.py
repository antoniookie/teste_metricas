import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging


# Configurar o layout para ser amplo
st.set_page_config(layout="wide")

# Configuração para suprimir logs de depuração do 'websockets'
logging.getLogger('websockets').setLevel(logging.WARNING)

# =====================================================================
# Funções para a Seção de Fundos
# =====================================================================

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

def calculate_volatility(data, column='Cota', window=126):
    data['Daily_Return'] = data[column].pct_change()  
    data['Volatility_12m'] = data['Daily_Return'].rolling(window).std() * np.sqrt(252) 
    return data.dropna()

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
            results['Fundo'].append(fund_return)  # Armazenar como float
            results['Benchmark'].append(benchmark_return)  # Armazenar como float
            results['Alpha'].append(alpha)  # Armazenar como float
        else:
            results['Período'].append(period_name)
            results['Fundo'].append(np.nan)
            results['Benchmark'].append(np.nan)
            results['Alpha'].append(np.nan)

    return pd.DataFrame(results)

def calculate_rolling_returns(data, column='Cota'):
    """
    Calcula os retornos móveis de 12m, 24m e 36m, ajustando para fundos com menos de 36 meses de dados.
    Agora utilizando deslocamentos mensais ao invés de dias.
    """
    data['Retorno 12m'] = ((data[column] / data[column].shift(252)) - 1) * 100  # 12 meses
    data['Retorno 24m'] = ((data[column] / data[column].shift(504)) - 1) * 100  # 24 meses
    data['Retorno 36m'] = ((data[column] / data[column].shift(756)) - 1) * 100  # 36 meses

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

# =====================================================================
# Funções para a Seção de Análise de Índices Bloomberg
# =====================================================================

def calcular_retorno_acumulado(df):
    df['Retorno'] = df['Cotação'].pct_change()
    df['Retorno Acumulado'] = (1 + df['Retorno']).cumprod()
    return df

def calcular_volatilidade_indice(df, janela=21):
    df['Volatilidade'] = df['Retorno'].rolling(window=janela).std()
    return df

def calcular_correlacao(dfs):
    """
    Calcula a matriz de correlação baseada em retornos mensais dos índices.
    """
    monthly_returns = {}
    for df in dfs:
        if 'Índice' not in df.columns:
            continue  # Pula DataFrames sem a coluna 'Índice'
        if df.empty:
            continue  # Pula DataFrames vazios
        ticker = df['Índice'].iloc[0]
        # Resample para mês-end e pegar a última cotação do mês
        temp_df = df.set_index('Data')['Cotação'].resample('M').last()
        # Calcular retorno mensal
        temp_returns = temp_df.pct_change().dropna()
        monthly_returns[ticker] = temp_returns

    # Criar DataFrame com retornos mensais
    returns_df = pd.DataFrame(monthly_returns)

    if returns_df.empty:
        return pd.DataFrame()

    # Remover meses com dados faltantes
    returns_df = returns_df.dropna()

    # Calcular matriz de correlação
    correlacao = returns_df.corr()

    return correlacao

@st.cache_data
def carregar_dados(nome_arquivo):
    try:
        df = pd.read_csv(nome_arquivo, delimiter=';', parse_dates=['Data'], dayfirst=True)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo '{nome_arquivo}': {e}")
        return pd.DataFrame()
    
    if 'Cotação' in df.columns:
        df['Cotação'] = pd.to_numeric(df['Cotação'].astype(str).str.replace(',', '.'), errors='coerce')
    else:
        st.error(f"O arquivo '{nome_arquivo}' não contém a coluna 'Cotação'.")
        return pd.DataFrame()
    
    df = df.sort_values(by='Data')
    return df

# =====================================================================
# Funções para a Seção de Detalhes dos Fundos (Integração do Segundo Código)
# =====================================================================

def formatar_moeda(valor):
    """Formata o valor para o padrão de moeda brasileiro."""
    return 'R$ {:,.2f}'.format(valor).replace(',', 'X').replace('.', ',').replace('X', '.')

def getMainData(font, fund_name):
    """Extrai os dados principais do Excel para o fundo selecionado."""
    try:
        data = pd.read_excel(font, sheet_name="Infos Gerais")
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {font}")
        return None
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Excel: {e}")
        return None

    data_fund = data[data['Nome Fundo'] == fund_name]

    if data_fund.empty:
        st.warning(f"Nenhum fundo encontrado com o nome: {fund_name}")
        return None

    try:
        cnpj = data_fund['CNPJ/GIIN'].iloc[0]
        aplicacao = data_fund['Aplicação'].iloc[0]
        resgate = data_fund['Resgate'].iloc[0]
        aplicacao_inicial = formatar_moeda(data_fund['Aplicação Inicial'].iloc[0])
        movimentacao_adicional = formatar_moeda(data_fund['Movimentação Adicional'].iloc[0])
        publico_alvo = data_fund['Público Alvo'].iloc[0]
        adm = data_fund['Adm'].iloc[0]
        custodiante = data_fund['Custodiante'].iloc[0]
        cotizacao = data_fund['Cotização'].iloc[0]
        isin = data_fund['ISIN'].iloc[0]
        tipo_estrategia = data_fund['Estratégia'].iloc[0]
        hedge = data_fund['Hedge'].iloc[0]

        disclaimer = ''
        if 'Disclaimers' in data_fund.columns:
            disc = data_fund['Disclaimers'].iloc[0]
            disclaimer = '' if pd.isna(disc) else str(disc).strip()

        return {
            "CNPJ/GIIN": cnpj,
            "Aplicação": aplicacao,
            "Resgate": resgate,
            "Aplicação Inicial": aplicacao_inicial,
            "Movimentação Adicional": movimentacao_adicional,
            "Público Alvo": publico_alvo,
            "Administração": adm,
            "Custodiante": custodiante,
            "Cotização": cotizacao,
            "ISIN": isin,
            "Estratégia": tipo_estrategia,
            "Hedge": hedge,
            "Disclaimers": disclaimer
        }
    except Exception as e:
        st.error(f"Erro ao processar os dados do fundo: {e}")
        return None

# =====================================================================
# Sidebar para Navegação
# =====================================================================

# Tentar carregar a logo
try:
    st.sidebar.image("logo.svg", width=200)
except FileNotFoundError:
    st.sidebar.warning("Arquivo de logo 'logo.svg' não encontrado. Continuando sem exibir a logo.")

menu = st.sidebar.radio("Navegação", ["Dashboard de Fundos", "Análise de Índices Bloomberg"])

# =====================================================================
# Seção: Dashboard de Fundos
# =====================================================================

if menu == "Dashboard de Fundos":
    st.title("Gama Investimentos")
    st.subheader("Dashboard de Fundos")

    file_path = 'HistCotas_29-nov-2024.csv'
    try:
        data = pd.read_csv(file_path, sep=';')
    except FileNotFoundError:
        st.error(f"O arquivo '{file_path}' não foi encontrado. Por favor, verifique o caminho e tente novamente.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao ler o arquivo '{file_path}': {e}")
        st.stop()

    required_cols = ['Data', 'Fundo', 'Cota', 'Benchmark', 'PL']
    if not all(c in data.columns for c in required_cols):
        st.error(f"O arquivo '{file_path}' deve conter as colunas: {', '.join(required_cols)}.")
        st.stop()

    data_cleaned = data[['Data', 'Fundo', 'Cota', 'Benchmark', 'PL']].copy()
    data_cleaned['Data'] = pd.to_datetime(data_cleaned['Data'], format='%d/%m/%Y', errors='coerce')
    data_cleaned.dropna(subset=['Data'], inplace=True) 

    for col in ['Cota', 'Benchmark', 'PL']:
        data_cleaned[col] = data_cleaned[col].astype(str).str.replace(',', '.')
        data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')
    data_cleaned.dropna(subset=['Cota', 'Benchmark', 'PL'], inplace=True)

    try:
        st.image("logo.svg", width=200)  
    except FileNotFoundError:
        st.warning("Arquivo de logo 'logo.svg' não foi encontrado. Continuando sem exibir a logo.")

    fundos_disponiveis = data_cleaned['Fundo'].unique()
    fundo_selecionado = st.selectbox('Selecione o fundo:', fundos_disponiveis)

    # Armazenar o fundo selecionado no session_state
    st.session_state.fundo_selecionado = fundo_selecionado

    dados_fundo = data_cleaned[data_cleaned['Fundo'] == fundo_selecionado].copy()

    if dados_fundo.empty:
        st.write("Nenhum dado disponível para o fundo selecionado.")
    else:
        dados_fundo['Data'] = pd.to_datetime(dados_fundo['Data'], format='%d/%m/%Y', errors='coerce')
        dados_fundo.dropna(subset=['Data'], inplace=True)  
        dados_fundo.set_index('Data', inplace=True)
        dados_fundo = dados_fundo[~dados_fundo.index.duplicated(keep='last')]

        # =====================================================================
        # Seção Adicional: Detalhes do Fundo (Integração do Segundo Código)
        # =====================================================================

        st.markdown("---")
        st.subheader("Detalhes do Fundo")

        # Caminho fixo para o arquivo Excel
        fonte_excel = r"Taxas Fundos.xlsx"
        
        # Carregar e exibir os detalhes do fundo
        detalhes_fundo = getMainData(font=fonte_excel, fund_name=fundo_selecionado)
        if detalhes_fundo:
            st.header(f"Detalhes do Fundo: {fundo_selecionado}")
            # Exibir informações detalhadas em colunas
            st.subheader("Informações Detalhadas")
            col1_det, col2_det = st.columns(2)

            with col1_det:
                st.markdown(f"**CNPJ/GIIN:** {detalhes_fundo['CNPJ/GIIN']}")
                st.markdown(f"**Aplicação:** {detalhes_fundo['Aplicação']}")
                st.markdown(f"**Resgate:** {detalhes_fundo['Resgate']}")
                st.markdown(f"**Aplicação Inicial:** {detalhes_fundo['Aplicação Inicial']}")
                st.markdown(f"**Movimentação Adicional:** {detalhes_fundo['Movimentação Adicional']}")

            with col2_det:
                st.markdown(f"**Público Alvo:** {detalhes_fundo['Público Alvo']}")
                st.markdown(f"**Administração:** {detalhes_fundo['Administração']}")
                st.markdown(f"**Custodiante:** {detalhes_fundo['Custodiante']}")
                st.markdown(f"**Cotização:** {detalhes_fundo['Cotização']}")
                st.markdown(f"**ISIN:** {detalhes_fundo['ISIN']}")
                st.markdown(f"**Estratégia:** {detalhes_fundo['Estratégia']}")
                st.markdown(f"**Hedge:** {detalhes_fundo['Hedge']}")

            if detalhes_fundo['Disclaimers']:
                st.warning(f"**Disclaimers:** {detalhes_fundo['Disclaimers']}")

        # =====================================================================
        # Gráficos e Tabelas Financeiras
        # =====================================================================
        
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
                yaxis_title='Rentabilidade',
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
                    name='Volatilidade - Rolling 12 meses',
                    line=dict(color='#BED754')  
                )
            )
            fig_vol.update_layout(
                xaxis_title='Data',
                yaxis_title='Volatilidade 12 meses (%)',
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
                tabela_periodos[['Fundo', 'Benchmark', 'Alpha']] = tabela_periodos[['Fundo', 'Benchmark', 'Alpha']].apply(pd.to_numeric, errors='coerce')

            # Adicionando o Gráfico de PL do Fundo
            st.subheader(f"PL do Fundo - {fundo_selecionado}")
            fig_pl = go.Figure()
            fig_pl.add_trace(
                go.Scatter(
                    x=dados_fundo.index,  
                    y=dados_fundo['PL'],  
                    mode='lines',
                    name='PL do Fundo',
                    line=dict(color='#BED754'),
                    fill='tozeroy'  
                )
            )
            fig_pl.update_layout(
                title=f'Patrimônio Líquido (PL) - {fundo_selecionado}',
                xaxis_title='Data',
                yaxis_title='Patrimônio Líquido (PL) em R$',
                template='plotly_white'
            )
            st.plotly_chart(fig_pl, use_container_width=True)

        # Gráfico de Retornos Móveis fora das colunas
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

# =====================================================================
# Seção: Análise de Índices Bloomberg
# =====================================================================

elif menu == "Análise de Índices Bloomberg":
    st.title("Análise de Índices Bloomberg")
    st.sidebar.title("Configurações de Índices")

    # Caminho fixo para o arquivo CSV
    arquivo = 'indices_bbg.csv'

    if arquivo:
        df = carregar_dados(arquivo)

        # Verificar se o fundo foi selecionado na Dashboard de Fundos
        if 'fundo_selecionado' not in st.session_state:
            st.warning("Por favor, selecione um fundo na seção 'Dashboard de Fundos' antes de realizar a análise de índices.")
            st.stop()

        fundo_selecionado = st.session_state.fundo_selecionado

        if df.empty:
            st.warning("O arquivo 'indices_bbg.csv' está vazio ou não foi lido corretamente.")
        else:
            # Verificar se a coluna 'Índice' existe
            if 'Índice' not in df.columns:
                st.error("A coluna 'Índice' não está presente em 'indices_bbg.csv'. Verifique os dados.")
                st.stop()

            tickers = df['Índice'].unique()
            dfs_por_ticker = [df[df['Índice'] == ticker].copy() for ticker in tickers]

            if len(tickers) == 0:
                st.warning("Nenhum índice encontrado no arquivo CSV.")
            else:
                st.sidebar.subheader("Selecione o índice para análise")
                ticker_selecionado = st.sidebar.selectbox("Índice", tickers)

                # Dados do índice selecionado
                df_selecionado = next((d for d in dfs_por_ticker if d['Índice'].iloc[0] == ticker_selecionado), None)
                if df_selecionado is not None and not df_selecionado.empty:
                    df_selecionado = calcular_retorno_acumulado(df_selecionado)
                    df_selecionado = calcular_volatilidade_indice(df_selecionado)

                    # Gráfico de retorno acumulado usando Plotly
                    st.subheader(f"Retorno Acumulado: {ticker_selecionado}")
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=df_selecionado['Data'],
                        y=df_selecionado['Retorno Acumulado'],
                        mode='lines',
                        name='Retorno Acumulado'
                    ))
                    fig1.update_layout(
                        title="Retorno Acumulado",
                        xaxis_title="Data",
                        yaxis_title="Retorno Acumulado",
                        template='plotly_white'
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                    # Gráfico de volatilidade usando Plotly
                    st.subheader(f"Volatilidade: {ticker_selecionado}")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=df_selecionado['Data'],
                        y=df_selecionado['Volatilidade'],
                        mode='lines',
                        name='Volatilidade (21 dias)',
                        line=dict(color='orange')
                    ))
                    fig2.update_layout(
                        title="Volatilidade",
                        xaxis_title="Data",
                        yaxis_title="Volatilidade",
                        template='plotly_white'
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    # Comparação de Retornos Mensais com o Fundo Selecionado
                    st.subheader("Comparação de Retornos Mensais com o Fundo Selecionado")
                    file_path_fundo = 'HistCotas_29-nov-2024.csv'
                    try:
                        data_fundo = pd.read_csv(file_path_fundo, sep=';')
                    except FileNotFoundError:
                        st.error(f"O arquivo '{file_path_fundo}' não foi encontrado. Por favor, verifique o caminho e tente novamente.")
                        st.stop()
                    except Exception as e:
                        st.error(f"Erro ao ler o arquivo '{file_path_fundo}': {e}")
                        st.stop()

                    required_columns_fundo = ['Data', 'Fundo', 'CNPJ', 'Cota', 'Benchmark', 'PL']
                    if not all(col in data_fundo.columns for col in required_columns_fundo):
                        st.error(f"O arquivo '{file_path_fundo}' deve conter as colunas: {', '.join(required_columns_fundo)}.")
                        st.stop()

                    fund_data_cleaned = data_fundo[['Data', 'Fundo', 'Cota']].copy()
                    fund_data_cleaned['Data'] = pd.to_datetime(fund_data_cleaned['Data'], format='%d/%m/%Y', errors='coerce')
                    fund_data_cleaned.dropna(subset=['Data'], inplace=True)
                    fund_data_cleaned['Cota'] = fund_data_cleaned['Cota'].astype(str).str.replace(',', '.')
                    fund_data_cleaned['Cota'] = pd.to_numeric(fund_data_cleaned['Cota'], errors='coerce')
                    fund_data_cleaned.dropna(subset=['Cota'], inplace=True)

                    dados_fundo_cleaned = fund_data_cleaned[fund_data_cleaned['Fundo'] == fundo_selecionado].copy()

                    if dados_fundo_cleaned.empty:
                        st.write("Nenhum dado disponível para o fundo selecionado.")
                        st.stop()

                    dados_fundo_cleaned.set_index('Data', inplace=True)
                    dados_fundo_cleaned = dados_fundo_cleaned[~dados_fundo_cleaned.index.duplicated(keep='last')]
                    dados_fundo_cleaned = dados_fundo_cleaned.resample('M').last()
                    dados_fundo_cleaned['Retorno Mensal Fundo'] = dados_fundo_cleaned['Cota'].pct_change() * 100
                    dados_fundo_cleaned = dados_fundo_cleaned.dropna()

                    # Calcular retornos mensais do índice selecionado
                    df_selecionado_resampled = df_selecionado.copy()
                    df_selecionado_resampled.set_index('Data', inplace=True)
                    df_selecionado_resampled = df_selecionado_resampled.resample('M').last()
                    df_selecionado_resampled['Retorno Mensal'] = df_selecionado_resampled['Cotação'].pct_change() * 100
                    df_selecionado_resampled = df_selecionado_resampled.dropna()

                    # Mesclar os retornos mensais do índice e do fundo
                    comparacao = pd.merge(
                        dados_fundo_cleaned[['Retorno Mensal Fundo']], 
                        df_selecionado_resampled[['Retorno Mensal']], 
                        left_index=True, 
                        right_index=True, 
                        how='inner',
                        suffixes=(f'_{fundo_selecionado}', f'_{ticker_selecionado}')
                    )

                    comparacao = comparacao.rename(columns={
                        'Retorno Mensal Fundo': 'Retorno Mensal Fundo (%)',
                        'Retorno Mensal': f'Retorno Mensal {ticker_selecionado} (%)'
                    })

                    comparacao['Alpha'] = comparacao['Retorno Mensal Fundo (%)'] - comparacao[f'Retorno Mensal {ticker_selecionado} (%)']

                    styled_comparacao = comparacao.style.format("{:.2f}%").applymap(highlight_values)
                    st.dataframe(styled_comparacao, use_container_width=True)

                    # =====================================================================
                    # Gráfico de Retorno Acumulado Normalizado (Fundo vs Índice Selecionado)
                    # =====================================================================
                    st.subheader("Retorno Acumulado Normalizado (Fundo vs Índice Selecionado)")
                    
                    # Selecionar datas comuns
                    common_dates = dados_fundo_cleaned.index.intersection(df_selecionado_resampled.index)
                    fund_returns = dados_fundo_cleaned['Retorno Mensal Fundo'].loc[common_dates] / 100.0
                    index_returns = df_selecionado_resampled['Retorno Mensal'].loc[common_dates] / 100.0

                    # Calcular retorno acumulado normalizado
                    fund_cum = 100 * (1 + fund_returns).cumprod()
                    index_cum = 100 * (1 + index_returns).cumprod()

                    # Criar DataFrame para plotagem
                    cum_returns_df = pd.DataFrame({
                        'Data': common_dates,
                        f'Fundo: {fundo_selecionado}': fund_cum.values,
                        f'Índice: {ticker_selecionado}': index_cum.values
                    })

                    # Plotar o gráfico
                    fig_norm = go.Figure()
                    fig_norm.add_trace(go.Scatter(
                        x=cum_returns_df['Data'],
                        y=cum_returns_df[f'Fundo: {fundo_selecionado}'],
                        mode='lines',
                        name=f'Fundo: {fundo_selecionado}'
                    ))
                    fig_norm.add_trace(go.Scatter(
                        x=cum_returns_df['Data'],
                        y=cum_returns_df[f'Índice: {ticker_selecionado}'],
                        mode='lines',
                        name=f'Índice: {ticker_selecionado}'
                    ))
                    fig_norm.update_layout(
                        title=f'Retorno Acumulado Normalizado - {fundo_selecionado} vs {ticker_selecionado}',
                        xaxis_title='Data',
                        yaxis_title='Retorno Acumulado (Base 100)',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_norm, use_container_width=True)

                    # =====================================================================
                    # Cálculo da correlação mensal fundo vs todos os índices
                    # =====================================================================
                    st.subheader("(EM BREVE) Matriz de Correlação (Mensal) - Fundo vs Índices")

                    # Calcular retornos mensais do fundo (já divididos por 100)
                    fund_monthly_returns = fund_returns.loc[common_dates]

                    # Criar DataFrame de retornos mensais de todos os índices
                    all_monthly_returns = {}
                    for tdf in dfs_por_ticker:
                        if 'Índice' not in tdf.columns or tdf.empty:
                            continue
                        ticker = tdf['Índice'].iloc[0]
                        tdf = tdf.copy()
                        tdf['Data'] = pd.to_datetime(tdf['Data'], errors='coerce')
                        tdf = tdf.dropna(subset=['Data'])
                        tdf.set_index('Data', inplace=True)
                        tdf_m = tdf.resample('M').last()
                        tdf_m['Retorno_Mensal'] = tdf_m['Cotação'].pct_change()
                        tdf_m = tdf_m.dropna()
                        # Renomear a coluna para seguir o padrão
                        all_monthly_returns[f'Retorno Mensal {ticker} (%)'] = tdf_m['Retorno_Mensal']

                    all_indices_returns = pd.DataFrame(all_monthly_returns)

                    # Alinhar as datas com o fundo
                    all_indices_returns = all_indices_returns[all_indices_returns.index >= fund_monthly_returns.index.min()]

                    # Intersecção das datas
                    common_dates_all = fund_monthly_returns.index.intersection(all_indices_returns.index)
                    fund_monthly_series = fund_monthly_returns.loc[common_dates_all].rename('Retorno Mensal Fundo (%)')
                    all_indices_returns = all_indices_returns.loc[common_dates_all]

                    combined_df = pd.concat([fund_monthly_series, all_indices_returns], axis=1)
                    combined_df = combined_df.dropna(how='any')

                    if combined_df.empty:
                        st.write("Não há dados suficientes após alinhar datas para calcular a correlação mensal com o fundo.")
                    else:
                        # correlation_matrix = combined_df.corr()
                        # fig_corr, ax = plt.subplots(figsize=(12, 10))
                        # sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
                        # plt.title("Matriz de Correlação: Retornos Mensais do Fundo e Índices")
                        # plt.xticks(rotation=45, ha='right')
                        # plt.yticks(rotation=0)
                        # st.pyplot(fig_corr)

                    # =====================================================================
                    # Gráfico de Evolução da Correlação entre o Índice Selecionado e o Fundo
                    # =====================================================================
                     st.subheader(f"(EM BREVE) Evolução da Correlação entre {fundo_selecionado} e {ticker_selecionado}")

                    # Definir o nome exato da coluna de retorno do índice selecionado
                    corr_column = f'Retorno Mensal {ticker_selecionado} (%)'
                    
                    # Verificar se a coluna existe antes de tentar acessar
                    if corr_column not in combined_df.columns:
                        st.error(f"A coluna '{corr_column}' não existe no DataFrame. Verifique os nomes das colunas.")
                        st.write("### Debug: Colunas em combined_df")
                        st.write(combined_df.columns.tolist())
                    else:
                        # Calcular a correlação rolante (12 meses) entre os retornos do fundo e do índice selecionado
                        window_size = 12  # 12 meses
                        rolling_corr = combined_df['Retorno Mensal Fundo (%)'].rolling(window=window_size).corr(combined_df[corr_column])

                        # Criar DataFrame para plotagem
                        rolling_corr_df = pd.DataFrame({
                            'Data': rolling_corr.index,
                            'Correlação Rolante (12 meses)': rolling_corr.values
                        }).dropna()

                        if rolling_corr_df.empty:
                            st.write("Não há dados suficientes para calcular a correlação rolante.")
                        else:
                            fig_corr_evolution = go.Figure()
                            fig_corr_evolution.add_trace(go.Scatter(
                                x=rolling_corr_df['Data'],
                                y=rolling_corr_df['Correlação Rolante (12 meses)'],
                                mode='lines',
                                name='Correlação Rolante (12 meses)',
                                line=dict(color='purple')
                            ))
                            fig_corr_evolution.update_layout(
                                title=f'Evolução da Correlação Rolante (12 meses) - {fundo_selecionado} vs {ticker_selecionado}',
                                xaxis_title='Data',
                                yaxis_title='Correlação',
                                yaxis=dict(range=[-1, 1]),
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_corr_evolution, use_container_width=True)
                else:
                    st.error("Índice selecionado não encontrado ou está vazio.")
    else:
        st.warning("Por favor, carregue um arquivo CSV para começar a análise de índices.")
