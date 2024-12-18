import streamlit as st
import pandas as pd

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

        if 'Disclaimers' in data_fund.columns:
            disclaimer = data_fund['Disclaimers'].iloc[0]
            disclaimer = '' if pd.isna(disclaimer) else str(disclaimer).strip()
        else:
            disclaimer = ''

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

def main():
    st.title("Visualizador de Fundos de Investimento")

    # Caminho fixo para o arquivo Excel
    fonte = r"C:\Users\AntônioRocha\OneDrive - Gama\Área de Trabalho\teste_metricas\Taxas Fundos.xlsx"
    
    # Verificar se o arquivo existe
    try:
        data = pd.read_excel(fonte, sheet_name="Infos Gerais")
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado no caminho especificado: {fonte}")
        return
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Excel: {e}")
        return

    # Obter os nomes dos fundos disponíveis
    try:
        fund_names = data['Nome Fundo'].unique()
    except KeyError:
        st.error("A coluna 'Nome Fundo' não foi encontrada no Excel.")
        return
    except Exception as e:
        st.error(f"Erro ao obter os nomes dos fundos: {e}")
        return

    # Selecionar o fundo
    fund_name = st.selectbox("Selecione o Fundo", options=sorted(fund_names))

    if fund_name:
        df = getMainData(font=fonte, fund_name=fund_name)
        if df:
            st.header(f"Detalhes do Fundo: {fund_name}")
            # Exibir os detalhes em formato de tabela
            df_display = pd.DataFrame(df, index=[0])
            st.table(df_display)

            # Exibir informações detalhadas em colunas
            st.subheader("Informações Detalhadas")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**CNPJ/GIIN:** {df['CNPJ/GIIN']}")
                st.markdown(f"**Aplicação:** {df['Aplicação']}")
                st.markdown(f"**Resgate:** {df['Resgate']}")
                st.markdown(f"**Aplicação Inicial:** {df['Aplicação Inicial']}")
                st.markdown(f"**Movimentação Adicional:** {df['Movimentação Adicional']}")

            with col2:
                st.markdown(f"**Público Alvo:** {df['Público Alvo']}")
                st.markdown(f"**Administração:** {df['Administração']}")
                st.markdown(f"**Custodiante:** {df['Custodiante']}")
                st.markdown(f"**Cotização:** {df['Cotização']}")
                st.markdown(f"**ISIN:** {df['ISIN']}")
                st.markdown(f"**Estratégia:** {df['Estratégia']}")
                st.markdown(f"**Hedge:** {df['Hedge']}")

            if df['Disclaimers']:
                st.warning(f"**Disclaimers:** {df['Disclaimers']}")

if __name__ == "__main__":
    main()
