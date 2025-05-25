# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ta  # Biblioteca de indicadores técnicos
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_auc_score
import io  # For capturing df.info()

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="GOOGL Stock Price Trend Predictor", page_icon="📈")


# --- Caching Functions ---
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            google_df = pd.read_csv(uploaded_file)
            st.success("Uploaded CSV file loaded successfully!")
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return None
    else:
        try:
            google_df = pd.read_csv('googl_daily_prices.csv')
        except FileNotFoundError:
            st.error("Erro: Arquivo 'googl_daily_prices.csv' não encontrado. Verifique o caminho ou faça upload.")
            return None
    return google_df


@st.cache_data
def preprocess_data(_df):
    df = _df.copy()
    if 'date' not in df.columns:
        # Try to find a date-like column if 'date' is not present
        date_col_found = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col].iloc[:5])  # Test conversion on a small sample
                    df.rename(columns={col: 'date'}, inplace=True)
                    date_col_found = 'date'
                    st.info(f"Column '{col}' auto-detected and renamed to 'date'.")
                    break
                except:
                    continue
        if not date_col_found:
            st.error(
                "Column 'date' not found and no suitable date-like column auto-detected. Please ensure your CSV has a recognizable date column.")
            return None

    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        st.error(f"Error converting 'date' column to datetime: {e}. Ensure 'date' is in a recognizable format.")
        return None

    df = df.set_index('date')
    df = df.sort_index()

    # Renomear colunas
    rename_map = {
        '1. open': 'open', 'open': 'open',
        '2. high': 'high', 'high': 'high',
        '3. low': 'low', 'low': 'low',
        '4. close': 'close', 'close': 'close',
        '5. volume': 'volume', 'volume': 'volume'
    }
    cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns and k != v}
    if cols_to_rename:
        df.rename(columns=cols_to_rename, inplace=True)

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(
            f"Missing required columns after renaming: {', '.join(missing_cols)}. Expected 'open', 'high', 'low', 'close', 'volume'. Check your CSV headers.")
        return None

    return df


@st.cache_data
def engineer_features(_df):
    df = _df.copy()
    # Retornos
    df['simple_return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Médias Móveis (SMA e EMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # Volatilidade
    df['volatility_20d'] = df['log_return'].rolling(window=20).std() * np.sqrt(20)

    # Indicadores Técnicos com 'ta'
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd_indicator = ta.trend.MACD(df['close'])
    df['MACD'] = macd_indicator.macd()
    df['MACD_signal'] = macd_indicator.macd_signal()
    df['MACD_diff'] = macd_indicator.macd_diff()

    bb_indicator = ta.volatility.BollingerBands(df['close'])
    df['BBL_upper'] = bb_indicator.bollinger_hband()
    df['BBL_lower'] = bb_indicator.bollinger_lband()
    df['BBL_mavg'] = bb_indicator.bollinger_mavg()
    df['BBL_width'] = bb_indicator.bollinger_wband()
    df['BBL_percent'] = bb_indicator.bollinger_pband()

    # Features de Lag
    df['close_lag_1'] = df['close'].shift(1)
    df['volume_lag_1'] = df['volume'].shift(1)
    df['return_lag_1'] = df['log_return'].shift(1)

    # Variável Alvo
    df['target_direction'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Remover NaNs
    df.dropna(inplace=True)
    return df


# --- Plotting Functions (Styled for EDA) ---
def plot_time_series_st(df, column, title, ylabel, use_st_line_chart=True):
    st.subheader(title)
    if use_st_line_chart and column in df.columns and pd.api.types.is_datetime64_any_dtype(df.index):
        chart_data = df[[column]].copy()
        st.line_chart(chart_data, use_container_width=True)
        st.caption(f"Série temporal para {column}. Eixo Y: {ylabel}")
    else:
        fig, ax = plt.subplots()
        ax.plot(df.index, df[column], label=column, color=sns.color_palette("viridis", 1)[0])
        ax.set_title(title)
        ax.set_xlabel('Data')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        st.pyplot(fig, use_container_width=True)


def plot_ohlc_styled(df):
    st.subheader('Preços Diários OHLC')
    fig, ax = plt.subplots()
    sns.lineplot(x=df.index, y=df['open'], label='Abertura (Open)', ax=ax, alpha=0.8)
    sns.lineplot(x=df.index, y=df['high'], label='Máxima (High)', ax=ax, alpha=0.8)
    sns.lineplot(x=df.index, y=df['low'], label='Mínima (Low)', ax=ax, alpha=0.8)
    sns.lineplot(x=df.index, y=df['close'], label='Fechamento (Close)', ax=ax, linewidth=2, color='black')
    ax.set_title('Preços Diários OHLC (Open, High, Low, Close)')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço (USD)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    st.caption("Visualização dos preços de abertura, máxima, mínima e fechamento.")


def plot_correlation_matrix_styled(df, numeric_cols):
    st.subheader('Matriz de Correlação')
    if not numeric_cols or not all(col in df.columns for col in numeric_cols):
        st.info("Nenhuma coluna numérica válida fornecida ou colunas ausentes para a matriz de correlação.")
        return
    correlation_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax, cbar=True,
                annot_kws={"size": 8})
    ax.set_title('Matriz de Correlação entre Variáveis Numéricas')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig, use_container_width=True)
    st.caption("Coeficientes de correlação de Pearson. Valores próximos de 1 ou -1 indicam forte correlação.")


def plot_distributions_styled(df, numeric_cols):
    st.subheader("Distribuição das Variáveis Selecionadas")
    if not numeric_cols or not all(col in df.columns for col in numeric_cols):
        st.info("Nenhuma coluna selecionada ou colunas ausentes para exibir distribuições.")
        return

    num_plots = len(numeric_cols)
    cols_per_row = st.slider("Colunas por linha para gráficos de distribuição:", 1, 4, 2, key="dist_cols")

    rows = (num_plots + cols_per_row - 1) // cols_per_row

    fig_height = 3.5 * rows
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 5, fig_height))
    axes = np.array(axes).flatten()

    palette = sns.color_palette("viridis", num_plots)

    for i, col_name in enumerate(numeric_cols):
        if i < len(axes):
            sns.histplot(df[col_name], kde=True, ax=axes[i], color=palette[i % len(palette)],
                         bins=30)  # Use modulo for palette
            axes[i].set_title(f'{col_name}', fontsize=10)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Frequência', fontsize=8)
            axes[i].tick_params(axis='both', which='major', labelsize=8)
            axes[i].grid(True, linestyle='--', alpha=0.5)

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    st.caption("Histogramas mostrando a distribuição de frequência de cada variável selecionada.")


def plot_boxplots_styled(df, numeric_cols):
    st.subheader("Boxplots das Variáveis Selecionadas")
    if not numeric_cols or not all(col in df.columns for col in numeric_cols):
        st.info("Nenhuma coluna selecionada ou colunas ausentes para exibir boxplots.")
        return

    num_plots = len(numeric_cols)
    cols_per_row = st.slider("Colunas por linha para boxplots:", 1, 5, 3, key="box_cols")

    rows = (num_plots + cols_per_row - 1) // cols_per_row

    fig_height = 4 * rows
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 3.5, fig_height))
    axes = np.array(axes).flatten()

    palette = sns.color_palette("viridis", num_plots)

    for i, col_name in enumerate(numeric_cols):
        if i < len(axes):
            sns.boxplot(y=df[col_name], ax=axes[i], color=palette[i % len(palette)],
                        width=0.5)  # Use modulo for palette
            axes[i].set_title(f'{col_name}', fontsize=10)
            axes[i].set_ylabel('')
            axes[i].tick_params(axis='y', which='major', labelsize=8)
            axes[i].grid(True, linestyle='--', alpha=0.5, axis='y')

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    st.caption("Boxplots mostrando a mediana, quartis e outliers para cada variável selecionada.")


# --- Plotting Functions for Model Evaluation ---
def plot_confusion_matrix_st(cm, model_name, class_names=['Baixa/Mesmo', 'Alta']):
    fig, ax = plt.subplots(figsize=(5, 4))  # Slightly smaller
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax, annot_kws={"size": 10})
    ax.set_xlabel('Previsto', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)
    ax.set_title(f'Matriz de Confusão: {model_name}', fontsize=12)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    st.pyplot(fig, use_container_width=True)


def plot_feature_importances_st(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted size
        sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax, palette="viridis")
        ax.set_title(f'Top 15 Feature Importances ({model_name})', fontsize=12)
        ax.set_xlabel('Importância', fontsize=10)
        ax.set_ylabel('Feature', fontsize=10)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info(f"O modelo {model_name} não possui o atributo 'feature_importances_'.")


# --- Model Training & Evaluation Function ---
def train_evaluate_model(model_name, model_instance, X_train_scaled, y_train, X_test_scaled, y_test, feature_names,
                         tune=False, param_grid=None, tscv_splits=3):
    results = {}
    st.markdown(f"#### Treinando e Avaliando: {model_name}")

    if tune and param_grid:
        tscv = TimeSeriesSplit(n_splits=tscv_splits)
        grid_search = GridSearchCV(estimator=model_instance,
                                   param_grid=param_grid,
                                   scoring='f1',
                                   cv=tscv,
                                   verbose=0,  # Reduce verbosity in Streamlit
                                   n_jobs=-1)
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Executando GridSearchCV para {model_name}... (Isso pode levar alguns minutos)")

        # Simplified progress for GridSearchCV
        # For a real progress bar, you'd need a custom callback or more complex logic
        grid_search.fit(X_train_scaled, y_train)  # This is the long step
        progress_bar.progress(100)  # Mark as complete after fit
        status_text.text(f"GridSearchCV para {model_name} concluído!")

        st.write(f"Melhores Parâmetros para {model_name}: `{grid_search.best_params_}`")
        trained_model = grid_search.best_estimator_
        results['Best Params'] = grid_search.best_params_
    else:
        with st.spinner(f"Treinando {model_name}..."):
            trained_model = model_instance.fit(X_train_scaled, y_train)

    y_pred = trained_model.predict(X_test_scaled)
    y_pred_proba = trained_model.predict_proba(X_test_scaled)[:, 1]

    results['Accuracy'] = accuracy_score(y_test, y_pred)
    results['Precision'] = precision_score(y_test, y_pred, zero_division=0)
    results['Recall'] = recall_score(y_test, y_pred, zero_division=0)
    results['F1-Score'] = f1_score(y_test, y_pred, zero_division=0)
    results['ROC AUC'] = roc_auc_score(y_test, y_pred_proba)

    cm = confusion_matrix(y_test, y_pred)
    class_report_df = pd.DataFrame(classification_report(y_test, y_pred, zero_division=0, output_dict=True)).transpose()

    # Layout for metrics and confusion matrix
    col_metrics, col_cm = st.columns([0.4, 0.6])  # Adjust ratio as needed

    with col_metrics:
        st.metric("Acurácia", f"{results['Accuracy']:.4f}")
        st.metric("Precisão", f"{results['Precision']:.4f}")
        st.metric("Recall", f"{results['Recall']:.4f}")
        st.metric("F1-Score", f"{results['F1-Score']:.4f}")
        st.metric("ROC AUC", f"{results['ROC AUC']:.4f}")

    with col_cm:
        plot_confusion_matrix_st(cm, model_name)

    st.markdown("###### Relatório de Classificação:")
    st.dataframe(class_report_df.style.format("{:.2f}"))  # Format numbers

    plot_feature_importances_st(trained_model, feature_names, model_name)
    st.markdown("---")  # Separator after each model

    return trained_model, results


# --- Main App Logic ---
st.title("📈 Preditor de Tendência de Preços de Ações")
st.markdown("""
Esta aplicação analisa dados históricos de preços de ações, realiza engenharia de features e treina modelos 
de Machine Learning para prever se o preço de fechamento do próximo dia será **maior** ou **menor/igual** 
que o do dia atual.
""")

# --- Sidebar ---
st.sidebar.header("⚙️ Configurações")
uploaded_file = st.sidebar.file_uploader("Carregar arquivo CSV de cotações", type="csv",
                                         help="O CSV deve conter colunas 'date', 'open', 'high', 'low', 'close', 'volume'.")
st.sidebar.info("Se nenhum arquivo for carregado, será utilizado o dataset `googl_daily_prices.csv` como exemplo.")

# Load and preprocess data
raw_df = load_data(uploaded_file)

if raw_df is not None:
    processed_df = preprocess_data(raw_df)

    if processed_df is not None:
        navigation = st.sidebar.radio("Navegação",
                                      ["Visão Geral dos Dados",
                                       "Análise Exploratória (EDA)",
                                       "Engenharia de Features",
                                       "Treinamento e Avaliação de Modelos"],
                                      captions=["Info, Nulos, Stats", "Gráficos Visuais", "Criação de Indicadores",
                                                "ML e Resultados"])
        st.sidebar.markdown("---")
        st.sidebar.markdown("Desenvolvido com ❤️ usando Streamlit.")

        if navigation == "Visão Geral dos Dados":
            st.header("📄 Visão Geral dos Dados")

            st.subheader("Dados Brutos Carregados")
            st.dataframe(raw_df.head())
            st.write(f"Shape do DataFrame original: {raw_df.shape}")

            with st.expander("Informações Detalhadas do DataFrame (raw_df.info())"):
                buffer = io.StringIO()
                raw_df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)

            with st.expander("Contagem de Valores Nulos (raw_df.isnull().sum())"):
                st.dataframe(raw_df.isnull().sum().to_frame(name='Nulos'))

            with st.expander("Estatísticas Descritivas (raw_df.describe())"):
                st.dataframe(raw_df.describe().style.format("{:.2f}"))

            st.subheader("Dados Pré-processados")
            st.markdown(f"""
            - Coluna 'date' convertida para datetime e definida como índice.
            - Dados ordenados por data.
            - Colunas renomeadas (se necessário) para: `open`, `high`, `low`, `close`, `volume`.
            """)
            st.dataframe(processed_df.head())
            st.write(f"Shape do DataFrame pré-processado: {processed_df.shape}")


        elif navigation == "Análise Exploratória (EDA)":
            st.header("📊 Análise Exploratória de Dados (EDA)")
            st.markdown("Visualizando os dados pré-processados com um estilo aprimorado e interativo.")

            current_theme = st.get_option("theme.base")
            if current_theme == "dark":
                sns.set_theme(style="darkgrid", palette="viridis")
                plt.style.use('dark_background')
            else:
                sns.set_theme(style="whitegrid", palette="viridis")
                plt.style.use('seaborn-v0_8-whitegrid')

            st.markdown("### Séries Temporais Fundamentais")
            col1, col2 = st.columns(2)
            with col1:
                plot_time_series_st(processed_df, 'close', 'Preço de Fechamento Diário', 'Preço (USD)',
                                    use_st_line_chart=True)
            with col2:
                plot_time_series_st(processed_df, 'volume', 'Volume Diário de Negociação', 'Volume',
                                    use_st_line_chart=True)

            plot_ohlc_styled(processed_df)

            st.markdown("---")
            st.markdown("### Análise de Correlação e Distribuição")
            numeric_cols_eda = ['open', 'high', 'low', 'close', 'volume']
            plot_correlation_matrix_styled(processed_df, numeric_cols_eda)

            st.markdown("#### Análise Detalhada por Variável")
            st.info("Selecione as variáveis numéricas abaixo para visualizar suas distribuições e boxplots.")

            default_cols_for_analysis = [col for col in numeric_cols_eda if col in processed_df.columns]

            selected_cols_for_dist = st.multiselect(
                "Selecione colunas para Distribuições e Boxplots:",
                options=default_cols_for_analysis,
                default=default_cols_for_analysis,
                key="eda_multiselect"
            )

            if selected_cols_for_dist:
                plot_distributions_styled(processed_df, selected_cols_for_dist)
                plot_boxplots_styled(processed_df, selected_cols_for_dist)
            else:
                st.warning("Por favor, selecione pelo menos uma coluna para visualizar as análises detalhadas.")

        elif navigation == "Engenharia de Features":
            st.header("🛠️ Engenharia de Features")
            df_featured = engineer_features(processed_df)

            st.markdown("""
            As seguintes features foram criadas e adicionadas ao conjunto de dados:
            - **Retornos:** Simples (`pct_change`) e Logarítmico (`log(price_t / price_t-1)`).
            - **Médias Móveis:** SMA (5, 10, 20, 50, 200 dias) e EMA (12, 26 dias).
            - **Volatilidade:** Desvio padrão dos retornos logarítmicos em uma janela de 20 dias (anualizado).
            - **Indicadores Técnicos (TA-Lib):**
                - `RSI` (Índice de Força Relativa)
                - `MACD` (Convergência/Divergência de Médias Móveis), incluindo linha de sinal e histograma.
                - `Bandas de Bollinger`: Superior, Inferior, Média Móvel central, Largura da banda e Percentual B.
            - **Features de Lag:** Preço de fechamento (`close_lag_1`), volume (`volume_lag_1`) e retorno logarítmico (`return_lag_1`) do dia anterior.
            - **Variável Alvo (`target_direction`):** `1` se o preço de fechamento do dia seguinte for maior que o atual, `0` caso contrário.

            *Observação: Linhas com valores NaN, resultantes da criação das features (especialmente médias móveis longas como SMA_200 e o shift da variável alvo), foram removidas para garantir a qualidade dos dados para modelagem.*
            """)
            with st.expander("Visualizar DataFrame com Features (Primeiras Linhas)"):
                st.dataframe(df_featured.head())
            st.write(f"Shape do DataFrame após engenharia de features e remoção de NaNs: **{df_featured.shape}**")

            with st.expander("Verificação de NaNs Após Engenharia de Features (df_featured.isnull().sum())"):
                null_sum = df_featured.isnull().sum()
                if null_sum.sum() == 0:
                    st.success("Nenhum valor nulo encontrado após a limpeza!")
                else:
                    st.dataframe(null_sum.to_frame(name='Nulos'))

            st.subheader("Distribuição da Variável Alvo (`target_direction`)")
            target_dist = df_featured['target_direction'].value_counts(normalize=True)
            st.bar_chart(target_dist)
            st.write(target_dist.apply(lambda x: f"{x:.2%}"))  # Display as percentage

        elif navigation == "Treinamento e Avaliação de Modelos":
            st.header("🧠 Treinamento e Avaliação de Modelos")
            df_featured = engineer_features(processed_df)

            features_to_drop = ['open', 'high', 'low', 'close', 'volume',
                                'simple_return', 'log_return', 'target_direction']
            # Ensure only existing columns are dropped
            X = df_featured.drop(columns=[col for col in features_to_drop if col in df_featured.columns])
            y = df_featured['target_direction']
            feature_names = X.columns.tolist()

            st.write(f"Shape das Features (X): **{X.shape}** | Shape da Target (y): **{y.shape}**")

            st.markdown("### Configuração do Treino e Teste")
            train_size_slider = st.slider("Percentual de Dados para Treino:", 0.5, 0.9, 0.8, 0.05,
                                          help="Define a proporção dos dados mais antigos a serem usados para treino.")
            train_size = int(len(X) * train_size_slider)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

            st.write(
                f"Conjunto de Treino: **{len(X_train)}** amostras ({X_train.index.min().date()} a {X_train.index.max().date()})")
            st.write(
                f"Conjunto de Teste: **{len(X_test)}** amostras ({X_test.index.min().date()} a {X_test.index.max().date()})")

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.caption("Features escalonadas usando MinMaxScaler.")

            st.markdown("---")
            st.markdown("### Seleção de Modelos para Treinamento")

            models_to_run = {
                "Regressão Logística": False,
                "Random Forest": False,
                "XGBoost (Padrão)": False,
                "XGBoost (Otimizado com GridSearchCV)": True  # Default this one
            }

            cols_checkbox = st.columns(len(models_to_run))
            idx = 0
            for model_name_key in models_to_run:
                models_to_run[model_name_key] = cols_checkbox[idx].checkbox(model_name_key,
                                                                            value=models_to_run[model_name_key],
                                                                            key=f"cb_{model_name_key.replace(' ', '_')}")
                idx += 1

            all_results_summary = []

            if st.button("🚀 Iniciar Treinamento e Avaliação dos Modelos Selecionados", type="primary",
                         use_container_width=True):
                if not any(models_to_run.values()):
                    st.warning("Por favor, selecione pelo menos um modelo para treinar.")
                else:
                    with st.spinner("Processando modelos... por favor, aguarde."):
                        if models_to_run["Regressão Logística"]:
                            lr_model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
                            _, results_lr = train_evaluate_model("Regressão Logística", lr_model,
                                                                 X_train_scaled, y_train, X_test_scaled, y_test,
                                                                 feature_names)
                            results_lr['Modelo'] = "Regressão Logística"
                            all_results_summary.append(results_lr)

                        if models_to_run["Random Forest"]:
                            rf_model = RandomForestClassifier(random_state=42, n_estimators=100,
                                                              class_weight='balanced')
                            _, results_rf = train_evaluate_model("Random Forest", rf_model,
                                                                 X_train_scaled, y_train, X_test_scaled, y_test,
                                                                 feature_names)
                            results_rf['Modelo'] = "Random Forest"
                            all_results_summary.append(results_rf)

                        if models_to_run["XGBoost (Padrão)"]:
                            xgb_orig_model = XGBClassifier(random_state=42, use_label_encoder=False,
                                                           eval_metric='logloss')
                            _, results_xgb_orig = train_evaluate_model("XGBoost (Padrão)", xgb_orig_model,
                                                                       X_train_scaled, y_train, X_test_scaled, y_test,
                                                                       feature_names)
                            results_xgb_orig['Modelo'] = "XGBoost (Padrão)"
                            all_results_summary.append(results_xgb_orig)

                        if models_to_run["XGBoost (Otimizado com GridSearchCV)"]:
                            neg_count = y_train.value_counts().get(0, 0)
                            pos_count = y_train.value_counts().get(1, 1)  # Avoid division by zero
                            scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0

                            st.info(
                                f"Calculado `scale_pos_weight` para XGBoost: {scale_pos_weight_val:.2f} (Neg: {neg_count}, Pos: {pos_count})")

                            xgb_tuned_model_base = XGBClassifier(random_state=42, use_label_encoder=False,
                                                                 eval_metric='logloss',
                                                                 scale_pos_weight=scale_pos_weight_val)

                            param_grid_xgb_lite = {  # Reduced for demo speed
                                'n_estimators': [50, 100],
                                'max_depth': [3, 4],
                                'learning_rate': [0.1],
                            }
                            st.caption("Grid de parâmetros para XGBoost (reduzido para demonstração):")
                            st.json(param_grid_xgb_lite)

                            _, results_xgb_tuned = train_evaluate_model("XGBoost (Otimizado)", xgb_tuned_model_base,
                                                                        X_train_scaled, y_train, X_test_scaled, y_test,
                                                                        feature_names,
                                                                        tune=True, param_grid=param_grid_xgb_lite,
                                                                        tscv_splits=2)  # Reduced splits
                            results_xgb_tuned['Modelo'] = "XGBoost (Otimizado)"
                            all_results_summary.append(results_xgb_tuned)

                    # --- Resumo dos Resultados ---
                    if all_results_summary:
                        st.header("🏆 Resumo Comparativo dos Resultados")

                        # Convert list of dicts to DataFrame
                        summary_df_list = []
                        for res_dict in all_results_summary:
                            summary_df_list.append({
                                'Modelo': res_dict.get('Modelo', 'N/A'),
                                'Acurácia': res_dict.get('Accuracy', np.nan),
                                'Precisão': res_dict.get('Precision', np.nan),
                                'Recall': res_dict.get('Recall', np.nan),
                                'F1-Score': res_dict.get('F1-Score', np.nan),
                                'ROC AUC': res_dict.get('ROC AUC', np.nan)
                            })

                        summary_df_display = pd.DataFrame(summary_df_list)

                        if not summary_df_display.empty:
                            st.dataframe(
                                summary_df_display.sort_values(by='F1-Score', ascending=False).set_index(
                                    'Modelo').style.format("{:.4f}").highlight_max(axis=0, color='lightgreen')
                            )
                        else:
                            st.info("Nenhum resultado de modelo para exibir no resumo.")
                    else:
                        st.info("Nenhum modelo foi treinado para exibir o resumo.")
    else:
        st.warning(
            "⚠️ Não foi possível carregar ou pré-processar os dados. Verifique o arquivo carregado ou as colunas do CSV.")
else:
    st.info(
        "ℹ️ Aguardando carregamento do arquivo CSV na barra lateral ou usando o conjunto de dados padrão `googl_daily_prices.csv`.")