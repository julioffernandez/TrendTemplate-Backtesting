import logging
import pandas as pd
import numpy as np
import yfinance as yf

# Configuración del sistema de registros (logs) para monitorear el proceso
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_historical_data(tickers, start_date, end_date):
    """
    Descarga y valida datos históricos OHLCV de forma masiva utilizando la API de yfinance.

    Parámetros:
    -----------
    tickers : list
        Lista de símbolos/tickers de activos financieros.
    start_date : str
        Fecha de inicio en formato 'YYYY-MM-DD'.
    end_date : str
        Fecha final en formato 'YYYY-MM-DD'.

    Retorna:
    --------
    tuple (dict, dict)
        - data_dict: Diccionario con DataFrames limpios y validados por ticker.
        - failed_tickers: Diccionario con los tickers descartados y la causa del fallo.
    """
    data_dict = {}
    failed_tickers = {}

    if not tickers:
        return data_dict, failed_tickers

    # Normaliza y elimina duplicados manteniendo el orden original
    ticker_list = list(dict.fromkeys([t.strip().upper() for t in tickers]))

    try:
        # Descarga vectorizada multihilo para optimizar tiempo de ejecución
        raw = yf.download(
            ticker_list,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,   # Ajusta automáticamente dividendos y splits en OHLC
            group_by='ticker',  # Agrupa las columnas por ticker para una ordenación óptima
            threads=True        # Habilita ejecución concurrente multihilo
        )
    except Exception as e:
        logger.error(f"Error crítico en yf.download batch: {e}")
        return data_dict, {t: str(e) for t in ticker_list}

    # Conjunto de columnas mínimas requeridas para análisis técnico y backtesting
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}

    for ticker in ticker_list:
        try:
            # 1. Extracción de datos agnóstica a la estructura de la respuesta (MultiIndex vs Single Index)
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0):
                    df = raw[ticker].copy()
                elif ticker in raw.columns.get_level_values(1):
                    df = raw.xs(ticker, axis=1, level=1).copy()
                else:
                    failed_tickers[ticker] = "No devuelto por la API"
                    continue
            else:
                df = raw.copy()

            # 2. Limpieza inicial: eliminación de filas completamente vacías
            df = df.dropna(how='all')
            if df.empty:
                failed_tickers[ticker] = "DataFrame vacío"
                continue

            # 3. Validación de estructura de columnas
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                failed_tickers[ticker] = f"Columnas ausentes: {missing}"
                continue

            # 4. Limpieza de datos esenciales de precio y volumen
            df = df.dropna(subset=['Close', 'Volume'])
            
            # Requisito de un año de trading (252 ruedas) para cálculo estable de Medias Móviles (SMA 200)
            if len(df) < 252:
                failed_tickers[ticker] = f"Histórico insuficiente ({len(df)} barras)"
                continue

            data_dict[ticker] = df.sort_index()

        except Exception as e:
            failed_tickers[ticker] = f"{type(e).__name__}: {str(e)}"
            continue

    logger.info(f"Descarga finalizada: {len(data_dict)} admitidos, {len(failed_tickers)} descartados.")
    return data_dict, failed_tickers


def get_earnings_calendar(tickers):
    """
    Descarga el historial completo de fechas de reportes de ganancias (Earnings Dates).
    
    Esta información es vital en el modelo Minervini para evitar operar
    durante eventos de alta volatilidad no controlada antes del reporte.

    Parámetros:
    -----------
    tickers : list
        Lista de activos a consultar.

    Retorna:
    --------
    tuple (dict, dict)
        - earnings_dict: Lista de fechas de earnings (sin zona horaria) por ticker.
        - failed_earnings: Detalle de errores por ticker.
    """
    earnings_dict = {}
    failed_earnings = {}

    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            # Intento de extracción del historial extendido de earnings
            cal_df = tk.get_earnings_dates(limit=100)
            
            dates = []
            if cal_df is not None and not cal_df.empty:
                # Extrae los timestamps del índice y remueve información de zona horaria para estandarizar
                dates = [
                    pd.to_datetime(d).tz_convert(None) if pd.to_datetime(d).tzinfo else pd.to_datetime(d) 
                    for d in cal_df.index
                ]
            else:
                # Estrategia de respaldo (Fallback) usando la propiedad básica 'calendar'
                cal = tk.calendar
                if cal is not None and isinstance(cal, dict) and 'Earnings Date' in cal:
                    dates = [pd.to_datetime(d).tz_localize(None) for d in cal['Earnings Date']]

            earnings_dict[ticker] = dates

        except Exception as e:
            failed_earnings[ticker] = str(e)
            earnings_dict[ticker] = []

    return earnings_dict, failed_earnings


def calculate_trend_template(data_dict, benchmark_df, rs_threshold=90.0):
    """
    Calcula las medias móviles, el rating de Fuerza Relativa (RS) estilo IBD y
    evalúa las 8 condiciones del Trend Template de Mark Minervini.

    Condiciones evaluadas:
    1. Precio actual > SMA 150 y SMA 200.
    2. SMA 150 > SMA 200.
    3. Pendiente de SMA 200 ascendente (mínimo durante 1 mes / 20 días de trading).
    4. SMA 50 > SMA 150 y SMA 200.
    5. Precio actual > SMA 50.
    6. Precio actual >= 25% por encima del mínimo de 52 semanas.
    7. Precio actual a no más de un 25% de distancia del máximo de 52 semanas.
    8. Fuerza Relativa (RS) >= umbral configurado (default e ideal: percentil 90, fuerza preferida por encima de 80, mínimo estricto aceptable 70).

    Parámetros:
    -----------
    data_dict : pd.DataFrame o dict de DataFrames
        Datos históricos de los activos.
    benchmark_df : pd.DataFrame
        DataFrame del índice de referencia (ej. QQQ / S&P 500) para RS relativo directo.
    rs_threshold : float, opcional
        Percentil mínimo de fuerza relativa transversal requerida (por defecto 90.0).

    Retorna:
    --------
    pd.DataFrame o dict
        Estructura de datos enriquecida con indicadores y señales booleanas de tendencia.
    """
    is_single_df = isinstance(data_dict, pd.DataFrame)
    working_dict = {'ASSET': data_dict.copy()} if is_single_df else {t: df.copy() for t, df in data_dict.items()}

    perf_dict = {}
    
    # --- PASO 1: Cálculo de medias móviles, rangos de 52 semanas y Score IBD para RS ---
    for ticker, df in working_dict.items():
        # Medias móviles simples (SMA)
        df['SMA_25'] = df['Close'].rolling(window=25).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_150'] = df['Close'].rolling(window=150).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Tendencia de la SMA de 200 días (compara el valor actual con el de hace 20 días)
        df['SMA_200_Slope'] = (df['SMA_200'] > df['SMA_200'].shift(20)).fillna(False)
        
        # Extrems de 52 semanas (252 días de negociación)
        df['Low_52w'] = df['Low'].rolling(window=252).min()
        df['High_52w'] = df['High'].rolling(window=252).max()

        # Score IBD ponderado para RS: $0.4 \times Retorno_{3m} + 0.3 \times Retorno_{6m} + 0.3 \times Retorno_{12m}$
        perf = (df['Close'] / df['Close'].shift(63)) * 0.4 + \
               (df['Close'] / df['Close'].shift(126)) * 0.3 + \
               (df['Close'] / df['Close'].shift(252)) * 0.3
        perf_dict[ticker] = perf
        working_dict[ticker] = df

    # --- PASO 2: Cálculo Transversal de Fuerza Relativa (RS Percentile) ---
    if not is_single_df and len(working_dict) >= 2:
        perf_matrix = pd.DataFrame(perf_dict)
        # Ranking transversal diario normalizado en escala 1-99
        rs_matrix = perf_matrix.rank(axis=1, pct=True) * 99.0
    else:
        rs_matrix = None

    # --- PASO 3: Evaluación de Criterios del Trend Template por Activo ---
    for ticker, df in working_dict.items():
        if rs_matrix is not None:
            df['RS_Percentile'] = rs_matrix[ticker]
            df['RS_Ratio'] = df['RS_Percentile']
            # Condición 8: Fuerza Relativa superior al umbral (ej. Top 10% del mercado)
            c8 = df['RS_Percentile'] >= rs_threshold
        else:
            # Fallback para activo único: Compara el rendimiento directo vs. Benchmark
            df['RS_Percentile'] = np.nan
            if benchmark_df is not None and 'Close' in benchmark_df.columns:
                bench_aligned = benchmark_df['Close'].reindex(df.index).ffill()
                df['RS_Ratio'] = (df['Close'] / df['Close'].shift(252)) / (bench_aligned / bench_aligned.shift(252))
                # Bate al benchmark en el último año (ratio >= 1.0)
                c8 = (df['RS_Ratio'] >= 1.15).fillna(False)
            else:
                df['RS_Ratio'] = np.nan
                c8 = pd.Series(False, index=df.index)

        # Evaluación de las Reglas Técnicas de Minervini
        c1 = (df['Close'] > df['SMA_150']) & (df['Close'] > df['SMA_200'])        # Regla 1: Precio > SMA 150 y 200
        c2 = df['SMA_150'] > df['SMA_200']                                        # Regla 2: SMA 150 > SMA 200
        c3 = df['SMA_200_Slope']                                                  # Regla 3: SMA 200 en fase ascendente
        c4 = (df['SMA_50'] > df['SMA_150']) & (df['SMA_50'] > df['SMA_200'])       # Regla 4: SMA 50 > SMA 150 y 200
        c5 = df['Close'] > df['SMA_50']                                           # Regla 5: Precio > SMA 50
        c6 = df['Close'] >= (df['Low_52w'] * 1.25)                                # Regla 6: Al menos 25% sobre el mínimo de 52 sem
        c7 = df['Close'] >= (df['High_52w'] * 0.75)                               # Regla 7: A menos de un 25% del máximo de 52 sem

        # Filtro estrictamente técnico (Condiciones 1 a 7)
        df['Technical_Template_Pass'] = c1 & c2 & c3 & c4 & c5 & c6 & c7
        
        # Filtro completo de Minervini incluyendo la prueba de Fuerza Relativa (Condiciones 1 a 8)
        df['Trend_Template_Pass'] = df['Technical_Template_Pass'] & c8
        working_dict[ticker] = df

    return working_dict['ASSET'] if is_single_df else working_dict