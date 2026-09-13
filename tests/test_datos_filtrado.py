from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from datos_filtrado import (
    calculate_trend_template,
    get_earnings_calendar,
    get_historical_data,
)


# =====================================================================
# Fixtures para reutilizar datos sintéticos
# =====================================================================
@pytest.fixture
def mock_ohlcv_df():
    """Genera un DataFrame sintético con 300 filas para probar indicadores técnicos."""
    dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
    np.random.seed(42)

    # Simular una tendencia alcista para que cumpla condiciones de Trend Template
    close_prices = np.linspace(100, 200, 300) + np.random.normal(0, 1, 300)

    df = pd.DataFrame(
        {
            "Open": close_prices - 1,
            "High": close_prices + 2,
            "Low": close_prices - 2,
            "Close": close_prices,
            "Volume": np.random.randint(1000, 50000, size=300),
        },
        index=dates,
    )
    return df


@pytest.fixture
def mock_benchmark_df():
    """Genera un DataFrame sintético de Benchmark (QQQ/SPY)."""
    dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
    prices = np.linspace(300, 400, 300)
    return pd.DataFrame({"Close": prices}, index=dates)


# =====================================================================
# Tests para get_historical_data
# =====================================================================
def test_get_historical_data_empty_input():
    """Comprueba el comportamiento cuando la lista de tickers está vacía."""
    data, failed = get_historical_data([], "2023-01-01", "2023-12-31")
    assert data == {}
    assert failed == {}


@patch("yfinance.download")
def test_get_historical_data_success(mock_yf_download, mock_ohlcv_df):
    """Simula una descarga exitosa de yfinance con MultiIndex."""
    # Preparamos un MultiIndex similar al que devuelve yfinance
    multi_cols = pd.MultiIndex.from_product(
        [["AAPL"], ["Open", "High", "Low", "Close", "Volume"]]
    )
    mock_raw = pd.DataFrame(mock_ohlcv_df.values, index=mock_ohlcv_df.index, columns=multi_cols)
    mock_yf_download.return_value = mock_raw

    data, failed = get_historical_data(["AAPL"], "2023-01-01", "2023-12-31")

    assert "AAPL" in data
    assert len(failed) == 0
    assert "Close" in data["AAPL"].columns
    assert len(data["AAPL"]) >= 252


@patch("yfinance.download")
def test_get_historical_data_insufficient_history(mock_yf_download, mock_ohlcv_df):
    """Verifica que se descarte un activo si tiene menos de 252 barras."""
    short_df = mock_ohlcv_df.iloc[:100]  # Solo 100 días
    multi_cols = pd.MultiIndex.from_product(
        [["FAIL_TICKER"], ["Open", "High", "Low", "Close", "Volume"]]
    )
    mock_raw = pd.DataFrame(short_df.values, index=short_df.index, columns=multi_cols)
    mock_yf_download.return_value = mock_raw

    data, failed = get_historical_data(["FAIL_TICKER"], "2023-01-01", "2023-12-31")

    assert "FAIL_TICKER" not in data
    assert "FAIL_TICKER" in failed
    assert "Histórico insuficiente" in failed["FAIL_TICKER"]


# =====================================================================
# Tests para get_earnings_calendar
# =====================================================================
@patch("yfinance.Ticker")
def test_get_earnings_calendar_success(mock_ticker):
    """Simula la extracción correcta de fechas de resultados corporativos."""
    mock_instance = MagicMock()
    mock_dates = pd.date_range("2023-01-01", periods=3, freq="QE")
    
    # Le añadimos una columna ficticia para que df.empty sea False
    mock_df = pd.DataFrame({"EPS Estimate": [0.5, 0.6, 0.7]}, index=mock_dates)
    mock_instance.get_earnings_dates.return_value = mock_df
    mock_ticker.return_value = mock_instance

    earnings, failed = get_earnings_calendar(["AAPL"])

    assert "AAPL" in earnings
    assert len(earnings["AAPL"]) == 3
    assert len(failed) == 0

@patch("yfinance.Ticker")
def test_get_earnings_calendar_fallback(mock_ticker):
    """Verifica el mecanismo de fallback cuando get_earnings_dates falla o es None."""
    mock_instance = MagicMock()
    mock_instance.get_earnings_dates.return_value = None
    mock_instance.calendar = {"Earnings Date": [pd.Timestamp("2023-05-01")]}
    mock_ticker.return_value = mock_instance

    earnings, failed = get_earnings_calendar(["MSFT"])

    assert "MSFT" in earnings
    assert len(earnings["MSFT"]) == 1
    assert earnings["MSFT"][0] == pd.Timestamp("2023-05-01")


# =====================================================================
# Tests para calculate_trend_template
# =====================================================================
def test_calculate_trend_template_single_dataframe(mock_ohlcv_df, mock_benchmark_df):
    """Comprueba el cálculo del Trend Template sobre un único DataFrame."""
    res_df = calculate_trend_template(mock_ohlcv_df, mock_benchmark_df)

    # Verificar que se crearon las columnas calculadas
    expected_cols = [
        "SMA_25",
        "SMA_50",
        "SMA_150",
        "SMA_200",
        "Technical_Template_Pass",
        "Trend_Template_Pass",
        "RS_Ratio",
    ]
    for col in expected_cols:
        assert col in res_df.columns

    assert isinstance(res_df["Technical_Template_Pass"].iloc[-1], (bool, np.bool_))


def test_calculate_trend_template_dict_multiple_assets(
    mock_ohlcv_df, mock_benchmark_df
):
    """Comprueba el ranking transversal (RS_Percentile) cuando hay múltiples activos."""
    # Crear un segundo activo cuya tendencia relativa sea inferior (rendimiento decreciente)
    df_asset2 = mock_ohlcv_df.copy()
    # Multiplicar por una rampa descendente para alterar los ratios de cambio porcentual
    df_asset2["Close"] = df_asset2["Close"] * np.linspace(1.0, 0.4, len(df_asset2))

    data_dict = {"A1": mock_ohlcv_df, "A2": df_asset2}

    processed_dict = calculate_trend_template(
        data_dict, mock_benchmark_df, rs_threshold=50.0
    )

    assert "A1" in processed_dict
    assert "A2" in processed_dict
    assert "RS_Percentile" in processed_dict["A1"].columns

    # A1 debe tener un percentil de Fuerza Relativa estrictamente mayor que A2 al final
    last_rs_a1 = processed_dict["A1"]["RS_Percentile"].iloc[-1]
    last_rs_a2 = processed_dict["A2"]["RS_Percentile"].iloc[-1]
    assert last_rs_a1 > last_rs_a2