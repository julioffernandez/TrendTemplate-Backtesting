import numpy as np
import pandas as pd

def calculate_metrics(equity_df, trades_df, benchmark_df=None, risk_free_rate=0.02):
    """
    Calcula métricas clave de rendimiento, riesgo y ejecución para una estrategia de trading.

    Parámetros:
    -----------
    equity_df : pd.DataFrame
        Historial de la curva de patrimonio (debe contener la columna 'Equity').
    trades_df : pd.DataFrame
        Historial de operaciones (requiere columnas 'Ticker', 'Entry Date', 'PnL ($)', 'Return (%)').
    benchmark_df : pd.DataFrame, opcional
        Serie de datos del índice o activo de referencia (debe contener la columna 'Close').
    risk_free_rate : float, opcional
        Tasa libre de riesgo anualizada en formato decimal (por defecto 0.02 = 2%).

    Retorna:
    --------
    dict
        Diccionario con las métricas de rendimiento y riesgo calculadas y redondeadas.
    """
    
    # 1. Validación de datos de entrada indispensables
    if equity_df is None or equity_df.empty or len(equity_df) < 2:
        return {
            "Retorno Total (%)": 0.0,
            "CAGR Anualizado (%)": 0.0,
            "Máximo Drawdown (%)": 0.0,
            "Ratio de Sharpe": 0.0,
            "Ratio de Calmar": 0.0,
            "Win Rate (%)": 0.0,
            "Win Rate Ejecuciones (%)": 0.0,
            "Profit Factor": 0.0,
            "Payoff Ratio": 0.0,
            "Total Operaciones": 0,
            "Total Posiciones": 0
        }

    # 2. Cálculo del Retorno Total y horizonte temporal
    equity = equity_df['Equity']
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    
    # Normalización del índice de fechas y cálculo del período en años
    start_dt = pd.to_datetime(equity.index[0])
    end_dt = pd.to_datetime(equity.index[-1])
    days = (end_dt - start_dt).days
    years = max(days / 365.25, 1e-6)  # Previene división por cero para períodos inferiores a un día
    
    # Tasa de Crecimiento Anual Compuesto (CAGR)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0
    
    # 3. Métricas de caída pico a valle (Drawdown) y Ratio de Calmar
    rolling_max = equity.cummax()  # Máximos históricos acumulados
    drawdowns = (equity - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()  # Máxima pérdida porcentual acumulada
    calmar = (cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    
    # 4. Ajuste de retornos diarios y Ratio de Sharpe
    daily_returns = equity.pct_change().dropna()
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0  # Tasa libre de riesgo diaria (base 252 días hábiles)
    excess_returns = daily_returns - daily_rf
    std_dev = daily_returns.std()
    
    # Sharpe Ratio anualizado
    sharpe = np.sqrt(252.0) * (excess_returns.mean() / std_dev) if std_dev > 1e-8 else 0.0
    
    # 5. Análisis del comportamiento de las operaciones (Trades)
    if trades_df is not None and not trades_df.empty:
        # Agrupa ejecuciones individuales en posiciones completas (útil para salidas parciales)
        pos_df = trades_df.groupby(['Ticker', 'Entry Date']).agg({
            'PnL ($)': 'sum',
            'Return (%)': 'mean'  # Retorno medio ponderado de la posición
        }).reset_index()

        # Posiciones ganadoras vs. perdedoras
        wins_pos = pos_df[pos_df['PnL ($)'] > 0]
        losses_pos = pos_df[pos_df['PnL ($)'] < 0]
        win_rate_pos = len(wins_pos) / len(pos_df) if len(pos_df) > 0 else 0.0

        # Payoff Ratio Porcentual (Ganancia media / Pérdida media)
        avg_win_pct = wins_pos['Return (%)'].mean() if len(wins_pos) > 0 else 0.0
        avg_loss_pct = abs(losses_pos['Return (%)'].mean()) if len(losses_pos) > 0 else 0.0
        payoff_ratio_pct = (avg_win_pct / avg_loss_pct) if avg_loss_pct > 0 else np.inf

        # Profit Factor monetario global (Ganancias totales / Pérdidas totales)
        total_gains = trades_df[trades_df['PnL ($)'] > 0]['PnL ($)'].sum()
        total_losses = abs(trades_df[trades_df['PnL ($)'] < 0]['PnL ($)'].sum())
        profit_factor = (total_gains / total_losses) if total_losses > 0 else np.inf

        # Tasa de acierto a nivel de ejecución individual
        win_rate_exec = len(trades_df[trades_df['PnL ($)'] > 0]) / len(trades_df)
        total_ops = len(trades_df)
        total_positions = len(pos_df)
    else:
        # Valores por defecto cuando no se provee DataFrame de trades
        win_rate_exec = 0.0
        win_rate_pos = 0.0
        profit_factor = 0.0
        payoff_ratio_pct = 0.0
        total_ops = 0
        total_positions = 0

    # 6. Estructuración y redondeo de las métricas principales
    metrics = {
        "Retorno Total (%)": round(total_return * 100, 2),
        "CAGR Anualizado (%)": round(cagr * 100, 2),
        "Máximo Drawdown (%)": round(max_drawdown * 100, 2),
        "Ratio de Sharpe": round(sharpe, 2),
        "Ratio de Calmar": round(calmar, 2),
        "Win Rate (%)": round(win_rate_pos * 100, 2),
        "Win Rate Ejecuciones (%)": round(win_rate_exec * 100, 2),
        "Profit Factor": round(profit_factor, 2),
        "Payoff Ratio": round(payoff_ratio_pct, 2),
        "Total Operaciones": total_ops,
        "Total Posiciones": total_positions
    }

    # 7. Comparación opcional contra un Benchmark (No se usa actualmente en el jupyter notebook, pero se deja el bloque para posible invocación directa)
    if benchmark_df is not None and not benchmark_df.empty and 'Close' in benchmark_df.columns:
        # Reindexación para alinear el benchmark exactamente con las fechas de la curva de equity
        bench_sub = benchmark_df['Close'].reindex(pd.to_datetime(equity.index)).ffill()
        
        if len(bench_sub) >= 2 and bench_sub.iloc[0] > 0:
            bench_ret = (bench_sub.iloc[-1] / bench_sub.iloc[0]) - 1.0
            bench_cagr = ((1.0 + bench_ret) ** (1.0 / years) - 1.0) if years > 0 else 0.0
            bench_max_dd = ((bench_sub - bench_sub.cummax()) / bench_sub.cummax()).min()
            
            # Adición de métricas relativas al resultado
            metrics["Benchmark Retorno (%)"] = round(bench_ret * 100, 2)
            metrics["Benchmark CAGR (%)"] = round(bench_cagr * 100, 2)
            metrics["Benchmark Máx DD (%)"] = round(bench_max_dd * 100, 2)
            metrics["Exceso Retorno Anualizado (%)"] = round((cagr - bench_cagr) * 100, 2)

    return metrics