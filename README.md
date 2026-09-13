# Minervini SEPA & Trend Template Backtesting Engine

Motor cuantitativo de backtesting en Python diseñado para evaluar estrategias sistemáticas de seguimiento de tendencia basadas en la metodología SEPA (Specific Entry Point Analysis) y el Trend Template de Mark Minervini sobre los componentes del índice Nasdaq-100.

El sistema incorpora gestión de riesgo asimétrica, protección ante anuncios de resultados corporativos (Earnings), dimensionamiento dinámico de posición y simulación de remuneración de tesorería indexada a los tipos de interés de la Reserva Federal.

## Características Principales

*   **Trend Template Cuantitativo (Fase 2 de Minervini):**
    *   Filtro técnico completo de 8 criterios: alineación de medias móviles (`SMA_50`, `SMA_150`, `SMA_200`), pendiente positiva de `SMA_200`, cercanía a máximos de 52 semanas y rebote desde mínimos de 52 semanas.
    *   Ranking transversal ponderado de Fuerza Relativa (RS) estilo IBD (ponderación trimestral, semestral y anual).
*   **Gestión Asimétrica del Riesgo:**
    *   Control estricto de pérdidas con Stop Loss configurable (5.0% - 8.0%).
    *   Subida automática de protección a Breakeven Neto (incluyendo comisiones) al alcanzar +2R.
    *   Toma de beneficios parcial (50%) en +3R.
    *   Salida secuencial en trailing stop por pérdida de la media móvil (`SMA_25`) con confirmación de buffer.
*   **Control de Eventos Corporativos (Earnings Protection):**
    *   Veto temporal de nuevas compras antes de la publicación de resultados (`earnings_entry_buffer`).
    *   Venta defensiva obligatoria previo al evento si la posición no ha acumulado un colchón mínimo de seguridad (`earnings_cushion_pct`).
*   **Filtros Institucionales de Entrada:**
    *   Confirmación de rotura de resistencia con volumen institucional ($\ge 2.0 \times$ media de 50 sesiones).
    *   Filtro anti-chasing para descartar aperturas en gap excesivamente sobreextendidas.
*   **Remuneración Dinámica de Tesorería:**
    *   Devengo diario de tipos reales de la Fed sobre el efectivo ocioso, mitigando el arrastre de liquidez (*cash drag*) en fases defensivas.
*   **Métricas Avanzadas y Desglose Anual:**
    *   Evaluación de ratios Sharpe, Calmar, Payoff Ratio, Win Rate a nivel de operación/posición y cálculo de Alfa anual frente a QQQ.

## Estructura del Repositorio

├── backtest_minervini.ipynb # Notebook principal: descarga, ejecución de modelos y visualización
├── motor.py # Motor de simulación y lógica de ejecución (MinerviniBacktest)
├── datos_filtrado.py # Descarga masiva, extracción de earnings y cálculo del Trend Template
├── evaluacion.py # Cálculo de métricas cuantitativas de riesgo y rendimiento
└── README.md # Documentación técnica

## Requisitos e Instalación

1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/tu-usuario/minervini-backtest.git](https://github.com/tu-usuario/minervini-backtest.git)
   cd minervini-backtest

2. Insalar dependencias:
python -m pip install pandas numpy yfinance matplotlib
Mediante VS Code tenener las extensiones de Jupyter y Python instaladas

3. Ejecutar el Jupyter Notebook en bloques de forma secuencial

Nota sobre el Comportamiento del Sistema: La fortaleza del algoritmo reside en la asimetría de capital: neutraliza prácticamente los mercados bajistas severos (ej. 2022). En periodos de rebote vertical con megacaps concentradas o rangos laterales de alta volatilidad (ej. 2023-2025), la estrategia asume un coste de oportunidad (cash drag y saltos de stop en falsos breakouts) a cambio de mantener un perfil de drawdown protegido por debajo del 12%.


# Guía de Uso

Ejecución básica desde Python o dentro de backtest_minervini.ipynb:

import pandas as pd
from datos_filtrado import get_historical_data, get_earnings_calendar, calculate_trend_template
from motor import MinerviniBacktest
from evaluacion import calculate_metrics
import yfinance as yf

## 1. Definir universo y fechas
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "COST"]
start_date = "2020-04-30"
end_date = "2026-06-30"

## 2. Descarga y cálculo de indicadores
raw_data, _ = get_historical_data(TICKERS, start_date=start_date, end_date=end_date)
benchmark = yf.download("QQQ", start=start_date, end=end_date, progress=False, auto_adjust=True)
if isinstance(benchmark.columns, pd.MultiIndex):
    benchmark.columns = benchmark.columns.get_level_values(0)

earnings_cal, _ = get_earnings_calendar(list(raw_data.keys()))
data_processed = calculate_trend_template(raw_data, benchmark)

## 3. Inicializar y correr el motor
bt = MinerviniBacktest(
    data=data_processed,
    benchmark_df=benchmark,
    earnings_calendar=earnings_cal,
    initial_capital=100000.0,
    max_positions=5,
    stop_loss_pct=0.055,          # Stop Loss inicial al 5.5%
    exit_sma_period=25,           # Salida en cruce de SMA 25
    vol_mult=2.0,                 # Exigencia de 200% de volumen en rotura
    sma_buffer_days=1,            # Salida en Open t+1 tras confirmar pérdida
    cash_yield_annual="dynamic",  # Tipos de la Fed sobre liquidez
    earnings_entry_buffer=14,     # No comprar 14 días antes de resultados
    earnings_exit_buffer=2,       # Evaluar salida 2 días antes de resultados
    earnings_cushion_pct=0.06     # Exigir +6% de beneficio para mantener en earnings
)

equity_df, trades_df = bt.run()

## 4. Métricas de rendimiento
metricas = calculate_metrics(equity_df, trades_df, benchmark)
for k, v in metricas.items():
    print(f"{k}: {v}")


# Licencia

Distribuido bajo la licencia MIT.