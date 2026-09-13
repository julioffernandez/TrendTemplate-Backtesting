import pandas as pd
import numpy as np


class MinerviniBacktest:
    """
    Motor de Backtesting para la Estrategia de Trading de Mark Minervini.
    
    Implementa gestión de riesgo asimétrica, trailing stops dinámicos por media móvil,
    acumulación de tesorería remunerada, filtros anti-chasing, control de volatilidad VCP
    y prevención de eventos de resultados corporativos (Earnings).
    """

    def __init__(self, data, benchmark_df, earnings_calendar=None, initial_capital=100000.0, 
                 max_positions=5, commission_rate=0.0025, stop_loss_pct=0.055, 
                 exit_sma_period=25, vol_mult=2.0, dynamic_slots=True, 
                 dynamic_concentration=True, sma_buffer_days=1, cash_yield_annual='dynamic', 
                 max_chase_pct=0.03, earnings_entry_buffer=14, earnings_exit_buffer=2, 
                 earnings_cushion_pct=0.06):
        """
        Inicializa los parámetros operativos y de riesgo del backtest.
        """
        # Filtros de protección ante publicación de resultados corporativos (Earnings)
        self.earnings_entry_buffer = earnings_entry_buffer  # Días previos a resultados donde se bloquean nuevas compras. Default: <= 14 días
        self.earnings_exit_buffer = earnings_exit_buffer    # Días previos a resultados donde se evalúa liquidación preventiva. Default: <= 2 días de resultados
        self.earnings_cushion_pct = earnings_cushion_pct    # Colchón mínimo de beneficio acumulado exigido para mantener la posición. Default: (6% ≈ +1R / +2R)
        
        # 1. Copia defensiva completa para aislar el motor de mutaciones externas en los dataframes
        self.data = {ticker: df.copy() for ticker, df in data.items()}
        self.benchmark_df = benchmark_df.copy()
        self.earnings_calendar = earnings_calendar or {}
        
        # Configuración de capital y comisiones
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.commission_rate = commission_rate
        
        # Reglas de salida y dimensión de posiciones
        self.stop_loss_pct = stop_loss_pct
        self.exit_sma_period = exit_sma_period
        self.vol_mult = vol_mult
        self.dynamic_slots = dynamic_slots
        self.dynamic_concentration = dynamic_concentration
        self.sma_buffer_days = sma_buffer_days        # 0 = Salida al cierre (MOC), >=1 = Salida en apertura (MOO) en t+1
        self.cash_yield_annual = cash_yield_annual    # Tasa de remuneración del efectivo ocioso
        self.max_chase_pct = max_chase_pct            # Tolerancia máxima a la sobreextensión del precio en la apertura (Anti-Chasing)
        
        # Estructuras de estado interno de la simulación
        self.positions = {}
        self.trade_history = []
        self.equity_curve = []
        
    def _has_upcoming_earnings(self, ticker, current_date, window_days):
        """
        Verifica si un activo tiene reporte de resultados dentro de la ventana de días especificada.
        """
        if ticker not in self.earnings_calendar or not self.earnings_calendar[ticker]:
            return False
        c_date = current_date.tz_localize(None) if getattr(current_date, 'tz', None) is not None else current_date
        window_end = c_date + pd.Timedelta(days=window_days)
        return any(c_date <= ed <= window_end for ed in self.earnings_calendar[ticker])
        
    def run(self):
        """
        Ejecuta la simulación cronológica del backtest día a día.
        """
        # 1. Reseteo explícito del estado para permitir reejecuciones sobre la misma instancia
        self.capital = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.equity_curve = []

        sma_col = f'SMA_{self.exit_sma_period}'
        
        # 2. Precomputación vectorizada completa de indicadores técnicos
        if 'SMA_200' not in self.benchmark_df.columns:
            self.benchmark_df['SMA_200'] = self.benchmark_df['Close'].rolling(window=200).mean()

        for ticker, df in self.data.items():
            if sma_col not in df.columns:
                df[sma_col] = df['Close'].rolling(window=self.exit_sma_period).mean()
            if 'Resist_40' not in df.columns:
                df['Resist_40'] = df['High'].shift(1).rolling(40).max()
            if 'Vol_SMA_50' not in df.columns:
                df['Vol_SMA_50'] = df['Volume'].shift(1).rolling(50).mean()

        # Definición de la ventana temporal de simulación (omitimos las primeras 252 barras de calentamiento)
        simulation_dates = self.benchmark_df.index[252:]
        factor_breakeven_neto = (1.0 + self.commission_rate) / (1.0 - self.commission_rate)

        # Calendario histórico de tipos oficiales de la Reserva Federal (remuneración de caja)
        HISTORICAL_FED_RATES = [
            ("2020-01-01", 0.0009),  # 0.09% (ZIRP post-COVID)
            ("2022-03-17", 0.0033),  # Primer alza (+25 bps)
            ("2022-05-05", 0.0083),  # +50 bps
            ("2022-06-16", 0.0158),  # +75 bps
            ("2022-07-28", 0.0233),  # +75 bps
            ("2022-09-22", 0.0308),  # +75 bps
            ("2022-11-03", 0.0383),  # +75 bps
            ("2022-12-15", 0.0433),  # +50 bps
            ("2023-02-02", 0.0458),  # +25 bps
            ("2023-03-23", 0.0483),  # +25 bps
            ("2023-05-04", 0.0508),  # +25 bps
            ("2023-07-27", 0.0533),  # Tipo terminal meseta (~5.33%)
            ("2024-09-19", 0.0483),  # Inicio ciclo recortes (-50 bps)
            ("2024-11-08", 0.0458),  # -25 bps
            ("2024-12-19", 0.0433),  # -25 bps (Rango 4.25%-4.50%)
            ("2025-09-18", 0.0408),  # -25 bps (Rango 4.00%-4.25%) tras pausa ene-ago
            ("2025-10-30", 0.0383),  # -25 bps (Rango 3.75%-4.00%)
            ("2025-12-11", 0.0358),  # -25 bps (Rango 3.50%-3.75%, reunión 10 dic)
        ]

        # Configuración de la serie de rendimiento diario del efectivo no invertido
        if isinstance(self.cash_yield_annual, (int, float)):
            annual_rates = pd.Series(float(self.cash_yield_annual), index=simulation_dates)
        elif isinstance(self.cash_yield_annual, pd.Series):
            annual_rates = self.cash_yield_annual.reindex(simulation_dates).ffill().fillna(0.0)
        elif str(self.cash_yield_annual).lower() in ('dynamic', 'historical'):
            sched_df = pd.DataFrame(HISTORICAL_FED_RATES, columns=['Date', 'Rate'])
            sched_df['Date'] = pd.to_datetime(sched_df['Date'])
            sim_tz = simulation_dates.tz if hasattr(simulation_dates, 'tz') else None
            if sim_tz is not None:
                sched_df['Date'] = sched_df['Date'].dt.tz_localize(sim_tz)
            sched_df = sched_df.set_index('Date')
            annual_rates = sched_df.reindex(sched_df.index.union(simulation_dates)).ffill().loc[simulation_dates]['Rate']
        else:
            annual_rates = pd.Series(0.0, index=simulation_dates)

        daily_yield_series = (1.0 + annual_rates) ** (1.0 / 252.0) - 1.0

        pending_buys = []
        pending_exits = []

        # Bucle principal de simulación diaria
        for i, current_date in enumerate(simulation_dates):
            is_last_day = (i == len(simulation_dates) - 1)
            
            # =========================================================================
            # FASE 1: APERTURA (t) - EJECUCIÓN DE ÓRDENES PENDIENTES Y REMUNERACIÓN DE CAJA
            # =========================================================================
            
            # 1.1 Remuneración diaria del capital ocioso en tesorería
            daily_rate = float(daily_yield_series.loc[current_date])
            if daily_rate > 0.001:
                daily_rate = 0.001  # Límite superior de seguridad diaria
                
            if self.capital > 0 and daily_rate > 0:
                self.capital += self.capital * daily_rate

            # 1.2 Ejecución de ventas pendientes activadas en t-1 (Cierre confirmado por SMA)
            for exit_order in pending_exits:
                ticker = exit_order['ticker']
                if ticker in self.positions and current_date in self.data[ticker].index:
                    pos = self.positions[ticker]
                    open_price = float(self.data[ticker].loc[current_date, 'Open'])
                    
                    gross_rev = pos['shares'] * open_price
                    net_rev = gross_rev * (1.0 - self.commission_rate)
                    pnl = net_rev - pos['total_cost']
                    pnl_pct = (open_price / pos['entry_price']) - 1.0
                    
                    self.capital += net_rev
                    self.trade_history.append({
                        'Ticker': ticker,
                        'Entry Date': pos['entry_date'].strftime('%Y-%m-%d'),
                        'Exit Date': current_date.strftime('%Y-%m-%d'),
                        'Entry Price': round(pos['entry_price'], 2),
                        'Exit Price': round(open_price, 2),
                        'PnL ($)': round(pnl, 2),
                        'Return (%)': round(pnl_pct * 100, 2),
                        'Reason': exit_order['reason']
                    })
                    del self.positions[ticker]

            pending_exits = [e for e in pending_exits if e['ticker'] in self.positions]

            # 1.3 Ejecución de compras pendientes cuya condición de breakout se validó al cierre de t-1
            if pending_buys and not is_last_day:
                for t, p in self.positions.items():
                    if current_date in self.data[t].index:
                        p['last_open'] = float(self.data[t].loc[current_date, 'Open'])
                        
                portfolio_value_open = self.capital + sum(
                    p['shares'] * p['last_open'] for p in self.positions.values()
                )
                
                for buy_order in pending_buys:
                    # Comprobación del límite de posiciones según slots dinámicos o fijos
                    if self.dynamic_slots:
                        current_occupied = sum(0.5 if p['partial_sold'] else 1.0 for p in self.positions.values())
                        if current_occupied >= self.max_positions:
                            break
                    else:
                        if len(self.positions) >= self.max_positions:
                            break

                    ticker = buy_order['ticker']
                    if ticker in self.positions or current_date not in self.data[ticker].index:
                        continue
                        
                    open_price = float(self.data[ticker].loc[current_date, 'Open'])
                    ref_close = buy_order.get('ref_close', open_price)

                    # Filtro Anti-Chasing: Descarta compras si la apertura es demasiado superior al cierre t-1
                    if self.max_chase_pct is not None:
                        if open_price > ref_close * (1.0 + self.max_chase_pct):
                            continue
                    
                    # Concentración dinámica de capital en la fase de arranque de la cartera
                    if self.dynamic_concentration and len(self.positions) <= 1:
                        target_size = portfolio_value_open / 3.0
                    else:
                        target_size = portfolio_value_open / self.max_positions
                        
                    allocated_cash = min(target_size, self.capital)
                    if allocated_cash < portfolio_value_open * 0.10:  # Mínimo del 10% del valor para abrir posición
                        continue
                        
                    shares = int(allocated_cash // (open_price * (1.0 + self.commission_rate)))
                    cost = shares * open_price * (1.0 + self.commission_rate)
                    
                    if shares > 0 and self.capital >= cost:
                        self.capital -= cost
                        self.positions[ticker] = {
                            'shares': shares,
                            'entry_price': open_price,
                            'entry_date': current_date,
                            'total_cost': cost,
                            'stop_loss': open_price * (1.0 - self.stop_loss_pct),
                            'partial_sold': False,
                            'reached_2r': False,
                            'sma_below_count': 0,
                            'last_open': open_price,
                            'last_close': open_price
                        }
            pending_buys.clear()

            # =========================================================================
            # FASE 2: INTRADÍA (t) - PRIORIDAD TEMPORAL Y GESTIÓN DE BENEFICIOS ASIMÉTRICOS
            # =========================================================================
            closed_intraday = []
            for ticker, pos in self.positions.items():
                if current_date not in self.data[ticker].index:
                    continue
                    
                df = self.data[ticker]
                row_today = df.loc[current_date]
                high_today = float(row_today['High'])
                low_today = float(row_today['Low'])
                open_today = float(row_today['Open'])
                close_today = float(row_today['Close'])
                
                # Umbrales de precios para trailing breakeven (+2R) y toma de beneficios parcial (+3R)
                breakeven_trigger = pos['entry_price'] * (1.0 + (2.0 * self.stop_loss_pct))
                stop_breakeven_neto = pos['entry_price'] * factor_breakeven_neto
                tp_trigger = pos['entry_price'] * (1.0 + (self.stop_loss_pct * 3.0))

                # ---------------------------------------------------------------------
                # SUBFASE 2.1: GAPS Y EVENTOS EN APERTURA (09:30 AM)
                # ---------------------------------------------------------------------
                # A) Apertura en Gap Down por debajo del Stop Loss
                if open_today <= pos['stop_loss']:
                    exit_price = open_today
                    gross_rev = pos['shares'] * exit_price
                    net_rev = gross_rev * (1.0 - self.commission_rate)
                    pnl = net_rev - pos['total_cost']
                    pnl_pct = (exit_price / pos['entry_price']) - 1.0
                    
                    self.capital += net_rev
                    reason = 'Stop Loss (Gap Down)' if pos['stop_loss'] < pos['entry_price'] else 'Breakeven Neto (Gap Down)'
                    
                    self.trade_history.append({
                        'Ticker': ticker,
                        'Entry Date': pos['entry_date'].strftime('%Y-%m-%d'),
                        'Exit Date': current_date.strftime('%Y-%m-%d'),
                        'Entry Price': round(pos['entry_price'], 2),
                        'Exit Price': round(exit_price, 2),
                        'PnL ($)': round(pnl, 2),
                        'Return (%)': round(pnl_pct * 100, 2),
                        'Reason': reason
                    })
                    closed_intraday.append(ticker)
                    continue

                # B) Apertura con Gap Up a nivel de Breakeven (+2R)
                if open_today >= breakeven_trigger:
                    pos['reached_2r'] = True
                    if pos['stop_loss'] < stop_breakeven_neto:
                        pos['stop_loss'] = stop_breakeven_neto

                # C) Apertura con Gap Up a nivel de Take Profit Parcial (+3R)
                if open_today >= tp_trigger and not pos['partial_sold']:
                    shares_to_sell = pos['shares'] // 2
                    if shares_to_sell > 0:
                        exec_price = open_today
                        gross_rev = shares_to_sell * exec_price
                        net_rev = gross_rev * (1.0 - self.commission_rate)
                        cost_sold = shares_to_sell * pos['entry_price'] * (1.0 + self.commission_rate)
                        pnl = net_rev - cost_sold
                        pnl_pct = (exec_price / pos['entry_price']) - 1.0
                        
                        self.capital += net_rev
                        self.trade_history.append({
                            'Ticker': ticker,
                            'Entry Date': pos['entry_date'].strftime('%Y-%m-%d'),
                            'Exit Date': current_date.strftime('%Y-%m-%d'),
                            'Entry Price': round(pos['entry_price'], 2),
                            'Exit Price': round(exec_price, 2),
                            'PnL ($)': round(pnl, 2),
                            'Return (%)': round(pnl_pct * 100, 2),
                            'Reason': 'TP Parcial (50% Gap Up)'
                        })
                        
                        pos['shares'] -= shares_to_sell
                        pos['total_cost'] -= cost_sold
                        pos['partial_sold'] = True
                        pos['reached_2r'] = True
                        if pos['stop_loss'] < stop_breakeven_neto:
                            pos['stop_loss'] = stop_breakeven_neto

                # ---------------------------------------------------------------------
                # SUBFASE 2.2: EVENTOS EN DESARROLLO INTRADÍA
                # ---------------------------------------------------------------------
                current_stop = pos['stop_loss']
                hits_stop = low_today <= current_stop
                hits_2r = high_today >= breakeven_trigger
                hits_tp = high_today >= tp_trigger and not pos['partial_sold']

                # Resolución de ambigüedad temporal intradía según el color de la vela diario
                if hits_stop and (hits_2r or hits_tp):
                    target_first = close_today < open_today  # En vela bajista, el máximo ocurrió primero
                else:
                    target_first = hits_2r or hits_tp

                if target_first and (hits_2r or hits_tp):
                    if hits_2r:
                        pos['reached_2r'] = True
                        if pos['stop_loss'] < stop_breakeven_neto:
                            pos['stop_loss'] = stop_breakeven_neto

                    if hits_tp:
                        shares_to_sell = pos['shares'] // 2
                        if shares_to_sell > 0:
                            exec_price = tp_trigger
                            gross_rev = shares_to_sell * exec_price
                            net_rev = gross_rev * (1.0 - self.commission_rate)
                            cost_sold = shares_to_sell * pos['entry_price'] * (1.0 + self.commission_rate)
                            pnl = net_rev - cost_sold
                            pnl_pct = (exec_price / pos['entry_price']) - 1.0

                            self.capital += net_rev
                            self.trade_history.append({
                                'Ticker': ticker,
                                'Entry Date': pos['entry_date'].strftime('%Y-%m-%d'),
                                'Exit Date': current_date.strftime('%Y-%m-%d'),
                                'Entry Price': round(pos['entry_price'], 2),
                                'Exit Price': round(exec_price, 2),
                                'PnL ($)': round(pnl, 2),
                                'Return (%)': round(pnl_pct * 100, 2),
                                'Reason': 'TP Parcial (50% Intradía)'
                            })
                            pos['shares'] -= shares_to_sell
                            pos['total_cost'] -= cost_sold
                            pos['partial_sold'] = True
                            pos['reached_2r'] = True
                            if pos['stop_loss'] < stop_breakeven_neto:
                                pos['stop_loss'] = stop_breakeven_neto

                    if hits_stop:
                        exit_price = pos['stop_loss']
                        gross_rev = pos['shares'] * exit_price
                        net_rev = gross_rev * (1.0 - self.commission_rate)
                        pnl = net_rev - pos['total_cost']
                        pnl_pct = (exit_price / pos['entry_price']) - 1.0

                        self.capital += net_rev
                        reason = 'Breakeven Neto (+2R)' if pos['stop_loss'] >= pos['entry_price'] else 'Stop Loss'
                        self.trade_history.append({
                            'Ticker': ticker,
                            'Entry Date': pos['entry_date'].strftime('%Y-%m-%d'),
                            'Exit Date': current_date.strftime('%Y-%m-%d'),
                            'Entry Price': round(pos['entry_price'], 2),
                            'Exit Price': round(exit_price, 2),
                            'PnL ($)': round(pnl, 2),
                            'Return (%)': round(pnl_pct * 100, 2),
                            'Reason': reason
                        })
                        closed_intraday.append(ticker)
                        continue

                elif hits_stop:
                    exit_price = current_stop
                    gross_rev = pos['shares'] * exit_price
                    net_rev = gross_rev * (1.0 - self.commission_rate)
                    pnl = net_rev - pos['total_cost']
                    pnl_pct = (exit_price / pos['entry_price']) - 1.0

                    self.capital += net_rev
                    reason = 'Stop Loss' if current_stop < pos['entry_price'] else 'Breakeven Neto (+2R)'
                    self.trade_history.append({
                        'Ticker': ticker,
                        'Entry Date': pos['entry_date'].strftime('%Y-%m-%d'),
                        'Exit Date': current_date.strftime('%Y-%m-%d'),
                        'Entry Price': round(pos['entry_price'], 2),
                        'Exit Price': round(exit_price, 2),
                        'PnL ($)': round(pnl, 2),
                        'Return (%)': round(pnl_pct * 100, 2),
                        'Reason': reason
                    })
                    closed_intraday.append(ticker)
                    continue

            # Eliminación de posiciones cerradas durante la sesión intradía
            for t in closed_intraday:
                del self.positions[t]

            # =========================================================================
            # FASE 3: CIERRE (t) - EVALUACIÓN DE REGRESION DE SMA, EARNINGS Y SELECCIÓN
            # =========================================================================
            for ticker, pos in self.positions.items():
                if current_date in self.data[ticker].index:
                    pos['last_close'] = float(self.data[ticker].loc[current_date, 'Close'])

            closed_at_close = []
            for ticker, pos in self.positions.items():
                # A) Cierre forzoso por finalización del período de backtest
                if is_last_day:
                    close_price = pos['last_close']
                    gross_rev = pos['shares'] * close_price
                    net_rev = gross_rev * (1.0 - self.commission_rate)
                    pnl = net_rev - pos['total_cost']
                    pnl_pct = (close_price / pos['entry_price']) - 1.0
                    self.capital += net_rev
                    self.trade_history.append({
                        'Ticker': ticker,
                        'Entry Date': pos['entry_date'].strftime('%Y-%m-%d'),
                        'Exit Date': current_date.strftime('%Y-%m-%d'),
                        'Entry Price': round(pos['entry_price'], 2),
                        'Exit Price': round(close_price, 2),
                        'PnL ($)': round(pnl, 2),
                        'Return (%)': round(pnl_pct * 100, 2),
                        'Reason': 'Cierre Fin Backtest'
                    })
                    closed_at_close.append(ticker)
                    continue

                if current_date not in self.data[ticker].index:
                    continue
                    
                df = self.data[ticker]
                row_today = df.loc[current_date]
                close_today = float(row_today['Close'])
                sma_exit = float(row_today[sma_col])

                # B) Liquidación preventiva por proximidad de publicación de resultados sin colchón suficiente
                if self._has_upcoming_earnings(ticker, current_date, window_days=self.earnings_exit_buffer):
                    unrealized_pnl = (close_today / pos['entry_price']) - 1.0
                    if not pos['reached_2r'] and unrealized_pnl < self.earnings_cushion_pct:
                        gross_rev = pos['shares'] * close_today
                        net_rev = gross_rev * (1.0 - self.commission_rate)
                        pnl = net_rev - pos['total_cost']
                        pnl_pct = (close_today / pos['entry_price']) - 1.0
                        
                        self.capital += net_rev
                        self.trade_history.append({
                            'Ticker': ticker,
                            'Entry Date': pos['entry_date'].strftime('%Y-%m-%d'),
                            'Exit Date': current_date.strftime('%Y-%m-%d'),
                            'Entry Price': round(pos['entry_price'], 2),
                            'Exit Price': round(close_today, 2),
                            'PnL ($)': round(pnl, 2),
                            'Return (%)': round(pnl_pct * 100, 2),
                            'Reason': f'Salida Earnings (Sin colchón: {unrealized_pnl*100:.1f}%)'
                        })
                        closed_at_close.append(ticker)
                        continue

                # C) Salida por pérdida confirmada de la media móvil (SMA Exit)
                if pos['reached_2r']:
                    if close_today < sma_exit:
                        pos['sma_below_count'] += 1
                        if self.sma_buffer_days == 0:
                            # Venta al cierre (Market On Close - MOC)
                            gross_rev = pos['shares'] * close_today
                            net_rev = gross_rev * (1.0 - self.commission_rate)
                            pnl = net_rev - pos['total_cost']
                            pnl_pct = (close_today / pos['entry_price']) - 1.0
                            self.capital += net_rev
                            self.trade_history.append({
                                'Ticker': ticker,
                                'Entry Date': pos['entry_date'].strftime('%Y-%m-%d'),
                                'Exit Date': current_date.strftime('%Y-%m-%d'),
                                'Entry Price': round(pos['entry_price'], 2),
                                'Exit Price': round(close_today, 2),
                                'PnL ($)': round(pnl, 2),
                                'Return (%)': round(pnl_pct * 100, 2),
                                'Reason': f'Cruce SMA {self.exit_sma_period} (MOC / Buffer 0d)'
                            })
                            closed_at_close.append(ticker)
                        elif pos['sma_below_count'] >= self.sma_buffer_days:
                            # Programación de venta en apertura del día siguiente (MOO t+1)
                            if not any(e['ticker'] == ticker for e in pending_exits):
                                pending_exits.append({
                                    'ticker': ticker,
                                    'reason': f'Cruce SMA {self.exit_sma_period} (Buffer {self.sma_buffer_days}d)'
                                })
                    else:
                        pos['sma_below_count'] = 0  # Reseteo de confirmaciones consecutivas

            for t in closed_at_close:
                del self.positions[t]

            # Registro diario del patrimonio neto ajustado a mercado (Mark-to-Market)
            portfolio_value_close = self.capital + sum(
                p['shares'] * p['last_close'] for p in self.positions.values()
            )
            self.equity_curve.append({'Date': current_date, 'Equity': portfolio_value_close})

            # Identificación de nuevas oportunidades de compra para ejecutar en la apertura t+1
            if not is_last_day:
                if self.dynamic_slots:
                    occupied_slots = sum(0.5 if p['partial_sold'] else 1.0 for p in self.positions.values())
                    slots_to_free = sum(0.5 if self.positions[e['ticker']]['partial_sold'] else 1.0 
                                        for e in pending_exits if e['ticker'] in self.positions)
                    net_occupied = max(0.0, occupied_slots - slots_to_free)
                    available_slots = int(np.floor(self.max_positions - net_occupied))
                else:
                    available_slots = self.max_positions - len(self.positions) + len(pending_exits)

                # Filtro de Régimen de Mercado: Exige que el Benchmark cotice por encima de su SMA de 200 días
                qqq_row = self.benchmark_df.loc[current_date]
                mercado_favorable = float(qqq_row['Close']) > float(qqq_row['SMA_200'])

                if available_slots > 0 and mercado_favorable:
                    candidates = []
                    for ticker, df in self.data.items():
                        if ticker in self.positions or any(e['ticker'] == ticker for e in pending_exits) or current_date not in df.index:
                            continue

                        # Veto por proximidad de entrega de resultados
                        if self._has_upcoming_earnings(ticker, current_date, window_days=self.earnings_entry_buffer):
                            continue
                        
                        row = df.loc[current_date]
                        trend_pass = row.get('Trend_Template_Pass', False)
                        if not pd.notna(trend_pass) or not bool(trend_pass):
                            continue
                        
                        try:
                            # Resistencia de consolidación de 40 sesiones (Base de 8 semanas)
                            resistance_40 = float(row['Resist_40']) if 'Resist_40' in row else float(df['High'].shift(1).rolling(40).max().loc[current_date])
                        except KeyError:
                            continue
                        vol_avg_50 = float(row['Vol_SMA_50'])
                        rs_ratio = float(row['RS_Ratio'])
                        high_52w = float(row['High_52w'])
                        close_price = float(row['Close'])

                        if (np.isnan(resistance_40) or np.isnan(vol_avg_50) or 
                            np.isnan(rs_ratio) or np.isnan(high_52w) or np.isnan(close_price)):
                            continue
                            
                        is_breakout = close_price > resistance_40
                        vol_institutional = float(row['Volume']) >= self.vol_mult * vol_avg_50
                        near_52w_high = close_price >= (high_52w * 0.85)
                        
                        if is_breakout and vol_institutional and near_52w_high:
                            candidates.append((ticker, rs_ratio))
                    
                    # Ordenación de candidatos por mayor fuerza relativa (Relative Strength)
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    for ticker, _ in candidates[:available_slots]:
                        row_candidate = self.data[ticker].loc[current_date]
                        pending_buys.append({
                            'ticker': ticker,
                            'ref_close': float(row_candidate['Close']),
                            'resistance': float(row_candidate['Resist_40'])
                        })

        # Construcción de los DataFrames de salida del backtest
        equity_df = pd.DataFrame(self.equity_curve).set_index('Date')
        trades_df = pd.DataFrame(self.trade_history)
        return equity_df, trades_df