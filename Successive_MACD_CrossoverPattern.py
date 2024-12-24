"""
Successfully grabs all SP500 stocks and undertakes 2-year backtests of the following successiveMACD strategy:
* BuyTrigger: any time there is an MACD crossover in negative territory.
* SuccessiveBuyTriggers: any time another MACD crossover occurs at a more positive point than the previous one.
* SellTrigger: any time the MACD falls below the most recent MACD crossover point

NEXT:
* Add the NASDAQ and Russell stocks
* Create a simpler version that simply sends notification triggers when MACD crossovers and successive MACD crossovers occur
* Formally assess the success of the strategy
* Add additional criteria to further optimize
"""

import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD
from concurrent.futures import ProcessPoolExecutor
import time
from datetime import datetime, timedelta
import re  # For sanitizing sector names
import os  # For debug prints

# --- Helper Functions ---

def calculate_max_drawdown(close_prices):
    """
    Calculates the maximum drawdown for a series of close prices.
    """
    cumulative_max = close_prices.cummax()
    drawdown = (close_prices - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100  # Convert to percentage
    return max_drawdown

def calculate_sharpe_ratio(close_prices, risk_free_rate=0):
    """
    Calculates the Sharpe ratio for a series of close prices.
    """
    returns = close_prices.pct_change().dropna()
    excess_returns = returns - (risk_free_rate / 252)  # Assuming 252 trading days
    if excess_returns.std() == 0:
        return np.nan
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)  # Annualized Sharpe Ratio
    return sharpe_ratio

###############################################################################
# NEW: identify_macd_sequences() to capture multiple sequences of crossovers
###############################################################################
def identify_macd_sequences(
    df, 
    min_days_between_crossovers=5, 
    macd_diff_threshold=0.05, 
    search_period_days=730
):
    """
    Returns a list of 'sequences', each sequence being a list of crossover datetimes
    for valid bullish MACD crossovers. When a reset occurs, we finalize that 
    sequence and start a new one.

    Criteria:
    1. First crossover requires MACD < 0.
    2. Successive crossovers require MACD be more positive than last crossover's MACD.
    3. Reset the current sequence if MACD < last_crossover_macd - 0.1.
    4. Must exceed macd_diff_threshold to be considered a valid crossover.
    5. Must be >= min_days_between_crossovers from the last crossover to be valid.
    """
    if 'MACD' not in df.columns or 'Signal_Line' not in df.columns:
        raise ValueError("DataFrame must contain 'MACD' and 'Signal_Line' columns")

    sequences = []
    current_sequence = []
    last_crossover_date = None
    last_crossover_macd = None

    latest_date = df.index.max()
    cutoff_date = latest_date - pd.Timedelta(days=search_period_days)

    for i in range(1, len(df)):
        current_date = df.index[i]
        if current_date < cutoff_date:
            continue

        current_macd = df['MACD'].iloc[i]
        prev_macd = df['MACD'].iloc[i - 1]
        current_signal = df['Signal_Line'].iloc[i]
        prev_signal = df['Signal_Line'].iloc[i - 1]

        # Check reset condition
        if last_crossover_macd is not None and current_macd < last_crossover_macd - 0.1:
            # finalize current sequence
            if current_sequence:
                sequences.append(current_sequence)
            current_sequence = []
            last_crossover_date = None
            last_crossover_macd = None

        # Detect bullish crossover
        if (current_macd > current_signal) and (prev_macd <= prev_signal):
            macd_diff = current_macd - current_signal
            if macd_diff >= macd_diff_threshold:
                if not current_sequence:
                    # First crossover in a new sequence: must be MACD < 0
                    if current_macd < 0:
                        current_sequence.append(current_date)
                        last_crossover_date = current_date
                        last_crossover_macd = current_macd
                else:
                    # Subsequent crossovers
                    if current_macd > last_crossover_macd:
                        days_since_last = (current_date - last_crossover_date).days
                        if days_since_last >= min_days_between_crossovers:
                            current_sequence.append(current_date)
                            last_crossover_date = current_date
                            last_crossover_macd = current_macd

    # finalize the last sequence
    if current_sequence:
        sequences.append(current_sequence)

    return sequences

def get_sp500_tickers_and_sectors():
    """
    Fetches the list of S&P 500 tickers and their corresponding sectors from Wikipedia.
    """
    try:
        wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(wiki_url)
        sp500_table = tables[0]
        
        if 'Symbol' not in sp500_table.columns or 'GICS Sector' not in sp500_table.columns:
            print("Required columns not found in the S&P 500 table.")
            return {}
        
        # Clean ticker symbols (e.g., BRK.B -> BRK-B)
        sp500_table['Symbol'] = sp500_table['Symbol'].str.replace('.', '-', regex=False)
        
        # Drop duplicates to ensure unique symbols
        sp500_table.drop_duplicates(subset=['Symbol'], inplace=True)

        ticker_sector_dict = dict(zip(sp500_table['Symbol'], sp500_table['GICS Sector']))
        
        print(f"Successfully fetched {len(ticker_sector_dict)} S&P 500 tickers and their sectors.")
        return ticker_sector_dict
    
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return {}
    
def get_nasdaq_tickers():
    url = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt"
    df = pd.read_csv(url, sep='|')

    # Filter out test/listed-only data
    df = df[df['Nasdaq Traded'] == 'Y']

    # Clean up ticker formatting
    df['Symbol'] = df['Symbol'].str.strip().str.upper()
    
    # Return just tickers as a list
    tickers = df['Symbol'].unique().tolist()
    return tickers
    
###########################################################
# Temporarily return just AMD and TSLA for debugging.
###########################################################
# def get_sp500_tickers_and_sectors():
#    """
#    
#    """
#    return {
#        "AMD": "Information Technology",
#        "TSLA": "Consumer Discretionary"
#    }
##########################################################

def get_macd_data(ticker, period="2y"):
    """
    Fetches historical data using yfinance's Ticker object with no caching/threads,
    then calculates MACD and returns a DataFrame.
    """
    # 1. Remove any hidden unicode or trailing spaces, uppercase.
    ticker_clean = ticker.strip().upper()
    print(f"[DEBUG] Original ticker='{ticker}' -> Cleaned ticker='{ticker_clean}'")

    # 2. Optionally clear requests_cache if installed, to disable any caching.
    try:
        import requests_cache
        requests_cache.clear()
        print("[DEBUG] Cleared requests_cache to disable all caching.")
    except ImportError:
        pass  # If you don't have requests_cache installed, we just skip

    # 3. Use Ticker(...).history(...) approach
    print(f"[DEBUG] Using yf.Ticker('{ticker_clean}').history(...) with period='{period}'...")
    ticker_obj = yf.Ticker(ticker_clean)
    df = ticker_obj.history(period=period, interval="1d", actions=False)

    if df.empty:
        print(f"[DEBUG] No data returned for {ticker_clean} with period='{period}'. Skipping.")
        return None

    days = (df.index.max() - df.index.min()).days
    print(f"[DEBUG] Downloaded {days} days of data for ticker={ticker_clean}, df.shape={df.shape}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    close_column = [col for col in df.columns if 'Close' in col]
    if not close_column:
        print(f"[DEBUG] Ticker {ticker_clean} has no 'Close' column. Skipping.")
        return None

    df['Close'] = df[close_column[0]].copy()

    macd_indicator = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd_indicator.macd().copy()
    df['Signal_Line'] = macd_indicator.macd_signal().copy()

    print(f"[DEBUG] Successfully computed MACD for {ticker_clean}. Returning DataFrame.")
    return df.copy()

def handle_insufficient_data(df, ticker):
    if df is None or df.empty:
        print(f"No valid data for {ticker}. Skipping.")
        return False
    if df.shape[0] < 252:
        print(f"Insufficient rows ({df.shape[0]}) for {ticker}. Skipping.")
        return False
    return True

def simulate_backtest_plan(df, buy_date, sell_threshold=0.1):
    """
    Simulates a backtest plan: buys at the buy_date and sells when MACD drops below 
    last_crossover_macd - sell_threshold.
    """
    try:
        buy_price = df.loc[buy_date, 'Close']
        last_crossover_macd = df.loc[buy_date, 'MACD']

        sell_condition = df['MACD'] < (last_crossover_macd - sell_threshold)
        sell_dates = df.loc[buy_date:].index[sell_condition.loc[buy_date:]]
        if not sell_dates.empty:
            sell_date = sell_dates[0]
        else:
            sell_date = df.index[-1]

        sell_price = df.loc[sell_date, 'Close']

        return_percentage = ((sell_price - buy_price) / buy_price) * 100

        holding_period = df.loc[buy_date:sell_date]
        max_drawdown = calculate_max_drawdown(holding_period['Close'])
        sharpe_ratio = calculate_sharpe_ratio(holding_period['Close'])

        return sell_date.date(), round(return_percentage, 2), round(max_drawdown, 2), round(sharpe_ratio, 2)

    except Exception as e:
        print(f"Error in backtest plan for buy_date {buy_date.date()}: {e}")
        return None, None, None, None

###############################################################################
# NEW: screen_stock() now uses identify_macd_sequences() to produce
# multiple rows (one per sequence) and accumulate summary_metrics for each.
###############################################################################
def screen_stock(ticker, sector, max_crossover_events=5):
    """
    Screens a single stock for MACD *sequences* and performs backtesting on 
    each crossover. Returns a list of rows (one row per sequence) plus summary_metrics.
    """
    print(f"\n=== Screening {ticker} in Sector: {sector} [PID {os.getpid()}] ===")
    df = get_macd_data(ticker, period="2y")
    if df is None or 'MACD' not in df:
        print(f"Skipping {ticker} due to missing or invalid data.")
        return None

    if not handle_insufficient_data(df, ticker):
        print(f"Skipping {ticker} due to insufficient data.")
        return None

    df = df.copy()
    df['12_month_avg'] = df['MACD'].rolling(window=252, min_periods=1).mean()
    df['MACD_std_12_month'] = df['MACD'].rolling(window=252, min_periods=1).std()
    df['5_day_avg'] = df['MACD'].rolling(window=5).mean()

    ###########################################################################
    # Instead of identify_macd_crossovers(...), we use identify_macd_sequences(...)
    ###########################################################################
    sequences = identify_macd_sequences(
        df,
        min_days_between_crossovers=5,
        macd_diff_threshold=0.05,
        search_period_days=730
    )

    if not sequences:
        print(f"No significant bullish MACD crossovers found for {ticker} within the last 2 years.")
        return None

    print(f"Detected {len(sequences)} sequences for {ticker}:")
    for seq in sequences:
        print("  ", [d.date() for d in seq])

    # We'll create one row per sequence and store relevant backtest data
    # in that row. We'll also accumulate summary_metrics for aggregator.
    per_sequence_rows = []
    summary_metrics = {}

    # Initialize aggregator counts
    total_wins = {idx: 0 for idx in range(1, max_crossover_events + 1)}
    total_returns = {idx: 0.0 for idx in range(1, max_crossover_events + 1)}
    event_counts = {idx: 0 for idx in range(1, max_crossover_events + 1)}

    # Build one row per sequence
    for seq_idx, seq_dates in enumerate(sequences, start=1):
        row = {
            "ticker": ticker,
            "sector": sector,
            "SequenceIndex": seq_idx
        }

        # Reference the last date in the sequence for additional metrics
        last_seq_date = seq_dates[-1]
        row["12_month_avg"] = df.loc[last_seq_date, "12_month_avg"]
        row["5_day_avg"] = df.loc[last_seq_date, "5_day_avg"]
        row["MACD"] = df.loc[last_seq_date, "MACD"]
        row["Signal_Line"] = df.loc[last_seq_date, "Signal_Line"]

        # Iterate over each crossover in the sequence
        for i, crossover_date in enumerate(seq_dates, start=1):
            if i > max_crossover_events:
                # Skip if exceeding max crossover events
                break

            # Store crossover date
            row[f"Crossover_{i}"] = crossover_date.date()

            # Perform backtest
            sell_date, return_pct, max_drawdown, sharpe_ratio = simulate_backtest_plan(df, crossover_date)

            if sell_date is not None:
                row[f"Sell_date_{i}"] = sell_date
                row[f"Return_%_{i}"] = return_pct
                row[f"MaxDD_{i}"] = max_drawdown
                row[f"Sharpe_{i}"] = sharpe_ratio

                # Update summary metrics
                if return_pct is not None:
                    event_counts[i] += 1
                    total_returns[i] += return_pct
                    if return_pct > 0:
                        total_wins[i] += 1

        # Append the row for this sequence
        per_sequence_rows.append(row)

    # Aggregate summary metrics across all sequences for this ticker
    for i in range(1, max_crossover_events + 1):
        total_trades = event_counts[i]
        if total_trades > 0:
            win_rate = (total_wins[i] / total_trades) * 100
            avg_return = total_returns[i] / total_trades
        else:
            win_rate = 0.0
            avg_return = 0.0

        summary_metrics[f'WinRate_%_{i}'] = round(win_rate, 2)
        summary_metrics[f'AvgReturnperTrade_%_{i}'] = round(avg_return, 2)
        summary_metrics[f'Number_of_Events_{i}'] = total_trades
        summary_metrics[f'Number_of_Wins_{i}'] = total_wins[i]
        summary_metrics[f'Total_Returns_{i}'] = total_returns[i]

    return per_sequence_rows, summary_metrics

def isolated_screen_stock(args):
    """
    Wrapper function to handle exceptions during stock screening.
    """
    ticker, sector = args
    print(f"[DEBUG] [PID {os.getpid()}] isolated_screen_stock -> ticker={ticker}, sector={sector}")
    try:
        return screen_stock(ticker, sector)
    except Exception as e:
        print(f"Error processing {ticker} in sector {sector}: {e}")
        return None

def screen_universe(ticker_sector_mapping, max_crossover_events=5):
    """
    Screens multiple tickers (serially) and aggregates their results.
    """
    per_stock_results = []
    summary_metrics_list = []
    
    # No concurrency – we just loop
    for ticker, sector in ticker_sector_mapping.items():
        print(f"[DEBUG] Running screen_stock for {ticker} in sector={sector} WITHOUT concurrency...")
        result = isolated_screen_stock((ticker, sector))
        if result:
            # result is (list_of_rows, summary_dict)
            rows, summary = result
            per_stock_results.extend(rows)
            summary_metrics_list.append(summary)
    
    # Initialize aggregated_summary
    aggregated_summary = {}
    for idx in range(1, max_crossover_events + 1):
        aggregated_summary[f'Total_Wins_{idx}'] = 0
        aggregated_summary[f'Total_Returns_{idx}'] = 0.0
        aggregated_summary[f'Number_of_Events_{idx}'] = 0
    
    # Summation
    for summary in summary_metrics_list:
        for idx in range(1, max_crossover_events + 1):
            number_of_wins = summary.get(f'Number_of_Wins_{idx}', 0)
            total_returns = summary.get(f'Total_Returns_{idx}', 0.0)
            number_of_events = summary.get(f'Number_of_Events_{idx}', 0)
            
            aggregated_summary[f'Total_Wins_{idx}'] += number_of_wins
            aggregated_summary[f'Total_Returns_{idx}'] += total_returns
            aggregated_summary[f'Number_of_Events_{idx}'] += number_of_events

    # Final stats
    summary_data = []
    for idx in range(1, max_crossover_events + 1):
        wins = aggregated_summary[f'Total_Wins_{idx}']
        rets = aggregated_summary[f'Total_Returns_{idx}']
        evts = aggregated_summary[f'Number_of_Events_{idx}']
        if evts > 0:
            win_rate = (wins / evts) * 100
            avg_return = rets / evts
        else:
            win_rate = 0.0
            avg_return = 0.0

        summary_data.append({
            "Strategy": f"MACD_Crossover_{idx}",
            "WinRate_%": round(win_rate, 2),
            "AvgReturnperTrade_%": round(avg_return, 2),
            "Number_of_Events": evts
        })

    summary_df = pd.DataFrame(summary_data)
    return per_stock_results, summary_df

# --- Core Function ---

def main():
    # Step 1: Fetch S&P 500 tickers and their sectors
    print("Fetching S&P 500 tickers and their sectors...")
    ticker_sector_mapping = get_sp500_tickers_and_sectors()
    if not ticker_sector_mapping:
        print("Failed to fetch S&P 500 tickers. Exiting.")
        return
    
    print(f"Total tickers fetched: {len(ticker_sector_mapping)}")
    
    # Step 2: Screen the universe of tickers
    per_stock_results, aggregated_summary = screen_universe(ticker_sector_mapping)
    
    if per_stock_results:
        results_df = pd.DataFrame(per_stock_results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'SuccessiveMACDstrategy_SP500_{timestamp}.csv'
        results_df.to_csv(csv_filename, index=False)
        print(f"\nBacktest results saved to '{csv_filename}'.")
        print("Matching stocks:")
        print(results_df.head())
    
        with open(csv_filename, 'a', newline='') as f:
            f.write('\n')
            aggregated_summary.to_csv(f, index=False, header=True)
    
        print(f"\nSummary statistics appended to '{csv_filename}'.")
        print("Summary Statistics per Strategy:")
        print(aggregated_summary)
    else:
        print("No stocks matched the criteria.")

# --- Execution Entry Point ---

if __name__ == "__main__":
    main()




