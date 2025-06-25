'''
V2 Gap Scalping, Mkt neutral (rupee neutral), net exposure = 0, scaling factor, NIFTY hedged
'''

import pandas as pd
import os
from datetime import datetime, time, timedelta
import glob
from collections import defaultdict
import numpy as np
import time as tm

class Backtester:
    def __init__(self):
        self.raw_data_path = "/home/t0013/ml_alpha/ankit_raghav/min_bars_generator/raw_data"
        self.universe_path = "/home/t0013/ml_alpha/intraday_momentum/universe_gap_scalping"
        self.signal_path = "/home/t0013/LR_open_gap_predictions"
        self.lot_size_data = pd.read_csv('/home/t0013/ml_alpha/intraday_momentum/contracts_n_lotsize_data/fut_contracts_n_lotsize_data_20250319.csv')
        self.mkt_cap_data = pd.read_csv('/home/t0013/ml_alpha/ankit_raghav/min_bars_generator/market_capitilization.csv')
        # Output paths
        self.tradesheet_path = "/home/t0013/gap_scalping/daily_tradesheets"
        self.results_path = "/home/t0013/gap_scalping/daily_results"
        self.summary_path = "/home/t0013/gap_scalping/daily_summary"
        self.positions_path = "/home/t0013/gap_scalping/positions"
        os.makedirs(self.tradesheet_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.summary_path, exist_ok=True)
        os.makedirs(self.positions_path, exist_ok=True)
        self.entry_time = "15:16:00"
        self.exit_time = "09:16:00"
        self.final_exit_time = "09:17:00"
        self.max_capital = 100000000
        self.reset_state()
        
    def reset_state(self):
        """Reset all state variables for a new backtest run"""
        self.signals = {}
        self.trade_count = defaultdict(dict)
        self.positions = []
        self.trades = []
        self.daily_capital_deployed = 0
        self.trade_sheet = pd.DataFrame(columns=['date', 'trade_id', 'tcp_id', 'ticker',
                                               'entry_time', 'exit_time', 'entry_price',
                                               'exit_price', 'qty', 'pnl', 'deployed_capital',
                                               'close_entry', 'close_exit'])
        self.trade_id = 0
        self.results = []
        self.traded_tcp_ids = set()
        
    def get_lot_size(self, tcp_id):
        """Get lot size for a given tcp_id"""
        lot_size = self.lot_size_data[self.lot_size_data['tcp_id'] == tcp_id]['lot_size'].values
        if len(lot_size) > 0:
            return lot_size[0]
        else:
            print(f"Skipping tcp_id {tcp_id}: No lot size found")
            return 0

    def get_lot_qty(self, tcp_id):
        label = self.mkt_cap_data[self.mkt_cap_data['tcp_id'] == tcp_id]['mkt_cap_label'].values
        try:
            if label[0] == 3:
                lot_qty = 2
            elif label[0] == 2:
                lot_qty = 3
            elif label[0] == 1:
                lot_qty = 1
            else:
                lot_qty = 1
            return lot_qty
        except Exception as e:
            print(f"Error in lot qty: {str(e)}")
            lot_qty = 1
            return lot_qty
    
    def capital_limit_reached(self, potential_trade_capital):
        """Check if adding a new trade would exceed the capital limit"""
        projected_capital = self.daily_capital_deployed + potential_trade_capital
        if projected_capital > self.max_capital:
            print(f"Capital limit would be exceeded. Current: {self.daily_capital_deployed}, "
                  f"New trade: {potential_trade_capital}, Limit: {self.max_capital}")
            return True
        return False
    
    def can_trade_tcp_id(self, tcp_id, trade_date, current_time):
        """Check if a tcp_id can be traded at the current time"""
        if any(pos['tcp_id'] == tcp_id for pos in self.positions):
            return False
        if tcp_id in self.traded_tcp_ids:
            return False
        for trade in self.trades:
            if trade['tcp_id'] == tcp_id and trade['exit_time'] == current_time:
                return False
        return True
            
    def filter_data_by_time_range(self, df):
        """Filter data to 14:59:00 to 15:30:00 time range"""
        return df[(df['time'] >= time(14, 39)) & (df['time'] <= time(15, 30))].sort_values('time')
    
    def get_next_trading_date(self, date_str):
        """Get the next trading date from the current date"""
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        next_date_obj = date_obj + timedelta(days=1)
        next_date_str = next_date_obj.strftime('%Y%m%d')
        raw_file = os.path.join(self.raw_data_path, f"{next_date_str}.pkl")
        if os.path.isfile(raw_file):
            return next_date_str
        date_files = glob.glob(os.path.join(self.raw_data_path, "*.pkl"))
        date_strs = sorted([os.path.basename(f).split('.')[0] for f in date_files])
        for potential_date in date_strs:
            if potential_date > date_str:
                return potential_date
        return None

    def load_signal_data(self, date):
        """Load signal data for a specific date"""
        date_formatted = datetime.strptime(date, '%Y%m%d').strftime('%Y%m%d')
        signal_file = os.path.join(self.signal_path, f"LR_gap_pred_{date_formatted}.csv")
        if not os.path.exists(signal_file):
            print(f"Signal file not found: {signal_file}")
            return pd.DataFrame()
        signal_df = pd.read_csv(signal_file)
        print(f"Loaded signal data for {date} - Rows: {len(signal_df)}")
        return signal_df

    def get_available_dates(self):
        """Get all dates available in both raw and universe data"""
        raw_files = glob.glob(os.path.join(self.raw_data_path, "*.pkl"))
        universe_files = glob.glob(os.path.join(self.universe_path, "universe_*.csv"))
        raw_dates = {os.path.basename(f).split('.')[0] for f in raw_files}
        universe_dates = {os.path.basename(f).split('_')[1].split('.')[0] for f in universe_files}
        valid_dates = []
        for date in sorted(list(raw_dates.intersection(universe_dates))):
            next_date = self.get_next_trading_date(date)
            if next_date:
                valid_dates.append(date) 
        return valid_dates
    
    def load_data_for_date(self, date):
        """Load and prepare data for a specific date"""
        raw_file = os.path.join(self.raw_data_path, f"{date}.pkl")
        universe_file = os.path.join(self.universe_path, f"universe_{date}.csv")
        print(f"Loading raw data from: {raw_file}")
        print(f"Loading universe data from: {universe_file}")
        raw_df = pd.read_pickle(raw_file)
        universe_df = pd.read_csv(universe_file)
        new_row = {'tcp_id': 10500001,'exchange_ticker': 'NIFTY','cs_zscore': 0, 'label': 3}
        universe_df = pd.concat([universe_df, pd.DataFrame([new_row])], ignore_index=True)
        raw_df['time'] = pd.to_datetime(raw_df['time'], format='%H:%M:%S').dt.time
        if 'trade_date' in universe_df.columns:
            universe_df = universe_df.drop(columns=['trade_date'])
        date_str = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
        raw_df['trade_date'] = date_str
        unique_tickers = universe_df['exchange_ticker'].unique()
        filtered_raw_df = raw_df[raw_df['exchange_ticker'].isin(unique_tickers)]
        filtered_raw_df = self.filter_data_by_time_range(filtered_raw_df)
        filtered_raw_df = filtered_raw_df.drop_duplicates()
        universe_df = universe_df.drop_duplicates()
        merged_df = pd.merge(
            filtered_raw_df,
            universe_df,
            on=['tcp_id', 'exchange_ticker'],
            how='inner'
        )
        print(f"Data loaded - Raw rows: {len(filtered_raw_df)}, Universe rows: {len(universe_df)}, Merged rows: {len(merged_df)}")
        return merged_df, universe_df
    
    def load_next_day_data_for_exit(self, date, position_tcp_ids):
        """Load next day's data for exit"""
        next_date = self.get_next_trading_date(date)
        if not next_date:
            print(f"No next trading date available after {date}")
            return None
        raw_file = os.path.join(self.raw_data_path, f"{next_date}.pkl")
        print(f"Loading next day raw data from: {raw_file} for exit")
        raw_df = pd.read_pickle(raw_file)
        raw_df['time'] = pd.to_datetime(raw_df['time'], format='%H:%M:%S').dt.time
        date_str = datetime.strptime(next_date, '%Y%m%d').strftime('%Y-%m-%d')
        raw_df['trade_date'] = date_str
        filtered_raw_df = raw_df[raw_df['tcp_id'].isin(position_tcp_ids)]
        exit_time_obj = datetime.strptime(self.exit_time, '%H:%M:%S').time()
        final_exit_time_obj = datetime.strptime(self.final_exit_time, '%H:%M:%S').time()
        filtered_exit_df = filtered_raw_df[
            (filtered_raw_df['time'] == exit_time_obj) | 
            (filtered_raw_df['time'] == final_exit_time_obj)
        ]
        print(f"Next day data loaded for exit - Rows: {len(filtered_exit_df)}")
        return filtered_exit_df
        


    def get_data_at_time(self, raw_df, target_time='15:16:00', tcp_id=10500001):
        """Get data at specified time or next available time up to 15:20"""
        try:
            base_time = datetime.strptime(target_time, '%H:%M:%S')
            max_time = datetime.strptime('15:20:00', '%H:%M:%S')
            current_time = base_time
            while current_time <= max_time:
                time_str = current_time.strftime('%H:%M:%S')
                time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
                # Filter data for current time
                raw_filtered = raw_df[raw_df['time'] == time_obj]
                # Merge filtered data
                current_data = raw_filtered
                # Filter for specific tcp_id
                tcp_data = current_data[current_data['tcp_id'] == tcp_id]
                if not tcp_data.empty:
                    return tcp_data.iloc[0], time_str
                current_time += timedelta(minutes=1)
            return None, None
        except Exception as e:
            print(f"Error in get_data_at_time: {str(e)}")
            return None, None

    

    def load_next_day_data_for_exit(self, date, position_tcp_ids):
        """Load next day's data for exit"""
        next_date = self.get_next_trading_date(date)
        if not next_date:
            print(f"No next trading date available after {date}")
            return None
        raw_file = os.path.join(self.raw_data_path, f"{next_date}.pkl")
        print(f"Loading next day raw data from: {raw_file} for exit")
        raw_df = pd.read_pickle(raw_file)
        raw_df['time'] = pd.to_datetime(raw_df['time'], format='%H:%M:%S').dt.time
        date_str = datetime.strptime(next_date, '%Y%m%d').strftime('%Y-%m-%d')
        raw_df['trade_date'] = date_str
        filtered_raw_df = raw_df[raw_df['tcp_id'].isin(position_tcp_ids)]
        exit_time_obj = datetime.strptime(self.exit_time, '%H:%M:%S').time()
        final_exit_time_obj = datetime.strptime(self.final_exit_time, '%H:%M:%S').time()
        filtered_exit_df = filtered_raw_df[
            (filtered_raw_df['time'] == exit_time_obj) | 
            (filtered_raw_df['time'] == final_exit_time_obj)
        ]
        print(f"Next day data loaded for exit - Rows: {len(filtered_exit_df)}")
        return filtered_exit_df


    def get_data_at_time(self, raw_df, target_time='15:16:00', tcp_id=10500001):
            """Get data at specified time or next available time up to 15:20"""
            try:
                base_time = datetime.strptime(target_time, '%H:%M:%S')
                max_time = datetime.strptime('15:20:00', '%H:%M:%S')
                current_time = base_time
                while current_time <= max_time:
                    time_str = current_time.strftime('%H:%M:%S')
                    time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
                    # Filter data for current time
                    raw_filtered = raw_df[raw_df['time'] == time_obj]
                    # Merge filtered data
                    current_data = raw_filtered
                    # Filter for specific tcp_id
                    tcp_data = current_data[current_data['tcp_id'] == tcp_id]
                    if not tcp_data.empty:
                        return tcp_data.iloc[0], time_str
                    current_time += timedelta(minutes=1)
                return None, None
            except Exception as e:
                print(f"Error in get_data_at_time: {str(e)}")
                return None, None


    def enter_trade(self, current_data, universe_df, entry_time):
            """Process entries for the current time period with enhanced market neutral strategy
            - Implements position size limits as percentage of portfolio
            - Uses scaling factors to precisely match long and short exposures
            - Processes all valid signals rather than targeting a fixed number of positions
            - Balances with NIFTY when one side has no valid candidates
            """
            current_date = universe_df['trade_date'].iloc[0].split()[0].replace('-', '') if 'trade_date' in universe_df.columns else current_data['trade_date'].iloc[0].split()[0].replace('-', '')
            next_date = self.get_next_trading_date(current_date)
            if not next_date:
                print("No next trading date available for signal data, skipping entries")
                return
            signal_df = self.load_signal_data(next_date)
            if signal_df.empty:
                print("No signal data available for next day, skipping entries")
                return
            entry_time_obj = datetime.strptime(entry_time, '%H:%M:%S').time()
            entry_data = current_data[current_data['time'] == entry_time_obj]
            if entry_data.empty:
                print(f"No data available at entry time {entry_time}")
                return
            # print(entry_data[entry_data['exchange_ticker']=='NIFTY'])
            new_row = {'tcp_id': 10500001,'exchange_ticker': 'NIFTY','cs_zscore': 0, 'label': 3}
            universe_df = pd.concat([universe_df, pd.DataFrame([new_row])], ignore_index=True)
            combined_df = pd.merge(
                universe_df,
                signal_df[['tcp_id', 'predicted_gap_sign']],
                on=['tcp_id'],
                how='inner'
            )
            if combined_df.empty:
                print("No matches found between universe and signal data")
                return
            # Filter potential long and short candidates based on labels and signals
            long_securities = combined_df[(combined_df['label'].isin([4, 5])) & (combined_df['predicted_gap_sign'] == 1)]
            short_securities = combined_df[(combined_df['label'].isin([1, 2])) & (combined_df['predicted_gap_sign'] == 0)]
            total_signals = len(long_securities) + len(short_securities)
            print(f"Found {len(long_securities)} potential long entries and {len(short_securities)} potential short entries (total signals: {total_signals})")
        
            # Create a potential trades list with all the info needed for selection
            potential_longs = []
            potential_shorts = []
        
            # Process long candidates
            for _, row in long_securities.iterrows():
                tcp_id = row['tcp_id']
                exchange_ticker = row['exchange_ticker']
                ticker_data = entry_data[entry_data['tcp_id'] == tcp_id]
                if ticker_data.empty:
                    continue
                if not self.can_trade_tcp_id(tcp_id, ticker_data.iloc[0]['trade_date'], entry_time):
                    continue
                lot_size = self.get_lot_size(tcp_id)
                if lot_size <= 0:
                    continue
                close_price = ticker_data.iloc[0]['close']
                potential_longs.append({
                    "tcp_id": tcp_id,
                    "exchange_ticker": exchange_ticker,
                    "lot_size": lot_size,
                    "close_price": close_price,
                    "position_value": lot_size * close_price,
                    "trade_date": ticker_data.iloc[0]['trade_date'],
                    "ticker_data": ticker_data.iloc[0]
                })
            
        
            # Process short candidates
            for _, row in short_securities.iterrows():
                tcp_id = row['tcp_id']
                exchange_ticker = row['exchange_ticker']
                ticker_data = entry_data[entry_data['tcp_id'] == tcp_id]
                if ticker_data.empty:
                    continue
                if not self.can_trade_tcp_id(tcp_id, ticker_data.iloc[0]['trade_date'], entry_time):
                    continue
                lot_size = self.get_lot_size(tcp_id)
                if lot_size <= 0:
                    continue
                close_price = ticker_data.iloc[0]['close']
                potential_shorts.append({
                    "tcp_id": tcp_id,
                    "exchange_ticker": exchange_ticker,
                    "lot_size": lot_size,
                    "close_price": close_price,
                    "position_value": lot_size * close_price,
                    "trade_date": ticker_data.iloc[0]['trade_date'],
                    "ticker_data": ticker_data.iloc[0]
                })
            
            print(f"Found {len(potential_longs)} valid long candidates and {len(potential_shorts)} valid short candidates")
        
            # Handle cases where one side has no candidates by balancing with NIFTY
            if not potential_longs and not potential_shorts:
                print("No valid candidates on either side, skipping entries")
                return
                
            # Calculate position limits as percentage of portfolio
            max_position_pct = 100 / max(total_signals, 1)  # Avoid division by zero
            max_position_value = self.max_capital * (max_position_pct / 100)
            print(f"Setting maximum position value to {max_position_value:.2f} ({max_position_pct}% of total capital)")
        
            # Limit position sizes based on max percentage
            for long_pos in potential_longs:
                if long_pos["position_value"] > max_position_value:
                    # Calculate how many lots we can use to stay under the limit
                    original_lots = long_pos["lot_size"]
                    max_lots = int(max_position_value / long_pos["close_price"])
                    # Ensure we have at least 1 lot
                    max_lots = max(1, max_lots)
                    long_pos["lot_size"] = max_lots
                    long_pos["position_value"] = max_lots * long_pos["close_price"]
                    print(f"Limiting {long_pos['exchange_ticker']} position from {original_lots} lots to {max_lots} lots to stay under position limit")
        
            for short_pos in potential_shorts:
                if short_pos["position_value"] > max_position_value:
                    # Calculate how many lots we can use to stay under the limit
                    original_lots = short_pos["lot_size"]
                    max_lots = int(max_position_value / short_pos["close_price"])
                    # Ensure we have at least 1 lot
                    max_lots = max(1, max_lots)
                    short_pos["lot_size"] = max_lots
                    short_pos["position_value"] = max_lots * short_pos["close_price"]
                    print(f"Limiting {short_pos['exchange_ticker']} position from {original_lots} lots to {max_lots} lots to stay under position limit")
        
            # Sort by signal strength (proxy: label value for longs, inverse label for shorts)
            # This ensures we prioritize strongest signals when selecting securities
            if potential_longs:
                long_securities_with_label = pd.merge(
                    pd.DataFrame(potential_longs),
                    combined_df[['tcp_id', 'label']],
                    on='tcp_id',
                    how='inner'
                )
                
                # Sort by label (higher is stronger for longs)
                potential_longs_sorted = []
                for _, row in long_securities_with_label.sort_values('label', ascending=False).iterrows():
                    for pos in potential_longs:
                        if pos['tcp_id'] == row['tcp_id']:
                            potential_longs_sorted.append(pos)
                            break
                potential_longs = potential_longs_sorted

            if potential_shorts:
                short_securities_with_label = pd.merge(
                    pd.DataFrame(potential_shorts),
                    combined_df[['tcp_id', 'label']],
                    on='tcp_id',
                    how='inner'
                )
                
                # Sort by label (lower is stronger for shorts)
                potential_shorts_sorted = []
                for _, row in short_securities_with_label.sort_values('label', ascending=True).iterrows():
                    for pos in potential_shorts:
                        if pos['tcp_id'] == row['tcp_id']:
                            potential_shorts_sorted.append(pos)
                            break
                potential_shorts = potential_shorts_sorted
        
            print(f"Selected {len(potential_longs)} long candidates and {len(potential_shorts)} short candidates based on signal strength")
        
            # Calculate maximum allowable capital per side (half of total max capital)
            max_side_capital = self.max_capital / 2
        
            # First calculate total values on each side
            long_total_value = sum(pos["position_value"] for pos in potential_longs)
            short_total_value = sum(pos["position_value"] for pos in potential_shorts)
        
            print(f"Initial long value: {long_total_value:.2f}, Initial short value: {short_total_value:.2f}")
            print(f"Imbalance before scaling: {abs(long_total_value - short_total_value):.2f}")
            
            # Check if we need to balance with NIFTY
            need_nifty_balance = False
            nifty_side = None
            nifty_balance_value = 0
            
            if not potential_shorts and potential_longs:
                # We have longs but no shorts - need to short NIFTY
                need_nifty_balance = True
                nifty_side = "Short"
                nifty_balance_value = min(long_total_value, max_side_capital)
                print(f"No short candidates available - will balance with SHORT position in NIFTY worth {nifty_balance_value:.2f}")
            elif not potential_longs and potential_shorts:
                # We have shorts but no longs - need to long NIFTY
                need_nifty_balance = True
                nifty_side = "Long"
                nifty_balance_value = min(short_total_value, max_side_capital)
                print(f"No long candidates available - will balance with LONG position in NIFTY worth {nifty_balance_value:.2f}")
            
            # Apply scaling factor to precisely match exposures for normal case (both longs and shorts available)
            if potential_longs and potential_shorts:
                # Calculate the target value (minimum of the two sides)
                target_value = min(long_total_value, short_total_value, max_side_capital)
            
                # Calculate scaling factors
                long_scaling_factor = target_value / long_total_value
                short_scaling_factor = target_value / short_total_value
            
                print(f"Long scaling factor: {long_scaling_factor:.4f}, Short scaling factor: {short_scaling_factor:.4f}")
            
                # Apply scaling factors to each position's lot size
                scaled_longs = []
                scaled_shorts = []
                adjusted_long_total = 0
                adjusted_short_total = 0
            
                # Scale long positions
                for pos in potential_longs:
                    # Scale the lots, rounding to ensure we have whole lots
                    original_lots = pos["lot_size"]
                    adjusted_lots = max(1, round(original_lots * long_scaling_factor))
                    adjusted_value = adjusted_lots * pos["close_price"]
                
                    # Create a copy of the position with adjusted values
                    adjusted_pos = pos.copy()
                    adjusted_pos["lot_size"] = adjusted_lots
                    adjusted_pos["position_value"] = adjusted_value
                    adjusted_pos["scaling_applied"] = long_scaling_factor
                    adjusted_pos["original_lots"] = original_lots
                
                    scaled_longs.append(adjusted_pos)
                    adjusted_long_total += adjusted_value
            
                # Scale short positions
                for pos in potential_shorts:
                    # Scale the lots, rounding to ensure we have whole lots
                    original_lots = pos["lot_size"]
                    adjusted_lots = max(1, round(original_lots * short_scaling_factor))
                    adjusted_value = adjusted_lots * pos["close_price"]
                
                    # Create a copy of the position with adjusted values
                    adjusted_pos = pos.copy()
                    adjusted_pos["lot_size"] = adjusted_lots
                    adjusted_pos["position_value"] = adjusted_value
                    adjusted_pos["scaling_applied"] = short_scaling_factor
                    adjusted_pos["original_lots"] = original_lots
                
                    scaled_shorts.append(adjusted_pos)
                    adjusted_short_total += adjusted_value
            
                print(f"After scaling - Long value: {adjusted_long_total:.2f}, Short value: {adjusted_short_total:.2f}")
                print(f"Remaining imbalance: {abs(adjusted_long_total - adjusted_short_total):.2f} ({abs(adjusted_long_total - adjusted_short_total)/target_value*100:.2f}%)")
            
                # Final adjustment: if there's still an imbalance, try to fine-tune one more position
                if abs(adjusted_long_total - adjusted_short_total) > 0:
                    imbalance = abs(adjusted_long_total - adjusted_short_total)
                    if adjusted_long_total > adjusted_short_total:
                        # Long side is larger, try to reduce one position
                        for i, pos in enumerate(scaled_longs):
                            if pos["lot_size"] > 1:  # Can't reduce below 1 lot
                                pos["lot_size"] -= 1
                                pos["position_value"] = pos["lot_size"] * pos["close_price"]
                                adjusted_long_total = sum(p["position_value"] for p in scaled_longs)
                                if adjusted_long_total <= adjusted_short_total:
                                    print(f"Fine-tuned {pos['exchange_ticker']} to reduce long exposure")
                                    break
                    else:
                        # Short side is larger, try to reduce one position
                        for i, pos in enumerate(scaled_shorts):
                            if pos["lot_size"] > 1:  # Can't reduce below 1 lot
                                pos["lot_size"] -= 1
                                pos["position_value"] = pos["lot_size"] * pos["close_price"]
                                adjusted_short_total = sum(p["position_value"] for p in scaled_shorts)
                                if adjusted_short_total <= adjusted_long_total:
                                    print(f"Fine-tuned {pos['exchange_ticker']} to reduce short exposure")
                                    break
            
                # Final calculation of total values
                final_long_total = sum(pos["position_value"] for pos in scaled_longs)
                final_short_total = sum(pos["position_value"] for pos in scaled_shorts)
            
                print(f"Final portfolio - Long value: {final_long_total:.2f}, Short value: {final_short_total:.2f}")
                print(f"Final imbalance: {abs(final_long_total - final_short_total):.2f} ({abs(final_long_total - final_short_total)/(final_long_total+final_short_total)*200:.2f}%)")
            
                # Enter the selected trades
                for long_pos in scaled_longs:
                    trade = {
                        "trade_date": long_pos["trade_date"],
                        "entry_time": entry_time,
                        "tcp_id": long_pos["tcp_id"],
                        "exchange_ticker": long_pos["exchange_ticker"],
                        "side": "Long",
                        "qty": long_pos["lot_size"],
                        "entry_price": long_pos["close_price"],
                        "close_entry": long_pos["close_price"]
                    }
                    self.positions.append(trade)
                    self.traded_tcp_ids.add(long_pos["tcp_id"])
                    self.daily_capital_deployed += long_pos["position_value"]
                    print(f"Entered Long position for {long_pos['exchange_ticker']} at {entry_time}, price: {long_pos['close_price']}, qty: {long_pos['lot_size']} (scaled from {long_pos.get('original_lots', 'N/A')})")
            
                for short_pos in scaled_shorts:
                    trade = {
                        "trade_date": short_pos["trade_date"],
                        "entry_time": entry_time,
                        "tcp_id": short_pos["tcp_id"],
                        "exchange_ticker": short_pos["exchange_ticker"],
                        "side": "Short",
                        "qty": short_pos["lot_size"],
                        "entry_price": short_pos["close_price"],
                        "close_entry": short_pos["close_price"]
                    }
                    self.positions.append(trade)
                    self.traded_tcp_ids.add(short_pos["tcp_id"])
                    self.daily_capital_deployed += short_pos["position_value"]
                    print(f"Entered Short position for {short_pos['exchange_ticker']} at {entry_time}, price: {short_pos['close_price']}, qty: {short_pos['lot_size']} (scaled from {short_pos.get('original_lots', 'N/A')})")
            
            # Handle NIFTY balancing case (when one side has no candidates)
            if need_nifty_balance:
                nifty_tcp_id = 10500001  # TCP ID for NIFTY index futures
                # print(current_data)
                nifty_ticker_data = current_data[current_data['exchange_ticker'] == "NIFTY"]
                
                if nifty_ticker_data.empty:
                    print(f"Error: Cannot find NIFTY data at {entry_time} for balancing")
                    return
                    
                nifty_price = nifty_ticker_data.iloc[0]['close']
                nifty_lot_size = self.get_lot_size(nifty_tcp_id)
                
                # Calculate how many lots needed to match the balance value
                nifty_lots_needed = max(1, int(nifty_balance_value / (nifty_price * nifty_lot_size)))
                nifty_actual_value = nifty_lots_needed * nifty_lot_size * nifty_price
                
                print(f"Using {nifty_lots_needed} lots of NIFTY to balance with {nifty_actual_value:.2f} worth of exposure")
                
                # Create and enter the NIFTY trade
                nifty_trade = {
                    "trade_date": nifty_ticker_data.iloc[0]['trade_date'],
                    "entry_time": entry_time,
                    "tcp_id": nifty_tcp_id,
                    "exchange_ticker": "NIFTY",
                    "side": nifty_side,
                    "qty": nifty_lots_needed * nifty_lot_size,
                    "entry_price": nifty_price,
                    "close_entry": nifty_price,
                    "is_balancing": True
                }
                
                self.positions.append(nifty_trade)
                self.traded_tcp_ids.add(nifty_tcp_id)
                self.daily_capital_deployed += nifty_actual_value
                print(f"Entered {nifty_side} position for NIFTY at {entry_time}, price: {nifty_price}, qty: {nifty_lots_needed * nifty_lot_size} to balance portfolio")
                
                # Enter the other side regular trades as well
                if potential_longs and not potential_shorts:
                    for long_pos in potential_longs:
                        trade = {
                            "trade_date": long_pos["trade_date"],
                            "entry_time": entry_time,
                            "tcp_id": long_pos["tcp_id"],
                            "exchange_ticker": long_pos["exchange_ticker"],
                            "side": "Long",
                            "qty": long_pos["lot_size"],
                            "entry_price": long_pos["close_price"],
                            "close_entry": long_pos["close_price"]
                        }
                        self.positions.append(trade)
                        self.traded_tcp_ids.add(long_pos["tcp_id"])
                        self.daily_capital_deployed += long_pos["position_value"]
                        print(f"Entered Long position for {long_pos['exchange_ticker']} at {entry_time}, price: {long_pos['close_price']}, qty: {long_pos['lot_size']}")
                elif potential_shorts and not potential_longs:
                    for short_pos in potential_shorts:
                        trade = {
                            "trade_date": short_pos["trade_date"],
                            "entry_time": entry_time,
                            "tcp_id": short_pos["tcp_id"],
                            "exchange_ticker": short_pos["exchange_ticker"],
                            "side": "Short",
                            "qty": short_pos["lot_size"],
                            "entry_price": short_pos["close_price"],
                            "close_entry": short_pos["close_price"]
                        }
                        self.positions.append(trade)
                        self.traded_tcp_ids.add(short_pos["tcp_id"])
                        self.daily_capital_deployed += short_pos["position_value"]
                        print(f"Entered Short position for {short_pos['exchange_ticker']} at {entry_time}, price: {short_pos['close_price']}, qty: {short_pos['lot_size']}")
                        
            print(f"Entry processing complete - {len(self.positions)} total positions in portfolio, capital deployed: {self.daily_capital_deployed:.2f}")




    def save_positions(self, date):
        """Save current positions to a CSV file"""
        if not self.positions:
            print("No positions to save")
            return
        positions_df = pd.DataFrame(self.positions)
        positions_path = os.path.join(self.positions_path, f"positions_{date}.csv")
        positions_df.to_csv(positions_path, index=False)
        print(f"Positions saved to {positions_path}")
            
    def exit_trade(self, current_data, exit_time):
        """Process exits for all open positions at the specified exit time"""
        if not self.positions:
            return
        print(f"Exiting trades at {exit_time}")
        tcp_data_map = {row['tcp_id']: row for _, row in current_data.iterrows() if row['time'] == datetime.strptime(exit_time, '%H:%M:%S').time()}
        positions_copy = self.positions.copy()
        for position in positions_copy:
            tcp_id = position['tcp_id']
            if tcp_id in tcp_data_map:
                self.close_trade(position, tcp_data_map[tcp_id], exit_time)
            else:
                print(f"Warning: No data found for {position['exchange_ticker']} at {exit_time}, using entry price for exit")
                dummy_row = {
                    'close': position['entry_price']
                }
                self.close_trade(position, dummy_row, exit_time)
    
    def close_trade(self, trade, row, exit_time):
        """Close a specific trade"""
        self.trade_id += 1
        exit_price = row['close']
        trade['exit_price'] = exit_price
        trade['exit_time'] = exit_time
        trade['close_exit'] = row['close']
        if trade['side'] == 'Short':
            pnl = (trade['entry_price'] - exit_price) * trade['qty']
        elif trade['side'] == 'Long':
            pnl = (exit_price - trade['entry_price']) * trade['qty']
        else:
            pnl = 0
        trade['pnl'] = pnl
        trade['deployed_capital'] = trade['qty'] * trade['entry_price']
        self.trades.append(trade.copy())  
        self.positions.remove(trade)
        self.update_trade_sheet(trade)
        print(f"Exited {trade['side']} position for {trade['exchange_ticker']} at {exit_time}, entry: {trade['entry_price']}, exit: {exit_price}, PnL: {pnl:.2f}")
    
    def update_trade_sheet(self, trade):
        """Update the trade sheet with a new trade"""
        new_row = pd.DataFrame({
            'date': [trade['trade_date']],
            'trade_id': [self.trade_id],
            'tcp_id': [trade['tcp_id']],
            'ticker': [trade['exchange_ticker']],
            'entry_time': [trade['entry_time']],
            'exit_time': [trade['exit_time']],
            'entry_price': [trade['entry_price']],
            'exit_price': [trade['exit_price']],
            'qty': [trade['qty']],
            'pnl': [trade['pnl']],
            'deployed_capital': [trade['deployed_capital']],
            'close_entry': [trade['close_entry']],
            'close_exit': [trade['close_exit']]
        })
        self.trade_sheet = pd.concat([self.trade_sheet, new_row], ignore_index=True)
    
    def capture_results(self):
        """Capture and store trading results for each time period"""
        if self.trade_sheet.empty:
            print("No trades to capture results for")
            return pd.DataFrame()
        date = self.trade_sheet['date'].iloc[0]
        results = [{
            "date": date,
            "time": self.entry_time,
            "deployed_capital": self.daily_capital_deployed,
            "pnl": 0,
            "return": 0
        }]
        exit_pnl = sum([trade['pnl'] for trade in self.trades if trade['exit_time'] == self.exit_time])
        results.append({
            "date": date,
            "time": self.exit_time,
            "deployed_capital": self.daily_capital_deployed,
            "pnl": exit_pnl,
            "return": exit_pnl / self.daily_capital_deployed if self.daily_capital_deployed > 0 else 0
        })
        final_pnl = sum([trade['pnl'] for trade in self.trades if trade['exit_time'] == self.final_exit_time])
        total_pnl = exit_pnl + final_pnl
        results.append({
            "date": date,
            "time": self.final_exit_time,
            "deployed_capital": self.daily_capital_deployed,
            "pnl": total_pnl,
            "return": total_pnl / self.daily_capital_deployed if self.daily_capital_deployed > 0 else 0
        })
        self.results = results
        results_df = pd.DataFrame(self.results)
        print("\nFinal Results:")
        print(results_df)
        return results_df
    
    def save_files(self, date):
        """Save the final files after processing completes"""
        if self.trade_sheet.empty:
            print("No trades to save.")
            return 0, 0, 0
        date_str = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
        trade_sheet_path = os.path.join(self.tradesheet_path, f"tradesheet_{date}.pkl")
        self.trade_sheet.to_pickle(trade_sheet_path)
        print(f"Trade sheet saved to {trade_sheet_path}")
        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(self.results_path, f"results_{date}.pkl")
        results_df.to_pickle(results_path)
        print(f"Results saved to {results_path}")
        total_pnl, total_return, total_trades = self.calculate_total_pnl_and_trade_count()
        summary_df = pd.DataFrame([{
            'date': date_str,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'total_trades': total_trades,
            'deployed_capital': self.daily_capital_deployed
        }])
        summary_path = os.path.join(self.summary_path, f"summary_{date}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
        return total_pnl, total_return, total_trades
    
    def calculate_total_pnl_and_trade_count(self):
        """Calculate and print final trading statistics"""
        total_pnl = self.trade_sheet['pnl'].sum()
        total_trades = len(self.trade_sheet)
        total_return = (total_pnl / self.daily_capital_deployed) * 100 if self.daily_capital_deployed > 0 else 0
        print(f"\nTotal PnL: {total_pnl:.2f}")
        print(f"Total Trades: {total_trades}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Deployed Capital: {self.daily_capital_deployed:.2f}")
        return total_pnl, total_return, total_trades
    
    def print_current_status(self, current_time):
        """Print current trading status"""
        current_positions = len(self.positions)
        completed_trades = len(self.trades)
        current_pnl = sum([trade['pnl'] for trade in self.trades])
        print(f"\nStatus at {current_time}:")
        print(f"Open Positions: {current_positions}")
        print(f"Completed Trades: {completed_trades}")
        print(f"Current PnL: {current_pnl:.2f}")
        print(f"Deployed Capital: {self.daily_capital_deployed:.2f}/{self.max_capital:.2f}")
    
    def process_single_date(self, date):
        """Process data for a single date"""
        print(f"\nProcessing date: {date}")
        self.reset_state()
        try:
            # Load data for this date
            data_df, universe_df = self.load_data_for_date(date)
            # Process entry at 15:00:00
            print(f"\n=== Processing entry at {self.entry_time} ===")
            self.enter_trade(data_df, universe_df, self.entry_time)
            self.print_current_status(self.entry_time)
            # Save positions
            self.save_positions(date)
            # Get the next day's data for exit
            if self.positions:
                position_tcp_ids = [pos['tcp_id'] for pos in self.positions]
                next_day_data = self.load_next_day_data_for_exit(date, position_tcp_ids)
                if next_day_data is not None:
                    # Exit at 09:16:00 on next day
                    print(f"\n=== Processing exit at {self.exit_time} (next day) ===")
                    self.exit_trade(next_day_data, self.exit_time)
                    self.print_current_status(self.exit_time)
                    # Final exit for any remaining positions at 09:17:00
                    if self.positions:
                        print(f"\n=== Processing final exit at {self.final_exit_time} (next day) ===")
                        self.exit_trade(next_day_data, self.final_exit_time)
                        self.print_current_status(self.final_exit_time)
                else:
                    print("Unable to exit positions: No next day data available")
            # Capture and save results
            self.capture_results()
            total_pnl, total_return, total_trades = self.save_files(date)
            print(f"\nDate {date} Processing Complete:")
            print(f"Total PnL: {total_pnl:.2f}")
            print(f"Total Return: {total_return:.2f}%")
            print(f"Total Trades: {total_trades}")
        except Exception as e:
            print(f"Error processing date {date}: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def process_all_dates(self):
        """Process all available dates"""
        # Get all available dates
        dates = self.get_available_dates()
        print(f"\nFound {len(dates)} dates to process")
        # Process each date
        all_summaries = []
        for date in dates:
            try:
                self.process_single_date(date)
                # Collect summary for overall statistics
                summary_file = os.path.join(self.summary_path, f"summary_{date}.csv")
                if os.path.exists(summary_file):
                    summary = pd.read_csv(summary_file)
                    all_summaries.append(summary)
            except Exception as e:
                print(f"Failed to process date {date}: {str(e)}")
                continue
        # Create overall summary
        if all_summaries:
            overall_summary = pd.concat(all_summaries, ignore_index=True)
            # Filter profitable and losing days
            profitable_days = overall_summary[overall_summary['total_pnl'] > 0]
            losing_days = overall_summary[overall_summary['total_pnl'] < 0]
            # Calculate statistics
            overall_stats = {
                'total_days': len(overall_summary),
                'total_pnl': overall_summary['total_pnl'].sum(),
                'average_daily_pnl': overall_summary['total_pnl'].mean(),
                'median_daily_pnl': overall_summary['total_pnl'].median(),
                'std_daily_pnl': overall_summary['total_pnl'].std(),
                'max_daily_pnl': overall_summary['total_pnl'].max(),
                'min_daily_pnl': overall_summary['total_pnl'].min(),
                'total_trades': overall_summary['total_trades'].sum(),
                'average_trades_per_day': overall_summary['total_trades'].mean(),
                'profitable_days': len(profitable_days),
                'losing_days': len(losing_days),
                'avg_gain': profitable_days['total_pnl'].mean() if not profitable_days.empty else 0,
                'max_gain': profitable_days['total_pnl'].max() if not profitable_days.empty else 0,
                'avg_loss': losing_days['total_pnl'].mean() if not losing_days.empty else 0,
                'max_loss': losing_days['total_pnl'].min() if not losing_days.empty else 0
            }
            print("\nBacktest Complete! Overall Statistics:")
            print(f"Total Days Processed: {overall_stats['total_days']}")
            print(f"Total PnL: {overall_stats['total_pnl']:.2f}")
            print(f"Average Daily PnL: {overall_stats['average_daily_pnl']:.2f}")
            print(f"Profitable Days: {overall_stats['profitable_days']} ({overall_stats['profitable_days']/overall_stats['total_days']*100:.1f}%)")
            print(f"Losing Days: {overall_stats['losing_days']} ({overall_stats['losing_days']/overall_stats['total_days']*100:.1f}%)")
            print(f"Total Trades: {overall_stats['total_trades']}")
            # Save overall statistics
            overall_summary_path = os.path.join(self.summary_path, "backtest_overall_summary.csv")
            pd.DataFrame([overall_stats]).to_csv(overall_summary_path, index=False)
            print(f"\nOverall summary saved to: {overall_summary_path}")

def main():
    try:
        print("Starting backtester...")
        backtester = Backtester()
        backtester.process_all_dates()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nBacktest processing completed.")


if __name__ == "__main__":
    main()