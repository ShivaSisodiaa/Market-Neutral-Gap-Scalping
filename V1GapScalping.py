# V1 Gap Scalping

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
        self.signal_path = "/home/t0013/ml_alpha/strat_ideas/opening_gap_pred/LR_open_gap_predictions"
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
        
    def enter_trade(self, current_data, universe_df, entry_time):
        """Process entries for the current time period with modified strategy"""
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
        combined_df = pd.merge(
            universe_df,
            signal_df[['tcp_id', 'predicted_gap_sign']],
            on=['tcp_id'],
            how='inner'
        )
        if combined_df.empty:
            print("No matches found between universe and signal data")
            return
        long_securities = combined_df[(combined_df['label'].isin([4, 5])) & (combined_df['predicted_gap_sign'] == 1)]
        short_securities = combined_df[(combined_df['label'].isin([1, 2])) & (combined_df['predicted_gap_sign'] == 0)]
        print(f"Found {len(long_securities)} potential long entries and {len(short_securities)} potential short entries")
        for _, row in long_securities.iterrows():
            tcp_id = row['tcp_id']
            exchange_ticker = row['exchange_ticker']
            ticker_data = entry_data[entry_data['tcp_id'] == tcp_id]
            if ticker_data.empty:
                print(f"No data for {exchange_ticker} (long) at entry time")
                continue
            ticker_row = ticker_data.iloc[0]
            if not self.can_trade_tcp_id(tcp_id, ticker_row['trade_date'], entry_time):
                continue
            qty = self.get_lot_size(tcp_id) * 1
            if qty <= 0:
                continue
            trade_signal = 'Long'
            trade_capital = qty * ticker_row['close']
            if self.capital_limit_reached(trade_capital):
                print(f"Skipping trade for {exchange_ticker} due to capital limit")
                continue
            self.daily_capital_deployed += trade_capital
            trade = {
                "trade_date": ticker_row['trade_date'],
                "entry_time": entry_time,
                "tcp_id": tcp_id,
                "exchange_ticker": exchange_ticker,
                "side": trade_signal,
                "qty": qty,
                "entry_price": ticker_row['close'],
                "close_entry": ticker_row['close']
            }
            self.positions.append(trade)
            self.traded_tcp_ids.add(tcp_id)
            print(f"Entered {trade_signal} position for {exchange_ticker} at {entry_time}, price: {ticker_row['close']}, qty: {qty}")
        for _, row in short_securities.iterrows():
            tcp_id = row['tcp_id']
            exchange_ticker = row['exchange_ticker']
            ticker_data = entry_data[entry_data['tcp_id'] == tcp_id]
            if ticker_data.empty:
                print(f"No data for {exchange_ticker} (short) at entry time")
                continue
            ticker_row = ticker_data.iloc[0]
            if not self.can_trade_tcp_id(tcp_id, ticker_row['trade_date'], entry_time):
                continue
            qty = self.get_lot_size(tcp_id) * 1
            if qty <= 0:
                continue
            trade_signal = 'Short'
            trade_capital = qty * ticker_row['close']
            if self.capital_limit_reached(trade_capital):
                print(f"Skipping trade for {exchange_ticker} due to capital limit")
                continue
            self.daily_capital_deployed += trade_capital
            trade = {
                "trade_date": ticker_row['trade_date'],
                "entry_time": entry_time,
                "tcp_id": tcp_id,
                "exchange_ticker": exchange_ticker,
                "side": trade_signal,
                "qty": qty,
                "entry_price": ticker_row['close'],
                "close_entry": ticker_row['close']
            }
            self.positions.append(trade)
            self.traded_tcp_ids.add(tcp_id)
            print(f"Entered {trade_signal} position for {exchange_ticker} at {entry_time}, price: {ticker_row['close']}, qty: {qty}")
        
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