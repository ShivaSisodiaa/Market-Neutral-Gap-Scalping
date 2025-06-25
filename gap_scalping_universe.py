# Cross_Sectional Z-Score of intraday returns, Momentum Filter

import os
import pandas as pd
import pickle as pkl
import numpy as np
from datetime import datetime, time, timedelta


class daily_universe:
    
    def __init__(self, raw_data_folder, momentum_filter_folder):
        self.raw_data_folder = raw_data_folder
        self.momentum_filter_folder = momentum_filter_folder
        self.columns = ['trade_date', 'time', 'tcp_id', 'exchange_ticker', 'open', 'high', 'low', 'close', 'volume']
        self.numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        self.data_files = self.sort_data_files()
        self.start_time = time(9, 16)  # 09:16:00
        self.end_time = time(14, 56)   # 14:56:00
    
    def sort_data_files(self):
        """Sort data files by date."""
        return sorted(
            [f for f in os.listdir(self.raw_data_folder) if f.endswith('.pkl')],
            key=lambda x: x.split('.')[0]
        )
    
    def load_momentum_filter(self, date_str):
        """Load momentum filter data for a specific date.
        
        Args:
            date_str: Date string in 'YYYYMMDD' format
            
        Returns:
            Set of tcp_ids to include, or None if file not found
        """
        try:
            # Format date to match the feature file naming convention
            feature_file_path = os.path.join(self.momentum_filter_folder, f"feature_{date_str}.csv")
            
            if not os.path.exists(feature_file_path):
                print(f"Momentum filter file not found: {feature_file_path}")
                return None
                
            # Load the feature file
            feature_df = pd.read_csv(feature_file_path)
            
            # Extract unique tcp_ids
            tcp_ids = set(feature_df['tcp_id'].unique())
            print(f"Loaded momentum filter with {len(tcp_ids)} unique tcp_ids")
            
            return tcp_ids
            
        except Exception as e:
            print(f"Error loading momentum filter: {str(e)}")
            return None
    
    def load_and_process_file(self, file_path, filter_tcp_ids=None):
        """Load and process a single pickle file.
        
        Args:
            file_path: Path to the pickle file
            filter_tcp_ids: Set of tcp_ids to filter by (optional)
            
        Returns:
            Processed DataFrame
        """
        with open(file_path, "rb") as f:
            data = pkl.load(f)
        
        df = pd.DataFrame(data, columns=self.columns)
        
        # Apply tcp_id filter if provided
        if filter_tcp_ids is not None:
            original_len = len(df)
            df = df[df['tcp_id'].isin(filter_tcp_ids)]
            filtered_len = len(df)
            print(f"Filtered raw data from {original_len} to {filtered_len} rows based on momentum filter")
            
            # Check if we have any data left after filtering
            if df.empty:
                print("Warning: No data remains after applying momentum filter")
                return None
        
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.drop_duplicates()
        
        # convert num to float
        df[self.numeric_columns] = df[self.numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        # Handle na data
        df['close'] = df.groupby('tcp_id')['close'].transform(lambda x: x.ffill())
        df['open'] = df['open'].fillna(df['close'])
        df['high'] = df['high'].fillna(df['close'])
        df['low'] = df['low'].fillna(df['close'])
        df['volume'] = df['volume'].fillna(0)
        
        # filter by time range
        start = datetime.strptime('09:15:00', '%H:%M:%S').time()
        end = datetime.strptime('15:31:00', '%H:%M:%S').time()
        df = df[(df['time'] > start) & (df['time'] < end)]
        
        return df
    
    def get_price_at_time(self, raw_df, target_time, tcp_id, price_type='close'):
        """Get price at specified time or next available time"""
        try:
            time_obj = target_time if isinstance(target_time, time) else datetime.strptime(target_time, '%H:%M:%S').time()
            # Filter data for target time and tcp_id
            price_data = raw_df[(raw_df['tcp_id'] == tcp_id) & (raw_df['time'] == time_obj)]
            if not price_data.empty:
                return price_data[price_type].iloc[0]
            # If exact time not found, find next available time
            later_times = raw_df[(raw_df['tcp_id'] == tcp_id) & (raw_df['time'] > time_obj)]
            if not later_times.empty:
                return later_times.iloc[0][price_type]
            return None
        except Exception as e:
            print(f"Error in get_price_at_time: {str(e)}")
            return None
    
    def calculate_cross_sectional_zscore(self, raw_df):
        """Calculate cross-sectional z-score for all tickers using time series data"""
        # Get unique tickers
        unique_ticker_data = raw_df.drop_duplicates(subset=['tcp_id', 'exchange_ticker'])
        ticker_data = []
        
        for _, row in unique_ticker_data.iterrows():
            tcp_id = row['tcp_id']
            ticker = row['exchange_ticker']
            
            # Get start and end prices
            start_price = self.get_price_at_time(raw_df, self.start_time, tcp_id, 'close')
            end_price = self.get_price_at_time(raw_df, self.end_time, tcp_id, 'close')
            
            if start_price is not None and end_price is not None and start_price > 0:
                # Calculate return
                return_value = ((end_price - start_price) * 100) / start_price
                ticker_data.append({
                    'trade_date': row['trade_date'],
                    'tcp_id': tcp_id,
                    'exchange_ticker': ticker,
                    'return': return_value
                })
        
        # Create DataFrame with returns
        returns_df = pd.DataFrame(ticker_data)
        if returns_df.empty:
            print("No valid return data found")
            return None
            
        # Calculate cross-sectional z-score
        returns_df['cs_zscore'] = (returns_df['return'] - returns_df['return'].mean()) / returns_df['return'].std()
        return returns_df
    
    def process_universe(self, input_date):
        """Process data for the given input date and generate universe"""
        input_date_str = input_date
        
        # Find next available date file
        date_index = next((i for i, f in enumerate(self.data_files) if f.split('.')[0] >= input_date_str), None)
        if date_index is None or date_index >= len(self.data_files) - 1:
            print(f"No next date file available after {input_date_str}")
            return None
            
        # Get current and next date files
        current_date_file = self.data_files[date_index]
        input_date_plus_1_file = self.data_files[date_index + 1]
        input_date_plus_1 = input_date_plus_1_file.split('.')[0]
        print(f"Processing date: {input_date_str}, Next date: {input_date_plus_1}")
        
        # Load momentum filter for current date
        filter_tcp_ids = self.load_momentum_filter(input_date_str)
        if filter_tcp_ids is None:
            print(f"No momentum filter data available for {input_date_str}. Processing all tickers.")
        
        # Load and process current date data with tcp_id filter
        current_file_path = os.path.join(self.raw_data_folder, current_date_file)
        raw_df = self.load_and_process_file(current_file_path, filter_tcp_ids)
        
        if raw_df is None or raw_df.empty:
            print(f"No data available for {input_date_str} after applying filters")
            return None
        
        # Calculate cross-sectional z-scores
        zscore_df = self.calculate_cross_sectional_zscore(raw_df)
        if zscore_df is None or zscore_df.empty:
            print(f"Could not calculate z-scores for {input_date_str}")
            return None
        
        # Exclude NIFTY
        zscore_df = zscore_df[zscore_df['exchange_ticker'] != 'NIFTY']
        
        # Calculate quintile labels safely with error handling
        try:
            # Check if we have enough unique values for quintiles
            unique_values = zscore_df['cs_zscore'].nunique()
            print(f"Number of unique z-score values: {unique_values}")
            
            if unique_values >= 5:
                # Enough unique values for 5 bins
                zscore_df['label'] = pd.qcut(zscore_df['cs_zscore'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            else:
                # Not enough unique values for 5 bins, use fewer bins
                n_bins = min(unique_values, 4)  # Use at most 4 bins
                print(f"Not enough unique values for 5 quintiles. Using {n_bins} bins instead.")
                
                # Create custom labels based on number of bins
                labels = list(range(1, n_bins + 1))
                zscore_df['label'] = pd.qcut(zscore_df['cs_zscore'], n_bins, labels=labels, duplicates='drop')
                
                # If we're using fewer than 5 bins, map the bins to the expected labels
                if n_bins == 4:
                    # Map bins to labels: 1,2,4,5 (skip middle bin)
                    label_map = {1: 1, 2: 2, 3: 4, 4: 5}
                    zscore_df['label'] = zscore_df['label'].map(label_map)
                elif n_bins == 3:
                    # Map bins to labels: 1,3,5 (represent low, mid, high)
                    label_map = {1: 1, 2: 3, 3: 5}
                    zscore_df['label'] = zscore_df['label'].map(label_map)
                elif n_bins == 2:
                    # Map bins to labels: 1,5 (represent low, high)
                    label_map = {1: 1, 2: 5}
                    zscore_df['label'] = zscore_df['label'].map(label_map)
        except Exception as e:
            print(f"Error creating quintile labels: {str(e)}")
            print("Using manual binning as fallback")
            
            # Fallback to manual quantile assignment
            sorted_zscores = zscore_df['cs_zscore'].sort_values()
            n_items = len(sorted_zscores)
            
            if n_items < 5:
                # For very few items, just use binary classification
                cutoff = sorted_zscores.median()
                zscore_df['label'] = np.where(zscore_df['cs_zscore'] <= cutoff, 1, 5)
            else:
                # Manual quintile assignment
                quintile_size = n_items // 5
                cutoffs = [
                    sorted_zscores.iloc[quintile_size],
                    sorted_zscores.iloc[quintile_size * 2],
                    sorted_zscores.iloc[quintile_size * 3],
                    sorted_zscores.iloc[quintile_size * 4]
                ]
                
                def assign_quintile(value):
                    if value <= cutoffs[0]:
                        return 1
                    elif value <= cutoffs[1]:
                        return 2
                    elif value <= cutoffs[2]:
                        return 3
                    elif value <= cutoffs[3]:
                        return 4
                    else:
                        return 5
                
                zscore_df['label'] = zscore_df['cs_zscore'].apply(assign_quintile)
        
        print("Z-score DataFrame with Labels sample:")
        print(zscore_df.head())
        
        # Count the number of stocks in each label
        label_counts = zscore_df['label'].value_counts().sort_index()
        print("Number of stocks in each label:")
        print(label_counts)
        
        # Filter for labels 1, 2, 4, and 5 (bottom and top quintiles)
        final_df = zscore_df[zscore_df['label'].isin([1, 2, 4, 5])]
        
        # Check if we have stocks in these categories
        if final_df.empty:
            print("No stocks found in the target labels (1, 2, 4, 5)")
            # Fallback: include all stocks with labels
            final_df = zscore_df.copy()
            print(f"Using all {len(final_df)} stocks as fallback")
        
        final_df = final_df.sort_values(by=['label', 'cs_zscore'])
        final_df = final_df[['trade_date', 'tcp_id', 'exchange_ticker', 'cs_zscore', 'label']]
        
        print(f"Final DataFrame - {len(final_df)} rows")
        print(final_df.head())
        
        # Save results
        base_dir = '/home/t0013/ml_alpha/intraday_momentum'
        universe_folder_dir = os.path.join(base_dir, 'universe_gap_scalping')
        
        # Create directory if it doesn't exist
        os.makedirs(universe_folder_dir, exist_ok=True)
        
        output_file = os.path.join(universe_folder_dir, f"universe_{input_date_str}.csv")
        final_df.to_csv(output_file, index=False)
        print(f"Saved universe to {output_file}")
        
        return input_date_plus_1
    
    def process_all_dates(self):
        """Process all dates in the raw data folder sequentially"""
        if not self.data_files:
            print("No data files found in the raw data folder")
            return
            
        current_date = self.data_files[0].split('.')[0]
        
        # Process each date sequentially
        processed_count = 0
        skipped_count = 0
        
        while True:
            print(f"\n{'='*50}")
            print(f"Processing date: {current_date}")
            print(f"{'='*50}")
            
            next_date = self.process_universe(current_date)
            
            if next_date is None:
                skipped_count += 1
                current_index = next((i for i, f in enumerate(self.data_files) 
                                     if f.split('.')[0] > current_date), None)
                                     
                if current_index is None or current_index >= len(self.data_files) - 1:
                    print(f"Reached the end of available data files after processing {processed_count} dates " 
                          f"and skipping {skipped_count} dates")
                    break
                    
                current_date = self.data_files[current_index].split('.')[0]
            else:
                processed_count += 1
                current_date = next_date
                
                if not any(f.split('.')[0] >= current_date for f in self.data_files):
                    print(f"Processed all available dates: {processed_count} dates processed, {skipped_count} dates skipped")
                    break
                    
        print(f"\nProcessing complete. Successfully processed {processed_count} dates. Skipped {skipped_count} dates.")


if __name__ == "__main__":
    raw_data_folder = "/home/t0013/ml_alpha/ankit_raghav/min_bars_generator/raw_data"
    momentum_filter_folder = "/home/t0013/ml_alpha/intraday_momentum/momentum_filter_for_gap_scalping"
    
    universe_generator = daily_universe(raw_data_folder, momentum_filter_folder)
    universe_generator.process_all_dates()