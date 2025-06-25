import os
import argparse
from datetime import datetime
from pathlib import Path
from data_processor import DataProcessor

def get_all_dates(raw_data_folder):
    """
    Get all available dates from raw data files.
    """
    raw_files = [f for f in os.listdir(raw_data_folder) if f.endswith('.pkl')]
    raw_dates = sorted(f.split('.')[0] for f in raw_files)
    return raw_dates


def filter_dates_by_range(all_dates, start_date, end_date):
    """
    Filter dates to include only those within the specified range (inclusive).
    """
    if start_date is None and end_date is None:
        return all_dates
    
    filtered_dates = []
    for date_str in all_dates:
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue
        filtered_dates.append(date_str)
    
    return filtered_dates


def process_single_date(date_str, raw_data_folder, feature_data_folder, processor):
    """
    Process data for a single date: load raw, compute features + RSI filter, and save to CSV.
    Returns True on success, False on error or insufficient data.
    """
    try:
        # 1. Load & basic processing
        processed_data = processor.load_and_process_data(date_str)
        if processed_data is None or processed_data.empty:
            print(f"Skipping {date_str}: no data after basic processing.")
            return False

        # 2. Calculate RSI values on processed_data
        processed_data = processor.calculate_rsi(processed_data)
        if 'RSI_7' not in processed_data.columns:
            print(f"Skipping {date_str}: RSI_7 not computed.")
            return False

        # 3. Filter for latest date
        if 'trade_date' in processed_data.columns:
            latest_date = processed_data['trade_date'].max()
            filtered_data = processed_data[processed_data['trade_date'] == latest_date]

        # 4. Identify extreme-RSI tcp_ids to drop
        extreme_ids = (
            filtered_data
            .loc[(filtered_data['RSI_7'] <= 20) | (filtered_data['RSI_7'] >= 80), 'tcp_id']
            .unique()
        )

        # 5. Keep only rows for tcp_ids NOT in extreme list
        filtered_data = filtered_data[~filtered_data['tcp_id'].isin(extreme_ids)]
        print(filtered_data)
        if filtered_data.empty:
            print(f"Skipping {date_str}: all data filtered out by RSI extremes.")
            return False

        # 6. 'trade_date', 'tcp_id', 'exchange_ticker', 'RSI_7'
        rsi_df = (
                filtered_data
                .drop_duplicates(subset=['tcp_id'])[['trade_date','tcp_id', 'exchange_ticker', 'RSI_7']]
                .reset_index(drop=True)
            )

        # 7. Save to CSV
        date_obj = rsi_df['trade_date'].iloc[0]
        date_str_fmt = date_obj.strftime("%Y%m%d")  # e.g., datetime -> '20241211'
        output_filepath = os.path.join(feature_data_folder, f"feature_{date_str_fmt}.csv")
        rsi_df.to_csv(output_filepath, index=False)
        print(f"Successfully processed and saved features for {date_str}")
        return True

    except Exception as e:
        print(f"Error processing {date_str}: {str(e)}")
        return False


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Process raw data dates to compute and save feature CSVs'
    )
    parser.add_argument(
        '--start_date', type=str,
        help='Start date in YYYYMMDD format (inclusive)'
    )
    parser.add_argument(
        '--end_date', type=str,
        help='End date in YYYYMMDD format (inclusive)'
    )
    args = parser.parse_args()

    # Define folders
    raw_data_folder = "/home/t0013/ml_alpha/ankit_raghav/min_bars_generator/raw_data"
    feature_data_folder = (
        "/home/t0013/ml_alpha/intraday_momentum/momentum_filter_for_gap_scalping"
    )

    # Create output folder if needed
    Path(feature_data_folder).mkdir(parents=True, exist_ok=True)

    # Discover available raw dates
    all_dates = get_all_dates(raw_data_folder)
    if not all_dates:
        print("No raw data files found.")
        return

    # Filter dates
    dates_to_process = filter_dates_by_range(
        all_dates, args.start_date, args.end_date
    )
    if not dates_to_process:
        print("No dates found in the specified range.")
        return

    print(f"Processing {len(dates_to_process)} dates from {dates_to_process[0]} to {dates_to_process[-1]}...")

    # Initialize processor
    processor = DataProcessor(raw_data_folder)

    # Loop over dates
    success_count = 0
    failure_count = 0
    for date_str in dates_to_process:
        print(f"\n--- Processing date: {date_str} ---")
        ok = process_single_date(
            date_str, raw_data_folder, feature_data_folder, processor
        )
        if ok:
            success_count += 1
        else:
            failure_count += 1

    # Summary
    print("\nProcessing Complete!")
    print(f"Successfully processed: {success_count} dates")
    print(f"Failed to process: {failure_count} dates")

if __name__ == "__main__":
    main()
