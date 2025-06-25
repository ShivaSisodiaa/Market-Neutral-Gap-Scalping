import pandas as pd
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 12})

summary_dir = "/home/t0013/gap_scalping/daily_summary"
result_df = pd.DataFrame(columns=['date', 'total_pnl', 'total_return', 'total_trades', 'deployed_capital'])

# Get all CSV files in the dir
csv_files = glob.glob(os.path.join(summary_dir, 'summary_*.csv'))

# Sort the files by date
csv_files.sort(key=lambda x: datetime.strptime(os.path.basename(x)[8:18], '%Y-%m-%d'))

# load each CSV file & append to the result dfs
all_dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    all_dfs.append(df)

if all_dfs:
    # Combine all dfs
    result_df = pd.concat(all_dfs, ignore_index=True)
    
    # Convert date strings to datetime objects
    result_df['date'] = pd.to_datetime(result_df['date'])
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(summary_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create bar chart for daily returns
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x='date', y='total_pnl', data=result_df, palette=['green' if x > 0 else 'red' for x in result_df['total_pnl']])
    plt.title('Daily PnL', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('PnL', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add values on top of each bar
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + (0.1 if bar.get_height() > 0 else -0.4),
            f"{result_df['total_pnl'].iloc[i]:.2f}",
            ha='center', va='bottom', fontsize=10
        )
    
    # Format x-axis to show fewer dates (prevent crowding)
    if len(result_df) > 15:
        every_nth = len(result_df) // 15 + 1
        for idx, label in enumerate(ax.xaxis.get_ticklabels()):
            if idx % every_nth != 0:
                label.set_visible(False)
    
    # Save the plot
    daily_returns_path = os.path.join(plots_dir, 'daily_pnl.png')
    plt.savefig(daily_returns_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate cumulative returns
    result_df['cumulative_pnl'] = result_df['total_pnl'].cumsum()
    
    # Create scatter plot with line for cumulative returns
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='date', y='cumulative_pnl', data=result_df, marker='o', linewidth=2)
    plt.title('Cumulative PnL Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative PnL', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add annotations for start and end points
    plt.annotate(f"Start: 0.00%", 
                xy=(result_df['date'].iloc[0], 0), 
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=10)
    
    plt.annotate(f"End: {result_df['cumulative_pnl'].iloc[-1]:.2f}%", 
                xy=(result_df['date'].iloc[-1], result_df['cumulative_pnl'].iloc[-1]), 
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=10)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    cumulative_returns_path = os.path.join(plots_dir, 'cumulative_pnl.png')
    plt.savefig(cumulative_returns_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the combined df with cumulative returns to a CSV file
    output_path = os.path.join(summary_dir, 'combined_summary.csv')
    result_df.to_csv(output_path, index=False)
    print(f"Combined summary saved to: {output_path}")
    
    # Create a performance summary figure with both charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Daily returns on top subplot
    colors = ['green' if x > 0 else 'red' for x in result_df['total_pnl']]
    ax1.bar(result_df['date'], result_df['total_pnl'], color=colors)
    ax1.set_title('Daily PnL', fontsize=16)
    ax1.set_ylabel('PnL', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Cumulative returns on bottom subplot
    ax2.plot(result_df['date'], result_df['cumulative_pnl'], marker='o', linewidth=2, color='blue')
    ax2.set_title('Cumulative PnL', fontsize=16)
    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('Cumulative PnL', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis to prevent crowding
    if len(result_df) > 10:
        every_nth = len(result_df) // 10 + 1
        for idx, label in enumerate(ax2.xaxis.get_ticklabels()):
            if idx % every_nth != 0:
                label.set_visible(False)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_path = os.path.join(plots_dir, 'trading_performance_summary.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {plots_dir}")
    
    # Calculate stats
    overall_stats = {
        'total_days': len(result_df),
        'total_pnl': result_df['total_pnl'].sum(),
        'average_daily_pnl': result_df['total_pnl'].mean(),
        'median_daily_pnl': result_df['total_pnl'].median(),
        'std_daily_pnl': result_df['total_pnl'].std(),
        'max_daily_pnl': result_df['total_pnl'].max(),
        'min_daily_pnl': result_df['total_pnl'].min(),
        'total_trades': result_df['total_trades'].sum(),
        'average_trades_per_day': result_df['total_trades'].mean(),
        'final_cumulative_pnl': result_df['cumulative_pnl'].iloc[-1]
    }

    # Filter profitable & losing days
    profitable_days = result_df[result_df['total_pnl'] > 0]
    losing_days = result_df[result_df['total_pnl'] < 0]
    overall_stats.update({
        'profitable_days': len(profitable_days),
        'losing_days': len(losing_days),
        'avg_gain': profitable_days['total_pnl'].mean() if not profitable_days.empty else 0,
        'max_gain': profitable_days['total_pnl'].max() if not profitable_days.empty else 0,
        'avg_loss': losing_days['total_pnl'].mean() if not losing_days.empty else 0,
        'max_loss': losing_days['total_pnl'].min() if not losing_days.empty else 0
    })

    # Print the stats
    print("\nBacktest Complete! Overall Statistics:")
    print(f"Total Days Processed: {overall_stats['total_days']}")
    print(f"Total PnL: {overall_stats['total_pnl']:.2f}")
    print(f"Final Cumulative Return: {overall_stats['final_cumulative_pnl']:.2f}")
    print(f"Average Daily PnL: {overall_stats['average_daily_pnl']:.2f}")
    print(f"Profitable Days: {overall_stats['profitable_days']} ({overall_stats['profitable_days']/overall_stats['total_days']*100:.1f}%)")
    print(f"Losing Days: {overall_stats['losing_days']} ({overall_stats['losing_days']/overall_stats['total_days']*100:.1f}%)")
    print(f"Total Trades: {overall_stats['total_trades']}")
    print(f"Average Trades Per Day: {overall_stats['average_trades_per_day']:.2f}")
    print(f"Median Daily PnL: {overall_stats['median_daily_pnl']:.2f}")
    print(f"Standard Deviation of Daily PnL: {overall_stats['std_daily_pnl']:.2f}")
    print(f"Maximum Daily PnL: {overall_stats['max_daily_pnl']:.2f}")
    print(f"Minimum Daily PnL: {overall_stats['min_daily_pnl']:.2f}")
    print(f"Average Win: {overall_stats['avg_gain']:.2f}")
    print(f"Average Loss: {overall_stats['avg_loss']:.2f}")
    print(f"Maximum Win: {overall_stats['max_gain']:.2f}")
    print(f"Maximum Loss: {overall_stats['max_loss']:.2f}")

    # Save overall stats
    overall_summary_path = os.path.join(summary_dir, "backtest_overall_summary.csv")
    pd.DataFrame([overall_stats]).to_csv(overall_summary_path, index=False)
    print(f"\nOverall summary saved to: {overall_summary_path}")
else:
    print("No CSV files found in the directory.")