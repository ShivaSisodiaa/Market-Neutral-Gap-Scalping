import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
def load_data(filepath):
    """Load data from CSV file and prepare for analysis."""
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df.iloc[:, 0])
    # Rename columns based on the data structure
    df.columns = ['date', 'tcp_id', 'symbol', 'open_gap']
    
    # Sort by tcp_id and date
    df = df.sort_values(['tcp_id', 'date'])
    
    print(f"Loaded data with shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")
    
    return df

# Feature engineering
def compute_features(df, window=5):
    """Compute features for the gap prediction model."""
    print("Computing features...")
    df = df.sort_values(['tcp_id', 'date']).copy()
    
    # Target: sign of next day's gap
    df['gap_sign'] = np.sign(df['open_gap'])
    df['target'] = df.groupby('tcp_id')['gap_sign'].shift(-1)
    
    df = df[df['target'] != 0]
    
    # Core features
    df['abs_gap'] = df['open_gap'].abs()
    
    # Z-score of gap
    df['zscore_gap'] = df.groupby('tcp_id')['open_gap'].transform(
        lambda x: (x - x.rolling(window).mean()) / x.rolling(window).std())
    
    # Standard deviation of gap
    df['std_gap'] = df.groupby('tcp_id')['open_gap'].transform(
        lambda x: x.rolling(window).std())
    
    # Cumulative momentum (sum of signs)
    df['cum_momentum_5'] = df.groupby('tcp_id')['gap_sign'].transform(
        lambda x: x.rolling(3).sum())
    
    # Win rate (proportion of positive gaps)
    df['win_rate_5'] = df.groupby('tcp_id')['gap_sign'].transform(
        lambda x: x.rolling(3).apply(lambda y: np.sum(y > 0) / len(y)))
    
    # Streak: consecutive days with same sign
    df['streak'] = df.groupby('tcp_id')['gap_sign'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    
    # Volatility ratio: recent volatility compared to longer-term
    df['vol_ratio'] = df.groupby('tcp_id')['open_gap'].transform(
        lambda x: x.rolling(3).std() / x.rolling(10).std())
    
    # Drop rows with NA values
    df_clean = df.dropna()
    
    print(f"Feature computation complete. Rows before cleaning: {df.shape[0]}, after: {df_clean.shape[0]}")
    
    return df_clean

# Improved walk-forward CV splits with true day-by-day predictions
def daily_walk_forward_splits(df, train_days=20, test_days=1):
    """Generate day-by-day walk-forward train/test splits."""
    # Get unique dates in sorted order
    unique_dates = sorted(df['date'].unique())
    total_days = len(unique_dates)
    
    splits = []
    for i in range(train_days, total_days - test_days + 1):
        train_start = i - train_days
        train_end = i
        test_day = i
        
        train_dates = unique_dates[train_start:train_end]
        test_date = unique_dates[test_day]
        
        splits.append((train_dates, [test_date]))
    
    return splits

# Model training and evaluation
def evaluate_models(df, feature_cols, target_col='target', train_days=20, test_days=1):
    """Evaluate multiple models using day-by-day walk-forward validation."""
    print("Evaluating models with daily walk-forward validation...")
    results = []
    feature_importance = pd.DataFrame()
    
    # Generate splits
    splits = daily_walk_forward_splits(df, train_days=train_days, test_days=test_days)
    
    # Only use a subset of splits to make it more efficient
    # Use every 5th split to cover the dataset while keeping runtime reasonable
    evaluation_splits = splits[::5]
    
    for i, (train_dates, test_dates) in enumerate(evaluation_splits):
        print(f"Processing split {i+1}/{len(evaluation_splits)}: "
              f"Train from {train_dates[0]} to {train_dates[-1]}, Test {test_dates[0]}")
        
        train = df[df['date'].isin(train_dates)]
        test = df[df['date'].isin(test_dates)]
        
        # Skip if either train or test is empty
        if train.empty or test.empty:
            print("  Empty train or test set, skipping...")
            continue
        
        X_train = train[feature_cols]
        y_train = train[target_col]
        X_test = test[feature_cols]
        y_test = test[target_col]
        
        # Convert target to binary (-1 -> 0, 1 -> 1) for compatibility with sklearn metrics
        y_train_binary = (y_train > 0).astype(int)
        y_test_binary = (y_test > 0).astype(int)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models with hyperparameters
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42)
        }
        
        split_results = []
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Use scaled data for all algorithms
            model.fit(X_train_scaled, y_train_binary)
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                y_pred = (y_prob > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            acc = accuracy_score(y_test_binary, y_pred)
            prec = precision_score(y_test_binary, y_pred, zero_division=0)
            rec = recall_score(y_test_binary, y_pred, zero_division=0)
            f1 = f1_score(y_test_binary, y_pred, zero_division=0)
            
            # Store results
            model_result = {
                'split': i,
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_date': test_dates[0],
                'model': name,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'samples': len(y_test)
            }
            split_results.append(model_result)
            
            # Extract feature importance if available
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                else:  # For logistic regression and SVM with linear kernel
                    importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                
                importance_df = pd.DataFrame({
                    'split': i,
                    'model': name,
                    'feature': feature_cols,
                    'importance': importances
                })
                feature_importance = pd.concat([feature_importance, importance_df])
        
        results.extend(split_results)
    
    results_df = pd.DataFrame(results)
    return results_df, feature_importance

# Plotting functions
def plot_results(results_df):
    """Plot model performance metrics."""
    plt.figure(figsize=(14, 8))
    
    # Plot average metrics by model
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    avg_results = results_df.groupby('model')[metrics].mean().reset_index()
    
    plt.subplot(2, 1, 1)
    sns.barplot(x='model', y='accuracy', data=avg_results)
    plt.title('Average Accuracy by Model')
    plt.ylim(0.45, 0.75)  # Set a reasonable range for accuracy visualization
    plt.xticks(rotation=45)
    
    plt.subplot(2, 1, 2)
    melted = pd.melt(avg_results, id_vars=['model'], value_vars=['precision', 'recall', 'f1_score'])
    sns.barplot(x='model', y='value', hue='variable', data=melted)
    plt.title('Precision, Recall, and F1-Score by Model')
    plt.ylim(0.3, 0.8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('model_performance.png')
    plt.close()
    
    # Plot performance across splits
    plt.figure(figsize=(14, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        sns.lineplot(x='split', y=metric, hue='model', data=results_df)
        plt.title(f'{metric.capitalize()} Across Splits')
        plt.xlabel('Split')
        plt.ylabel(metric.capitalize())
    plt.tight_layout()
    plt.savefig('performance_across_splits.png')
    plt.close()
    
    return "Plots saved as 'model_performance.png' and 'performance_across_splits.png'"

def plot_feature_importance(feature_importance_df):
    """Plot feature importance for models that support it."""
    # Average feature importance across splits
    avg_importance = feature_importance_df.groupby(['model', 'feature'])['importance'].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    models = avg_importance['model'].unique()
    n_models = len(models)
    
    for i, model_name in enumerate(models, 1):
        plt.subplot(int(np.ceil(n_models/2)), 2, i)
        model_data = avg_importance[avg_importance['model'] == model_name].sort_values('importance', ascending=False)
        sns.barplot(x='importance', y='feature', data=model_data)
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
    
    plt.savefig('feature_importance.png')
    plt.close()
    
    return "Feature importance plot saved as 'feature_importance.png'"

# Main execution
def main(data_path):
    """Main function to run the entire pipeline."""
    # Load data
    print("Starting gap prediction analysis pipeline...")
    df = load_data(data_path)
    
    # Compute features
    feature_df = compute_features(df)
    
    # Define feature columns
    feature_cols = [
        'open_gap', 
        'abs_gap', 
        'gap_sign',  # Current day's gap sign
        'zscore_gap', 
        'std_gap', 
        'cum_momentum_5', 
        'win_rate_5',
        'streak',     # New feature: consecutive days with same sign
        'vol_ratio'   # New feature: volatility ratio
    ]
    
    # EDA
    print("\nFeature statistics:")
    print(feature_df[feature_cols + ['target']].describe())
    
    # Correlation analysis
    corr = feature_df[feature_cols + ['target']].corr()
    print("\nFeature correlations with target:")
    print(corr['target'].sort_values(ascending=False))
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Check class balance
    target_counts = feature_df['target'].value_counts(normalize=True)
    print("\nTarget class distribution:")
    print(target_counts)
    
    # Evaluate models with daily walk-forward
    results, feature_importance = evaluate_models(
        feature_df, 
        feature_cols, 
        target_col='target',
        train_days=20,  # Train on 20 days
        test_days=1     # Test on 1 day (next day)
    )
    
    # Generate plots
    plot_results(results)
    if not feature_importance.empty:
        plot_feature_importance(feature_importance)
    
    # Print summary results
    print("\nAverage performance by model:")
    avg_results = results.groupby('model')[['accuracy', 'precision', 'recall', 'f1_score']].mean()
    print(avg_results.sort_values('accuracy', ascending=False))
    
    # Print class-specific performance for the best model
    best_model = avg_results.sort_values('accuracy', ascending=False).index[0]
    best_model_results = results[results['model'] == best_model]
    print(f"\nBest model: {best_model}")
    print(f"Average accuracy: {best_model_results['accuracy'].mean():.4f}")
    print(f"Average precision: {best_model_results['precision'].mean():.4f}")
    print(f"Average recall: {best_model_results['recall'].mean():.4f}")
    print(f"Average F1-score: {best_model_results['f1_score'].mean():.4f}")
    
    return results, feature_importance

if __name__ == '__main__':
    data_path = '/home/t0013/ml_alpha/strat_ideas/opening_gap_pred/open_gaps_dataset.csv'
    results, feature_importance = main(data_path)