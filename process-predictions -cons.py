import pandas as pd
import os
from datetime import datetime
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

def process_betting_results(predictions_folder, games_file):
    """
    Process all prediction files and combine with actual game results using a conservative betting strategy.
    Only places bets when:
    1. Model predicts >50% win probability
    2. Model's probability exceeds the implied probability from odds
    """
    # Read and process all prediction files
    all_predictions = []
    for file in glob.glob(os.path.join(predictions_folder, '*.csv')):
        try:
            df = pd.read_csv(file)
            df['source_file'] = os.path.basename(file)
            all_predictions.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not all_predictions:
        raise ValueError("No prediction files found or processed")
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
    
    # Check for duplicate games (same date, home team, and away team)
    print("Checking for duplicate game entries...")
    
    # Create a unique identifier for each game
    predictions_df['game_id'] = predictions_df['Date'].dt.strftime('%Y-%m-%d') + '_' + predictions_df['Away Team'] + '_' + predictions_df['Home Team']
    
    # Check if duplicates exist
    duplicate_count = predictions_df.duplicated('game_id', keep=False).sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate game entries. Keeping only the first occurrence of each game.")
        # Keep only the first occurrence of each game
        predictions_df = predictions_df.drop_duplicates('game_id', keep='first')
    else:
        print("No duplicate game entries found.")
    
    # Drop the temporary game_id column
    predictions_df = predictions_df.drop('game_id', axis=1)
    
    # Read games results
    games_df = pd.read_csv(games_file)
    
    def parse_game_result(game_str):
        try:
            date_str, score_str = game_str.split(' - ')
            team1, score1, team2, score2 = score_str.replace(',', '').split(' ')
            return {
                'date': datetime.strptime(date_str, '%Y-%m-%d').date(),
                'team1': team1,
                'score1': int(score1),
                'team2': team2,
                'score2': int(score2)
            }
        except Exception as e:
            print(f"Error parsing game result '{game_str}': {e}")
            return None

    game_results = []
    for game in games_df['Game'].unique():
        result = parse_game_result(game)
        if result:
            game_results.append(result)
    
    results_df = pd.DataFrame(game_results)
    
    game_outcomes = {}
    for _, row in results_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        game_outcomes[(date_str, row['team1'], row['team2'])] = {
            'score': (row['score1'], row['score2']),
            'winner': row['team1'] if row['score1'] > row['score2'] else row['team2']
        }
        game_outcomes[(date_str, row['team2'], row['team1'])] = {
            'score': (row['score2'], row['score1']),
            'winner': row['team1'] if row['score1'] > row['score2'] else row['team2']
        }

    def get_game_result(row):
        date_str = row['Date'].strftime('%Y-%m-%d')
        away_team = row['Away Team']
        home_team = row['Home Team']
        game_key = (date_str, away_team, home_team)
        
        if game_key in game_outcomes:
            result = game_outcomes[game_key]
            away_score, home_score = result['score']
            predicted_winner = away_team if row['Away Probability'] > row['Home Probability'] else home_team
            return pd.Series({
                'actual_away_score': away_score,
                'actual_home_score': home_score,
                'actual_winner': result['winner'],
                'prediction_correct': result['winner'] == predicted_winner
            })
        return pd.Series({
            'actual_away_score': None,
            'actual_home_score': None,
            'actual_winner': None,
            'prediction_correct': None
        })

    final_df = predictions_df.join(predictions_df.apply(get_game_result, axis=1))
    
    # Calculate implied probabilities
    final_df['Away Team Implied Probability'] = final_df.apply(
        lambda x: 100 / (x['Away Team Odds'] + 100) if x['Away Team Odds'] > 0 
        else -x['Away Team Odds'] / (-x['Away Team Odds'] + 100), 
        axis=1
    )
    
    final_df['Home Team Implied Probability'] = final_df.apply(
        lambda x: 100 / (x['Home Team Odds'] + 100) if x['Home Team Odds'] > 0 
        else -x['Home Team Odds'] / (-x['Home Team Odds'] + 100), 
        axis=1
    )
    
    def should_bet(row):
        """Determine if we should place a bet based on conservative criteria"""
        # Check if model probability exceeds implied probability
        away_value = row['Away Probability'] > row['Away Team Implied Probability']
        home_value = row['Home Probability'] > row['Home Team Implied Probability']
        
        # Check if model is confident (>50%)
        away_confident = row['Away Probability'] > 0.5
        home_confident = row['Home Probability'] > 0.5
        
        # Determine if we should bet and on which team
        if away_confident and away_value:
            return 'away'
        elif home_confident and home_value:
            return 'home'
        return None

    def calculate_conservative_bet_outcome(row):
        """Calculate betting outcome using conservative strategy"""
        if pd.isna(row['prediction_correct']):
            return None
            
        bet_decision = should_bet(row)
        if bet_decision is None:
            return 0  # No bet placed
            
        odds = row['Away Team Odds'] if bet_decision == 'away' else row['Home Team Odds']
        predicted_winner = row['Away Team'] if bet_decision == 'away' else row['Home Team']
        
        if row['actual_winner'] == predicted_winner:
            if odds > 0:
                return odds/100
            else:
                return -100/odds
        return -1 if bet_decision else 0

    final_df['bet_placed'] = final_df.apply(should_bet, axis=1)
    final_df['bet_outcome'] = final_df.apply(calculate_conservative_bet_outcome, axis=1)
    
    # Add additional metrics
    final_df['predicted_winner'] = final_df.apply(
        lambda x: x['Away Team'] if x['Away Probability'] > x['Home Probability'] else x['Home Team'],
        axis=1
    )
    final_df['prediction_confidence'] = final_df.apply(
        lambda x: max(x['Away Probability'], x['Home Probability']),
        axis=1
    )
    final_df['month'] = final_df['Date'].dt.to_period('M')
    
    return final_df

def analyze_conservative_results(final_df):
    """Generate summary statistics for conservative betting strategy"""
    # Filter to only completed games
    completed_games = final_df.dropna(subset=['actual_winner'])
    
    # Get games where bets were placed
    bet_games = completed_games[completed_games['bet_placed'].notna()]
    
    # Basic statistics
    total_games = len(completed_games)
    total_bets_placed = len(bet_games)
    correct_bets = len(bet_games[bet_games['prediction_correct'] == True])
    
    # Calculate ROI
    total_return = bet_games['bet_outcome'].sum()
    roi = total_return / total_bets_placed if total_bets_placed > 0 else 0
    
    # Team performance
    team_stats = {}
    teams = set(completed_games['Home Team'].unique()) | set(completed_games['Away Team'].unique())
    
    for team in teams:
        team_games = completed_games[
            ((completed_games['Home Team'] == team) | (completed_games['Away Team'] == team)) &
            (completed_games['bet_placed'].notna())
        ]
        
        if len(team_games) == 0:
            continue
            
        correct_count = team_games['prediction_correct'].sum()
        team_stats[team] = {
            'total_bets': len(team_games),
            'correct_bets': correct_count,
            'accuracy': correct_count / len(team_games) if len(team_games) > 0 else 0,
            'roi': team_games['bet_outcome'].mean(),
            'total_profit': team_games['bet_outcome'].sum() * 100
        }
    
    team_performance = pd.DataFrame.from_dict(team_stats, orient='index')
    
    # Monthly performance
    monthly_stats = bet_games.groupby('month').agg({
        'bet_placed': 'count',
        'prediction_correct': ['sum', 'mean'],
        'bet_outcome': ['mean', 'sum']
    }).round(3)
    
    # Reset multi-level column names to flat names
    monthly_stats.columns = [
        'bets_placed', 'correct_bets', 'accuracy', 'avg_roi', 'total_profit'
    ]
    monthly_stats['total_profit'] = monthly_stats['total_profit'] * 100
    
    # Calculate cumulative metrics for time series visualization
    bet_games_sorted = bet_games.sort_values('Date')
    bet_games_sorted['cumulative_return'] = bet_games_sorted['bet_outcome'].cumsum()
    bet_games_sorted['cumulative_bets'] = range(1, len(bet_games_sorted) + 1)
    bet_games_sorted['cumulative_roi'] = bet_games_sorted['cumulative_return'] / bet_games_sorted['cumulative_bets']
    
    # Edge calculation - avoiding duplicate index issues
    edges = []
    for _, row in bet_games.iterrows():
        if row['bet_placed'] == 'away':
            edge = row['Away Probability'] - row['Away Team Implied Probability']
        elif row['bet_placed'] == 'home':
            edge = row['Home Probability'] - row['Home Team Implied Probability']
        else:
            edge = None
        edges.append(edge)
    
    # Create a new DataFrame with reset index
    edge_analysis_df = pd.DataFrame({
        'edge': edges,
        'prediction_correct': bet_games['prediction_correct'].values,
        'bet_outcome': bet_games['bet_outcome'].values
    })
    
    # Calculate edge performance
    edge_performance = None
    if len(edge_analysis_df) >= 4:  # Need at least 4 data points for quartiles
        try:
            edge_bins = pd.qcut(edge_analysis_df['edge'].dropna(), 4)
            edge_performance = edge_analysis_df.dropna(subset=['edge']).groupby(edge_bins).agg({
                'prediction_correct': ['count', 'mean'],
                'bet_outcome': 'mean'
            })
        except ValueError:  # Handle case if qcut fails due to duplicate values
            print("Warning: Could not create quartiles for edge analysis due to duplicate values.")
            edge_performance = pd.DataFrame()
    else:
        edge_performance = pd.DataFrame()
    
    return {
        'total_games': total_games,
        'total_bets': total_bets_placed,
        'bet_rate': total_bets_placed / total_games if total_games > 0 else 0,
        'correct_bets': correct_bets,
        'accuracy': correct_bets / total_bets_placed if total_bets_placed > 0 else 0,
        'roi': roi,
        'total_profit_loss': total_return * 100,
        'team_performance': team_performance,
        'monthly_performance': monthly_stats,
        'edge_performance': edge_performance,
        'time_series_data': bet_games_sorted,
        'all_bet_games': bet_games
    }

def create_visualizations(analysis, output_folder='betting_visualizations'):
    """Create and save visualizations based on betting analysis"""
    print("Creating visualizations...")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Set up visualization style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = [12, 6]
    
    # 1. Cumulative Profit Over Time
    time_data = analysis['time_series_data']
    
    if not time_data.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(time_data['Date'], time_data['cumulative_return'] * 100, 'b-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.title('Cumulative Profit Over Time (Based on $100 Bets)', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Profit ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'cumulative_profit_over_time.png'), dpi=300)
        plt.close()
        print("✓ Created cumulative profit chart")
    else:
        print("✗ Not enough time series data for cumulative profit chart")
    
    # 2. Team Performance (Top teams by ROI)
    team_perf = analysis['team_performance']
    if not team_perf.empty:
        # Sort and get top performers
        team_perf_sorted = team_perf.sort_values('roi', ascending=False)
        top_teams = min(10, len(team_perf_sorted))
        top_team_perf = team_perf_sorted.head(top_teams)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(top_team_perf.index, top_team_perf['roi'], color='skyblue')
        
        plt.title(f'Top {top_teams} Teams by Return on Investment', fontsize=14)
        plt.xlabel('ROI', fontsize=12)
        plt.ylabel('Team', fontsize=12)
        plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
        
        # Add bet count annotations
        for i, bar in enumerate(bars):
            team = top_team_perf.index[i]
            bet_count = top_team_perf.loc[team, 'total_bets']
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'Bets: {int(bet_count)}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'top_teams_by_roi.png'), dpi=300)
        plt.close()
        print("✓ Created team performance chart")
    else:
        print("✗ Not enough team data for team performance chart")
    
    # 3. Monthly Performance
    monthly = analysis['monthly_performance'].reset_index()
    if not monthly.empty:
        monthly['month_str'] = monthly['month'].astype(str)
        
        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(111)
        
        bars = ax1.bar(monthly['month_str'], monthly['total_profit'], color='steelblue', alpha=0.7)
        ax1.set_ylabel('Profit/Loss ($)', fontsize=12)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        # Add a second y-axis for accuracy
        ax2 = ax1.twinx()
        ax2.plot(monthly['month_str'], monthly['accuracy'], 'ro-', linewidth=2)
        ax2.set_ylabel('Prediction Accuracy', fontsize=12, color='r')
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        plt.title('Monthly Betting Performance', fontsize=14)
        plt.xticks(rotation=45)
        
        # Add bet count annotations
        for i, bar in enumerate(bars):
            bet_count = monthly.iloc[i]['bets_placed']
            plt.text(bar.get_x() + bar.get_width()/2, 10, 
                    f'Bets: {int(bet_count)}', ha='center', va='bottom', rotation=90)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'monthly_performance.png'), dpi=300)
        plt.close()
        print("✓ Created monthly performance chart")
    else:
        print("✗ Not enough monthly data for monthly performance chart")
    
    # 4. Edge Analysis - Accuracy and ROI by Edge Size
    edge_perf = analysis['edge_performance']
    if edge_perf is not None and not edge_perf.empty:
        try:
            edge_perf_reset = edge_perf.reset_index()
            edge_perf_reset.columns = ['edge_bin', 'bet_count', 'accuracy', 'roi']
            edge_perf_reset['edge_bin_str'] = edge_perf_reset['edge_bin'].astype(str)
            
            plt.figure(figsize=(12, 6))
            ax1 = plt.subplot(111)
            
            bars = ax1.bar(edge_perf_reset['edge_bin_str'], edge_perf_reset['roi'], color='steelblue', alpha=0.7)
            ax1.set_ylabel('Return on Investment', fontsize=12)
            ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            
            # Add a second y-axis for accuracy
            ax2 = ax1.twinx()
            ax2.plot(edge_perf_reset['edge_bin_str'], edge_perf_reset['accuracy'], 'ro-', linewidth=2)
            ax2.set_ylabel('Prediction Accuracy', fontsize=12, color='r')
            ax2.set_ylim(0, 1)
            ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
            
            plt.title('Performance by Edge Size (Model Probability - Implied Probability)', fontsize=14)
            plt.xlabel('Edge Size (Grouped by Quartiles)', fontsize=12)
            plt.xticks(rotation=45)
            
            # Add bet count annotations
            for i, bar in enumerate(bars):
                bet_count = edge_perf_reset.iloc[i]['bet_count']
                plt.text(bar.get_x() + bar.get_width()/2, 0.01, 
                        f'Bets: {int(bet_count)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'edge_analysis.png'), dpi=300)
            plt.close()
            print("✓ Created edge analysis chart")
        except Exception as e:
            print(f"✗ Error creating edge analysis chart: {e}")
    else:
        print("✗ Not enough edge data for edge analysis chart")
    
    # 5. Prediction Confidence vs. Accuracy
    bet_games = analysis['all_bet_games']
    if not bet_games.empty:
        try:
            # Create confidence bins
            conf_values = bet_games['prediction_confidence'].values
            correct_values = bet_games['prediction_correct'].values
            
            if len(conf_values) >= 5:  # Need at least 5 data points for 5 bins
                conf_df = pd.DataFrame({
                    'prediction_confidence': conf_values,
                    'prediction_correct': correct_values
                })
                
                confidence_bins = pd.cut(conf_df['prediction_confidence'], bins=5)
                confidence_accuracy = conf_df.groupby(confidence_bins)['prediction_correct'].agg(['count', 'mean'])
                confidence_accuracy.columns = ['bet_count', 'accuracy']
                confidence_accuracy = confidence_accuracy.reset_index()
                confidence_accuracy['confidence_bin_str'] = confidence_accuracy['prediction_confidence'].astype(str)
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(confidence_accuracy['confidence_bin_str'], confidence_accuracy['accuracy'], 
                            color=sns.color_palette("viridis", len(confidence_accuracy)))
                plt.title('Prediction Accuracy by Model Confidence', fontsize=14)
                plt.xlabel('Model Confidence Range', fontsize=12)
                plt.ylabel('Accuracy', fontsize=12)
                plt.ylim(0, 1)
                plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
                plt.xticks(rotation=45)
                
                # Add bet count annotations
                for i, bar in enumerate(bars):
                    bet_count = confidence_accuracy.iloc[i]['bet_count']
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                            f'Bets: {int(bet_count)}', ha='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, 'confidence_vs_accuracy.png'), dpi=300)
                plt.close()
                print("✓ Created confidence analysis chart")
            else:
                print("✗ Not enough confidence data for confidence analysis chart")
        except Exception as e:
            print(f"✗ Error creating confidence analysis chart: {e}")
    else:
        print("✗ Not enough bet data for confidence analysis chart")
    
    # 6. Summary Dashboard
    try:
        plt.figure(figsize=(12, 10))
        plt.suptitle('NHL Betting Model Performance Dashboard', fontsize=16, y=0.98)
        
        # Key metrics
        plt.subplot(3, 2, 1)
        metrics = ['Total Games', 'Bets Placed', 'Accuracy', 'ROI', 'Profit/Loss']
        values = [
            f"{analysis['total_games']}",
            f"{analysis['total_bets']} ({analysis['bet_rate']:.1%})",
            f"{analysis['accuracy']:.1%}",
            f"{analysis['roi']:.1%}",
            f"${analysis['total_profit_loss']:.2f}"
        ]
        
        # Create a table for key metrics
        plt.axis('off')
        table = plt.table(
            cellText=[values],
            colLabels=metrics,
            cellLoc='center',
            loc='center',
            bbox=[0, 0.2, 1, 0.6]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        plt.title('Key Performance Metrics', fontsize=12)
        
        # Add mini versions of the other charts
        # Cumulative profit
        plt.subplot(3, 2, 2)
        if not time_data.empty:
            plt.plot(time_data['Date'], time_data['cumulative_return'] * 100, 'b-')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            plt.title('Cumulative Profit ($)', fontsize=10)
            plt.xticks(rotation=45, fontsize=8)
            plt.yticks(fontsize=8)
        else:
            plt.text(0.5, 0.5, 'No time data available', 
                    horizontalalignment='center',
                    transform=plt.gca().transAxes)
            plt.axis('off')
        
        # Top 5 teams
        plt.subplot(3, 2, 3)
        if not team_perf.empty:
            top5_teams = team_perf.sort_values('roi', ascending=False).head(5)
            plt.barh(top5_teams.index, top5_teams['roi'], color=sns.color_palette("viridis", 5))
            plt.title('Top 5 Teams by ROI', fontsize=10)
            plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
        else:
            plt.text(0.5, 0.5, 'No team data available', 
                    horizontalalignment='center',
                    transform=plt.gca().transAxes)
            plt.axis('off')
        
        # Monthly performance
        plt.subplot(3, 2, 4)
        if not monthly.empty:
            plt.bar(monthly['month_str'], monthly['total_profit'], color='steelblue', alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            plt.title('Monthly Profit/Loss ($)', fontsize=10)
            plt.xticks(rotation=45, fontsize=8)
        else:
            plt.text(0.5, 0.5, 'No monthly data available', 
                    horizontalalignment='center',
                    transform=plt.gca().transAxes)
            plt.axis('off')
        
        # Edge analysis
        plt.subplot(3, 2, 5)
        if edge_perf is not None and not edge_perf.empty:
            plt.bar(edge_perf_reset['edge_bin_str'], edge_perf_reset['roi'], color='steelblue', alpha=0.7)
            plt.plot(edge_perf_reset['edge_bin_str'], edge_perf_reset['accuracy'], 'ro-')
            plt.title('Performance by Edge Size', fontsize=10)
            plt.xticks(rotation=45, fontsize=8)
        else:
            plt.text(0.5, 0.5, 'No edge data available', 
                    horizontalalignment='center',
                    transform=plt.gca().transAxes)
            plt.axis('off')
        
        # Confidence vs accuracy
        plt.subplot(3, 2, 6)
        if not bet_games.empty and len(conf_values) >= 5:
            plt.bar(confidence_accuracy['confidence_bin_str'], confidence_accuracy['accuracy'], 
                    color=sns.color_palette("viridis", len(confidence_accuracy)))
            plt.title('Accuracy by Model Confidence', fontsize=10)
            plt.xticks(rotation=45, fontsize=8)
        else:
            plt.text(0.5, 0.5, 'No confidence data available', 
                    horizontalalignment='center',
                    transform=plt.gca().transAxes)
            plt.axis('off')
        
        plt.subplots_adjust(hspace=0.6, wspace=0.3)
        plt.savefig(os.path.join(output_folder, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created summary dashboard")
    except Exception as e:
        print(f"✗ Error creating summary dashboard: {e}")
    
    print(f"Visualizations saved to '{output_folder}' directory")

if __name__ == "__main__":
    predictions_folder = "24_25_results"
    games_file = "24_25_games.csv"
    
    print(f"Loading prediction files from {predictions_folder}")
    print(f"Loading game results from {games_file}")
    
    try:
        # Process results with conservative strategy
        results_df = process_betting_results(predictions_folder, games_file)
        print(f"Processed {len(results_df)} prediction records")
        
        results_df.to_csv("conservative_betting_results.csv", index=False)
        print("Saved processed results to 'conservative_betting_results.csv'")
        
        # Generate analysis
        analysis = analyze_conservative_results(results_df)
        print("Completed betting analysis")
        
        # Create and save visualizations
        create_visualizations(analysis)
        
        # Print summary information
        print("\nConservative Betting Strategy Analysis:")
        print(f"Total Games Analyzed: {analysis['total_games']}")
        print(f"Total Bets Placed: {analysis['total_bets']}")
        print(f"Bet Rate: {analysis['bet_rate']:.1%}")
        print(f"Betting Accuracy: {analysis['accuracy']:.1%}")
        print(f"ROI: {analysis['roi']:.1%}")
        print(f"Total Profit/Loss (assuming $100 bets): ${analysis['total_profit_loss']:.2f}")
        
        print("\nTeam Performance (sorted by ROI):")
        if not analysis['team_performance'].empty:
            print(analysis['team_performance'].sort_values('roi', ascending=False))
        else:
            print("No team performance data available")
        
        print("\nMonthly Performance:")
        if not analysis['monthly_performance'].empty:
            print(analysis['monthly_performance'])
        else:
            print("No monthly performance data available")
        
        print("\nPerformance by Edge Size (larger edge = bigger difference between model and odds):")
        if analysis['edge_performance'] is not None and not analysis['edge_performance'].empty:
            print(analysis['edge_performance'])
        else:
            print("No edge performance data available")
            
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error processing betting data: {e}")
        import traceback
        traceback.print_exc()