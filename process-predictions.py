import pandas as pd
import os
from datetime import datetime
import glob
import numpy as np

def process_betting_results(predictions_folder, games_file):
    """
    Process all prediction files and combine with actual game results.
    
    Args:
        predictions_folder (str): Path to folder containing daily prediction CSV files
        games_file (str): Path to the season games results CSV file
    
    Returns:
        pandas.DataFrame: Combined dataset with predictions and results
    """
    # Read and process all prediction files
    all_predictions = []
    for file in glob.glob(os.path.join(predictions_folder, '*.csv')):
        try:
            df = pd.read_csv(file)
            # Add source file name for reference
            df['source_file'] = os.path.basename(file)
            all_predictions.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not all_predictions:
        raise ValueError("No prediction files found or processed")
    
    # Combine all predictions into one DataFrame
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Convert Date to datetime
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
    
    # Read games results
    games_df = pd.read_csv(games_file)
    
    # Process games results to get game outcomes
    def parse_game_result(game_str):
        """Parse game string into structured format"""
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

    # Process each game result
    game_results = []
    for game in games_df['Game'].unique():
        result = parse_game_result(game)
        if result:
            game_results.append(result)
    
    # Convert game results to DataFrame
    results_df = pd.DataFrame(game_results)
    
    # Create a mapping of actual game outcomes
    game_outcomes = {}
    for _, row in results_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        # Store both team perspectives
        game_outcomes[(date_str, row['team1'], row['team2'])] = {
            'score': (row['score1'], row['score2']),
            'winner': row['team1'] if row['score1'] > row['score2'] else row['team2']
        }
        game_outcomes[(date_str, row['team2'], row['team1'])] = {
            'score': (row['score2'], row['score1']),
            'winner': row['team1'] if row['score1'] > row['score2'] else row['team2']
        }

    # Add actual results to predictions
    def get_game_result(row):
        """Get actual game result for a prediction"""
        date_str = row['Date'].strftime('%Y-%m-%d')
        away_team = row['Away Team']
        home_team = row['Home Team']
        game_key = (date_str, away_team, home_team)
        
        if game_key in game_outcomes:
            result = game_outcomes[game_key]
            away_score, home_score = result['score']
            return pd.Series({
                'actual_away_score': away_score,
                'actual_home_score': home_score,
                'actual_winner': result['winner'],
                'prediction_correct': (
                    (result['winner'] == away_team and row['Away Probability'] > row['Home Probability']) or
                    (result['winner'] == home_team and row['Home Probability'] > row['Away Probability'])
                )
            })
        return pd.Series({
            'actual_away_score': None,
            'actual_home_score': None,
            'actual_winner': None,
            'prediction_correct': None
        })

    # Add game results to predictions
    final_df = predictions_df.join(predictions_df.apply(get_game_result, axis=1))
    
    # Calculate betting outcomes
    def calculate_bet_outcome(row):
        """Calculate betting outcome if bet was placed on predicted winner"""
        if pd.isna(row['prediction_correct']):
            return None
            
        predicted_winner = 'away' if row['Away Probability'] > row['Home Probability'] else 'home'
        odds = row['Away Team Odds'] if predicted_winner == 'away' else row['Home Team Odds']
        
        if row['prediction_correct']:
            if odds > 0:
                return odds/100  # For positive odds (e.g., +150 pays 1.5)
            else:
                return -100/odds  # For negative odds (e.g., -150 pays 0.667)
        return -1  # Lost bet

    final_df['bet_outcome'] = final_df.apply(calculate_bet_outcome, axis=1)
    
    # Add some summary statistics
    final_df['predicted_winner'] = final_df.apply(
        lambda x: x['Away Team'] if x['Away Probability'] > x['Home Probability'] else x['Home Team'],
        axis=1
    )
    final_df['prediction_confidence'] = final_df.apply(
        lambda x: max(x['Away Probability'], x['Home Probability']),
        axis=1
    )
    
    # Add month column for time-based analysis
    final_df['month'] = final_df['Date'].dt.to_period('M')
    
    return final_df

def analyze_team_performance(df):
    """
    Analyze model performance for each team
    
    Args:
        df (pandas.DataFrame): Processed predictions and results
    
    Returns:
        dict: Team performance metrics
    """
    # Initialize containers for team stats
    team_stats = {}
    
    # Get unique teams
    teams = set(df['Home Team'].unique()) | set(df['Away Team'].unique())
    
    for team in teams:
        # Get all games where this team played
        team_games = df[
            ((df['Home Team'] == team) | (df['Away Team'] == team)) &
            df['prediction_correct'].notna()
        ]
        
        if len(team_games) == 0:
            continue
            
        # Calculate team-specific metrics
        total_games = len(team_games)
        correct_predictions = team_games['prediction_correct'].sum()
        total_profit = team_games['bet_outcome'].sum()
        
        # Store team stats
        team_stats[team] = {
            'games': total_games,
            'correct_predictions': correct_predictions,
            'accuracy': correct_predictions / total_games,
            'roi': total_profit / total_games,
            'total_profit': total_profit * 100  # Convert to dollars
        }
    
    return pd.DataFrame.from_dict(team_stats, orient='index')

def analyze_time_trends(df):
    """
    Analyze model performance trends over time
    
    Args:
        df (pandas.DataFrame): Processed predictions and results
    
    Returns:
        pandas.DataFrame: Monthly performance metrics
    """
    # Group by month and calculate metrics
    monthly_stats = df[df['prediction_correct'].notna()].groupby('month').agg({
        'prediction_correct': ['count', 'sum', 'mean'],
        'bet_outcome': ['mean', 'sum']
    }).round(3)
    
    # Flatten column names
    monthly_stats.columns = [
        'total_games', 'correct_predictions', 'accuracy', 'avg_roi', 'total_profit'
    ]
    
    # Convert profit to dollars
    monthly_stats['total_profit'] = monthly_stats['total_profit'] * 100
    
    # Add cumulative metrics
    monthly_stats['cumulative_accuracy'] = monthly_stats['correct_predictions'].cumsum() / monthly_stats['total_games'].cumsum()
    monthly_stats['cumulative_roi'] = (monthly_stats['total_profit'].cumsum() / 
                                     (monthly_stats['total_games'].cumsum() * 100))
    
    return monthly_stats

def analyze_home_away_performance(df):
    """
    Analyze model performance for home vs away predictions
    
    Args:
        df (pandas.DataFrame): Processed predictions and results
    
    Returns:
        dict: Home/away performance metrics
    """
    completed_games = df[df['prediction_correct'].notna()]
    
    # Create home/away prediction indicators
    completed_games['predicted_home_win'] = completed_games['Home Probability'] > completed_games['Away Probability']
    completed_games['actual_home_win'] = completed_games['actual_winner'] == completed_games['Home Team']
    
    # Analyze home predictions
    home_predictions = completed_games[completed_games['predicted_home_win']]
    away_predictions = completed_games[~completed_games['predicted_home_win']]
    
    def calculate_location_stats(data):
        if len(data) == 0:
            return {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0,
                'avg_roi': 0,
                'total_profit': 0
            }
            
        return {
            'total_predictions': len(data),
            'correct_predictions': data['prediction_correct'].sum(),
            'accuracy': data['prediction_correct'].mean(),
            'avg_roi': data['bet_outcome'].mean(),
            'total_profit': data['bet_outcome'].sum() * 100  # Convert to dollars
        }
    
    return {
        'home_predictions': calculate_location_stats(home_predictions),
        'away_predictions': calculate_location_stats(away_predictions)
    }

def analyze_results(final_df):
    """
    Generate comprehensive summary statistics from the processed results
    
    Args:
        final_df (pandas.DataFrame): Processed predictions and results
    
    Returns:
        dict: Summary statistics
    """
    # Filter to only completed games
    completed_games = final_df.dropna(subset=['actual_winner'])
    
    # Calculate basic statistics
    total_predictions = len(completed_games)
    correct_predictions = completed_games['prediction_correct'].sum()
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Calculate ROI
    total_bets = len(completed_games.dropna(subset=['bet_outcome']))
    total_return = completed_games['bet_outcome'].sum()
    roi = (total_return / total_bets) if total_bets > 0 else 0
    
    # Accuracy by confidence level
    confidence_bins = pd.cut(completed_games['prediction_confidence'], 
                           bins=[0, 0.55, 0.6, 0.65, 0.7, 1],
                           labels=['50-55%', '55-60%', '60-65%', '65-70%', '70%+'])
    accuracy_by_confidence = completed_games.groupby(confidence_bins)['prediction_correct'].agg(['count', 'mean'])
    
    # Get team performance
    team_performance = analyze_team_performance(completed_games)
    
    # Get time trends
    time_trends = analyze_time_trends(completed_games)
    
    # Get home/away performance
    home_away_performance = analyze_home_away_performance(completed_games)
    
    return {
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'roi': roi,
        'total_profit_loss': total_return,
        'accuracy_by_confidence': accuracy_by_confidence,
        'team_performance': team_performance,
        'time_trends': time_trends,
        'home_away_performance': home_away_performance
    }

if __name__ == "__main__":
    # Example usage
    predictions_folder = "24_25_results"
    games_file = "24_25_games.csv"
    
    # Process all results
    results_df = process_betting_results(predictions_folder, games_file)
    
    # Save combined results to CSV
    results_df.to_csv("combined_betting_results.csv", index=False)
    
    # Generate and print analysis
    analysis = analyze_results(results_df)
    
    print("\nOverall Model Performance Summary:")
    print(f"Total Predictions: {analysis['total_predictions']}")
    print(f"Correct Predictions: {analysis['correct_predictions']}")
    print(f"Accuracy: {analysis['accuracy']:.1%}")
    print(f"ROI: {analysis['roi']:.1%}")
    print(f"Total Profit/Loss (assuming $100 bets): ${analysis['total_profit_loss']*100:.2f}")
    
    print("\nAccuracy by Confidence Level:")
    print(analysis['accuracy_by_confidence'])
    
    print("\nTeam Performance Summary:")
    print(analysis['team_performance'].sort_values('roi', ascending=False))
    
    print("\nMonthly Performance Trends:")
    print(analysis['time_trends'])
    
    print("\nHome vs Away Performance:")
    home_stats = analysis['home_away_performance']['home_predictions']
    away_stats = analysis['home_away_performance']['away_predictions']
    
    print("\nHome Predictions:")
    print(f"Total Predictions: {home_stats['total_predictions']}")
    print(f"Accuracy: {home_stats['accuracy']:.1%}")
    print(f"ROI: {home_stats['avg_roi']:.1%}")
    print(f"Total Profit/Loss: ${home_stats['total_profit']:.2f}")
    
    print("\nAway Predictions:")
    print(f"Total Predictions: {away_stats['total_predictions']}")
    print(f"Accuracy: {away_stats['accuracy']:.1%}")
    print(f"ROI: {away_stats['avg_roi']:.1%}")
    print(f"Total Profit/Loss: ${away_stats['total_profit']:.2f}")
