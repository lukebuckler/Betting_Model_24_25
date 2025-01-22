import pandas as pd
import os
from datetime import datetime
import glob
import numpy as np

def process_betting_results(predictions_folder, games_file):
    """
    Process all prediction files and combine with actual game results using a conservative betting strategy.
    Only places bets when:
    1. Model predicts >50% win probability
    2. Model's probability exceeds the implied probability from odds
    """
    # [Previous file reading and initial processing code remains the same until betting calculations]
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
    
    def should_bet(row):
        """Determine if we should place a bet based on conservative criteria"""
        # Convert odds to implied probabilities
        if row['Away Team Odds'] > 0:
            away_implied_prob = 100 / (row['Away Team Odds'] + 100)
        else:
            away_implied_prob = -row['Away Team Odds'] / (-row['Away Team Odds'] + 100)
            
        if row['Home Team Odds'] > 0:
            home_implied_prob = 100 / (row['Home Team Odds'] + 100)
        else:
            home_implied_prob = -row['Home Team Odds'] / (-row['Home Team Odds'] + 100)
            
        # Check if model probability exceeds implied probability
        away_value = row['Away Probability'] > away_implied_prob
        home_value = row['Home Probability'] > home_implied_prob
        
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
    correct_bets = len(bet_games[bet_games['prediction_correct']])
    
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
            
        team_stats[team] = {
            'total_bets': len(team_games),
            'correct_bets': team_games['prediction_correct'].sum(),
            'accuracy': team_games['prediction_correct'].mean(),
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
    
    monthly_stats.columns = [
        'bets_placed', 'correct_bets', 'accuracy', 'avg_roi', 'total_profit'
    ]
    monthly_stats['total_profit'] = monthly_stats['total_profit'] * 100
    
    # Value analysis
    def calculate_edge(row):
        if row['bet_placed'] == 'away':
            return row['Away Probability'] - row['Away Team Implied Probability']
        elif row['bet_placed'] == 'home':
            return row['Home Probability'] - row['Home Team Implied Probability']
        return None

    bet_games['edge'] = bet_games.apply(calculate_edge, axis=1)
    edge_performance = bet_games.groupby(pd.qcut(bet_games['edge'], 4)).agg({
        'prediction_correct': ['count', 'mean'],
        'bet_outcome': 'mean'
    })
    
    return {
        'total_games': total_games,
        'total_bets': total_bets_placed,
        'bet_rate': total_bets_placed / total_games,
        'correct_bets': correct_bets,
        'accuracy': correct_bets / total_bets_placed if total_bets_placed > 0 else 0,
        'roi': roi,
        'total_profit_loss': total_return * 100,
        'team_performance': team_performance,
        'monthly_performance': monthly_stats,
        'edge_performance': edge_performance
    }

if __name__ == "__main__":
    predictions_folder = "24_25_results"
    games_file = "24_25_games.csv"
    
    # Process results with conservative strategy
    results_df = process_betting_results(predictions_folder, games_file)
    results_df.to_csv("conservative_betting_results.csv", index=False)
    
    # Generate and print analysis
    analysis = analyze_conservative_results(results_df)
    
    print("\nConservative Betting Strategy Analysis:")
    print(f"Total Games Analyzed: {analysis['total_games']}")
    print(f"Total Bets Placed: {analysis['total_bets']}")
    print(f"Bet Rate: {analysis['bet_rate']:.1%}")
    print(f"Betting Accuracy: {analysis['accuracy']:.1%}")
    print(f"ROI: {analysis['roi']:.1%}")
    print(f"Total Profit/Loss (assuming $100 bets): ${analysis['total_profit_loss']:.2f}")
    
    print("\nTeam Performance (sorted by ROI):")
    print(analysis['team_performance'].sort_values('roi', ascending=False))
    
    print("\nMonthly Performance:")
    print(analysis['monthly_performance'])
    
    print("\nPerformance by Edge Size (larger edge = bigger difference between model and odds):")
    print(analysis['edge_performance'])
