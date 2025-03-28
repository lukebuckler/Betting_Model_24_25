import pandas as pd
import os
from datetime import datetime
import glob

def process_model_results():
    predictions = []
    pred_files = glob.glob('24_25_results/*.csv')
    
    for file in pred_files:
        try:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'])
            predictions.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not predictions:
        raise ValueError("No prediction files found")
    
    all_predictions = pd.concat(predictions, ignore_index=True)
    
    games_df = pd.read_csv('24_25_games.csv')
    
    def parse_game_result(row):
        game_str = row['Game']
        date_str, score_str = game_str.split(' - ')
        team1, score1, team2, score2 = score_str.replace(',', '').split(' ')
        
        return pd.Series({
            'Game_Date': pd.to_datetime(date_str),
            'Team1': team1,
            'Team2': team2,
            'Score1': int(score1),
            'Score2': int(score2),
            'Winner': team1 if int(score1) > int(score2) else team2
        })
    
    results = games_df.groupby('Game').first().reset_index()
    results = results.apply(parse_game_result, axis=1)
    
    def calculate_bet_decision(row):
        away_implied_prob = (100 / (row['Away Team Odds'] + 100) if row['Away Team Odds'] > 0 
                           else -row['Away Team Odds'] / (-row['Away Team Odds'] + 100))
        home_implied_prob = (100 / (row['Home Team Odds'] + 100) if row['Home Team Odds'] > 0 
                           else -row['Home Team Odds'] / (-row['Home Team Odds'] + 100))
        
        away_edge = row['Away Probability'] - away_implied_prob
        home_edge = row['Home Probability'] - home_implied_prob
        
        if row['Away Probability'] > 0.5 and away_edge > 0:
            return pd.Series({
                'Bet_Team': row['Away Team'],
                'Bet_Odds': row['Away Team Odds'],
                'Bet_Prob': row['Away Probability'],
                'Edge': away_edge
            })
        elif row['Home Probability'] > 0.5 and home_edge > 0:
            return pd.Series({
                'Bet_Team': row['Home Team'],
                'Bet_Odds': row['Home Team Odds'],
                'Bet_Prob': row['Home Probability'],
                'Edge': home_edge
            })
        return pd.Series({
            'Bet_Team': None,
            'Bet_Odds': None,
            'Bet_Prob': None,
            'Edge': None
        })

    all_predictions = pd.concat([
        all_predictions,
        all_predictions.apply(calculate_bet_decision, axis=1)
    ], axis=1)
    
    merged_df = pd.merge(
        all_predictions,
        results,
        left_on=['Date', 'Away Team', 'Home Team'],
        right_on=['Game_Date', 'Team1', 'Team2'],
        how='left'
    )
    
    def calculate_bet_result(row):
        if pd.isna(row['Bet_Team']):
            return pd.Series({
                'Bet_Placed': False,
                'Bet_Won': None,
                'Bet_Return': 0
            })
            
        bet_won = row['Bet_Team'] == row['Winner']
        
        if bet_won:
            if row['Bet_Odds'] > 0:
                return_amount = row['Bet_Odds'] / 100
            else:
                return_amount = -100 / row['Bet_Odds']
        else:
            return_amount = -1
            
        return pd.Series({
            'Bet_Placed': True,
            'Bet_Won': bet_won,
            'Bet_Return': return_amount * 100
        })
    
    result_columns = merged_df.apply(calculate_bet_result, axis=1)
    merged_df = pd.concat([merged_df, result_columns], axis=1)
    
    merged_df['Predicted_Winner'] = merged_df.apply(
        lambda x: x['Away Team'] if x['Away Probability'] > x['Home Probability'] else x['Home Team'],
        axis=1
    )
    merged_df['Prediction_Correct'] = merged_df['Predicted_Winner'] == merged_df['Winner']
    merged_df['Prediction_Confidence'] = merged_df.apply(
        lambda x: max(x['Away Probability'], x['Home Probability']),
        axis=1
    )
    
    final_columns = [
        'Date', 
        'Away Team', 'Home Team',
        'Away Team Odds', 'Home Team Odds',
        'Away Probability', 'Home Probability',
        'Predicted_Winner', 'Winner',
        'Score1', 'Score2',
        'Prediction_Correct', 'Prediction_Confidence',
        'Bet_Placed', 'Bet_Team', 'Bet_Odds', 'Bet_Prob', 'Edge',
        'Bet_Won', 'Bet_Return'
    ]
    
    analysis_df = merged_df[final_columns].copy()
    
    # Remove duplicate game entries keeping only the first prediction for each game
    analysis_df = analysis_df.drop_duplicates(subset=['Date', 'Away Team', 'Home Team'], keep='first')
    
    # Sort by date to ensure proper cumulative calculations
    analysis_df = analysis_df.sort_values('Date')
    
    # Calculate cumulative metrics
    analysis_df['Cumulative_Return'] = analysis_df['Bet_Return'].cumsum()
    analysis_df['Running_Accuracy'] = analysis_df['Prediction_Correct'].expanding().mean()
    analysis_df['Running_Win_Rate'] = analysis_df[analysis_df['Bet_Placed']]['Bet_Won'].expanding().mean()
    
    # Save to CSV
    analysis_df.to_csv('model_performance_analysis.csv', index=False)
    
    # Print summary statistics
    completed_games = analysis_df[analysis_df['Winner'].notna()]
    total_predictions = len(completed_games)
    correct_predictions = completed_games['Prediction_Correct'].sum()
    total_bets = completed_games['Bet_Placed'].sum()
    winning_bets = completed_games[completed_games['Bet_Won'] == True].shape[0]
    total_return = completed_games['Bet_Return'].sum()
    
    print(f"\nModel Performance Summary:")
    print(f"Total Predictions: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions} ({correct_predictions/total_predictions:.1%})")
    print(f"Total Bets Placed: {total_bets}")
    print(f"Winning Bets: {winning_bets} ({winning_bets/total_bets:.1%})")
    print(f"Total Return: ${total_return:,.2f}")
    print(f"ROI: {total_return/(total_bets*100):.1%}")

if __name__ == "__main__":
    process_model_results()