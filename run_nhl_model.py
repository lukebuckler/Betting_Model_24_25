import requests
from bs4 import BeautifulSoup
import pandas as pd
import joblib
from datetime import datetime
import os
import re
import datetime

#wesbite with current season's games and stats
url = 'https://www.naturalstattrick.com/games.php?fromseason=20242025&thruseason=20242025&stype=2&sit=sva&loc=B&team=All&rate=n'

response = requests.get(url)

#use beautiful soup to parse the webpage's content
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table
table = soup.find('table')

# Read the table using pandas
df = pd.read_html(str(table))[0]
#drop all columns with a %
cols_to_drop = [col for col in df.columns if '%' in col]
    
#drop other unneccesary columns
df.drop(columns=cols_to_drop, inplace=True)
df.drop('Unnamed: 2', axis=1, inplace=True)
df.drop('Attendance', axis=1, inplace=True)
df.drop('TOI', axis=1, inplace=True)
df.to_csv('24_25_games.csv', index=False)

# use the mapping dictionary to replace the team names with the abbreviations
abbreviations_df = pd.read_csv('teams.csv')
mapping_dict = dict(zip(abbreviations_df['Team'], abbreviations_df['Abbrv']))

def update_and_overwrite_file(file_path, mapping_dict):
    df = pd.read_csv(file_path)
    df['Team'] = df['Team'].map(mapping_dict)
    df.to_csv(file_path, index=False)  

file_paths = ['24_25_Games.csv']

for file_path in file_paths:
    update_and_overwrite_file(file_path, mapping_dict)

# use the mapping dictionary to replace the team names in the game string with the abbreviations
abbreviations_df = pd.read_csv('teams.csv')
mapping_dict = dict(zip(abbreviations_df['Name'], abbreviations_df['Abbrv']))

def replace_team_names(game_string, mapping_dict):
    pattern = r'(\d{4}-\d{2}-\d{2}) - ([\w\s]+) (\d+), ([\w\s]+) (\d+)'
    match = re.match(pattern, game_string)

    if match:
        date, team1, score1, team2, score2 = match.groups()
        team1_abbr = mapping_dict.get(team1.strip(), team1)
        team2_abbr = mapping_dict.get(team2.strip(), team2) 
        return f"{date} - {team1_abbr} {score1}, {team2_abbr} {score2}"
    else:
        return game_string

def update_and_overwrite_file(file_path, mapping_dict):
    df = pd.read_csv(file_path)
    df['Game'] = df['Game'].apply(lambda x: replace_team_names(x, mapping_dict))
    df.to_csv(file_path, index=False)

file_paths = ['24_25_games.csv']

for file_path in file_paths:
    update_and_overwrite_file(file_path, mapping_dict)

import os
import pandas as pd

# Define the input file and output directory
input_file = "C:/Users/luken/Desktop/Betting_Model_24_25/24_25_games.csv"
base_output_directory = "C:/Users/luken/Desktop/Betting_Model_24_25/24_25_team"

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"Error: The file '{input_file}' does not exist.")
else:
    # Extract the year from the file name
    year = input_file.split('_games.csv')[0]

    # Create the output directory if it doesn't exist
    if not os.path.exists(base_output_directory):
        os.makedirs(base_output_directory)

    # Read the input file
    df = pd.read_csv(input_file)

    # Extract game details using regex
    pattern = r'(\d{4}-\d{2}-\d{2}) - ([\w\s]+) (\d+), ([\w\s]+) (\d+)'
    df[['Date', 'AwayTeam', 'AwayScore', 'HomeTeam', 'HomeScore']] = df['Game'].str.extract(pattern)

    # Add a 'HomeResult' column
    df['HomeResult'] = 'Draw'
    df.loc[df['HomeScore'] > df['AwayScore'], 'HomeResult'] = 'Won'
    df.loc[df['HomeScore'] < df['AwayScore'], 'HomeResult'] = 'Lost'

#sort each team's games into a it's own file for the current season
    team_subsets = {team: df[df['Team'] == team] for team in df['Team'].unique()}

    for team, subset in team_subsets.items():
        team_file_name = f"{team}.csv"
        team_file_path = os.path.join(base_output_directory, team_file_name)
        subset.to_csv(team_file_path, index=False)


#calculate days of rest
dir_path = "C:/Users/luken/Desktop/Betting_Model_24_25/24_25_team"


all_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.csv')]

for file in all_files:
    file_path = os.path.join(dir_path, file)
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        date_col = 'Date'
    df[date_col] = pd.to_datetime(df[date_col])

    df['time_diff'] = df[date_col].diff()
    df['days_of_rest'] = df['time_diff'].dt.days - 1
    df.drop('time_diff', axis=1, inplace=True)
    df.to_csv(file_path, index=False)


#calculate days of rest
dir_path = "C:/Users/luken/Desktop/Betting_Model_24_25/24_25_team"


all_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.csv')]

for file in all_files:
    file_path = os.path.join(dir_path, file)
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        date_col = 'Date'
    df[date_col] = pd.to_datetime(df[date_col])

    df['time_diff'] = df[date_col].diff()
    df['days_of_rest'] = df['time_diff'].dt.days - 1
    df.drop('time_diff', axis=1, inplace=True)
    df.to_csv(file_path, index=False)

    
#ODDS SCRAPING
# Fetch NHL odds data
url = 'https://sports.yahoo.com/nhl/odds/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract tables from the webpage
tables = soup.find_all('table')
data = []

for table in tables:
    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append(cols)

# Convert table data to DataFrame
df = pd.DataFrame(data)

# Filter rows where the first column length is greater than 5
filtered_df = df[df[0].astype(str).map(len) > 5]

# Extract team names and odds
filtered_df['Team'] = filtered_df[0].apply(lambda x: x.split('(')[0].strip())
filtered_df['Odds'] = filtered_df[1].apply(lambda x: x[-4:].strip())

# Create a new DataFrame with relevant columns
new_df = filtered_df[['Team', 'Odds']]

# Load team abbreviations and create mapping dictionary
abbreviations_df = pd.read_csv('teams.csv')

# Clean team names in abbreviations_df (strip spaces but keep original capitalization)
#def clean_team_name(name):
    #return name.strip()

#abbreviations_df['Team'] = abbreviations_df['Team'].apply(clean_team_name)
mapping_dict = dict(zip(abbreviations_df['Name'], abbreviations_df['Abbrv']))

# Clean team names in filtered_df
#filtered_df['Team'] = filtered_df['Team'].apply(clean_team_name)

# Map team names to abbreviations
def map_team_name(team):
    if team in mapping_dict:
        return mapping_dict[team]
    else:
        print(f"Unmapped team: {team}")  # Log missing teams
        return team  # Optionally keep original name for unmapped teams

new_df['Team'] = filtered_df['Team'].apply(map_team_name)

# Convert American odds to decimal odds
def convert_american_to_decimal(american_odds):
    if american_odds.startswith('+'):
        return 1 + int(american_odds[1:]) / 100
    elif american_odds.startswith('-'):
        return 1 + 100 / abs(int(american_odds))
    else:
        return None

new_df['Decimal Odds'] = new_df['Odds'].apply(convert_american_to_decimal)
new_df['Decimal Odds'] = new_df['Decimal Odds'].round(2)

# Convert decimal odds to implied probability
def decimal_to_implied_probability(decimal_odds):
    if decimal_odds > 0:
        return round(1 / decimal_odds, 2)
    else:
        return None

new_df['Implied Probability'] = new_df['Decimal Odds'].apply(decimal_to_implied_probability)

# Save the final DataFrame to a CSV file
new_df.to_csv('nhl_odds.csv', index=False)

print("Script completed. Check 'nhl_odds.csv' for the results.")

#FINAL VERSION OF GAME SCRAPING
# URL of the page to scrape
url = 'https://www.naturalstattrick.com/'

# Send an HTTP request to the website
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all tables with class "boxscore"
tables = soup.find_all('table', class_='boxscore')

# Load team abbreviations
abbreviations_df = pd.read_csv('teams.csv')  # Replace with the correct path to your CSV
mapping_dict = dict(zip(abbreviations_df['Name'], abbreviations_df['Abbrv']))

# Prepare the data
games_data = []

for table in tables:
    # Find all rows in the table
    rows = table.find_all('tr')
    if len(rows) >= 2:  # Ensure there are at least two rows (away and home teams)
        away_team = rows[0].find('td', style='text-align: left').text.strip()
        home_team = rows[1].find('td', style='text-align: left').text.strip()
        today_date = datetime.datetime.now().strftime('%Y-%m-%d')  # Current date

        # Map team names to abbreviations
        away_team_abbr = mapping_dict.get(away_team, away_team)  # Default to original name if not found
        home_team_abbr = mapping_dict.get(home_team, home_team)  # Default to original name if not found

        # Append the extracted data with Game column
        games_data.append({
            'AwayTeam': away_team_abbr,
            'HomeTeam': home_team_abbr,
            'Date': today_date,
            'Game': f"{today_date} - {away_team_abbr}, {home_team_abbr}"
        })

# Convert the data into a DataFrame
games_df = pd.DataFrame(games_data)

# Save the DataFrame to a CSV file
output_file = 'games_today.csv'
if not games_df.empty:
    games_df.to_csv(output_file, index=False)
    print(f"Data scraped and saved to {output_file}")
else:
    print("No data found. Check the selectors or website structure.")

current_date = pd.Timestamp(datetime.date.today())

def calculate_rolling_average(df, columns, window=10):
    rolling_df = df[columns].rolling(window=window, min_periods=10).mean()
    return rolling_df


base_path = '24_25_Team'
df4 = pd.DataFrame()
# New top-level folder for averaged data
average_data_path = '24_25_test'
os.makedirs(average_data_path, exist_ok=True)
# Iterate over each year's folder
# Iterate over each team's CSV file
for team_file in os.listdir(base_path):
    if team_file.endswith('.csv'):
        team_path = os.path.join(base_path, team_file)
        # Read the CSV file
        df = pd.read_csv(team_path)

        # Select the desired columns and calculate the rolling averages
        selected_columns = ['Game', 'Team', 'Date', 'AwayTeam', 'AwayScore', 'HomeTeam', 'HomeScore', 'HomeResult', 'days_of_rest']
        other_columns = df.columns.difference(selected_columns)
        rolling_df = calculate_rolling_average(df, other_columns)

    
        # Combine the selected columns with the rolling averages
        combined_df = pd.concat([df[selected_columns], rolling_df], axis=1)
        cols_to_drop = ['Game','AwayTeam', 'AwayScore', 'HomeTeam', 'HomeScore', 'HomeResult','days_of_rest']
       
    #drop other unneccesary columns
        combined_df.drop(columns=cols_to_drop, inplace=True)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df['time_diff']=(current_date-combined_df['Date'])
        combined_df['days_of_rest'] = combined_df['time_diff'].dt.days - 1
        combined_df.drop('time_diff', axis=1, inplace=True)
        combined_df.insert(1, 'days_of_rest', combined_df.pop('days_of_rest'))
        last_row_df = combined_df.tail(1)
        df4 = pd.concat([df4, last_row_df], ignore_index=True, axis=0)

df4.to_csv('test.csv', index=False)

#pull form the file we just created and the list of today's game and get the stats for the home and away teams with the prefixes added and save it to the "todays_games.csv"
df_test = pd.read_csv('test.csv')
df_today= pd.read_csv('games_today.csv')

def add_prefix_to_columns(df, prefix):
    df.columns = [prefix + col for col in df.columns]
    return df

def find_team_data_modified(team, df_test):
    team_data = df_test[df_test['Team'] == team].drop(columns=['Team', 'Date'])
    return team_data
combined_data = []
for index, row in df_today.iterrows():
    game_info = row['Game']
    away_team = row['AwayTeam']
    home_team = row['HomeTeam']
    away_team_data = add_prefix_to_columns(find_team_data_modified(away_team, df_test).copy(), 'a_')
    home_team_data = add_prefix_to_columns(find_team_data_modified(home_team, df_test).copy(), 'h_')
    if not away_team_data.empty:
        away_team_data = away_team_data.iloc[0]
    if not home_team_data.empty:
        home_team_data = home_team_data.iloc[0]

    combined_row = pd.concat([away_team_data, home_team_data])
    combined_row['Game'] = game_info
    combined_data.append(combined_row)
df_combined_corrected = pd.DataFrame(combined_data)
df_combined_corrected.to_csv('today_games.csv', index=False)


# Load data
df_test = pd.read_csv('test.csv')
df_today = pd.read_csv('games_today.csv')

# Function to add prefix to column names
def add_prefix_to_columns(df, prefix):
    df.columns = [prefix + col for col in df.columns]
    return df

# Function to find data for a specific team
def find_team_data_modified(team, df_test):
    team_data = df_test[df_test['Team'] == team].drop(columns=['Team', 'Date'], errors='ignore')
    return team_data

# Combine data for today's games
combined_data = []
for index, row in df_today.iterrows():
    game_info = row['Game']
    away_team = row['AwayTeam']
    home_team = row['HomeTeam']
    
    # Find data for away and home teams
    away_team_data = add_prefix_to_columns(find_team_data_modified(away_team, df_test).copy(), 'a_')
    home_team_data = add_prefix_to_columns(find_team_data_modified(home_team, df_test).copy(), 'h_')

    # Select the first row of each team if available
    away_team_data = away_team_data.iloc[0] if not away_team_data.empty else pd.Series()
    home_team_data = home_team_data.iloc[0] if not home_team_data.empty else pd.Series()

    # Combine data into a single row
    combined_row = pd.concat([away_team_data, home_team_data], axis=0)
    combined_row['Game'] = game_info  # Add game info as a new column
    combined_data.append(combined_row.to_dict())  # Convert to dict before appending

# Create a DataFrame from the combined data
df_combined_corrected = pd.DataFrame(combined_data)

# Save to CSV
df_combined_corrected.to_csv('today_games.csv', index=False)
print("Data saved to 'today_games.csv'.")

#Load the model and scaler
svm_model = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler.joblib')
df5 = pd.read_csv('today_games.csv')

game_ids = df5['Game']

features = df5.drop(columns=['Game'])

#Scale features
scaled_features = scaler.transform(features)

#Predict probabilities
probabilities = svm_model.predict_proba(scaled_features)
predicted_labels = svm_model.predict(scaled_features)
#Create a dataframe for the output
prob_df = pd.DataFrame(probabilities, columns=[f'Prob_{label}' for label in svm_model.classes_])
prob_df['Predicted Label'] = predicted_labels
prob_df['Game ID'] = game_ids

# Assuming df is your DataFrame with games data
# Splitting the 'Game ID' column to extract away and home team names
prob_df[['Date', 'Teams']] = prob_df['Game ID'].str.split(' - ', expand=True)
prob_df[['Away Team', 'Home Team']] = prob_df['Teams'].str.split(', ', expand=True)
prob_df.drop(columns=['Game ID', 'Teams'], inplace=True)

# Load the NHL odds data
nhl_odds_df = pd.read_csv('nhl_odds.csv')

# Merging the odds information for the away team
prob_df = pd.merge(prob_df, nhl_odds_df, left_on='Away Team', right_on='Team', how='left')
prob_df.rename(columns={'Odds': 'Away Team Odds', 'Decimal Odds': 'Away Team Decimal Odds', 
                   'Implied Probability': 'Away Team Implied Probability'}, inplace=True)

# Merging the odds information for the home team
prob_df = pd.merge(prob_df, nhl_odds_df, left_on='Home Team', right_on='Team', how='left')
prob_df.rename(columns={'Odds': 'Home Team Odds', 'Decimal Odds': 'Home Team Decimal Odds', 
                   'Implied Probability': 'Home Team Implied Probability'}, inplace=True)
prob_df.drop(columns=['Team_x', 'Team_y'], inplace=True)

prob_df = prob_df[['Date', 'Away Team', 'Away Team Odds', 'Away Team Decimal Odds', 'Away Team Implied Probability', 
         'Prob_0', 'Prob_1', 'Home Team Implied Probability', 'Home Team Decimal Odds', 'Home Team Odds', 'Home Team']]
prob_df.rename(columns={'Prob_0': 'Away Probability', 'Prob_1': 'Home Probability'}, inplace=True)

#save it to a csv
formatted_date = current_date.strftime('%Y-%m-%d')
folder_name = "24_25_results"
filename = f"{folder_name}/{formatted_date}.csv"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)


prob_df.to_csv(filename, index=False)
print("Results can be found in 24_25_results in the file called " + filename)