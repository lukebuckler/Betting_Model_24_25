# Betting_Model_24_25
 NHL game projection and betting model
# Betting Model 24-25

This repository contains the code and resources for a betting model designed for the 2024-2025 NHL season. The project scrapes data from various sources, processes team statistics, and generates insights to inform betting strategies.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Usage](#usage)
- [Data Sources](#data-sources)

## Introduction

The goal of this project is to automate the collection and analysis of NHL data to enhance betting decision-making. The model aggregates daily game data, maps team statistics to corresponding games, and calculates additional metrics such as days of rest and team performance trends.

## Features

- Processes NHL game data from the previous 3 seasons from [Natural Stat Trick](https://www.naturalstattrick.com/).
- Trains a SVM model in order to predict outcome probabilites for upcoming NHL games.
- Scrapes daily game data from [Natural Stat Trick](https://www.naturalstattrick.com/).
- Maps rolling 10 game averages for team statistics to upcoming games.
- Calculates key metrics, such as days of rest and game results.
- Model takes in the home and away team's average statistics from the prrevious 10 games and outputs the predicted pobability of either team winning the game. 
- Outputs clean CSV files for analysis.


## Usage

The training set was processed in a jupyter labs notebook to prepare a training set of about 5000 NHL games with the statistics of each team's perfromance in the previous 10 games, as well as the results of said game.

The script nhl_model_run.py is run each day and it scrapes betting odds, and all the NHL games for far this season, processes the games to get the 10 game averages for each team, then puts together the input data for each of the games for the day and saves the outputs to a CSV titled by the date. 


## Data Sources

- [Natural Stat Trick](https://www.naturalstattrick.com/): Source for all NHL games data.


