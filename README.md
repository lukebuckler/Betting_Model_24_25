# NHL Game Prediction Model

## Overview
This project implements a machine learning model to predict NHL (National Hockey League) game outcomes. The system uses Support Vector Machine (SVM) classification to analyze historical game data and make predictions about future game results.

## Project Structure
The repository contains two main components:
- `run_nhl_model.py`: Script for data collection, preprocessing, and real-time predictions
- `SVM_model_training.ipynb`: Jupyter notebook containing model training and evaluation

## Features
- Scrapes real-time NHL game data from multiple sources
- Processes and transforms game statistics into meaningful features
- Implements SVM classification for game outcome prediction
- Includes comprehensive model evaluation metrics
- Provides probability estimates for game outcomes

## Model Performance
The model achieves the following performance metrics:
- Accuracy: 59%
- Mean Cross-Validation Score: 0.59
- ROC-AUC Score available in training notebook

## Requirements
- Python 3.x
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - beautifulsoup4
  - requests
  - joblib

## Usage
1. Train the model:
   ```python
   # Run the Jupyter notebook
   SVM_model_training.ipynb
 
2. Make Predictions
   ```python
   # Run the prediction script
   python run_nhl_model.py


## Data
The model uses historical NHL game data including:

Team statistics
Game scores
Performance metrics
Rest days between games

Results are stored in CSV format with predictions and probability estimates for each game.
These results are then used in comparison to the betting lines in order to idetnify positive expected value bets on NHL games.


## Author
Luke Buckler
## Acknowledgments

Natural Stat Trick for providing NHL statistics
Yahoo Sports for odds data


## Data Sources

- [Natural Stat Trick](https://www.naturalstattrick.com/): Source for all NHL games data.


