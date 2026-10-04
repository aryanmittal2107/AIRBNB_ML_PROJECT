# Airbnb Price Prediction — Multi-City Machine Learning Analysis

## Overview

This project analyzes Airbnb listing data from three cities — **New York City, Paris, and Berlin** — to investigate how listing characteristics influence Airbnb prices and to compare the performance of multiple machine learning models across different cities.

The project follows a common machine learning workflow for all three cities:

1. Dataset inspection
2. Data cleaning and preprocessing
3. Price extraction and conversion
4. Missing-value handling
5. Feature selection
6. Model training
7. Model evaluation
8. Feature importance analysis
9. Cross-city comparison

The main objective is to understand whether the same features and machine learning approaches perform consistently across different Airbnb markets.

---

## Cities Analyzed

- New York City
- Paris
- Berlin

Each city has its own Airbnb listings dataset.

---

## Project Structure

```text
AIRBNB_ML_PROJECT/
│
├── data/
│   ├── NYC/
│   │   └── listings.csv.gz
│   │
│   ├── Paris/
│   │   └── listings (1).csv.gz
│   │
│   └── berlin/
│       └── listings (2).csv.gz
│
├── notebooks/
│   └── 01_data_inspection.ipynb
│
├── .gitignore
└── README.md
```
Dataset
The datasets contain Airbnb listing information such as:
- Price
- Room type
- Number of guests accommodated
- Bedrooms
- Beds
- Bathrooms
- Number of reviews
- Reviews per month
- Availability
- Latitude and longitude
- Minimum nights
- Review scores
- Reviews in the last 12 months
The original datasets contain a large number of columns. A subset of relevant variables is selected for the machine learning experiments.
Data Preprocessing
Price Cleaning
The price column is stored as a string containing currency symbols and separators.
For example:
$226.75

The price values are converted into numerical values by removing currency symbols and separators.
A new price_clean column is created and used as the prediction target.
Rows with missing target prices are removed.
Missing Values
For the core feature set, missing numerical feature values are handled using median imputation.
The expanded feature set uses stricter cleaning and removes rows containing missing values in the selected expanded features.
Outlier Analysis
Interquartile Range (IQR) analysis is performed for each city's price distribution.
Extreme prices are identified as potential outliers, but they are not automatically removed because unusually expensive Airbnb listings may represent genuine listings in the dataset.
Features
Core Feature Set
The initial model uses:
room_type
accommodates
bedrooms
beds
bathrooms
number_of_reviews
reviews_per_month
availability_365

Expanded Feature Set
The expanded model additionally includes:
latitude
longitude
minimum_nights
review_scores_rating
number_of_reviews_ltm

Therefore, the expanded model uses both listing characteristics and additional geographic, booking, and review-related information.
Machine Learning Models
Three regression models are evaluated:
1. Linear Regression
A baseline linear regression model is used to establish a simple relationship between the selected features and Airbnb prices.
2. Random Forest
Random Forest regression is used to capture nonlinear relationships between listing characteristics and price.
3. XGBoost
XGBoost regression is used as a gradient-boosting approach to model more complex relationships in the Airbnb data.
Model Evaluation
The models are evaluated using:
Mean Absolute Error (MAE)
Measures the average absolute difference between the predicted and actual prices.
Lower MAE indicates better performance.
Root Mean Squared Error (RMSE)
Measures the square root of the average squared prediction error.
RMSE gives greater weight to large prediction errors.
Lower RMSE indicates better performance.
R² Score
Measures how much of the variation in the target price is explained by the model.
Higher R² indicates better explanatory performance.
Results
The notebook evaluates both the core and expanded feature sets for each city.
New York City
The expanded feature set improved the overall predictive performance compared with the initial core feature set.
The strongest XGBoost features included:
- Bathrooms
- Minimum nights
- Number of reviews in the last 12 months
- Room type
- Number of reviews
- Accommodates
The expanded XGBoost model achieved an R² of approximately 0.53.
Paris
The expanded feature set substantially improved the linear regression model compared with the core feature set.
The expanded models produced:
Model	MAE	RMSE	R²
Linear Regression	140.13	340.31	0.2991
Random Forest	126.28	382.93	0.1125
XGBoost	136.33	556.30	-0.8729


For Paris, the XGBoost model showed instability caused by very large prediction errors on extreme-priced listings.
The most important XGBoost features included:
- Beds
- Latitude
- Longitude
- Bedrooms
- Review score
- Minimum nights
- Accommodates
Berlin
For Berlin, Linear Regression produced the strongest overall performance when considering RMSE and R².
The expanded models produced:
Model	MAE	RMSE	R²
Linear Regression	64.83	107.89	0.4847
Random Forest	61.30	153.77	-0.0468
XGBoost	62.02	150.50	-0.0027


Although Random Forest achieved the lowest MAE, Linear Regression produced substantially lower RMSE and the highest R².
Important XGBoost features for Berlin included:
- Availability
- Latitude
- Accommodates
- Room type
- Bathrooms
- Bedrooms
- Minimum nights
Key Observations
The analysis demonstrates that model performance varies considerably between cities.
New York City
The expanded feature set provided a significant improvement, particularly for tree-based models.
Paris
The Paris dataset contains highly variable and extreme prices. XGBoost was particularly sensitive to these extreme listings, resulting in large prediction errors.
Berlin
Linear Regression performed strongly after adding the expanded features, achieving the highest R² and lowest RMSE among the evaluated models.
These differences suggest that Airbnb pricing relationships are city-dependent and that a single machine learning model does not necessarily perform equally well across different markets.
Feature Importance
Feature importance is examined using the XGBoost models.
This helps identify which listing characteristics contribute most strongly to the model's predictions.
The important features differ between cities, highlighting differences in Airbnb markets.
For example:
- New York City: bathrooms, minimum nights, and recent review activity were highly influential.
- Paris: beds and geographic coordinates were particularly influential.
- Berlin: availability, latitude, and accommodates were among the strongest features.
Notebook
The complete analysis is contained in:
notebooks/01_data_inspection.ipynb

The notebook contains:
- Dataset inspection
- Data cleaning
- Exploratory analysis
- Feature engineering
- Core model experiments
- Expanded model experiments
- Model evaluation
- Feature importance analysis
- City-specific observations
Requirements
The project uses Python and the following main libraries:
Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib

A Python virtual environment can be used to install the required dependencies.
Example:
python -m venv .venv

Activate the environment and install the required libraries:
pip install pandas numpy scikit-learn xgboost matplotlib

Running the Project
Clone the repository:
git clone https://github.com/aryanmittal2107/AIRBNB_ML_PROJECT.git

Navigate into the project:
cd AIRBNB_ML_PROJECT

Activate the virtual environment.
On Windows PowerShell:
.\.venv\Scripts\activate

Open the notebook:
notebooks/01_data_inspection.ipynb

Make sure the datasets are placed in the expected data/ directories before running the notebook.
Reproducibility
The notebook uses relative paths so that it can be executed from the repository structure:
../data/NYC/listings.csv.gz
../data/Paris/listings (1).csv.gz
../data/berlin/listings (2).csv.gz

The project uses a fixed random state for train-test splitting to make model evaluation reproducible.
Future Work
The next stage of the project will include:
- Final comparison of NYC, Paris, and Berlin
- Cross-city model performance visualization
- Comparison of feature importance across cities
- Analysis of why certain models perform differently across markets
- Additional model tuning and evaluation
