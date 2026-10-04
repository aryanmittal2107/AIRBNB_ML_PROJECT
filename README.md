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
