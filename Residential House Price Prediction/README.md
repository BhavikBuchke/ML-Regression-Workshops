# House Price Prediction Analysis

This project aims to provide accurate real estate valuation advice by identifying the underlying distribution of house prices and evaluating how structural features influence market value. The analysis follows a structured five-phase data science methodology to transform raw housing data into a robust predictive engine.

## Project Overview

The primary objective is to model house prices based on various physical and temporal characteristics. By leveraging machine learning pipelines, the project establishes a baseline for predictive accuracy and improves upon it using ensemble methods.

### Key Technical Achievements

* **Target Transformation:** Identified a log-normal distribution in sales prices, applying log transformations to ensure mathematical convergence in regression models.
* **Automated Pipeline:** Developed a streamlined preprocessing pipeline using `scikit-learn` to handle imputation, scaling, and categorical encoding in a single call.
* **Model Optimization:** Evaluated multiple algorithms, ultimately selecting a Random Forest regressor for its ability to capture complex feature interactions and reduce error.

## Dataset Description

The analysis utilizes a comprehensive real estate dataset containing 32 features. Key variables explored include:

* **Target Variable:** `SALEPRICE`
* **Temporal Features:** `YEARBUILT`, `YRSOLD`, `MOSOLD`
* **Physical Characteristics:** `LOTAREA`, `HOUSESTYLE`, `OVERALLCOND`, `GARAGECARS`, `POOLAREA`
* **Categorical Metadata:** Building type, foundation style, heating quality, and more.

## Project Phases

### 1. Business & Data Understanding

Initial analysis focused on target distribution. Visualizations (including 3D scatter plots) were used to map the relationships between lot area, house age, and price.

### 2. Data Preprocessing

To prepare the data for modeling, the following steps were integrated into a `ColumnTransformer` pipeline:

* **Numerical Data:** Missing values were handled via median imputation, followed by standard scaling.
* **Categorical Data:** Imputed with the most frequent values and transformed using One-Hot Encoding.

### 3. Model Evaluation

Four distinct modeling frameworks were tested to establish benchmarks:

1. **Simple Linear Regression:** Established the initial  baseline.
2. **Multiple Linear Regression:** Integrated over 30 features into a high-dimensional model.
3. **Decision Tree:** Utilized to capture non-linear feature interactions.
4. **Random Forest (Winner):** An ensemble approach that significantly reduced predictive error.

## Results and Impact

The project successfully established a high-performance model with the following technical outcomes:

* **Efficiency:** The automated pipeline handles all preprocessing steps seamlessly, allowing for easy integration of new data.
* **Accuracy:** The Random Forest ensemble model achieved the lowest error rate among all tested frameworks.
* **Metrics:** Models were evaluated based on Root Mean Squared Error (RMSE) and Variance Score ().

## Installation and Requirements

To reproduce the analysis, ensure you have the following Python libraries installed:

* `pandas` (v2.2.2)
* `matplotlib` (v3.8.4)
* `seaborn` (v0.13.2)
* `scikit-learn` (v1.4.2)
* `numpy`

### Usage

1. Clone the repository.
2. Ensure the dataset is accessible via the provided URL or local path.
3. Execute the `Predict_house_prices.ipynb` notebook to run the full training and evaluation pipeline.
