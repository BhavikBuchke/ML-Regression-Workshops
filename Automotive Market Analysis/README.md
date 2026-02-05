# 🛞 Automobile Price Prediction Analysis

This repository contains a comprehensive data science project focused on building and comparing predictive models to estimate the market price of used automobiles based on technical specifications and usage history.

## 📌 Project Objective

The core goal of this analysis is to develop an automated valuation model to assist resale businesses in scaling efficiently. By leveraging machine learning, we aim to predict car prices using features such as engine power, mileage, brand, and fuel type.

## 📊 Dataset Overview

The analysis is performed on a dataset of 100 car records. Key features include:

* **Technical Specs:** Engine capacity, Power, Mileage, and Seats.
* **Usage History:** Kilometers Driven, Year of manufacture, and Owner Type.
* **Categorical Data:** Brand, Model, Fuel Type, and Transmission.

The data profiling phase confirmed high data integrity with **zero missing values** across all 13 columns.

## 🛠️ Preprocessing Strategy

To prepare raw data for machine learning algorithms, a robust preprocessing pipeline was implemented:

* **One-Hot Encoding:** Transformed categorical variables (Brand, Fuel Type, etc.) into binary format.
* **Standard Scaling:** Normalized numerical features to ensure high-magnitude values (like Kilometers Driven) do not disproportionately bias the model.
* **Pipeline Integration:** Used Scikit-Learn's `ColumnTransformer` and `Pipeline` to ensure a clean, reproducible workflow from raw data to prediction.

## 🤖 Models Evaluated

The project conducts a head-to-head experiment between two distinct modeling philosophies:

1. **Multiple Linear Regression:** Used as a baseline, assuming a direct linear relationship between features and price.
2. **Random Forest Regressor:** An ensemble method using multiple decision trees to capture complex, non-linear patterns.

## 📈 Performance Results

The models were evaluated using **R-Squared (R2)** and **Root Mean Squared Error (RMSE)**.

| Model | R2 Score | RMSE |
| --- | --- | --- |
| **Multiple Linear Regression** | **0.91** | **302,448** |
| Random Forest Regressor | 0.88 | 343,186 |

### **Key Insights**

* **Linear Superiority:** Interestingly, the simpler Linear Regression model outperformed the Random Forest. This suggests that the primary relationships in this specific dataset—such as the inverse correlation between age/mileage and price—are significantly linear.
* **Data Volume:** The Random Forest model likely requires a larger dataset to effectively capture non-linear interactions and surpass the accuracy of the baseline linear model.

## 🚀 Getting Started

### Prerequisites

The following libraries are required to run the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

```

### Usage

1. Open the `automobile_analysis.ipynb` notebook.
2. Run the cells sequentially to perform data loading, preprocessing, model training, and visualization.
3. The notebook includes custom functions `model_metrics` and `plot_metrics` for streamlined evaluation.
