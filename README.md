# Optiver - Trading at the Close Competition Project

## Introduction
This project is an exploration into predicting closing price movements of Nasdaq listed stocks using data from the daily ten-minute closing auction on the NASDAQ stock exchange. The primary goal is to forecast future price movements relative to a synthetic index. This analysis plays a crucial role in enhancing market efficiency, especially during the final minutes of trading, which are often marked by heightened volatility.

## Dataset
The dataset provided by the Optiver Kaggle competition includes the following key variables:

- `stock_id`: Unique identifier for the stock
- `date_id`: Unique identifier for the date
- `imbalance_size`: Unmatched amount at the current reference price (USD)
- `imbalance_buy_sell_flag`: Indicator of auction imbalance direction
- `reference_price`: Near price bounded between the best bid and ask price
- `matched_size`: Amount that can be matched at the current reference price (USD)
- `far_price`: Crossing price for maximizing shares matched based on auction interest
- `near_price`: Crossing price for maximizing shares matched including continuous market orders
- `bid/ask_price`: Most competitive buy/sell level in the non-auction book
- `bid/ask_size`: Dollar notional amount on the most competitive buy/sell level in the non-auction book
- `wap`: Weighted average price in the non-auction book
- `seconds_in_bucket`: Seconds elapsed since the start of the day's closing auction
- `target`: 60-second future move in the weighted average price of the stock less the synthetic index move

## Feature Engineering and Selection
Due to Kaggle's RAM limitations and the extensive dataset size (approximately 5 million data points), the project utilized Recursive Feature Elimination (RFE) with cross-validation and a basic XGBoost model for feature selection. This process reduced the feature count to 54, making the model manageable.

## Models Used
- **Elastic Net**: A linear model with L1 and L2 penalties, useful for handling multicollinearity and overfitting.
- **LightGBM**: A gradient boosting model known for its efficiency with large-scale data.
- **XGBoost**: A robust model for structured data, capable of parallel computation and effective handling of missing data.
- **Multilayer Perceptron (MLP)**: A neural network model for capturing complex non-linear patterns.

## Methodology
The project adopted a time series split approach for cross-validation to maintain the chronological order of the data, crucial for time series analysis.

### Hyperparameter Tuning
Each model underwent specific hyperparameter tuning:
- Elastic Net: Focused on alpha (L2 penalty), l1_ratio (L1 penalty), and max_iter (maximum iterations).
- XGBoost: Two-phase tuning focusing on n_estimators, learning_rate, max_depth, and colsample_bynode.
- LightGBM: Similar two-phase approach as XGBoost, tuning n_estimators, learning_rate, num_leaves, and colsample_bytree.
- MLP: Tuned mainly on the activation function, choosing LeakyReLU for better performance.

## Evaluation Metrics
- Mean Absolute Error (MAE) was used for ElasticNet, XGBoost, and LightGBM.
- Smooth L1 loss was used for MLP.

## Results
XGBoost emerged as the top performer, closely followed by LightGBM. ElasticNet showed respectable results, while MLP lagged behind in this specific dataset.

## Variable Importance Analysis
- Elastic Net: Focused on imbalance_support_v2, wap, and reference_price.
- XGBoost and LightGBM: Highlighted the importance of seconds_in_bucket, date_id, and various imbalance indicators.

## Challenges and Learnings
The project faced several challenges, including handling Kaggle's 30GB RAM limitation, managing a large dataset, time-consuming hyperparameter tuning, feature engineering and selection, and MLP structure design. Strategies like using Numba for feature calculation significantly sped up the computation process.

## Conclusion
The project underscored the importance of various features across different models and highlighted the trade-offs between model complexity and interpretability.


_This project is part of the CompSci671 course and was submitted on December 5th, 2023 by Qijun Yang._

