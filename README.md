# Stock_Trend_Predictor
SOC - Stock Trend Predictor

This is my Summer of Code project. The goal was to predict whether a stock's
price will go up or down the next day, using past price/volume data and some
basic machine learning models. Did the whole pipeline from scratch - downloading
data, cleaning it, building features, training models, and evaluating them.

Mentor gave weekly tasks (Week 1 to Week 4 so far), this README covers all of
that.

Week 1 - Getting Familiar

Downloaded 2 years of Apple (AAPL) stock data using yfinance. Just got used to
basic stuff here - what Open/High/Low/Close/Volume actually mean, printing
data, understanding the shape of a stock dataset. No charts or ML yet, just
getting comfortable with the data.

Week 2 - Cleaning and Visualizing

Cleaned the data (checked for missing values, duplicates), then started
plotting it - closing price over time, volume, moving averages (MA20/MA50),
daily returns, and a correlation heatmap. This is where I actually started
seeing patterns in the data instead of just numbers in a table.

Saved the cleaned data as AAPL_clean.csv at the end.

Week 3 - Feature Engineering

This week was about turning raw price data into something a model can
actually learn from. Built 8 features like MA5, MA5_to_MA20, lag returns,
5-day volatility, volume change, etc. Also created the Target column - 1 if
the stock went up the next day, 0 if it didn't.

My own 2 features (assignment task):


RSI_14 - the classic momentum indicator, tells you if a stock is
overbought or oversold
Close_to_MA50 - same idea as Price_to_MA20 but using the 50-day average,
gives a longer term view of price position


Saved the final ML-ready dataset as AAPL_features.csv (482 rows, 10
features + Target, no missing values).

Week 4 - First ML Models

Split the data 80/20 by time (no shuffling, since this is time series data -
shuffling would let the model cheat by seeing the future). Trained two
models:


Logistic Regression
Decision Tree (max_depth=4)


Results on AAPL:

ModelTrain AccTest AccLogistic Regression52.9%48.4%Decision Tree64.0%51.6%

Also tried a Decision Tree with no depth limit just to see overfitting
happen live - it hit 100% train accuracy but dropped to 44% on test data.
Classic overfitting, model just memorised the training data instead of
learning anything useful.

Interesting finding: my two Week 3 features (RSI_14, Close_to_MA50)
ended up with zero feature importance in the Decision Tree. The model
didn't use them at all - other features like MA5 and Volatility_5d were
already covering that same momentum information. Doesn't mean the features
were wrong to build, just that they turned out redundant for this
particular model.

Assignment - comparing with another stock (RELIANCE.NS)

Ran the same full pipeline (features + both models) on Reliance Industries
to see if results change with a different stock.

ModelRELIANCE TrainRELIANCE TestLogistic Regression60.1%57.8%Decision Tree68.5%47.8%

Logistic Regression actually did better on Reliance than on AAPL - real
edge above the 50% random baseline. Decision Tree did worse and overfit
more (bigger gap between train and test accuracy). Looking at the confusion
matrix, the Decision Tree was basically predicting "Up" almost every time
on Reliance data, which is why it had a lot of false positives.

Conclusion: Logistic Regression was more reliable across both stocks.
Decision Tree tends to overfit more and gets biased toward one class. Stock
choice also matters - same model, same features, different results just by
changing the ticker.

(Side note: hit a ValueError: Input X contains infinity error while doing
this part - turned out Volume_Change had a few infinite values from a
divide-by-zero on a low-volume day. Fixed by replacing inf with NaN and
dropping those rows.)

Files in this repo


Week1.ipynb / Week2.ipynb / week3.ipynb / week4.ipynb - notebooks
for each week
AAPL_clean.csv - cleaned data from Week 2
AAPL_features.csv - final ML-ready dataset with all 10 features +
Target from Week 3
