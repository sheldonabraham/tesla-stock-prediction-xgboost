# Tesla Stock Price Prediction: Can a Model Beat "Same as Yesterday"?

Predicting Tesla's next-day closing price with XGBoost, and testing whether the model does better than a simple "no change" guess.

**Main finding:** it doesn't. Past daily price moves did not help predict tomorrow's move. Comparing against simple baselines is what shows this.

## Results (test period: 19 Feb 2020 to 13 Aug 2021, 376 trading days)

| Measure | Model | Simple baseline |
|---|---|---|
| Average error, in % points | 3.558 | 3.427 ("no change") |
| Days with the direction (up/down) right | 50.8% | 53.5% ("always up") |
| Average error, in dollars | $14.77 | $14.15 ("same as yesterday") |

The chart of actual vs predicted price looks almost perfect. That is only because each prediction starts from yesterday's real price. The dollar errors in the table show the model adds nothing.

## Data

- `TSLA.csv`: 1,258 trading days of Tesla prices, 16 Aug 2016 to 13 Aug 2021 (Open, High, Low, Close, Adj Close, Volume)
- Only the closing price is used
- No missing values
- The price rose from about $45 to about $717, peaking at $883, with most of the rise from 2020 onward

## Method

1. **Target:** the daily % change in closing price, not the price itself.
2. **Inputs:** the previous 5 days' % changes.
3. **Split by date, not at random:** train on 24 Aug 2016 to 18 Feb 2020 (876 days), test on the next 376 days. Shuffling time series data would let the model see the future.
4. **Model:** XGBoost regressor (200 trees, max depth 3, learning rate 0.05).
5. **Evaluation:** compared with "no change", "always up" and "same as yesterday" baselines.

## Why % change instead of price

The first version predicted the price directly from the previous 50 prices. It was off by about **$356 a day**, and its predictions were an almost flat line between $158 and $166.

Tree-based models like XGBoost cannot predict values outside the range they were trained on. The training data never went above about $177, but the test period went up to $883. Daily % changes look similar in 2017 and 2021, so the model can learn from them. This change cut the error from about $356 to $14.77.

## What I learned

- Always compare a model with a simple baseline. A good-looking chart can hide a useless model.
- Split time series by date, never at random.
- Tree models can't extrapolate beyond their training range.
- Daily stock moves carry very little information about the next day's move.

## Limitations

- Only past price changes were used. News, trading volume and market indices were not tested.
- The test period (2020–21) was unusually volatile. The average absolute daily move was 2.15% in training and 3.43% in testing.
- One stock and one train/test split.
- This is a learning project, not investment advice.

## Repository contents

```
├── TSLA_StockPred_Proj.ipynb   # the full analysis
├── TSLA.csv                    # daily price data
└── README.md
```

## How to run

```bash
pip install pandas numpy matplotlib plotly xgboost scikit-learn jupyter
jupyter notebook TSLA_StockPred_Proj.ipynb
```

Keep `TSLA.csv` in the same folder as the notebook, then run all cells.

## Tools

Python, pandas, NumPy, XGBoost, scikit-learn, Matplotlib, Plotly
