vimport pandas as pd
import numpy as np

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False


def seasonal_naive(values, season_length=365):
    preds = []
    for i in range(len(values)):
        if i - season_length >= 0:
            preds.append(values[i - season_length])
        else:
            # fallback: mean of last 7 values
            preds.append(values[max(0, i-7):i+1].mean() if i > 0 else values[0])
    return preds


if __name__ == "__main__":
    df = pd.read_csv("multivariate_series.csv")

    # target variable
    target_col = "target"
    df["ds"] = pd.to_datetime(df["date"])  # Prophet format
    df["y"] = df[target_col]

    if PROPHET_AVAILABLE:
        print("Prophet detected — training baseline model...")

        model = Prophet()
        model.fit(df[["ds", "y"]])

        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)

        preds = forecast["yhat"].iloc[:len(df)].values

    else:
        print("Prophet NOT installed — using seasonal naive baseline.")
        preds = seasonal_naive(df[target_col].values, season_length=365)

    # save predictions
    out = pd.DataFrame({
        "date": df["date"],
        "y_true": df[target_col],
        "y_pred_prophet": preds
    })

    out.to_csv("prophet_predictions.csv", index=False)
    print("Saved prophet_predictions.csv")
