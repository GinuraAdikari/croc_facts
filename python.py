import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------------
# Helpers
# ----------------------------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def fit_and_forecast_residuals(residuals_train, steps, order, seasonal_order, trend):
    """
    Fit SARIMAX on residuals_train and forecast `steps` residuals.
    Returns np.array of shape (steps,).
    """
    model = SARIMAX(
        residuals_train,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    # forecast returns a Series/array-like length=steps
    return np.asarray(fit.forecast(steps=steps), dtype=float), fit


def chronos_predict_quantile_05(
    runtime,
    endpoint_name,
    train_series_smoothed,
    past_covariates,
    prediction_length=12,
    quantiles=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
):
    """
    Calls your Chronos endpoint once and returns the q0.5 forecast as np.array (length=prediction_length).
    Adjust extraction logic to match your endpoint response structure.
    """
    quantiles = list(quantiles)
    quantile_keys = [str(q) for q in quantiles]

    payload = {
        "inputs": [{
            "target": train_series_smoothed,
            "past_covariates": past_covariates,
        }],
        "parameters": {
            "prediction_length": prediction_length,
            "quantile_levels": quantiles,
        }
    }

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read().decode())

    # From your screenshot: result["predictions"][0][key]
    q05_key = "0.5"
    chronos_q05 = np.array(result["predictions"][0][q05_key], dtype=float)
    return chronos_q05


# ----------------------------
# Your grid
# ----------------------------
p = [0, 1, 2, 3, 4, 5]
d = [0, 1]
q = [0, 1, 2, 3, 4]
P = [0, 1]
D = [0, 1]
Q = [0, 1, 2]
trends = ["n", "c", "t"]   # keep simple
m = 12
H = 12  # forecast horizon

_grid = list(itertools.product(p, d, q, P, D, Q, trends))

# ----------------------------
# 0) Prepare aligned train/test targets
# ----------------------------
# You must define these:
# residuals_train: residual series used to fit SARIMA (index/time-ordered)
# y_true_12: actual target for the next 12 months (aligned with your forecast horizon)
#
# Example if you already have test_df of next 12 months:
# y_true_12 = test_df["METRIC_VALUE"].values.astype(float)

# ----------------------------
# 1) Get Chronos q0.5 ONCE (recommended)
# ----------------------------
# chronos_q05 = chronos_predict_quantile_05(runtime, endpoint_name, train_series_smoothed, past_covariates, prediction_length=H)

# If you already computed chronos forecasts earlier, just reuse it:
# chronos_q05 = your_existing_array_length_12

# ----------------------------
# 2) Grid search by FINAL MAPE
# ----------------------------
rows = []
fail_count = 0

for (pp, dd, qq, PP, DD, QQ, tr) in _grid:
    try:
        # a) residual forecast
        resid_fc, fitted = fit_and_forecast_residuals(
            residuals_train=residuals_train,
            steps=H,
            order=(pp, dd, qq),
            seasonal_order=(PP, DD, QQ, m),
            trend=tr,
        )

        # b) combine with Chronos median
        y_pred_12 = chronos_q05 + resid_fc

        # c) MAPE
        mape_val = mape(y_true_12, y_pred_12)

        if np.isnan(mape_val):
            continue

        rows.append({
            "order": (pp, dd, qq),
            "seasonal_order": (PP, DD, QQ, m),
            "trend": tr,
            "mape": float(mape_val),
            # Optional diagnostics:
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
        })

    except Exception as e:
        fail_count += 1
        # Debugging line (keep during development):
        # print(f"FAILED order={(pp,dd,qq)} seasonal={(PP,DD,QQ,m)} trend={tr} -> {e}")
        continue

print(f"Successful fits: {len(rows)} | Failed fits: {fail_count}")


resid_fc, _ = fit_and_forecast_residuals(
    residuals_train, 12, order=(0,0,0), seasonal_order=(0,0,0,12), trend="n"
)
print(resid_fc)
df_tuned = pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)

top5 = df_tuned.head(5)
top5
