import numpy as np

def forward_feature_selection_chronos(
    df,
    target_col,
    feature_list,
    train_mask,
    val_mask,
    runtime,
    endpoint_name,
    max_features=None,
):
    selected_features = []
    remaining_features = feature_list.copy()
    
    best_global_mape = float("inf")
    history = []

    # Pre-extract target once (efficiency)
    train_target = df.loc[train_mask, target_col].values
    val_target = df.loc[val_mask, target_col].values

    iteration = 0

    while remaining_features:
        print(f"\n--- Iteration {iteration} ---")
        
        best_feature = None
        best_feature_mape = float("inf")
        best_forecast = None

        for feature in remaining_features:
            
            current_features = selected_features + [feature]

            # Prepare covariates
            train_cov = df.loc[train_mask, current_features]
            val_cov = df.loc[val_mask, current_features]

            #  Single Chronos call per feature
            forecast_dict = chronos_predict_quantile_05(
                runtime=runtime,
                endpoint_name=endpoint_name,
                train_series_smoothed=train_target,
                past_covariates=train_cov,
                prediction_length=len(val_target),
            )

            # extract 0.5 quantile
            y_pred = forecast_dict[0.5]

            current_mape = mape(val_target, y_pred)

            print(f"Feature: {feature} | MAPE: {current_mape:.4f}")

            if current_mape < best_feature_mape:
                best_feature_mape = current_mape
                best_feature = feature
                best_forecast = y_pred

        # stopping condition (no improvement)
        if best_feature_mape < best_global_mape:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_global_mape = best_feature_mape

            history.append({
                "iteration": iteration,
                "feature_added": best_feature,
                "mape": best_feature_mape,
                "features": selected_features.copy()
            })

            print(f"\n Selected: {best_feature} | New Best MAPE: {best_feature_mape:.4f}")

        else:
            print("\n No improvement — stopping")
            break

        iteration += 1

        if max_features and len(selected_features) >= max_features:
            break

    return selected_features, history
