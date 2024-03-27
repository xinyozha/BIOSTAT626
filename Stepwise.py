import statsmodels.api as sm

def forward_stepwise_regression(X_train, y_train, significance_level=0.05):

    initial_features = X_train.columns.tolist()
    selected_features = []
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(selected_features))
        pvalues = []
        for new_feature in remaining_features:
            model = sm.OLS(y_train, sm.add_constant(X_train[selected_features + [new_feature]])).fit()
            pvalues.append((new_feature, model.pvalues[new_feature]))
        pvalues.sort(key=lambda x: x[1])
        if pvalues[0][1] < significance_level:
            selected_features.append(pvalues[0][0])
        else:
            break
    final_model = sm.OLS(y_train, sm.add_constant(X_train[selected_features])).fit()
    return final_model

def backward_stepwise_regression(X_train, y_train, significance_level=0.05):
    features = X_train.columns.tolist()
    while len(features) > 0:
        model = sm.OLS(y_train, sm.add_constant(X_train[features])).fit()
        # Find the predictor with the highest p-value
        max_p_value = max(model.pvalues[1:])  # Skip the intercept
        feature_to_remove = model.pvalues[1:].idxmax()  # Skip the intercept
        if max_p_value > significance_level:
            features.remove(feature_to_remove)
        else:
            break
    final_model = sm.OLS(y_train, sm.add_constant(X_train[features])).fit()
    return final_model
