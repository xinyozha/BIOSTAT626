from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate(y_test, y_pred):
    """Evaluate the performance of a regression model.
    Parameters:
    - y_test: array, true values of response variable.
    - y_pred: array, predicted values of response variable.
    Returns:
    - A dictionary containing RMSE, MAE, MSE, and R-squared metrics.
    """
    metrics = {}
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics['MAE'] = mean_absolute_error(y_test, y_pred)
    metrics['MSE'] = mean_squared_error(y_test, y_pred)
    metrics['R2'] = r2_score(y_test, y_pred)
    metrics['NMSE'] = metrics['MSE'] / np.var(y_test)
    return metrics
