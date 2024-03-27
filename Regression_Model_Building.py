import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from Stepwise import forward_stepwise_regression, backward_stepwise_regression
from Evaluation import evaluate
import statsmodels.api as sm

# Data Processing
df = pd.read_csv('avg_intensity.csv', index_col=0)
outcome = pd.read_csv('train/train_data.csv', index_col=0)
combined_df = df.join(outcome)
X_train, X_test, y_train, y_test = train_test_split(combined_df.drop(columns=['OSmonth']), combined_df['OSmonth'], test_size=0.2, random_state=42)
#scaler = StandardScaler()
#scaler.fit(X_train)
#X_train_scaled = scaler.transform(X_train)
#X_test_scaled = scaler.transform(X_test)
#X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
#X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# Linear Regression Model
linear_mod = LinearRegression().fit(X_train,y_train)
y_pred_linear = linear_mod.predict(X_test)
linear_metrics = evaluate(y_test, y_pred_linear)
print("Linear Regression Metrics:", linear_metrics)

# Downward Stepwise Regression Model
backward_model = backward_stepwise_regression(X_train, y_train)
selected_features = backward_model.model.exog_names
selected_features.remove('const')
X_test_selected = X_test[selected_features]
y_pred_backward = backward_model.predict(sm.add_constant(X_test_selected))
backward_metrics = evaluate(y_test, y_pred_backward)
print("Backward Stepwise Regression Metrics:", backward_metrics)

# Ridge Regression Model
ridge_mod = Ridge(alpha=1.0).fit(X_train, y_train)
y_pred_ridge = ridge_mod.predict(X_test)
ridge_metrics = evaluate(y_test, y_pred_ridge)
print("Ridge Regression Metrics:", ridge_metrics)

# Lasso Regression Model
lasso_mod = Lasso(alpha=1.0).fit(X_train, y_train)
y_pred_lasso = lasso_mod.predict(X_test)
lasso_metrics = evaluate(y_test, y_pred_lasso)
print("Lasso Regression Metrics:", lasso_metrics)

# Elastic Net Regression Model
elastic_mod = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X_train,y_train)
y_pred_elastic = elastic_mod.predict(X_test)
elastic_metrics = evaluate(y_test, y_pred_elastic)
print("Elastic Net Regression Metrics:", elastic_metrics)