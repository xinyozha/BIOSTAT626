from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn import metrics
from Evaluation import evaluate
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVR
import pandas as pd
import numpy as np

df = pd.read_csv('avg_intensity.csv', index_col=0)
outcome = pd.read_csv('D:/626/projectgroup9/train/train_data.csv', index_col=0)
combined_df = df.join(outcome)
X_train, X_test, y_train, y_test = train_test_split(combined_df.drop(columns=['OSmonth']), combined_df['OSmonth'], test_size=0.2, random_state=42)

# standard
Stand_X = StandardScaler()
Stand_Y = StandardScaler()
train_data = Stand_X.fit_transform(X_train)
test_data = Stand_X.transform(X_test)
train_target = Stand_Y.fit_transform(y_train.values.reshape(-1,1))
test_target = Stand_Y.transform(y_test.values.reshape(-1,1))

# parameters
cv = KFold(n_splits=5, shuffle=True, random_state=42)
clf = GridSearchCV(SVR(),param_grid={'kernel':['linear','poly','sigmoid','rbf'],'C': [0.1,1,10],'gamma': [0.1,1,10]},cv=5, scoring='neg_mean_squared_error')
clf.fit(train_data,y_train)
best_model = clf.best_estimator_
print("best_param:",clf.best_params_)
print("Best RMSE:", np.sqrt(-clf.best_score_))

predictions = best_model.predict(test_data)
svmoutput = evaluate(y_test, predictions)
print("Test SVM Output: ", svmoutput)
