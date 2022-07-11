import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

url = 'https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
df = pd.read_csv(url)

fig = plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Hours', y='Scores')
# plt.show()
plt.close()

# splitting the data into train and test values.
X = df[['Hours']].values
y = df['Scores'].values

# taking 30% fo the data for testing and rest for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

# training a linearRegression model.
lr = LinearRegression()
lr.fit(X_train, y_train)

# predicting the scores for given hours in test dataset.
yhat_train = lr.coef_ * X_train + lr.intercept_

fig = plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train)
plt.plot(X_train, yhat_train, '-r')
# plt.show()
plt.close()

# predicting values on test data
yhat_test = lr.predict(X_test)

fig = plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test)
plt.plot(X_test, yhat_test, '-r')
# plt.show()
plt.close()

# evaluating the model based on test results.
r_score = r2_score(y_test, yhat_test)
MSE = mean_squared_error(y_test, yhat_test)
MAE = mean_absolute_error(y_test, yhat_test)

eval_df = pd.DataFrame({"Evalution Method": ['R2 score', 'Mean Sqaured Error', 'Mean Absolute Error'], "Values": [r_score, MSE, MAE]})
print(eval_df)
print(lr.predict([[9.25]]))

