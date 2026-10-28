import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = pd.read_csv("SalaryData.csv")

x_train = np.array(data['YearsExperience']).reshape(-1,1)
y_train = np.array(data['Salary']).reshape(-1,1)

model = LinearRegression()
model.fit(x_train , y_train)

x_values = np.linspace(min(x_train) , max(x_train) , 100)
y_values = model.predict(x_values)

y_preds = model.predict(x_train)

print(f"Mean square error: {mean_squared_error(y_train , y_preds)}")

plt.scatter(x_train , y_train , color='black')

plt.xlabel("Years experience")
plt.ylabel("Salary")

plt.plot(x_values , y_values , color="red" , label="Regression line")
plt.legend()

plt.show()

