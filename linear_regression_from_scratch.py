import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("SalaryData.csv")

x_train = data.YearsExperience
y_train = data.Salary

#-----------------------------------------------------------------------------------------------

def MeanSquareError(x , y , m , c):

    n = len(x)
    error = 0

    for i in range(n):
        y_pred = m*x[i] + c
        error += (y_pred - y[i])**2
    
    total_error = (1/(2*n)) * error
    return total_error

#-----------------------------------------------------------------------------------------------

def GradientFunction(x , y , m , c):

    n = len(x)

    dE_dm = 0
    dE_dc = 0
    
    for i in range(n):
        y_pred = m*x[i] + c

        dE_dm += (1/n)*(x[i])*(y_pred - y[i])
        dE_dc += (1/n)*(y_pred - y[i])

    return dE_dm , dE_dc

#-----------------------------------------------------------------------------------------------v

def GradientDescent(x , y , L , iterations):

    m = 0
    c = 0

    for i in range(iterations):
        dE_dm , dE_dc = GradientFunction(x , y , m , c)

        m = m - L*(dE_dm)
        c = c - L*(dE_dc)

    return m , c

#-----------------------------------------------------------------------------------------------

learning_rate = 0.001
iterations = 20000

m_final , c_final = GradientDescent(x_train , y_train , learning_rate , iterations)

print(f"m: {m_final} , c: {c_final}")
print(f"Mean square error: {MeanSquareError(x_train , y_train , m_final , c_final)}")

#-----------------------------------------------------------------------------------------------

plt.scatter(x_train , y_train , color="black")

x_values = np.linspace(min(x_train) , max(x_train) , 100)
y_values = m_final * x_values + c_final

plt.xlabel("Years experience")
plt.ylabel("Salary")

plt.plot(x_values , y_values , color="red" , label="Regression line")
plt.legend()

plt.show()







