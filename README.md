# **Linear regression: from scratch vs scikit-learn**

## Overview:

This project explores the mathematical foundations of linear regression by implementing the algorithm from scratch in Python and comparing the results with scikit-learn's implementation.

## Objectives:

- Understand the mathematical principles underlying linear regression
- Implement Linear regression from scratch using python
- Compare the results of my custom model to scikit-learns built-in Linear regression model

## Method:

- The function is for my model is y = mx + c ('m' is the gradient, 'c' is the intercept)
- Create a cost function, I will be using the mean square error
- Create a gradient function to compute the derivatives of the mean square error
- Create a function that carries out the gradient descent algorithm
- Use the gradient descent algorithm to eventually find the best values for _m_ and _c_
- Display the results
- Implement a second model using scikit-learn's LinearRegression class

#### Gradient descent algorithm:

- starting with initial values of _m_ and _c_
- repeatedly update _m_ and _c_ in small increments for a certain number of iterations to reduce the mean square error and converge to the minimum value
- this is done using a formula that includes the partial derivatives of the mean square error and a fixed value for the learning rate to control the step size

#### Mathematical formulas:

found in 'Linear regression Maths.pdf'

## Libraries needed:

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Results and evaluation:

#### Mean Square Error:

I evaluated the performance of the two implementations using the mean squared error (MSE) which measures the average squared difference between actual and predicted values

- My custom implementation (gradient descent): 15,648,488
- Scikit-learns implementation: 31,270,951

My value for the MSE is about 50% lower than the value for Scikit-learns implementation, suggesting that gradient descent performed better here and provided more optimised paramaters

#### Visualisation:

My custom implementation:

<img width="797" height="569" alt="image" src="https://github.com/user-attachments/assets/1a7f9c35-d2ba-45fc-bd95-210e82e141ad" />

Scikit-learns implementation:

<img width="797" height="569" alt="image" src="https://github.com/user-attachments/assets/66c9c8fe-07a4-4938-876a-a9a65eb10648" />


## Future Improvements:

- Investigate how the implementations behave on different datasets
- Explore the effect of different parameter estimation methods
