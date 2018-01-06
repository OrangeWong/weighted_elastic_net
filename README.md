# Creating Weighted Elastic Net estimator in Scikit-learn

## What is Weighted Elastic Net?
Weighted elastic net handles heteroskedasticity in data and applys regularization to the model to address overfitting.

## Introduction
We start from the improved version of elastic net from Section 13.5.3.3 in Machine Learning (by Kevin P. Murphy). 
Weight is not considered in this formulation. Therefore, we formulate a weighted improved version of elastic net (weighted elastic net) in order to handle the heteroskedasticity in data.

The cost function that calculates the error of the regression fit is:
\begin{align}
J &= \beta^T \left( \frac{X^T W X + \lambda_2  I}{1 + \lambda_2} \right) \beta - 2 y^T W X \beta + \lambda_1 || \beta ||_1
\end{align}
where $\beta$ is the coefficients of the fit, $W$ is the weight with respect to the observations, $X$ is the design matrix representating the data, $\lambda_1$ and $\lambda_2$ represents the degree of lasso and ridge regularizations, respectively.











