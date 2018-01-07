# Weighted Elastic Net Estimator

## What is Weighted Elastic Net?
Weighted elastic net handles heteroskedasticity in data and applys regularization to the model to address overfitting.

## Introduction
We start from the improved version of elastic net from Section 13.5.3.3 in Machine Learning (by Kevin P. Murphy). The details can be found in the Report.pdf, in which how the regularization can address the overfitting is briefly discussed. 
However, weight is not considered in this formulation. Therefore, we formulate a weighted improved version of elastic net (weighted elastic net) in order to handle the heteroskedasticity in data. The formualtion is shown in Report.pdf. 

## Implementation
A WEN() class is wirtten in weighted_elastic_net.py. Two optimization approaches are applied to get the mimimun of the cost function. 
The first is the sub-gradient descent since the L1 norm is not differentialbe. The second approach is the corrdinate descent method, and an anlytical coordinate gradient can be obtained and written in form of soft-thresholding function. The derivation is sohwn in Report.pdf.

The implmenetation is demonstrated in Weighted_elastic_net.ipynb. 










