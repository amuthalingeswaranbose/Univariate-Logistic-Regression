# Univariate-Logistic-Regression
Univariate Logistic Regression Class Implementation using Numpy Python Library (From Scratch).

## Logistic Regression - LR

  * Logistic Regression is used to predict the relationship between the y(dependent variable) and the x (independent variables), where the y (dependent varible) is binary (0 or 1) nature.α * learning_rate)
  
  * LR is used for classification problems.

## Formulas,

  1. Finding of y, y = (m * x) + c (m - slope, c - intercept, x - input, y - predicted output) linear equation.
  2. Calcualte sigmoid of y, a = 1 / (1 + exp(-y))
  3. Calculate loss, loss = -(y*log(a) + (1-y)*log(1-a))
  4. Find derivative of m, dm = ((a - y)* x)
  5. Find derivative of b, db = (a - y)
  6. Update the m (slope), m = m - (learning_rate * dm)
  7. Update the b (intercept), b = b - (learning_rate * db)

## Required Libraries

  1. Numpy .
  2. Matplotlib.
  3. random.
  
## Files

  1. Univariate_Logistic_Regression.py - contains Univariate_Logistic_Regression Class and methods (fit, pred, pred_prob).
  2. univariate-class-test.ipynb - contains the examples of random value generate, fit training data and find pred and pred_prob of given testing samples.

## How-to

 1. Install required libraries.
 2. clone this repo.
 3. open univariate-class-test.ipynb jupyter notebook and start run the code.
 4. after completing the notebook, try your own input and check how it works.
