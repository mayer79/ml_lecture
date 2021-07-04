# Introduction to Machine Learning

#### Michael Mayer

## Overview

Welcome to this little lecture on machine learning (ML). 

ML can be viewed as a collection of statistical algorithms used to

1. predict values (supervised ML) or to
2. investigate data structure (unsupervised ML).

Our focus is on *supervised ML*. Depending on if we predict numbers or classes, we talk about *regression* or *classification*.

## Copyright

This lecture is being distributed under the [creative commons license](https://creativecommons.org/licenses/by/2.0/).

## How to cite?

Michael Mayer. Introduction to Machine Learning (2020). Web: https://github.com/mayer79/ml_lecture/

## Organization

The lecture is split into four chapters, each of which is accompanied with an R resp. Python notebook. You will find them in the corresponding subfolders.

1. Basics and Linear Models 
    - Basics
    - Linear regression
    - Generalized Linear Model
2. Model Selection and Validation
3. Trees
    - Decision trees
    - Random forests
    - Gradient boosting
4. Neural Nets

All examples are self-contained.

Each chapter will take us about two hours and includes exercises.

You can preview the html outputs of the R lecture by clicking on these links:

1. [Basics and Linear Models](https://htmlpreview.github.io/?https://github.com/mayer79/ml_lecture/blob/master/r/1_Basics_and_Linear_Models.html)
2. [Model Selection and Validation](https://htmlpreview.github.io/?https://github.com/mayer79/ml_lecture/blob/master/r/2_Model_Selection_and_Validation.html)
3. [Trees](https://htmlpreview.github.io/?https://github.com/mayer79/ml_lecture/blob/master/r/3_Trees.html)
4. [Neural Nets](https://htmlpreview.github.io/?https://github.com/mayer79/ml_lecture/blob/master/r/4_Neural_Nets.html)

## Prerequisites

In order to be able to follow the lecture, you should be familiar with

- Descriptive statistics
- Linear regression
- R/Python

To get the lecture notes both as html and code-notebooks, clone this repository by running 

```
git clone https://github.com/mayer79/ml_lecture.git
```

### Software for the R version of the lecture

- R version >= 3.6
- Successfully installed packages:
    - `tidyverse`
    - `FNN`
    - `rpart`
    - `rpart.plot`
    - `ranger`
    - `xgboost`
    - `keras`
    - `flashlight`
    - `splitTools`
    - `insuranceData`

`keras` requires a Python installation with Tensorflow >=2.0. If you do not have Python installed and are not behind a company firewall, you can simply run the following two lines to get it.

```
library(keras)
install_keras()
```

Run the following code to test `keras`.

```
library(keras)
# use_python(path to Python)

input <- layer_input(shape = 3)

output <- input %>% 
  layer_dense(5, activation = "relu") %>% 
  layer_dense(1)

simple_model <- keras_model(input, output)

simple_model %>% 
  compile(loss = "mse", optimizer = optimizer_adam(lr = 0.1))

simple_model %>% 
  fit(x = data.matrix(iris[2:4]),
      y = iris[, 1],
      epochs = 20,
      batch_size = 10)
```

### Software for the Python version of the lecture

to do


## Further Reading

- James, G., Witten, D., Hastie, T., Tibshirani, R. (2013). *An Introduction to Statistical Learning - with Applications in R*. New York, USA: Springer New York Inc.

- Hastie, T., Tibshirani, R., Friedman, J. (2001). *The Elements of Statistical Learning*. New York, USA: Springer New York Inc.

- Wickham, H., Grolemund, G. (2017). *R for Data Science: Import, Tidy, Transform, Visualize, and Model Data*. O'Reilly Media. 

- VanderPlas, J. (2016). *Python data science handbook : essential tools for working with data*. Sebastopol, CA: O'Reilly Media, Inc.

- Chollet, F. (2017). *Deep Learning with Python*. Manning.

- Chollet, F., Allaire, J. J. (2018). *Deep Learning with R*. Manning.



