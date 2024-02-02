# Introduction to Machine Learning<a href='https://github.com/mayer79/ml_lecture'><img src='logo.png' align="right" height="200" /></a>

#### Michael Mayer

## Overview

Welcome to this training course on machine learning (ML). 
It is both available in R and Python.

ML can be viewed as a collection of statistical algorithms used to

1. predict values (supervised ML) or to
2. investigate data structure (unsupervised ML).

Our focus is on *supervised ML*. Depending on if we predict numbers or classes, we talk about *regression* or *classification*.

## Copyright

This lecture is being distributed under the [creative commons license](https://creativecommons.org/licenses/by/2.0/).

## How to cite?

Michael Mayer. Introduction to Machine Learning (2021). Web: https://github.com/mayer79/ml_lecture/

## Organization

The lecture is split into four chapters, each of which is accompanied with an R/Python notebook. You will find them in the corresponding subfolders.

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

Each chapter will take us about two hours and is accompanied by exercises.

You can preview the html outputs of the R lecture by clicking on these links:

1. [Basics and Linear Models](https://mayer79.github.io/ml_lecture/1_Basics_and_Linear_Models.html)
2. [Model Selection and Validation](https://mayer79.github.io/ml_lecture/2_Model_Selection_and_Validation.html)
3. [Trees](https://mayer79.github.io/ml_lecture/3_Trees.html)
4. [Neural Nets](https://mayer79.github.io/ml_lecture/4_Neural_Nets.html)

[Exercise Solutions](https://mayer79.github.io/ml_lecture/exercise_solutions.html)

The Python html files can be downloaded from the repository as well, but are not linked.

## Prerequisites

In order to be able to follow the lecture, you should be familiar with

- Descriptive statistics
- Linear regression
- R or Python

To get the lecture notes both as html and code-notebooks, clone this repository by running 

```
git clone https://github.com/mayer79/ml_lecture.git
```

### Software (and data) for the Python version of the lecture

We will use the conda environment specified in the "py" folder.

1. Install Anaconda (if not yet available on your machine)
2. Open the conda prompt and use it to navigate to your "ml_lecture/py" folder
3. Type in the conda prompt: "conda env create -f environment.yml"
4. Use this environment to run the code of the lecture, e.g., in [Visual Studio Code](https://code.visualstudio.com/) or [Jupyterlab](https://jupyterlab.readthedocs.io/en/stable/)

Furthermore, download the "**Car** [Data - csv]" dataset on the following [link](http://www.businessandeconomics.mq.edu.au/our_departments/Applied_Finance_and_Actuarial_Studies/research/books/GLMsforInsuranceData/data_sets)
and put it as "car.csv" into the "py" folder. It is about 4 MB small.

Note: we use default [black](https://github.com/psf/black) as autoformatter.

### Software for the R version of the lecture

- R version >= 4.1
- Successfully installed packages:
    - tidyverse
    - FNN
    - withr
    - rpart.plot
    - ranger
    - xgboost
    - keras
    - hstats
    - MetricsWeighted
    - insuranceData
    - lightgbm (optional)

For the last chapter, we will use Python with TensorFlow >= 2.15. You can install it by running the R command `keras::install_keras(version = "release-cpu")`. If the following code works, you are all set. (Some red start-up messages/warnings are okay.)

```
library(tensorflow)
tf$constant("Hello Tensorflow!")
```

## Optional Topics to Discuss

- Basics and Linear Models: random effects; penalized regression
- Model Selection and Validation: is it always necessary?
- Trees: supervised clustering; SHAP; interaction constraints
- Neural Nets: transfer learning; PyTorch; image data; autoencoders
- General: distribution of predictions; quantile regression; when is overfitting (un-)problematic?
- XAI: How to use insights from complex ML model to improve GLM?

## Further Reading

- James, G., Witten, D., Hastie, T., Tibshirani, R. (2013). *An Introduction to Statistical Learning - with Applications in R*. New York: Springer.

- Hastie, T., Tibshirani, R., Friedman, J. (2001). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. New York: Springer.

- Wickham, H., Grolemund, G. (2017). *R for Data Science: Import, Tidy, Transform, Visualize, and Model Data*. O'Reilly Media. 

- VanderPlas, J. (2016). *Python data science handbook : essential tools for working with data*. O'Reilly Media.

- Chollet, F. (2017). *Deep Learning with Python*. Manning Publications Co.

- Chollet, F., Allaire, J. J. (2018). *Deep Learning with R*. Manning Publications Co.



