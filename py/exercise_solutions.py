#=============================================================================
# Exercises
#=============================================================================

#=============================================================================
# Chapter 1) Basics and Linear Models
#=============================================================================

#=============================================================================
# Exercise on linear regression
#=============================================================================

# As alternative to the multiple linear regression on diamond prices with 
# logarithmic price and logarithmic carat, consider the same model without 
# logarithms. Interpret the output of the model. 
# Does it make sense from a practical perspective?
  
from math import gamma

from sklearn.dummy import DummyRegressor


library(tidyverse)
dia <- diamonds %>% 
  mutate_at(c("color", "cut", "clarity"), function(z) factor(z, ordered = FALSE))
fit <- lm(price ~ carat + color + cut + clarity, data = dia)
summary(fit)

# Comments
# Model quality: About 92% of price variations are explained by covariates.
#  Typical prediction error is 1157 USD.
# Effects: All effects point into the intuitively right direction
#  (larger stones are more expensive, worse color are less expensive etc.)
# Practical perspective: Additivity in color, cut and clarity are not 
#  making sense. Their effects should get larger with larger diamond size. 
#  This can be solved by adding interaction terms with carat or, much easier,
#  to switch to a logarithmic response.

#=============================================================================
# Exercise on GLMs
#=============================================================================

# Fit a Gamma regression with log-link to explain diamond prices by 
# `log(carat)`, `color`, `cut`, and `clarity`. Compare the coefficients with 
# those from the corresponding linear regression with `log(price)` as response. 
# Use dummy coding for the three categorical variables.

library(tidyverse)
dia <- diamonds %>% 
  mutate_at(c("color", "cut", "clarity"), function(z) factor(z, ordered = FALSE))
fit <- glm(price ~ log(carat) + color + cut + clarity, 
           data = dia, family = Gamma(link = "log"))
summary(fit)
mean(dia$price) / mean(predict(fit, type = "response")) - 1

# Comment: The coefficients are very similar to the linear regression with
#  log(price) as response. This makes sense, in the end we interpret the 
#  coefficients in the same way! The bias is only 0.3%, i.e. much smaller
#  than the 3% of the OLS with log(price) as response. Still, because
#  log is not the natural link of the Gamma regression, it is not exactly 0.

#=============================================================================
# Chapter 2) Model Selection and Validation
#=============================================================================

#=============================================================================
# Exercise 1
#=============================================================================

# Apply *stratified* splitting (on the response `price`) throughout the 
# last example. Do the results change?

library(tidyverse)
library(FNN)
library(splitTools)
library(MetricsWeighted)

# Covariates
x <- c("carat", "color", "cut", "clarity")

dia <- diamonds[, c("price", x)]

# Split diamonds into 90% for training and 10% for testing
ix <- partition(dia$price, p = c(train = 0.9, test = 0.1), 
                seed = 9838, type = "stratified")

train <- dia[ix$train, ]
test <- dia[ix$test, ]

y_train <- train$price
y_test <- test$price

# Standardize training data
X_train <- scale(data.matrix(train[, x]))

# Apply training scale to test data
X_test <- scale(data.matrix(test[, x]),
                center = attr(X_train, "scaled:center"),
                scale = attr(X_train, "scaled:scale"))

# Split training data into folds
nfolds <- 5
folds <- create_folds(y_train, k = nfolds, seed = 9838, type = "stratified")

# Cross-validation performance of k-nearest-neighbour for k = 1-20
paramGrid <- data.frame(RMSE = NA, k = 1:20)

for (i in 1:nrow(paramGrid)) {
  k <- paramGrid[i, "k"]
  scores <- c()
  
  for (fold in folds) {
    pred <- knn.reg(X_train[fold, ], test = X_train[-fold, ], 
                    k = k, y = y_train[fold])$pred
    scores[length(scores) + 1] <- rmse(y_train[-fold], pred)
  }
  paramGrid[i, "RMSE"] <- mean(scores)
}

# Best k along with its CV performance
paramGrid[order(paramGrid$RMSE)[1], ]

# Cross-validation performance of linear regression
rmse_reg <- c()

for (fold in folds) {
  fit <- lm(price ~ log(carat) + color + cut + carat, data = train[fold, ])
  pred <- predict(fit, newdata = train[-fold, ])
  rmse_reg[length(rmse_reg) + 1] <- rmse(y_train[-fold], pred)
}
(rmse_reg <- mean(rmse_reg))

# The overall best model is 6-nearest-neighbour
pred <- knn.reg(X_train, test = X_test, k = 6, y = y_train)$pred

# Test performance for the best model
rmse(y_test, pred)

# Comment: The CV results are slightly better and the test performance 
# is clearly worse. Both might be by chance.

#=============================================================================
# Exercise 2
#=============================================================================

# Regarding the problem that some diamonds seem to appear multiple times in 
# the data: As an alternative to *grouped* splitting, repeat the last example 
# also on data deduplicated by `price` and all covariates. Do the results change? 
# Do you think these results are overly pessimistic?

library(tidyverse)
library(FNN)
library(splitTools)
library(MetricsWeighted)

# Covariates
x <- c("carat", "color", "cut", "clarity")

dia <- diamonds[, c("price", x)]
dia <- unique(dia)

# Split diamonds into 90% for training and 10% for testing
ix <- partition(dia$price, p = c(train = 0.9, test = 0.1), seed = 9838, type = "basic")

train <- dia[ix$train, ]
test <- dia[ix$test, ]

y_train <- train$price
y_test <- test$price

# Standardize training data
X_train <- scale(data.matrix(train[, x]))

# Apply training scale to test data
X_test <- scale(data.matrix(test[, x]),
                center = attr(X_train, "scaled:center"),
                scale = attr(X_train, "scaled:scale"))

# Split training data into folds
nfolds <- 5
folds <- create_folds(y_train, k = nfolds, seed = 9838, type = "basic")

# Cross-validation performance of k-nearest-neighbour for k = 1-20
paramGrid <- data.frame(RMSE = NA, k = 1:20)

for (i in 1:nrow(paramGrid)) {
  k <- paramGrid[i, "k"]
  scores <- c()
  
  for (fold in folds) {
    pred <- knn.reg(X_train[fold, ], test = X_train[-fold, ], 
                    k = k, y = y_train[fold])$pred
    scores[length(scores) + 1] <- rmse(y_train[-fold], pred)
  }
  paramGrid[i, "RMSE"] <- mean(scores)
}

# Best k along with its CV performance
paramGrid[order(paramGrid$RMSE)[1], ]

# Cross-validation performance of linear regression
rmse_reg <- c()

for (fold in folds) {
  fit <- lm(price ~ log(carat) + color + cut + carat, data = train[fold, ])
  pred <- predict(fit, newdata = train[-fold, ])
  rmse_reg[length(rmse_reg) + 1] <- rmse(y_train[-fold], pred)
}
(rmse_reg <- mean(rmse_reg))

# The overall best model is 5-nearest-neighbour
pred <- knn.reg(X_train, test = X_test, k = 5, y = y_train)$pred

# Test performance for the best model
rmse(y_test, pred)

# Comments: The test performance of the best model (5-nn) seems clearly worse than the one
# without deduplication (~700 USD RMSE vs ~600). CV performance well corresponds to test performance.
# Overall, this is probably the more realistic performance than the one obtained from the original data set.
# Still, as certain rows could be identical by chance, our deduplication approach might be slightly too conservative.
# The true performance will probably be somewhere between the two approaches.

#=============================================================================
# Exercise 3
#=============================================================================

# Use cross-validation to select the best polynomial degree to represent 
# `log(carat)` in the Gamma GLM with log-link (with additional covariates 
# `color`, `cut`, and `clarity`). Evaluate the result on an independent 
# test data.

library(tidyverse)
library(splitTools)
library(MetricsWeighted)

dia <- diamonds %>% 
  mutate_at(c("color", "cut", "clarity"), function(z) factor(z, ordered = FALSE))

# Split diamonds into 90% for training and 10% for testing
ix <- partition(dia$price, p = c(train = 0.9, test = 0.1), seed = 9838, type = "basic")
train <- dia[ix$train, ]
test <- dia[ix$test, ]

# manual GridSearchCV
nfolds <- 5
folds <- create_folds(train$price, k = nfolds, seed = 9838, type = "basic")
paramGrid <- data.frame(Deviance = NA, k = 1:12)

for (i in 1:nrow(paramGrid)) {
  k <- paramGrid[i, "k"]
  scores <- c()
  
  for (fold in folds) {
    fit <- glm(price ~ poly(log(carat), degree = k) + color + cut + clarity, 
               data = train[fold, ], family = Gamma(link = "log"))
    
    pred <- predict(fit, train[-fold, ], type = "response")
    scores[length(scores) + 1] <- deviance_gamma(train$price[-fold], pred)
  }
  paramGrid[i, "Deviance"] <- mean(scores)
}

paramGrid

# Fit model on full training data
fit <- glm(price ~ poly(log(carat), degree = 8) + color + cut + clarity, 
           data = train, family = Gamma(link = "log"))

# Evaluate on test
pred <- predict(fit, test, type = "response")
deviance_gamma(test$price, pred) # 0.01710076
r_squared_gamma(test$price, pred) # 0.982464 relative deviance gain

# Comments: The optimal degree seems to be 8 with a CV deviance of 0.01575.
# There seems to be some amount of CV overfit as the deviance evaluated on 
# the test data is worse.

#=============================================================================
# Exercise 4
#=============================================================================

# Optional: Compare the linear regression for `price` (using `log(carat)`, 
# `color`, `cut`, and `clarity` as covariates) with a corresponding Gamma GLM 
# with log-link by simple validation. Use once (R)MSE for comparison and once 
# Gamma deviance. What do you observe?

# -> solution not shown here
  
#=============================================================================
# Chapter 3) Trees
#=============================================================================

#=============================================================================
# Exercises on Random Forests
#=============================================================================

#=============================================================================
# Exercise 1
#=============================================================================

# In above example, replace carat by its logarithm. Do the results change 
# compared to the example without logs?
library(tidyverse)
library(splitTools)
library(ranger)
library(MetricsWeighted)

dia <- diamonds %>% 
  mutate(log_carat = log(carat))

# Train/test split
ix <- partition(dia$price, p = c(train = 0.8, test = 0.2), seed = 9838)

fit <- ranger(price ~ log_carat + color + cut + clarity, 
              num.trees = 500,
              data = dia[ix$train, ], 
              importance = "impurity",
              seed = 83)
fit

# Performance on test data
pred <- predict(fit, dia[ix$test, ])$predictions
rmse(dia$price[ix$test], pred)

# Comment: The results are essentially identical because log is a monotonic trafo.
# Differences might come from implementation tricks of ranger.

#=============================================================================
# Exercise 2
#=============================================================================

# Fit a random forest on the claims data, predicting the binary variable `clm` 
# by the covariates `veh_value`, `veh_body`, `veh_age`, `gender`, `area`, 
# and `agecat`. Choose a suitable tree-depth by maximizing OOB error on the 
# training data. Make sure to fit a *probability random forest*, i.e. 
# predicting probabilities, not classes. Additionally, make sure to work with 
# a relevant loss function (information/cross-entropy or Gini gain). 
# Use a simple train/test split. Interpret the results by split gain importance 
# and partial dependence plots.

library(tidyverse)
library(splitTools)
library(ranger)
library(MetricsWeighted)
library(insuranceData)
data(dataCar)

# Train/test split (stratified on response)
ix <- partition(dataCar$clm, p = c(train = 0.8, test = 0.2), seed = 9838)

# Instead of systematic grid search, manually select good tree depth by OOB
fit <- ranger(clm ~ veh_value + veh_body + veh_age + gender + area + agecat,
              data = dataCar[ix$train, ], probability = TRUE, max.depth = 5,
              importance = "impurity")
fit # OOB prediction using Brier score (= MSE) 0.06340884 

# Note: Brier score is the same as the MSE, applied to binary data. It is
# not a bad evaluation criterion in such situation, 
# so we will use this within this example.
pred <- predict(fit, dataCar[ix$test, ])$predictions[, 2]
mse(dataCar[ix$test, "clm"], pred)  # 0.06337011
r_squared(dataCar[ix$test, "clm"], pred) # 0.002069925

# Alternative to Brier score: relative log-loss resp. deviance improvement
r_squared_bernoulli(dataCar[ix$test, "clm"], pred) # 0.004246408

# Comment: Test performance with small tree depth 5 seem to be best
# according to OOB results. When studying relative performance metrics
# like the relative deviance gain, we can see that performance of the 
# model is very low. TPL claims seem to be mostly determined by bad luck,
# which makes sense.

# Variable importance regarding Gini improvement
imp <- sort(importance(fit))
imp <- imp / sum(imp)
barplot(imp, horiz = TRUE, col = "orange", cex.names = 0.8)

# Partial dependence plot for the strongest predictor "veh_value"
fl <- flashlight(model = fit, 
                 y = "clm", 
                 data = dataCar[ix$train, ], 
                 label = "rf", 
                 predict_function = function(m, X) predict(m, X)$predictions[, 2])
plot(light_profile(fl, v = "veh_value", breaks = seq(0, 5, by = 0.1)))

#=============================================================================
# Exercises on Boosting
#=============================================================================

#=============================================================================
# Exercise 1
#=============================================================================

# Study the documentation of XGBoost to figure out how to make the model 
# monotonically increasing in carat. 
# Test your insights without rerunning the grid search in our last example, 
# i.e. just be refitting the final model. 
# How does the partial dependence plot for `carat` look now?

library(tidyverse)
library(xgboost)
library(splitTools)

# As XGBoost does not understand strings, we encode them by integers
prep_xgb <- function(X, x = c("carat", "color", "cut", "clarity")) {
  to_int <- c("color", "cut", "clarity")
  X[, to_int] <- lapply(X[, to_int], as.integer)
  data.matrix(X[, x])
}

# Split into train and test
ix <- partition(diamonds$price, p = c(train = 0.8, test = 0.2), seed = 9838)

y_train <- diamonds$price[ix$train]
X_train <- prep_xgb(diamonds[ix$train, ])

# XGBoost data handler
dtrain <- xgb.DMatrix(X_train, label = y_train)

# Load grid and select best iteration
grid <- readRDS("r/gridsearch/diamonds_xgb.rds")
grid <- grid[order(grid$score), ]

# Monotone constraints for carat
params <- as.list(grid[1, -(1:2)])
params$monotone_constraints <- c(1, 0, 0, 0)

# Fit final, tuned model with monotone constraints for carat
fit <- xgb.train(
  params = params, 
  data = dtrain, 
  nrounds = grid[1, "iteration"]
)

# Partial dependence plot for carat
library(flashlight)
fl <- flashlight(model = fit, 
                 y = "price", 
                 data = diamonds[ix$train, ], 
                 label = "xgb", 
                 predict_function = function(m, X) predict(m, prep_xgb(X)))

plot(light_profile(fl, v = "carat", n_bins = 40)) +
  labs(title = "Partial dependence plot for carat", y = "price")

# Comment: The argument is called "monotone_constraints". For each covariate,
# a value 0 means no constraint, a value -1 means a negative constraints,
# and a value 1 means positive constraint. Applying the constraint now leads
# to a monotonically increasing partial dependence plot. This is extremely
# useful in practice. Besides monotonic constraints, also interaction 
# constraints are possible.

#=============================================================================
# Exercise 2
#=============================================================================

# Develop a strong XGBoost model for the claims data set with binary response 
# `clm` and covariates `veh_value`, `veh_body`, `veh_age`, `gender`, `area`, 
# and `agecat`. Use a clean cross-validation/test approach. 
# Use log loss both as objective and evaluation metric. Interpret its results.

# We just adapt the template from the script
library(tidyverse)
library(xgboost)
library(splitTools)
library(insuranceData)
data(dataCar)

# As XGBoost does not understand strings, we encode them by integers
prep_xgb <- function(X, x = c("veh_value", "veh_body", "veh_age", 
                              "gender", "area", "agecat")) {
  to_int <- c("veh_body", "gender", "area")
  X[, to_int] <- lapply(X[, to_int], as.integer)
  data.matrix(X[, x])
}

# Split into train and test
ix <- partition(dataCar$clm, p = c(train = 0.8, test = 0.2), seed = 9838)

y_train <- dataCar$clm[ix$train]
X_train <- prep_xgb(dataCar[ix$train, ])

# XGBoost data handler
dtrain <- xgb.DMatrix(X_train, label = y_train)

# If grid search is to be run again, set tune <- TRUE
tune <- FALSE

if (tune) {
  # Use default parameters to set learning rate with suitable number of rounds
  params <- list(
    learning_rate = 0.03,
    objective = "binary:logistic",
    eval_metric = "logloss"
  )
  
  # Cross-validation
  cvm <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 5000,
    nfold = 5,
    early_stopping_rounds = 10,
    verbose = 0
  )
  cvm 
  
  # Grid
  grid <- expand.grid(
    iteration = NA,
    score = NA,
    learning_rate = 0.03,
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 3:6, 
    min_child_weight = c(1, 10),
    colsample_bytree = c(0.8, 1), 
    subsample = c(0.8, 1), 
    reg_lambda = c(0, 2.5, 5, 7.5),
    reg_alpha = c(0, 4),
    #   tree_method = "hist",   # when data is large
    min_split_loss = c(0, 1e-04)
  )
  
  # Grid search or randomized search if grid is too large
  max_size <- 20
  grid_size <- nrow(grid)
  if (grid_size > max_size) {
    grid <- grid[sample(grid_size, max_size), ]
    grid_size <- max_size
  }
  
  # Loop over grid and fit XGBoost with five-fold CV and early stopping
  pb <- txtProgressBar(0, grid_size, style = 3)
  for (i in seq_len(grid_size)) {
    cvm <- xgb.cv(
      params = as.list(grid[i, -(1:2)]),
      data = dtrain,
      nrounds = 5000,
      nfold = 5,
      early_stopping_rounds = 10,
      verbose = 0
    )
    
    # Store result
    grid[i, 1] <- cvm$best_iteration
    grid[i, 2] <- cvm$evaluation_log[[4]][cvm$best_iteration]
    setTxtProgressBar(pb, i)
    
    # Save grid to survive hard crashs
    saveRDS(grid, file = "r/gridsearch/claims_xgb.rds")
  }
}

# Load grid and select best iteration
grid <- readRDS("r/gridsearch/claims_xgb.rds")
grid <- grid[order(grid$score), ]

# Fit final, tuned model
fit <- xgb.train(
  params = as.list(grid[1, -(1:2)]), 
  data = dtrain, 
  nrounds = grid[1, "iteration"]
)

# Interpretation
library(MetricsWeighted)
library(flashlight)

# Performance on test data
pred <- predict(fit, prep_xgb(dataCar[ix$test, ]))
deviance_bernoulli(dataCar$clm[ix$test], pred)

# Relative performance
r_squared_bernoulli(dataCar$clm[ix$test], pred) # 0.00427
# Relative performance gain is very low, but very slightly better than 
# with the tuned random forest.

# Variable importance regarding loss improvement
imp <- xgb.importance(model = fit)
xgb.plot.importance(imp)

# Partial dependence plots
fl <- flashlight(model = fit, 
                 y = "clm", 
                 data = dataCar[ix$train, ], 
                 label = "xgb", 
                 predict_function = function(m, X) predict(m, prep_xgb(X)))

plot(light_profile(fl, v = "veh_value", breaks = seq(0, 5, by = 0.1))) %>% 
  labs(title = "Partial dependence plot for veh_value", y = "price")

plot(light_profile(fl, v = "veh_body"), rotate_x = TRUE) +
  labs(title = "Partial dependence plot for veh_body", y = "price")

plot(light_profile(fl, v = "area")) +
  labs(title = "Partial dependence plot for area", y = "price")

plot(light_profile(fl, v = "agecat")) +
  labs(title = "Partial dependence plot for agecat", y = "price")

#=============================================================================
# Chapter 4) Neural Nets
#=============================================================================

#=============================================================================
# Exercise 1
#=============================================================================

# Fit diamond prices by gamma deviance loss with log-link (i.e. exponential 
# output activation), using the custom loss function defined below. Tune the 
# model by simple validation and evaluate it on an independent test data set. 
# Interpret the final model. (Hints: I used a smaller learning rate 
# and had to replace the "relu" activations by "tanh". Furthermore, the 
# response needed to be transformed from int to float32)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE

from plotnine.data import diamonds

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Cast price as float
diamonds["price"] = diamonds["price"].astype("float32")

# Ordinal encoder for cut, color, and clarity
ord_features = ["cut", "color", "clarity"]
ord_levels = [
    diamonds[x].cat.categories.to_list() for x in ord_features
]
ord_encoder = OrdinalEncoder(categories=ord_levels)

# Data preprocessing pipeline
preprocessor = make_pipeline(
    ColumnTransformer(
        transformers=[
            ("ordinal", ord_encoder, ord_features),
            ("numeric", "passthrough", ["carat"])
        ]
    ),
    StandardScaler()
)

# Train/valid split
df_train, df_valid, y_train, y_valid = train_test_split(
    diamonds, 
    diamonds["price"], 
    test_size=0.2, 
    random_state=341
)

X_train = preprocessor.fit_transform(df_train)
X_valid = preprocessor.transform(df_valid)

# Input layer: we have 4 covariates
inputs = keras.Input(shape=(4,))

# One hidden layer with 5 nodes
x = layers.Dense(30, activation="tanh")(inputs)
x = layers.Dense(15, activation="tanh")(x)

# Output layer now connected to the last hidden layer
outputs = layers.Dense(1, activation=K.exp)(x)

# Create model
model = keras.Model(inputs=inputs, outputs=outputs)
# model.summary()

# Define Gamma loss
import keras.backend as K
def loss_gamma(y_true, y_pred):
  return -K.log(y_true / y_pred) + y_true / y_pred

# Compile model
model.compile(
    loss=loss_gamma,
    optimizer=keras.optimizers.Adam(learning_rate=0.001)
)

# Callbacks
cb = [
    keras.callbacks.EarlyStopping(patience=20),
    keras.callbacks.ReduceLROnPlateau(patience=5)
]

# Fit model
tf.random.set_seed(88)
history = model.fit(
    x=X_train,
    y=y_train,
    epochs=1000,
    batch_size=400, 
    validation_data=(X_valid, y_valid),
    callbacks=cb,
    verbose=1
)     

# Training RMSE over epochs
import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"], color="orange")
plt.title("Training Gamma loss over epochs")
plt.legend(["Training", "Validation"])
plt.ylim((0, 3))
plt.ylabel("Gamma loss")
plt.xlabel("Epoch")
plt.grid()
plt.show()

# Interpretation
from sklearn.metrics import mean_gamma_deviance
from sklearn.dummy import DummyRegressor
import dalex as dx

# Performance
dummy = DummyRegressor().fit(X_train, y_train)
dev_null = mean_gamma_deviance(y_valid, dummy.predict(X_valid))
dev = mean_gamma_deviance(y_valid, model.predict(X_valid))
deviance_explained = (dev_null - dev) / dev_null
print(f"% deviance explained: {deviance_explained:.2%}")

# Set up explainer
def pred_fun(m, X):
    return m.predict(preprocessor.transform(X), batch_size=1000).flatten()

exp = dx.Explainer(
    model, 
    data=df_valid[ord_features + ["carat"]], 
    y=y_valid, 
    predict_function=pred_fun, 
    verbose=False
)

# Permutation importance
vi = exp.model_parts()
vi.plot()

# Partial dependence
pdp_num = exp.model_profile(
    type="partial",
    label="Partial depencence for numeric variables",
    variables=["carat"],
    verbose=False
)
pdp_num.plot()

pdp_ord = exp.model_profile(
    type="partial",
    label="Partial depencence for ordinal variables",
    variables=ord_features,
    variable_type="categorical",
    variable_splits=dict(zip(ord_features, ord_levels)),
    verbose=False
)
pdp_ord.plot()

#=============================================================================
# Exercise 2
#=============================================================================

# Study either the optional claims data example or build your own neural net, 
# predicting claim yes/no. For simplicity, you can represent the categorical 
# feature `veh_body` by integers.

# -> see lecture notes for a solution with embeddings