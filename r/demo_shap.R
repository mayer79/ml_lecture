#==========================================================================
# Demo of SHAP - to be run after fitting the diamonds data with XGBoost,
# i.e., the last example in the chapter on trees.
#==========================================================================

library(SHAPforxgboost)
library(MetricsWeighted)
library(tidyverse)
library(withr)

# Crunch SHAP decomposition for 2000 observations
with_seed(8345,
  X_small <- X_train[sample(1:nrow(X_train), 2000, ), ]  
)
shap <- shap.prep(fit, X_train = X_small)

# SHAP importance plot
shap.plot.summary(shap)

# Or simply as values / barplot
shap.importance(shap)

# Dependence plots
shap.plot.dependence(
  shap, x = "carat", color_feature = "auto", 
  alpha = 0.4, smooth = FALSE
) + coord_cartesian(xlim = c(0, 2.5))

shap.plot.dependence(
  shap, x = "clarity", color_feature = "auto", alpha = 0.4, 
  jitter_width = 0.05, smooth = FALSE
)

shap.plot.dependence(
  shap, x = "color", color_feature = "auto", alpha = 0.4, 
  jitter_width = 0.05, smooth = FALSE
)

shap.plot.dependence(
  shap, x = "cut", color_feature = "auto", alpha = 0.4, 
  jitter_width = 0.05, smooth = FALSE
)

# Improve linear regression by insights gained from XAI methods
fit_lm <- lm(
  price ~ poly(log(carat), 3) * (clarity + cut + color), 
  data = diamonds[ix$train, ]
)
summary(fit_lm)

# Out-of-sample result: 0.966
r_squared(diamonds$price[ix$test], predict(fit_lm, diamonds[ix$test, ]))
