library(ggplot2)

dia <- diamonds
dia$color <- factor(dia$color, ordered = FALSE)

head(cbind(color = dia$color, model.matrix(~ color + 0, data = dia)), 10)

