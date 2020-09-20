library(ggplot2)
require(gridExtra)
library(grid)
n <- 100000
set.seed(1)
distr <- data.frame(
  Normal = rnorm(n),
  Binary = factor(sample(0:1, size = n, TRUE, p = c(0.7, 0.3))),
  Poisson = rpois(n, 0.1),
  Gamma = rgamma(n, 3, 3),
  Multinomial = sample(c("A", "B", "C", "D"), size = n, TRUE, p = c(0.4, 0.2, 0.3, 0.1))
)

p <- list()

for (d in names(distr)) { # d <- "Normal"
  if (d %in% c("Normal", "Gamma")) {
    geom <- function(...) geom_histogram(..., bins = 100)
  } else {
    geom <- geom_bar
  }
  p[[d]] <- ggplot(distr, aes_string(x = d)) +
    geom(fill = "orange") + 
    ylab(element_blank()) +
    theme_gray(base_size = 15) +
    theme(axis.ticks.y = element_blank(), 
          axis.text.y = element_blank())
}

main_title <- textGrob("Empirical distributions of 100'000 values", gp = gpar(fontsize = 20))

do.call(grid.arrange, c(p, list(nrow = 1, top = main_title)))
