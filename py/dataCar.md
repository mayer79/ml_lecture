# Australian Car Claims Data

The dataset is part of the {insuranceData} R package publicly available on CRAN.

It contains 67k rows and ten columns:

- veh_value: vehicle value, in $10,000s
- exposure: 0-1
- clm: occurrence of claim (0 = no, 1 = yes)
- numclaims: number of claims
- claimcst0: claim amount (0 if no claim)
- veh_body: vehicle body, coded as BUS CONVT COUPE HBACK HDTOP MCARA MIBUS PANVN RDSTR SEDAN STNWG TRUCK UTE
- veh_age: 1 (youngest), 2, 3, 4
- gender: a factor with levels F M
- area: a factor with levels A B C D E F
- agecat: 1 (youngest), 2, 3, 4, 5, 6

Source: http://www.acst.mq.edu.au/GLMsforInsuranceData

Reference: De Jong P., Heller G.Z. (2008), Generalized linear models for insurance data, Cambridge University Press

## R code to store the data

```r
library(insuranceData)
library(arrow)

data("dataCar")
dataCar |> head()
write_parquet(dataCar[-ncol(dataCar)], sink = "dataCar.parquet")
```
