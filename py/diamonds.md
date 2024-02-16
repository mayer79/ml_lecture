# Diamonds Data

The dataset comes with {seaborn}, but requires a tolerant firewall to fetch it from
the internet. Thus, we store it as Parquet file.

## R code to store the data

```py
import pandas as pd
import seaborn as sns

diamonds = sns.load_dataset("diamonds")
diamonds.to_parquet("diamonds.parquet", index=False)
```
