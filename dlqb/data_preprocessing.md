---
layout: page
title: Data Preprocessing
---

### Use Pandas to load data from `data_file = os.path.join('..', 'data', 'house_tiny.csv')`.
```
import pandas as pd

data = pd.read_csv(data_file)
```

### For numerical values in `inputs` that are missing, replace the “NaN” entries with the mean value of the same column
`inputs = inputs.fillna(inputs.mean())`

### What does `inputs = pd.get_dummies(inputs, dummy_na=True)` do?
For categorical or discrete values in inputs, we consider “NaN” as a category. If the “Alley” column only takes two types of categorical values “Pave” and “NaN”, pandas can automatically convert this column to two columns “Alley_Pave” and “Alley_nan”. A row whose alley type is “Pave” will set values of “Alley_Pave” and “Alley_nan” to 1 and 0. A row with a missing alley type will set their values to 0 and 1.

### Convert `inputs` dataframes into the tensor format
`torch.tensor(inputs.values)`