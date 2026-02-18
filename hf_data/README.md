---
language:
  - en
  - zh
license: apache-2.0
task_categories:
  - time-series-forecasting
  - other
tags:
  - finance
  - quantitative
  - qlib
  - factor
  - time-series
pretty_name: LLMStrat Qlib CSI300 Dataset
---

# LLMStrat Qlib CSI300 Dataset

**Usage reference:** 

[![GitHub](https://img.shields.io/badge/GitHub-LLMStrat-181717?logo=github)](https://github.com/Aroesler1/LLMStrat)

Qlib market data and pre-computed HDF5 files for LLMStrat factor mining (A-share, CSI 300).

## Dataset description

| Filename       | Description                                      |
| -------------- | ------------------------------------------------- |
| daily_pv.h5    | Adjusted daily price and volume data.            |
| daily_pv_debug.h5 | Debug subset (smaller) of price-volume data. |

## How to load from Hugging Face

```python
from huggingface_hub import hf_hub_download
import pandas as pd

# Download a file from this dataset
path = hf_hub_download(
    # Default public dataset source; replace with your own mirror if needed.
    repo_id="LLMStrat/qlib_csi300",
    filename="daily_pv.h5",
    repo_type="dataset"
)
df = pd.read_hdf(path, key="data")
```

**Note:** The key is always `"data"` for all HDF5 files in this dataset.

## How to read the files locally

If you have already downloaded the files:

```python
import pandas as pd
df = pd.read_hdf("daily_pv.h5", key="data")
```

## Field description (daily price and volume)

| Field    | Description                          |
| -------- | ------------------------------------ |
| open     | Open price of the stock on that day  |
| close    | Close price of the stock on that day |
| high     | High price of the stock on that day  |
| low      | Low price of the stock on that day   |
| volume   | Trading volume on that day           |
| factor   | Adjusted factor value                |
