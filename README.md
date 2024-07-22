---
jupyter:
  kernelspec:
    display_name: Python (learn-env)
    language: python
    name: learn-env
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.8.5
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .markdown}
# **Final Project Submission**

**Students names:**

-   BRIAN KARIUKI
-   ISAAC WANG\'ANG\'A
-   FAITH MUTISYA
-   PAMELA CHEBII
-   JENIFFER GATHARIA

Student pace: part time

Scheduled project review date/time: 19/07/2024

Instructor name: SAMUEL G MWANGI, SAMUEL KARU AMD WINNIE ANYOSO
:::

::: {.cell .markdown}
# **House Sale Analysis & Regression Model For King County** {#house-sale-analysis--regression-model-for-king-county}
:::

::: {.cell .markdown}
## Introduction
:::

::: {.cell .markdown}
In the vibrant housing market of King County, understanding the factors
that influence home prices is essential for real estate professionals,
homeowners, and potential buyers. As the region continues to grow and
evolve, stakeholders must analyze the dynamics of the housing landscape
to make informed decisions.

This project aims to analyze the King County House Sales dataset to
uncover key determinants of housing prices. For this project, we will
use both linear and multi-linear regression modeling to analyze house
sales in King County, a northwestern county in Washington State. The
goal is to understand factors affecting house prices and provide
actionable insights for a real estate agency.
:::

::: {.cell .markdown}
## Problem Statement
:::

::: {.cell .markdown}
The stakeholder is a real estate agency that helps homeowners buy and
sell homes. The task at hand is to provide advice on how different
features of a house, including potential renovations, can affect its
estimated value. The goal is to determine by how much certain
renovations or features might increase the estimated value of homes.
:::

::: {.cell .markdown}
### Objectives:
:::

::: {.cell .markdown}
This project aims to address the following key objectives:

1.  **Analyze the Impact of Square Footage on Housing Prices:** Use
    simple linear regression to evaluate how sqft_living and sqft_above
    individually impact house prices in King County. This will quantify
    the relationship between square footage and property values,
    providing clear insights for homeowners and real estate investors.

2.  **Identify Key Determinants of Housing Prices:** Utilize multiple
    linear regression modeling to analyze the King County House Sales
    dataset and determine the primary factors that influence housing
    prices.

3.  **Develop Predictive Model for House Pricing:** Create and refine a
    predictive model using multiple linear regression to accurately
    estimate house prices based on the identified key features. This
    model will assist real estate professionals and homeowners in making
    informed pricing and investment decisions, enhancing the efficiency
    and effectiveness of the home buying and selling process.
:::

::: {.cell .markdown}
## Data Understanding
:::

::: {.cell .markdown}
### Import Libraries
:::

::: {.cell .code execution_count="1"}
``` python
#imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
```
:::

::: {.cell .markdown}
### Data Loading
:::

::: {.cell .code execution_count="2"}
``` python
df = pd.read_csv("./data/kc_house_data.csv")
df.head()
```

::: {.output .execute_result execution_count="2"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="3"}
``` python
# Inspecting the column names of the data
df.columns
```

::: {.output .execute_result execution_count="3"}
    Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
           'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
           'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
           'lat', 'long', 'sqft_living15', 'sqft_lot15'],
          dtype='object')
:::
:::

::: {.cell .code execution_count="4"}
``` python
df.shape
```

::: {.output .execute_result execution_count="4"}
    (21597, 21)
:::
:::

::: {.cell .markdown}
-   **Number of Rows (Observations):** There are 21597 rows in the
    DataFrame.
-   **Number of Columns (Variables):** There are 21 columns (or
    variables) in the DataFrame.
:::

::: {.cell .markdown}
### King County Housing Dataset Column Descriptions

1.  id - Unique identifier for a house
2.  date - Date when the house was sold
3.  price - Sale price of the house
4.  bedrooms - Number of bedrooms in the house
5.  bathrooms - Number of bathrooms in the house
6.  sqft_living - Square footage of the interior living space
7.  sqft_lot - Square footage of the lot
8.  floors - Number of floors (levels) in the house
9.  waterfront - Indicates whether the house has a view of the
    waterfront
10. view - Number of times the house has been viewed
11. condition - Overall condition of the house
12. grade - Overall grade of the house based on the King County grading
    system
13. sqft_above - Square footage of the house apart from the basement
14. sqft_basement - Square footage of the basement
15. yr_built - Year the house was built
16. yr_renovated - Year the house was renovated
17. zipcode - ZIP code of the house\'s location
18. lat - Latitude coordinate of the house
19. long - Longitude coordinate of the house
20. sqft_living15 - Square footage of interior living space for the
    nearest 15 neighbors
21. sqft_lot15 - Square footage of the land lots of the nearest 15
    neighbors
:::

::: {.cell .markdown}
## EDA And Data Cleaning
:::

::: {.cell .markdown}
Getting information about our data
:::

::: {.cell .code execution_count="5"}
``` python
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             21597 non-null  int64  
     1   date           21597 non-null  object 
     2   price          21597 non-null  float64
     3   bedrooms       21597 non-null  int64  
     4   bathrooms      21597 non-null  float64
     5   sqft_living    21597 non-null  int64  
     6   sqft_lot       21597 non-null  int64  
     7   floors         21597 non-null  float64
     8   waterfront     19221 non-null  float64
     9   view           21534 non-null  float64
     10  condition      21597 non-null  int64  
     11  grade          21597 non-null  int64  
     12  sqft_above     21597 non-null  int64  
     13  sqft_basement  21597 non-null  object 
     14  yr_built       21597 non-null  int64  
     15  yr_renovated   17755 non-null  float64
     16  zipcode        21597 non-null  int64  
     17  lat            21597 non-null  float64
     18  long           21597 non-null  float64
     19  sqft_living15  21597 non-null  int64  
     20  sqft_lot15     21597 non-null  int64  
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.5+ MB
:::
:::

::: {.cell .markdown}
Data description for numerical columns
:::

::: {.cell .code execution_count="6"}
``` python
df.describe()
```

::: {.output .execute_result execution_count="6"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.159700e+04</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>19221.000000</td>
      <td>21534.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>17755.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.580474e+09</td>
      <td>5.402966e+05</td>
      <td>3.373200</td>
      <td>2.115826</td>
      <td>2080.321850</td>
      <td>1.509941e+04</td>
      <td>1.494096</td>
      <td>0.007596</td>
      <td>0.233863</td>
      <td>3.409825</td>
      <td>7.657915</td>
      <td>1788.596842</td>
      <td>1970.999676</td>
      <td>83.636778</td>
      <td>98077.951845</td>
      <td>47.560093</td>
      <td>-122.213982</td>
      <td>1986.620318</td>
      <td>12758.283512</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.876736e+09</td>
      <td>3.673681e+05</td>
      <td>0.926299</td>
      <td>0.768984</td>
      <td>918.106125</td>
      <td>4.141264e+04</td>
      <td>0.539683</td>
      <td>0.086825</td>
      <td>0.765686</td>
      <td>0.650546</td>
      <td>1.173200</td>
      <td>827.759761</td>
      <td>29.375234</td>
      <td>399.946414</td>
      <td>53.513072</td>
      <td>0.138552</td>
      <td>0.140724</td>
      <td>685.230472</td>
      <td>27274.441950</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000102e+06</td>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>370.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.123049e+09</td>
      <td>3.220000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1190.000000</td>
      <td>1951.000000</td>
      <td>0.000000</td>
      <td>98033.000000</td>
      <td>47.471100</td>
      <td>-122.328000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.904930e+09</td>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.618000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1560.000000</td>
      <td>1975.000000</td>
      <td>0.000000</td>
      <td>98065.000000</td>
      <td>47.571800</td>
      <td>-122.231000</td>
      <td>1840.000000</td>
      <td>7620.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.308900e+09</td>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.068500e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>2210.000000</td>
      <td>1997.000000</td>
      <td>0.000000</td>
      <td>98118.000000</td>
      <td>47.678000</td>
      <td>-122.125000</td>
      <td>2360.000000</td>
      <td>10083.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.900000e+09</td>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>9410.000000</td>
      <td>2015.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
Checking for null variables
:::

::: {.cell .code execution_count="7"}
``` python
df.isna().sum()
```

::: {.output .execute_result execution_count="7"}
    id                  0
    date                0
    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront       2376
    view               63
    condition           0
    grade               0
    sqft_above          0
    sqft_basement       0
    yr_built            0
    yr_renovated     3842
    zipcode             0
    lat                 0
    long                0
    sqft_living15       0
    sqft_lot15          0
    dtype: int64
:::
:::

::: {.cell .markdown}
:::

::: {.cell .markdown}
-   From the Output there are 2376, 63 and 3842 null values in
    thewaterfront, view and yr_renovated columns respectively.
-   The rest of the columns do not have missing values
:::

::: {.cell .code execution_count="8"}
``` python
# fill null values with mean
df = df.fillna(df.mean())
```
:::

::: {.cell .markdown}
Recheck for null values
:::

::: {.cell .code execution_count="9"}
``` python
df.isna().sum()
```

::: {.output .execute_result execution_count="9"}
    id               0
    date             0
    price            0
    bedrooms         0
    bathrooms        0
    sqft_living      0
    sqft_lot         0
    floors           0
    waterfront       0
    view             0
    condition        0
    grade            0
    sqft_above       0
    sqft_basement    0
    yr_built         0
    yr_renovated     0
    zipcode          0
    lat              0
    long             0
    sqft_living15    0
    sqft_lot15       0
    dtype: int64
:::
:::

::: {.cell .markdown}
Checking for Duplicates
:::

::: {.cell .code execution_count="10"}
``` python
df.duplicated().sum()
```

::: {.output .execute_result execution_count="10"}
    0
:::
:::

::: {.cell .markdown}
Create a histogram for all the variables to learn the relationship
:::

::: {.cell .code execution_count="11"}
``` python
df.hist(figsize= (24,21), bins="auto");
```

::: {.output .display_data}
![](vertopal_6a9fab4a037f4822bd8ec2f92e9807a0/59753f0cc283070d6bde7be534210284f1e36178.png)
:::
:::

::: {.cell .markdown}
# Record observations on the data

-   Price, sqft_living, sqft_lot, sqft_above, sqft_living15 and
    sqft_lot15 are all continuous data
-   Most values are bunched towards the lower end while there are a few
    very large values
-   From the bedrooms feature it appears most houses have around 2
    bedrrooms.
-   From the bathrooms feature it appears most houses have between two
    and three bedrooms.
-   We can see that there is an increase in the number of houses built
    as time goes on.
-   Most houses sold were built in the 2000s
-   Most houses have only one floor
:::

::: {.cell .markdown}
Pairplots to check the relationships
:::

::: {.cell .code execution_count="12"}
``` python
sns.pairplot(data=df)
```

::: {.output .execute_result execution_count="12"}
    <seaborn.axisgrid.PairGrid at 0x1d8639fb370>
:::

::: {.output .display_data}
![](vertopal_6a9fab4a037f4822bd8ec2f92e9807a0/75ce9090336310c2bb5b66d126a2f56a62d52dd2.png)
:::
:::

::: {.cell .markdown}
# Correlation Heatmap
:::

::: {.cell .code execution_count="13"}
``` python
plt.figure(figsize=(20, 12))
sns.heatmap(df.corr(), cmap="coolwarm", annot=True)
```

::: {.output .execute_result execution_count="13"}
    <AxesSubplot:>
:::

::: {.output .display_data}
![](vertopal_6a9fab4a037f4822bd8ec2f92e9807a0/a51702e894e6018e663d04cb1a66c661cf5dde75.png)
:::
:::

::: {.cell .code execution_count="14"}
``` python
df.corr()['price'].sort_values(ascending=False)
```

::: {.output .execute_result execution_count="14"}
    price            1.000000
    sqft_living      0.701917
    grade            0.667951
    sqft_above       0.605368
    sqft_living15    0.585241
    bathrooms        0.525906
    view             0.393749
    bedrooms         0.308787
    lat              0.306692
    waterfront       0.264116
    floors           0.256804
    yr_renovated     0.118125
    sqft_lot         0.089876
    sqft_lot15       0.082845
    yr_built         0.053953
    condition        0.036056
    long             0.022036
    id              -0.016772
    zipcode         -0.053402
    Name: price, dtype: float64
:::
:::

::: {.cell .markdown}
After analyzing the correlation heatmap, it is evident that the most
important features influencing house prices are `bathrooms`,
`sqft_living`, `grade`, `sqft_above`, and `sqft_lot15`. This conclusion
is based on their strong correlation values with the `price` variable.

-   `bathrooms` (0.53): Indicates a moderate positive correlation,
    suggesting that houses with more bathrooms tend to have higher
    prices.
-   `sqft_livin`g (0.70): Shows a strong positive correlation, meaning
    that larger living areas significantly increase house prices.
-   `Grade` (0.67): Demonstrates a strong positive correlation, implying
    that higher quality and better-graded houses are priced higher.
-   `sqft_above` (0.61): Reflects a strong positive correlation,
    indicating that houses with more above-ground living space are more
    expensive.
-   `sqft_lot15` (0.59): Indicates a moderate to strong positive
    correlation, suggesting that larger lots in the vicinity (nearest 15
    neighbors) tend to increase a house\'s value.

These correlation values highlight the significant impact of these
features on house prices, guiding homeowners and real estate
professionals in their decision-making processes.
:::

::: {.cell .markdown}
## Feature Selection
:::

::: {.cell .code execution_count="15"}
``` python
df = df[[ "price", "bathrooms", "sqft_living", "grade", "sqft_above", "bedrooms"]]
```
:::

::: {.cell .markdown}
Pair Plot
:::

::: {.cell .code execution_count="16"}
``` python
sns.pairplot(data=df)
```

::: {.output .execute_result execution_count="16"}
    <seaborn.axisgrid.PairGrid at 0x1d870235f70>
:::

::: {.output .display_data}
![](vertopal_6a9fab4a037f4822bd8ec2f92e9807a0/0f601e3aaf4fd4882c930468ba3c95fd7aeef538.png)
:::
:::

::: {.cell .markdown}
:::

::: {.cell .markdown}
# Data analysis
:::

::: {.cell .markdown}
### Analysis 1
:::

::: {.cell .markdown}
**Visualizing the relationship between Square Footage of the living area
and Housing Prices:**
:::

::: {.cell .code execution_count="17"}
``` python
plt.figure(figsize=(8, 6))
sns.scatterplot(x= 'sqft_living', y= 'price', color='orange', data=df)
plt.title('Price vs Square footage of Living')
plt.xlabel('sqft_living')
plt.ylabel('Price')
plt.show()
```

::: {.output .display_data}
![](vertopal_6a9fab4a037f4822bd8ec2f92e9807a0/26f38887352420bcb2406273928c2b2d6ab15c24.png)
:::
:::

::: {.cell .markdown}
From the Scatter plot we observe that the price and square foot living
area have a strong co-orelation with a positive trend . As the Square
foot living area increases the price of the house also increases.

This implies that houses with larger square foot living areas, command
higher prices
:::

::: {.cell .markdown}
**Visualizing the relationship between housing price and grade**
:::

::: {.cell .code execution_count="18"}
``` python
plt.figure(figsize=(8, 6))
sns.scatterplot(x='grade', y='price', data=df)
plt.title('Price vs Grade')
plt.xlabel('grade')
plt.ylabel('price')
plt.show()
```

::: {.output .display_data}
![](vertopal_6a9fab4a037f4822bd8ec2f92e9807a0/51f7bf23a0d926196e4ac03371d0cf3c4264fdd9.png)
:::
:::

::: {.cell .markdown}
**Observation**

The scatter plot indicates that there is a positive correlation between
the quality grade of houses and their prices.

Higher grade houses tend to have higher prices, but there is variation
within each grade level. This trend is consistent across the entire
range of grades, showing a strong relationship between house quality and
market value.
:::

::: {.cell .markdown}
**Visualizing the relationship between Square Footage Living Area and
Grade**
:::

::: {.cell .code execution_count="19"}
``` python
plt.figure(figsize=(8, 6))
sns.scatterplot(x='grade', y='sqft_living', color= "violet", data=df)
plt.title('Square footage Living Area vs Grade')
plt.xlabel('grade')
plt.ylabel('sqft_living')
plt.show()
```

::: {.output .display_data}
![](vertopal_6a9fab4a037f4822bd8ec2f92e9807a0/25824a9deb8a535a33664d483ab83c76aaaf2969.png)
:::
:::

::: {.cell .markdown}
**Observation**

From the scatterplot above, we observe that there is a clear positive
correlation between the grade of the house and the living area.
Higher-grade houses tend to have larger living areas.

This implies that Higher grade houses can justify their higher prices
with one of the factors being having a larger living area
:::

::: {.cell .markdown}
**Visualizing the relationship between Price and bathrooms**
:::

::: {.cell .code execution_count="20"}
``` python
plt.figure(figsize=(8, 6))
sns.scatterplot(x='bathrooms', y='price', color= 'red', data=df)
plt.title('Price vs bathrooms')
plt.xlabel('bathrooms')
plt.ylabel('price')
plt.show()
```

::: {.output .display_data}
![](vertopal_6a9fab4a037f4822bd8ec2f92e9807a0/45677f487a4770cca747714cacf0b8572c13df9b.png)
:::
:::

::: {.cell .markdown}
**Observation**

From the scatterplot above, we observe that there is a clear positive
correlation between the no of bathrooms and the price.

This implies that houses with more bathrooms fetch higher prices in the
market.
:::

::: {.cell .markdown}
## Simple Linear Regression
:::

::: {.cell .markdown}
-   Selecting the feature (sqft_living) and target (price)
:::

::: {.cell .code execution_count="21"}
``` python
# Selecting the feature (sqft_living) and target (price)
X = df['sqft_living']
y = df['price']

# Adding a constant term to the predictor
X = sm.add_constant(X)
```
:::

::: {.cell .markdown}
-   Fitting the Simple Linear Regression Model
:::

::: {.cell .code execution_count="22"}
``` python
# Fitting the simple linear regression model
model = sm.OLS(y, X).fit()

# Making predictions based on the model
predictions = model.predict(X)

# Printing the model summary to evaluate performance
print(model.summary())
```

::: {.output .stream .stdout}
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.493
    Model:                            OLS   Adj. R-squared:                  0.493
    Method:                 Least Squares   F-statistic:                 2.097e+04
    Date:                Mon, 22 Jul 2024   Prob (F-statistic):               0.00
    Time:                        18:53:52   Log-Likelihood:            -3.0006e+05
    No. Observations:               21597   AIC:                         6.001e+05
    Df Residuals:                   21595   BIC:                         6.001e+05
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const       -4.399e+04   4410.023     -9.975      0.000   -5.26e+04   -3.53e+04
    sqft_living   280.8630      1.939    144.819      0.000     277.062     284.664
    ==============================================================================
    Omnibus:                    14801.942   Durbin-Watson:                   1.982
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           542662.604
    Skew:                           2.820   Prob(JB):                         0.00
    Kurtosis:                      26.901   Cond. No.                     5.63e+03
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 5.63e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
:::
:::

::: {.cell .markdown}
The R-squared value of 0.493 indicates that approximately 49.3% of the
variability in house prices can be explained by the square footage of
the living area (sqft_living). This suggests a moderate level of
explanatory power of the model.

The model coefficients show that the intercept (const) is -43,999, which
is statistically significant with a t-statistic of -9.975 and a p-value
of 0.000. The coefficient for sqft_living is 280.8630, also highly
significant with a t-statistic of 144.819 and a p-value of 0.000,
indicating a strong positive relationship between square footage and
house prices.
:::

::: {.cell .markdown}
#### Visualizing the Relationship Between Square Foot Living and Price
:::

::: {.cell .code execution_count="23"}
``` python
# Plotting the regression line along with the data points
plt.figure(figsize=(10, 6))
sns.regplot(x='sqft_living', y='price', data=df, line_kws={"color": "red"})

# Adding title and displaying the plot
plt.title('Simple Linear Regression: sqft_living vs Price')
plt.show()
```

::: {.output .display_data}
![](vertopal_6a9fab4a037f4822bd8ec2f92e9807a0/05a5ad61e08e5fa4895ae4dbcdb4553fc8317c63.png)
:::
:::

::: {.cell .markdown}
-   The above visualization shows the actual data estimates of the
    living area square footage

-   We note from the above that the price of the house increases as the
    living room square footage increases.
:::

::: {.cell .markdown}
### **Analysis 2**
:::

::: {.cell .markdown}
## Multiple linear Regression
:::

::: {.cell .markdown}
Identifying Key Determinants of Housing Prices through Multiple Linear
Regression
:::

::: {.cell .markdown}
Selecting dependent and independent variables
:::

::: {.cell .code execution_count="24"}
``` python
x = df.drop('price', axis=1)
y= df.price
```
:::

::: {.cell .markdown}
### Train Test Split at test size 20% and Train size 80%
:::

::: {.cell .markdown}
:::

::: {.cell .code execution_count="25"}
``` python
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
```
:::

::: {.cell .markdown}
check shape of both train and test data
:::

::: {.cell .code execution_count="26"}
``` python
x_train.shape, y_train.shape, x_test.shape, y_test.shape
```

::: {.output .execute_result execution_count="26"}
    ((17277, 5), (17277,), (4320, 5), (4320,))
:::
:::

::: {.cell .markdown}
### Scale Data
:::

::: {.cell .code execution_count="27"}
``` python
scaler = StandardScaler()
```
:::

::: {.cell .code execution_count="28"}
``` python
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```
:::

::: {.cell .markdown}
## Modeling
:::

::: {.cell .code execution_count="29"}
``` python
model = LinearRegression()
```
:::

::: {.cell .code execution_count="30"}
``` python
# training the model
model.fit(x_train_scaled,y_train)
```

::: {.output .execute_result execution_count="30"}
    LinearRegression()
:::
:::

::: {.cell .code execution_count="31"}
``` python
y_pred = model.predict(x_test_scaled)
```
:::

::: {.cell .code execution_count="32"}
``` python
df = pd.DataFrame({"true":y_test,"pred":y_pred})
df.head()
```

::: {.output .execute_result execution_count="32"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>true</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3686</th>
      <td>132500.0</td>
      <td>158924.670706</td>
    </tr>
    <tr>
      <th>10247</th>
      <td>415000.0</td>
      <td>317610.607159</td>
    </tr>
    <tr>
      <th>4037</th>
      <td>494000.0</td>
      <td>407643.943822</td>
    </tr>
    <tr>
      <th>3437</th>
      <td>355000.0</td>
      <td>376037.116514</td>
    </tr>
    <tr>
      <th>19291</th>
      <td>606000.0</td>
      <td>413370.613977</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
## Validation
:::

::: {.cell .markdown}
validating models perfomance
:::

::: {.cell .code execution_count="33"}
``` python
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2  = r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R Squared:" ,r2)
```

::: {.output .stream .stdout}
    Mean Squared Error: 60197315834.49932
    Mean Absolute Error: 158764.49085394148
    R Squared: 0.5377128283646696
:::
:::

::: {.cell .markdown}
-   The Multiple regression model has a lower mae and mse and an better
    r2 score making it a better model.
:::

::: {.cell .markdown}
## Conclusion
:::

::: {.cell .markdown}
### Objective 1:

1.  **Analyze the Impact of Square Footage on Housing Prices:**
    -   **Findings:** - The analysis reveals a strong positive
        correlation (`r = 0.7019`) between size of the living area
        (measured by square foot, `sqft_living`) and housing prices
        (`prices`). Larger living areas tend to command higher prices in
        the King County housing market.

    -   **Implications:** - This implies that houses with larger square
        foot living areas, command higher prices

### Objective 2:

1.  **Identify Key Determinants of Housing Prices:**
    -   **Findings:** - The analysis identified that the key
        determinants of housing prices are the square footage of living
        space (correlation: 0.7019), housing grade (correlation:
        0.6680), square footage above ground (correlation: 0.6054),
        number of bathrooms (correlation: 0.5259), and number of
        bedrooms (correlation: 0.3088).

    -   **Implications:** - These findings suggest that improvements in
        these areas, particularly increasing living space and enhancing
        the grade and quality of the house, can significantly enhance
        the value of a property. This information guides homeowners and
        real estate professionals in making strategic decisions about
        renovations and investments to maximize property value.

### Objective 3:

1.  **Develop Predictive Model for House Pricing:**

    **Simple Linear Regression**

    -   **Findings:** - The simple linear regression model,
        incorporating feature `sqft_living`, achieves an R-squared
        (`R2`) score of `0.493` on the test set. This indicates that
        49.3% of the variance in housing prices (`prices`) can be
        explained by these predictor.

    -   **Implications:** - The strong relationship between square
        footage of living areas and housing prices suggests that
        increasing living space can significantly enhance property
        value, guiding strategic investment decisions.

    **Multi-linear Regression**

    -   **Findings:** - The simple linear regression model,
        incorporating features `bathrooms`, `sqft_living`, `grade`,
        `sqft_above` and `bedrooms` , the model achieves an R-squared
        (`R2`) score of `0.5377` on the test set. This indicates that
        53.77 % of the variance in housing prices (`prices`) can be
        explained by these predictors.

-   **Implications:**- The inclusion of multiple key features provides a
    more accurate estimation of housing prices, assisting real estate
    professionals in making informed pricing and investment decisions to
    optimize property value.
:::

::: {.cell .markdown}
# Recommendations
:::

::: {.cell .markdown}
1.  **Invest in Increasing Living Space:** 1Homeowners should consider
    expanding their living areas, as the analysis shows a strong
    positive correlation between square footage and housing prices. This
    investment can significantly enhance property value.

2.  **Focus on Key Features for Renovations:** Real estate professionals
    and homeowners should prioritize improvements in key determinants
    like housing grade, number of bathrooms, and square footage above
    ground to maximize property value and attract higher prices.

3.  **Utilize Predictive Modeling for Pricing Strategies:** Implementing
    predictive models that incorporate multiple key features will help
    real estate professionals and homeowners make more informed pricing
    and investment decisions, optimizing returns and efficiency in the
    housing market.
:::
