```python
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from bs4 import BeautifulSoup as bs
import warnings
import seaborn as sns
import webbrowser
import pandas as pd
from tempfile import NamedTemporaryFile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats as stats
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
def makedf(arr):
    url_base = "https://www.basketball-reference.com/leagues/NBA_"
    count_df = pd.DataFrame(columns=["Year", "Rk", "Player", "Pos", "Age", "Tm", "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
                           "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"])
    adv_df = pd.DataFrame(columns=["Year", "Rk", "Player", "Pos", "Age", "Tm", "G", "MP", "PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "AST%",
                               "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS", "WS/48", "OBPM", "DBPM", "BPM", "VORP"])
    for year in arr:
        r1 = requests.get(url_base + str(year) + "_per_game.html")
        if(r1.status_code == 429):
            print(str(year) + "| Too many requests, try again in " + r1.headers["Retry-After"] + " seconds")
            return
        r2 = requests.get(url_base + str(year) + "_advanced.html")
        count_soup = bs(r1.content)
        adv_soup = bs(r2.content)
        count_table = count_soup.find("table")
        adv_table = adv_soup.find("table")

        temp = pd.read_html(str(count_table))[0]
        count_df = pd.concat([count_df, temp], ignore_index=True)

        temp = pd.read_html(str(adv_table))[0]
        adv_df = pd.concat([adv_df, temp], ignore_index=True)

        count_df["Year"] = count_df["Year"].replace(np.NaN, year).astype(int)
        count_df["Player"] = count_df["Player"].str.replace('*', '')
        adv_df["Year"] = adv_df["Year"].replace(np.NaN, year).astype(int)
        adv_df["Player"] = adv_df["Player"].str.replace('*', '')
        time.sleep(4)
    adv_df = adv_df.drop(columns=["Rk", "Pos", "G", "Age", "MP"])
    count_df = count_df.drop(columns=["Rk"])
    df = count_df.merge(adv_df, "inner", ["Player", "Year", "Tm"])
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    return df[df['Player'] != 'Player']

def positionChart(df, col, title, ylabel):
    fig, ax = plt.subplots(5, figsize=(15, 20), layout="constrained")
    count = 0
    for pos in df['Pos'].unique():
        info = []
        for year in df['Year'].unique():
            info.append(df[(df['Year'] == year) & (df['Pos'].str.contains(pos))][col].values)
        dict = {df['Year'].unique()[i]:info[i] for i in range(44)}
        years = sorted(dict.keys())
        ax[count].boxplot(info)
        ax[count].set_title(pos)
        ax[count].set_xticklabels(years, rotation=45)
        ax[count].set_xlabel("Year")
        ax[count].set_ylabel(ylabel)
        count += 1
    fig.suptitle(title + " Over the Years", fontsize=16, horizontalalignment='center')
    fig.savefig(col + '.png')

def getTables(range, url_prefix, url_suffix, df):
    for year in range:
        req = requests.get(url_prefix + str(year) + url_suffix)
        #print(str(year) + " | " + str(req.status_code))
        temp = pd.read_html(str(bs(req.content).find("table")))[0]
        temp["Year"] = year
        if df.shape[0] == 0:
            df = temp
        else:
            df = pd.concat([df, temp], ignore_index=True)
        time.sleep(4)
    return df
```


```python
df = makedf(np.arange(1980, 2024))
```

Exploratory Data Analysis

One of the main talking points when discussing differences between eras in the NBA is the difference in scoring environments. In 2024 scoring is at an all-time high and it seems like it's been increasing every year for the last 10 years, showing no sign of slowing down. I graphed several metrics from 1980 (The year the three-point shot was added) to 2023 to show how these changes in the league and see what other trends we can see.


```python
#Retrieve league-wide NBA data from Basketball-Reference
r3 = requests.get("https://www.basketball-reference.com/leagues/NBA_stats_per_game.html")
print(r3.status_code)
league_soup = bs(r3.content)
league_table = league_soup.find("table")
league_df = pd.read_html(str(league_table))[0]
league_df = league_df[(league_df['Per Game', 'MP'] != 'Per Game') | (league_df['Per Game','MP'] != 'MP')]
league_df = league_df.drop(index=[20, 21, 42, 43])
league_df = league_df.iloc[1:45]
cpy = league_df.copy()
cpy.columns = ['_'.join(col).strip() for col in league_df.columns.values]
df_flat = cpy.reset_index()

for cols in df_flat.columns:
    try:
        df_flat[cols] = pd.to_numeric(df_flat[cols])
    except:
        continue

league_df = df_flat
league_df.drop(columns='index')
league_df.columns = [label.split('_')[-1] for label in league_df.columns.values]
league_df = league_df.drop(columns=['index', 'Rk', 'Lg', 'Age', 'Ht', 'Wt', 'G', 'MP'])
league_df['Season'] = [int(years.split('-')[0]) + 1 for years in league_df['Season'].values]

#Plot league-wide statistics
x = league_df['Season'] 
data = league_df.drop(columns=['Season'])
for stat in data.columns.values:
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.scatter(x, data[stat])
    plt.title(stat + " from 1985-2023")
    plt.xlabel("year")
    plt.ylabel(stat)
    plt.axvspan(1997, 2004, color='orange', alpha=0.3)
    plt.show()

league_df
```

    200



    
![png](GOATdebate_files/GOATdebate_4_1.png)
    



    
![png](GOATdebate_files/GOATdebate_4_2.png)
    



    
![png](GOATdebate_files/GOATdebate_4_3.png)
    



    
![png](GOATdebate_files/GOATdebate_4_4.png)
    



    
![png](GOATdebate_files/GOATdebate_4_5.png)
    



    
![png](GOATdebate_files/GOATdebate_4_6.png)
    



    
![png](GOATdebate_files/GOATdebate_4_7.png)
    



    
![png](GOATdebate_files/GOATdebate_4_8.png)
    



    
![png](GOATdebate_files/GOATdebate_4_9.png)
    



    
![png](GOATdebate_files/GOATdebate_4_10.png)
    



    
![png](GOATdebate_files/GOATdebate_4_11.png)
    



    
![png](GOATdebate_files/GOATdebate_4_12.png)
    



    
![png](GOATdebate_files/GOATdebate_4_13.png)
    



    
![png](GOATdebate_files/GOATdebate_4_14.png)
    



    
![png](GOATdebate_files/GOATdebate_4_15.png)
    



    
![png](GOATdebate_files/GOATdebate_4_16.png)
    



    
![png](GOATdebate_files/GOATdebate_4_17.png)
    



    
![png](GOATdebate_files/GOATdebate_4_18.png)
    



    
![png](GOATdebate_files/GOATdebate_4_19.png)
    



    
![png](GOATdebate_files/GOATdebate_4_20.png)
    



    
![png](GOATdebate_files/GOATdebate_4_21.png)
    



    
![png](GOATdebate_files/GOATdebate_4_22.png)
    



    
![png](GOATdebate_files/GOATdebate_4_23.png)
    



    
![png](GOATdebate_files/GOATdebate_4_24.png)
    





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
      <th>Season</th>
      <th>FG</th>
      <th>FGA</th>
      <th>3P</th>
      <th>3PA</th>
      <th>FT</th>
      <th>FTA</th>
      <th>ORB</th>
      <th>DRB</th>
      <th>TRB</th>
      <th>...</th>
      <th>PTS</th>
      <th>FG%</th>
      <th>3P%</th>
      <th>FT%</th>
      <th>Pace</th>
      <th>eFG%</th>
      <th>TOV%</th>
      <th>ORB%</th>
      <th>FT/FGA</th>
      <th>ORtg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023</td>
      <td>42.0</td>
      <td>88.3</td>
      <td>12.3</td>
      <td>34.2</td>
      <td>18.4</td>
      <td>23.5</td>
      <td>10.4</td>
      <td>33.0</td>
      <td>43.4</td>
      <td>...</td>
      <td>114.7</td>
      <td>0.475</td>
      <td>0.361</td>
      <td>0.782</td>
      <td>99.2</td>
      <td>0.545</td>
      <td>12.5</td>
      <td>24.0</td>
      <td>0.208</td>
      <td>114.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022</td>
      <td>40.6</td>
      <td>88.1</td>
      <td>12.4</td>
      <td>35.2</td>
      <td>16.9</td>
      <td>21.9</td>
      <td>10.3</td>
      <td>34.1</td>
      <td>44.5</td>
      <td>...</td>
      <td>110.6</td>
      <td>0.461</td>
      <td>0.354</td>
      <td>0.775</td>
      <td>98.2</td>
      <td>0.532</td>
      <td>12.3</td>
      <td>23.2</td>
      <td>0.192</td>
      <td>112.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021</td>
      <td>41.2</td>
      <td>88.4</td>
      <td>12.7</td>
      <td>34.6</td>
      <td>17.0</td>
      <td>21.8</td>
      <td>9.8</td>
      <td>34.5</td>
      <td>44.3</td>
      <td>...</td>
      <td>112.1</td>
      <td>0.466</td>
      <td>0.367</td>
      <td>0.778</td>
      <td>99.2</td>
      <td>0.538</td>
      <td>12.4</td>
      <td>22.2</td>
      <td>0.192</td>
      <td>112.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020</td>
      <td>40.9</td>
      <td>88.8</td>
      <td>12.2</td>
      <td>34.1</td>
      <td>17.9</td>
      <td>23.1</td>
      <td>10.1</td>
      <td>34.8</td>
      <td>44.8</td>
      <td>...</td>
      <td>111.8</td>
      <td>0.460</td>
      <td>0.358</td>
      <td>0.773</td>
      <td>100.3</td>
      <td>0.529</td>
      <td>12.8</td>
      <td>22.5</td>
      <td>0.201</td>
      <td>110.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019</td>
      <td>41.1</td>
      <td>89.2</td>
      <td>11.4</td>
      <td>32.0</td>
      <td>17.7</td>
      <td>23.1</td>
      <td>10.3</td>
      <td>34.8</td>
      <td>45.2</td>
      <td>...</td>
      <td>111.2</td>
      <td>0.461</td>
      <td>0.355</td>
      <td>0.766</td>
      <td>100.0</td>
      <td>0.524</td>
      <td>12.4</td>
      <td>22.9</td>
      <td>0.198</td>
      <td>110.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2018</td>
      <td>39.6</td>
      <td>86.1</td>
      <td>10.5</td>
      <td>29.0</td>
      <td>16.6</td>
      <td>21.7</td>
      <td>9.7</td>
      <td>33.8</td>
      <td>43.5</td>
      <td>...</td>
      <td>106.3</td>
      <td>0.460</td>
      <td>0.362</td>
      <td>0.767</td>
      <td>97.3</td>
      <td>0.521</td>
      <td>13.0</td>
      <td>22.3</td>
      <td>0.193</td>
      <td>108.6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2017</td>
      <td>39.0</td>
      <td>85.4</td>
      <td>9.7</td>
      <td>27.0</td>
      <td>17.8</td>
      <td>23.1</td>
      <td>10.1</td>
      <td>33.4</td>
      <td>43.5</td>
      <td>...</td>
      <td>105.6</td>
      <td>0.457</td>
      <td>0.358</td>
      <td>0.772</td>
      <td>96.4</td>
      <td>0.514</td>
      <td>12.7</td>
      <td>23.3</td>
      <td>0.209</td>
      <td>108.8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2016</td>
      <td>38.2</td>
      <td>84.6</td>
      <td>8.5</td>
      <td>24.1</td>
      <td>17.7</td>
      <td>23.4</td>
      <td>10.4</td>
      <td>33.3</td>
      <td>43.8</td>
      <td>...</td>
      <td>102.7</td>
      <td>0.452</td>
      <td>0.354</td>
      <td>0.757</td>
      <td>95.8</td>
      <td>0.502</td>
      <td>13.2</td>
      <td>23.8</td>
      <td>0.209</td>
      <td>106.4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2015</td>
      <td>37.5</td>
      <td>83.6</td>
      <td>7.8</td>
      <td>22.4</td>
      <td>17.1</td>
      <td>22.8</td>
      <td>10.9</td>
      <td>32.4</td>
      <td>43.3</td>
      <td>...</td>
      <td>100.0</td>
      <td>0.449</td>
      <td>0.350</td>
      <td>0.750</td>
      <td>93.9</td>
      <td>0.496</td>
      <td>13.3</td>
      <td>25.1</td>
      <td>0.205</td>
      <td>105.6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2014</td>
      <td>37.7</td>
      <td>83.0</td>
      <td>7.7</td>
      <td>21.5</td>
      <td>17.8</td>
      <td>23.6</td>
      <td>10.9</td>
      <td>31.8</td>
      <td>42.7</td>
      <td>...</td>
      <td>101.0</td>
      <td>0.454</td>
      <td>0.360</td>
      <td>0.756</td>
      <td>93.9</td>
      <td>0.501</td>
      <td>13.6</td>
      <td>25.5</td>
      <td>0.215</td>
      <td>106.6</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2013</td>
      <td>37.1</td>
      <td>82.0</td>
      <td>7.2</td>
      <td>20.0</td>
      <td>16.7</td>
      <td>22.2</td>
      <td>11.2</td>
      <td>31.0</td>
      <td>42.1</td>
      <td>...</td>
      <td>98.1</td>
      <td>0.453</td>
      <td>0.359</td>
      <td>0.753</td>
      <td>92.0</td>
      <td>0.496</td>
      <td>13.7</td>
      <td>26.5</td>
      <td>0.204</td>
      <td>105.8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2012</td>
      <td>36.5</td>
      <td>81.4</td>
      <td>6.4</td>
      <td>18.4</td>
      <td>16.9</td>
      <td>22.5</td>
      <td>11.4</td>
      <td>30.8</td>
      <td>42.2</td>
      <td>...</td>
      <td>96.3</td>
      <td>0.448</td>
      <td>0.349</td>
      <td>0.752</td>
      <td>91.3</td>
      <td>0.487</td>
      <td>13.8</td>
      <td>27.0</td>
      <td>0.208</td>
      <td>104.6</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2011</td>
      <td>37.2</td>
      <td>81.2</td>
      <td>6.5</td>
      <td>18.0</td>
      <td>18.6</td>
      <td>24.4</td>
      <td>10.9</td>
      <td>30.5</td>
      <td>41.4</td>
      <td>...</td>
      <td>99.6</td>
      <td>0.459</td>
      <td>0.358</td>
      <td>0.763</td>
      <td>92.1</td>
      <td>0.498</td>
      <td>13.4</td>
      <td>26.4</td>
      <td>0.229</td>
      <td>107.3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2010</td>
      <td>37.7</td>
      <td>81.7</td>
      <td>6.4</td>
      <td>18.1</td>
      <td>18.6</td>
      <td>24.5</td>
      <td>11.0</td>
      <td>30.8</td>
      <td>41.7</td>
      <td>...</td>
      <td>100.4</td>
      <td>0.461</td>
      <td>0.355</td>
      <td>0.759</td>
      <td>92.7</td>
      <td>0.501</td>
      <td>13.3</td>
      <td>26.3</td>
      <td>0.228</td>
      <td>107.6</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2009</td>
      <td>37.1</td>
      <td>80.9</td>
      <td>6.6</td>
      <td>18.1</td>
      <td>19.1</td>
      <td>24.7</td>
      <td>11.0</td>
      <td>30.3</td>
      <td>41.3</td>
      <td>...</td>
      <td>100.0</td>
      <td>0.459</td>
      <td>0.367</td>
      <td>0.771</td>
      <td>91.7</td>
      <td>0.500</td>
      <td>13.3</td>
      <td>26.7</td>
      <td>0.236</td>
      <td>108.3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2008</td>
      <td>37.3</td>
      <td>81.5</td>
      <td>6.6</td>
      <td>18.1</td>
      <td>18.8</td>
      <td>24.9</td>
      <td>11.2</td>
      <td>30.8</td>
      <td>42.0</td>
      <td>...</td>
      <td>99.9</td>
      <td>0.457</td>
      <td>0.362</td>
      <td>0.755</td>
      <td>92.4</td>
      <td>0.497</td>
      <td>13.2</td>
      <td>26.7</td>
      <td>0.231</td>
      <td>107.5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2007</td>
      <td>36.5</td>
      <td>79.7</td>
      <td>6.1</td>
      <td>16.9</td>
      <td>19.6</td>
      <td>26.1</td>
      <td>11.1</td>
      <td>29.9</td>
      <td>41.1</td>
      <td>...</td>
      <td>98.7</td>
      <td>0.458</td>
      <td>0.358</td>
      <td>0.752</td>
      <td>91.9</td>
      <td>0.496</td>
      <td>14.2</td>
      <td>27.1</td>
      <td>0.246</td>
      <td>106.5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2006</td>
      <td>35.8</td>
      <td>79.0</td>
      <td>5.7</td>
      <td>16.0</td>
      <td>19.6</td>
      <td>26.3</td>
      <td>11.2</td>
      <td>29.8</td>
      <td>41.0</td>
      <td>...</td>
      <td>97.0</td>
      <td>0.454</td>
      <td>0.358</td>
      <td>0.745</td>
      <td>90.5</td>
      <td>0.490</td>
      <td>13.7</td>
      <td>27.3</td>
      <td>0.248</td>
      <td>106.2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2005</td>
      <td>35.9</td>
      <td>80.3</td>
      <td>5.6</td>
      <td>15.8</td>
      <td>19.7</td>
      <td>26.1</td>
      <td>12.0</td>
      <td>29.8</td>
      <td>41.9</td>
      <td>...</td>
      <td>97.2</td>
      <td>0.447</td>
      <td>0.356</td>
      <td>0.756</td>
      <td>90.9</td>
      <td>0.482</td>
      <td>13.6</td>
      <td>28.7</td>
      <td>0.245</td>
      <td>106.1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2004</td>
      <td>35.0</td>
      <td>79.8</td>
      <td>5.2</td>
      <td>14.9</td>
      <td>18.2</td>
      <td>24.2</td>
      <td>12.1</td>
      <td>30.1</td>
      <td>42.2</td>
      <td>...</td>
      <td>93.4</td>
      <td>0.439</td>
      <td>0.347</td>
      <td>0.752</td>
      <td>90.1</td>
      <td>0.471</td>
      <td>14.2</td>
      <td>28.6</td>
      <td>0.228</td>
      <td>102.9</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2003</td>
      <td>35.7</td>
      <td>80.8</td>
      <td>5.1</td>
      <td>14.7</td>
      <td>18.5</td>
      <td>24.4</td>
      <td>12.0</td>
      <td>30.3</td>
      <td>42.3</td>
      <td>...</td>
      <td>95.1</td>
      <td>0.442</td>
      <td>0.349</td>
      <td>0.758</td>
      <td>91.0</td>
      <td>0.474</td>
      <td>14.0</td>
      <td>28.5</td>
      <td>0.229</td>
      <td>103.6</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2002</td>
      <td>36.2</td>
      <td>81.3</td>
      <td>5.2</td>
      <td>14.7</td>
      <td>17.9</td>
      <td>23.8</td>
      <td>12.2</td>
      <td>30.2</td>
      <td>42.4</td>
      <td>...</td>
      <td>95.5</td>
      <td>0.445</td>
      <td>0.354</td>
      <td>0.752</td>
      <td>90.7</td>
      <td>0.477</td>
      <td>13.6</td>
      <td>28.9</td>
      <td>0.221</td>
      <td>104.5</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2001</td>
      <td>35.7</td>
      <td>80.6</td>
      <td>4.8</td>
      <td>13.7</td>
      <td>18.6</td>
      <td>24.9</td>
      <td>12.0</td>
      <td>30.5</td>
      <td>42.5</td>
      <td>...</td>
      <td>94.8</td>
      <td>0.443</td>
      <td>0.354</td>
      <td>0.748</td>
      <td>91.3</td>
      <td>0.473</td>
      <td>14.1</td>
      <td>28.2</td>
      <td>0.231</td>
      <td>103.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2000</td>
      <td>36.8</td>
      <td>82.1</td>
      <td>4.8</td>
      <td>13.7</td>
      <td>19.0</td>
      <td>25.3</td>
      <td>12.4</td>
      <td>30.5</td>
      <td>42.9</td>
      <td>...</td>
      <td>97.5</td>
      <td>0.449</td>
      <td>0.353</td>
      <td>0.750</td>
      <td>93.1</td>
      <td>0.478</td>
      <td>14.2</td>
      <td>28.9</td>
      <td>0.231</td>
      <td>104.1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1999</td>
      <td>34.2</td>
      <td>78.2</td>
      <td>4.5</td>
      <td>13.2</td>
      <td>18.8</td>
      <td>25.8</td>
      <td>12.6</td>
      <td>29.1</td>
      <td>41.7</td>
      <td>...</td>
      <td>91.6</td>
      <td>0.437</td>
      <td>0.339</td>
      <td>0.728</td>
      <td>88.9</td>
      <td>0.466</td>
      <td>14.6</td>
      <td>30.2</td>
      <td>0.240</td>
      <td>102.2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1998</td>
      <td>35.9</td>
      <td>79.7</td>
      <td>4.4</td>
      <td>12.7</td>
      <td>19.4</td>
      <td>26.3</td>
      <td>13.1</td>
      <td>28.5</td>
      <td>41.5</td>
      <td>...</td>
      <td>95.6</td>
      <td>0.450</td>
      <td>0.346</td>
      <td>0.737</td>
      <td>90.3</td>
      <td>0.478</td>
      <td>14.5</td>
      <td>31.4</td>
      <td>0.243</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1997</td>
      <td>36.1</td>
      <td>79.3</td>
      <td>6.0</td>
      <td>16.8</td>
      <td>18.7</td>
      <td>25.3</td>
      <td>12.7</td>
      <td>28.4</td>
      <td>41.1</td>
      <td>...</td>
      <td>96.9</td>
      <td>0.455</td>
      <td>0.360</td>
      <td>0.738</td>
      <td>90.1</td>
      <td>0.493</td>
      <td>14.8</td>
      <td>30.8</td>
      <td>0.236</td>
      <td>106.7</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1996</td>
      <td>37.0</td>
      <td>80.2</td>
      <td>5.9</td>
      <td>16.0</td>
      <td>19.5</td>
      <td>26.4</td>
      <td>12.6</td>
      <td>28.6</td>
      <td>41.3</td>
      <td>...</td>
      <td>99.5</td>
      <td>0.462</td>
      <td>0.367</td>
      <td>0.740</td>
      <td>91.8</td>
      <td>0.499</td>
      <td>14.7</td>
      <td>30.6</td>
      <td>0.243</td>
      <td>107.6</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1995</td>
      <td>38.0</td>
      <td>81.5</td>
      <td>5.5</td>
      <td>15.3</td>
      <td>19.9</td>
      <td>27.1</td>
      <td>13.0</td>
      <td>28.5</td>
      <td>41.6</td>
      <td>...</td>
      <td>101.4</td>
      <td>0.466</td>
      <td>0.359</td>
      <td>0.737</td>
      <td>92.9</td>
      <td>0.500</td>
      <td>14.6</td>
      <td>31.4</td>
      <td>0.245</td>
      <td>108.3</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1994</td>
      <td>39.3</td>
      <td>84.4</td>
      <td>3.3</td>
      <td>9.9</td>
      <td>19.6</td>
      <td>26.6</td>
      <td>13.9</td>
      <td>29.1</td>
      <td>43.0</td>
      <td>...</td>
      <td>101.5</td>
      <td>0.466</td>
      <td>0.333</td>
      <td>0.734</td>
      <td>95.1</td>
      <td>0.485</td>
      <td>14.3</td>
      <td>32.2</td>
      <td>0.232</td>
      <td>106.3</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1993</td>
      <td>40.7</td>
      <td>86.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>20.9</td>
      <td>27.7</td>
      <td>13.8</td>
      <td>29.3</td>
      <td>43.1</td>
      <td>...</td>
      <td>105.3</td>
      <td>0.473</td>
      <td>0.336</td>
      <td>0.754</td>
      <td>96.8</td>
      <td>0.491</td>
      <td>14.0</td>
      <td>32.0</td>
      <td>0.243</td>
      <td>108.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1992</td>
      <td>41.3</td>
      <td>87.3</td>
      <td>2.5</td>
      <td>7.6</td>
      <td>20.2</td>
      <td>26.7</td>
      <td>14.4</td>
      <td>29.3</td>
      <td>43.7</td>
      <td>...</td>
      <td>105.3</td>
      <td>0.472</td>
      <td>0.331</td>
      <td>0.759</td>
      <td>96.6</td>
      <td>0.487</td>
      <td>13.6</td>
      <td>32.9</td>
      <td>0.232</td>
      <td>108.2</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1991</td>
      <td>41.4</td>
      <td>87.2</td>
      <td>2.3</td>
      <td>7.1</td>
      <td>21.3</td>
      <td>27.9</td>
      <td>14.0</td>
      <td>29.3</td>
      <td>43.3</td>
      <td>...</td>
      <td>106.3</td>
      <td>0.474</td>
      <td>0.320</td>
      <td>0.765</td>
      <td>97.8</td>
      <td>0.487</td>
      <td>13.9</td>
      <td>32.3</td>
      <td>0.245</td>
      <td>107.9</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1990</td>
      <td>41.5</td>
      <td>87.2</td>
      <td>2.2</td>
      <td>6.6</td>
      <td>21.8</td>
      <td>28.5</td>
      <td>13.8</td>
      <td>29.3</td>
      <td>43.1</td>
      <td>...</td>
      <td>107.0</td>
      <td>0.476</td>
      <td>0.331</td>
      <td>0.764</td>
      <td>98.3</td>
      <td>0.489</td>
      <td>13.9</td>
      <td>32.1</td>
      <td>0.250</td>
      <td>108.1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1989</td>
      <td>42.5</td>
      <td>89.0</td>
      <td>2.1</td>
      <td>6.6</td>
      <td>22.1</td>
      <td>28.8</td>
      <td>14.5</td>
      <td>29.4</td>
      <td>43.9</td>
      <td>...</td>
      <td>109.2</td>
      <td>0.477</td>
      <td>0.323</td>
      <td>0.768</td>
      <td>100.6</td>
      <td>0.489</td>
      <td>14.5</td>
      <td>33.0</td>
      <td>0.249</td>
      <td>107.8</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1988</td>
      <td>42.1</td>
      <td>87.7</td>
      <td>1.6</td>
      <td>5.0</td>
      <td>22.3</td>
      <td>29.1</td>
      <td>14.2</td>
      <td>29.2</td>
      <td>43.4</td>
      <td>...</td>
      <td>108.2</td>
      <td>0.480</td>
      <td>0.316</td>
      <td>0.766</td>
      <td>99.6</td>
      <td>0.489</td>
      <td>14.3</td>
      <td>32.8</td>
      <td>0.254</td>
      <td>108.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1987</td>
      <td>42.6</td>
      <td>88.8</td>
      <td>1.4</td>
      <td>4.7</td>
      <td>23.2</td>
      <td>30.5</td>
      <td>14.7</td>
      <td>29.3</td>
      <td>44.0</td>
      <td>...</td>
      <td>109.9</td>
      <td>0.480</td>
      <td>0.301</td>
      <td>0.763</td>
      <td>100.8</td>
      <td>0.488</td>
      <td>14.3</td>
      <td>33.4</td>
      <td>0.262</td>
      <td>108.3</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1986</td>
      <td>43.2</td>
      <td>88.6</td>
      <td>0.9</td>
      <td>3.3</td>
      <td>22.9</td>
      <td>30.3</td>
      <td>14.1</td>
      <td>29.4</td>
      <td>43.6</td>
      <td>...</td>
      <td>110.2</td>
      <td>0.487</td>
      <td>0.282</td>
      <td>0.756</td>
      <td>102.1</td>
      <td>0.493</td>
      <td>14.9</td>
      <td>32.4</td>
      <td>0.258</td>
      <td>107.2</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1985</td>
      <td>43.8</td>
      <td>89.1</td>
      <td>0.9</td>
      <td>3.1</td>
      <td>22.4</td>
      <td>29.4</td>
      <td>14.3</td>
      <td>29.2</td>
      <td>43.5</td>
      <td>...</td>
      <td>110.8</td>
      <td>0.491</td>
      <td>0.282</td>
      <td>0.764</td>
      <td>102.1</td>
      <td>0.496</td>
      <td>14.9</td>
      <td>32.9</td>
      <td>0.252</td>
      <td>107.9</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1984</td>
      <td>43.5</td>
      <td>88.4</td>
      <td>0.6</td>
      <td>2.4</td>
      <td>22.6</td>
      <td>29.7</td>
      <td>14.2</td>
      <td>28.8</td>
      <td>43.0</td>
      <td>...</td>
      <td>110.1</td>
      <td>0.492</td>
      <td>0.250</td>
      <td>0.760</td>
      <td>101.4</td>
      <td>0.495</td>
      <td>15.0</td>
      <td>33.0</td>
      <td>0.255</td>
      <td>107.6</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1983</td>
      <td>43.5</td>
      <td>89.7</td>
      <td>0.5</td>
      <td>2.3</td>
      <td>20.9</td>
      <td>28.3</td>
      <td>14.8</td>
      <td>29.6</td>
      <td>44.5</td>
      <td>...</td>
      <td>108.5</td>
      <td>0.485</td>
      <td>0.238</td>
      <td>0.740</td>
      <td>103.1</td>
      <td>0.488</td>
      <td>15.8</td>
      <td>33.4</td>
      <td>0.233</td>
      <td>104.7</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1982</td>
      <td>43.3</td>
      <td>88.2</td>
      <td>0.6</td>
      <td>2.3</td>
      <td>21.3</td>
      <td>28.6</td>
      <td>14.3</td>
      <td>29.1</td>
      <td>43.5</td>
      <td>...</td>
      <td>108.6</td>
      <td>0.491</td>
      <td>0.262</td>
      <td>0.746</td>
      <td>100.9</td>
      <td>0.495</td>
      <td>15.0</td>
      <td>33.0</td>
      <td>0.241</td>
      <td>106.9</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1981</td>
      <td>43.0</td>
      <td>88.4</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>21.7</td>
      <td>28.9</td>
      <td>14.5</td>
      <td>28.9</td>
      <td>43.5</td>
      <td>...</td>
      <td>108.1</td>
      <td>0.486</td>
      <td>0.245</td>
      <td>0.751</td>
      <td>101.8</td>
      <td>0.489</td>
      <td>15.6</td>
      <td>33.5</td>
      <td>0.245</td>
      <td>105.5</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1980</td>
      <td>43.6</td>
      <td>90.6</td>
      <td>0.8</td>
      <td>2.8</td>
      <td>21.3</td>
      <td>27.8</td>
      <td>15.1</td>
      <td>29.9</td>
      <td>44.9</td>
      <td>...</td>
      <td>109.3</td>
      <td>0.481</td>
      <td>0.280</td>
      <td>0.764</td>
      <td>103.1</td>
      <td>0.486</td>
      <td>15.5</td>
      <td>33.5</td>
      <td>0.235</td>
      <td>105.3</td>
    </tr>
  </tbody>
</table>
<p>44 rows × 25 columns</p>
</div>



There was an interesting trend that emerged for a few statistics. For FG, FGA, PTS, and Pace there is a clear nonlinear pattern in the data. Specifically, the graphs for these metrics have a parabolic shape, decreasing as time progresses starting in 1980, bottoming out somewhere in 1997-2004, and increasing from there. Interestingly, 3 significant rule changes in this period may have affected scoring during this time. The first one was in 1997 when the restricted area, an area within the paint where you can't draw charges, was instituted. This made defense more difficult and would lead to more offense in theory. In 2001, the NBA eliminated the illegal defense rule where defenders could not be more than an arm's length away from an offensive player for more than three seconds and instituted another, the defensive three seconds, which mandates a defender must not remain in the paint for more than 3 seconds. This rule change essentially legalized zone defense, making it easier for defenders to guard isolation-heavy players who, before the rule change, would essentially be given an entire half of the court to operate 1-on-1 with a defender. This may have had an unintentional consequence of increasing pace by hampering isolation players and encouraging more ball movement which tends to be quicker offense. Increasing pace, measured in team possessions per 48 minutes, would lead to more FGA which leads to more FG and more PTS. Lastly, in 2004 the NBA started enforcing handchecking where defenders were able to keep their hands on offensive players and impeded their movement. This rule change would have benefited the offensive side of the ball and made for more scoring. eFG% and ORtg show a similar trend, but there is not much of a decrease before these rule changes, there is more of a constant trend, then there is a dip during the rule change period of 1997-2004 and an increase not too long after that. These statistical trends could also be a result of broader NBA trends like the more three-point shooting, the success of Mike D'Antoni's 7-seconds-or-less offense and the teams that have tried to mimic it since, the growth of positionless basketball, some combination of everything I've mentioned, or none of them entirely. Whatever the cause may be, it's clear to see that the NBA underwent seismic changes during this time making comparisons across eras difficult, even within the same position.


```python
#change datatypes of df
df['Age'] = df['Age'].astype(int)
for cols in df.columns:
    try:
        df[cols] = pd.to_numeric(df[cols])
    except:
        continue
#Seperating the data for starters
#Starter - a player who starts more games than they come off the bench in the games that they play
starters = df[(df["G"] - (df["GS"]) <= df["GS"])& (df["GS"] >= 20)]
pts = []
temp = []
for year in starters['Year'].unique():
    temp = starters[starters['Year'] == year]['PTS'].tolist()
    pts.append(temp)
    
max_len = max(len(row) for row in pts)
d = [row + [None] * (max_len - len(row)) for row in pts]

#Retrieve play-by-play data from Basketball-Reference. Will be used to resolve player positions to the single position they play the most
year = 1997
req = requests.get(f"https://www.basketball-reference.com/leagues/NBA_{year}_play-by-play.html")

pbp_soup = bs(req.content)
pbp_table = pbp_soup.find("table")
pbp = pd.read_html(str(pbp_table))[0]
pbp["Year"] = year

for year in range(1998, 2024):
    req = requests.get(f"https://www.basketball-reference.com/leagues/NBA_{year}_play-by-play.html")
    pbp_soup = bs(req.content)
    pbp_table = pbp_soup.find("table")
    temp = pd.read_html(str(pbp_table))[0]
    temp["Year"] = year
    pbp = pd.concat([pbp, temp], ignore_index=True)
    time.sleep(4)

#Fix pbp columns
cpy = pbp.copy()
cpy.columns = ['_'.join(col).strip() for col in pbp.columns.values]
df_flat = cpy.reset_index()
df_flat = df_flat.drop(columns="index")
pbp.columns = [label.split('_')[-1] for label in df_flat.columns.values]

pbp = pbp.rename(columns={'': "Year"})
pbp = pbp[pbp["C%"] != "C%"]
#pbp = pbp.drop(columns=["Rk", "Age", "G", "MP", "OnCourt", "On-Off", "BadPass", "LostBall", "Shoot", "Off.", "Shoot", "Off.", "PGA", "And1", "Blkd"])
for cols in pbp.columns:
    try:
        pbp[cols] = pd.to_numeric(pbp[cols].str.rstrip('%'))
    except:
        continue

cols = pbp.columns.tolist()
cols = cols[-2:] + cols[:-2]
pbp = pbp[cols]
pbp = pbp.loc[:, "Year":"C%"]
pbp = pbp.drop(columns=["Rk", "MP"])

starters = starters.reset_index().drop(columns="index")
pbp = pbp.reset_index().drop(columns="index")
```


```python
final = starters.merge(pbp, "left")
#Retrieve per-possesssion statistics from Basketball-Reference for Defensive Ratings
poss = getTables(np.arange(1980, 2024), "https://www.basketball-reference.com/leagues/NBA_", "_per_poss.html", pd.DataFrame())
poss["Player"] = poss["Player"].str.replace('*', '')
poss = poss.drop(columns=poss.loc[:, "MP":"ORtg"])
poss = poss.drop(columns="Rk")
poss = poss[poss["Player"] != "Player"]
for cols in poss.columns:
    try:
        poss[cols] = pd.to_numeric(poss[cols])
    except:
        continue
final = final.merge(poss, 'left')
#Set each players position to be the single most column they play most according to the play-by-play data, if they do not have play-by-play data
#reset the players position to what it was before
final["Pos"] = final.loc[:, "PG%":"C%"].idxmax(axis="columns")
final["Pos"] = final["Pos"].fillna(starters["Pos"])
final["Pos"] = final["Pos"].str.rstrip("%")
final["Pos"] = [pos.split("-")[0] for pos in final["Pos"].values]
starters = final
```

Cross-position comparisons are even more difficult. As mentioned above, the NBA has trended towards positionless basketball where the positional responsibilities typically attributed to certain positions are being loosened. Point guards once thought to be past first and focused on assists now are among the most prolific scorers in the league. Centers once were tethered to the low block, responsible for patrolling the paint on defense, and seldomly took any three-point jump shots. In today's NBA, big men who can't shoot the three-pointer are often played off the floor, especially if you can't keep up with perimeter players on defense. This means that the average player at each position looks different during these eras. The box and whisker plots above display these changes.

In order to better compare players across positions and eras, I will normalize the data with respect to the position and year for 8 statistical categories: Points, Assists, Rebounds, Blocks, Steals, Turnovers, True Shooting Percentage, and Defensive Rating. I chose these 8 stats because I thought they were a good mix of stats that would encapsulate a player's performance on both sides of the ball and also take into account efficiency. There is also little overlap with these stats, with the exception of Defensive Rating where Steals and Blocks are used in the calculation. After I selected the stats I'd use in my comparison found the z-score for all 8 of these stats. The Z-score is calculated by subtracting the sample mean from the observed value and dividing by the standard deviation. The resulting value represents how many standard deviations the observed value is away from the mean. This can be interpreted as how far a player is for average for that particular stat. For defensive rating, a low value is better, but when the Z-score would be negative for good defenders. To get around this I negated the z-score so that good defenders get positive z-scores and bad defenders get negative z-scores like all the other stats. The same holds true for turnovers. After calculating the z-score for all 8 stats, I took the mean to get the average z-score. This is the metric I will use for player comparison.


```python
positionChart(starters, "PTS", "Scoring", "Points Per Game")
positionChart(starters, "TRB", "Rebounding", "Rebounds Per Game")
positionChart(starters, "AST", "Assists", "Assists Per Game")
positionChart(starters, "STL", "Steals", "Steals Per Game")
positionChart(starters, "BLK", "Blocks", "Blocks Per Game")
positionChart(starters, "TOV", "Turnovers", "Turnovers Per Game")
positionChart(starters, "TS%", "True Shooting Percentage", "TS%")
positionChart(starters, "DWS", "Defensive Win Shares", "Wins Added")
positionChart(starters, "DRtg", "Defensive Rating", "Points")
```


    
![png](GOATdebate_files/GOATdebate_9_0.png)
    



    
![png](GOATdebate_files/GOATdebate_9_1.png)
    



    
![png](GOATdebate_files/GOATdebate_9_2.png)
    



    
![png](GOATdebate_files/GOATdebate_9_3.png)
    



    
![png](GOATdebate_files/GOATdebate_9_4.png)
    



    
![png](GOATdebate_files/GOATdebate_9_5.png)
    



    
![png](GOATdebate_files/GOATdebate_9_6.png)
    



    
![png](GOATdebate_files/GOATdebate_9_7.png)
    



    
![png](GOATdebate_files/GOATdebate_9_8.png)
    



```python
starters = starters.drop(columns=(starters.loc[:, "FG%":"DRB"] + starters.loc[:, "GS":"FGA"] + starters.loc[:, "3P":"3PA"] + starters.loc[:, "3PAr":"TOV%"] + starters.loc[:, "WS":"C%"] + starters.loc[:, "2P":"2P%"]))
starters = starters.drop(columns=["OWS", "TOV", "PF", "PER", "USG%", "DWS"])
```


```python
starters["PTSZ"] = starters.groupby(["Pos", "Year"])["PTS"].transform(stats.zscore)
starters["ASTZ"] = starters.groupby(["Pos", "Year"])["AST"].transform(stats.zscore)
starters["BLKZ"] = starters.groupby(["Pos", "Year"])["BLK"].transform(stats.zscore)
starters["STLZ"] = starters.groupby(["Pos", "Year"])["STL"].transform(stats.zscore)
#starters["TOV_ZSCORE"] = starters.groupby(["Pos", "Year"])["TOV"].transform(stats.zscore) * -1
starters["TRBZ"] = starters.groupby(["Pos", "Year"])["TRB"].transform(stats.zscore)
starters["TS%Z"] = starters.groupby(["Pos", "Year"])["TS%"].transform(stats.zscore)
#Defensive ratings are a measure of how many points a player allows per 100 possesions. High DRtg is bad
starters["DRtgZ"] = starters.groupby(["Pos", "Year"])["DRtg"].transform(stats.zscore) * -1
```


```python
starters
```




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
      <th>Year</th>
      <th>Player</th>
      <th>Pos</th>
      <th>Age</th>
      <th>Tm</th>
      <th>G</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>PTS</th>
      <th>TS%</th>
      <th>DRtg</th>
      <th>PTSZ</th>
      <th>ASTZ</th>
      <th>BLKZ</th>
      <th>STLZ</th>
      <th>TRBZ</th>
      <th>TS%Z</th>
      <th>DRtgZ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>Tiny Archibald</td>
      <td>PG</td>
      <td>31</td>
      <td>BOS</td>
      <td>80</td>
      <td>2.5</td>
      <td>8.4</td>
      <td>1.3</td>
      <td>0.1</td>
      <td>14.1</td>
      <td>0.574</td>
      <td>105.0</td>
      <td>-0.630432</td>
      <td>1.577593</td>
      <td>-1.093216</td>
      <td>-1.341065</td>
      <td>-0.844583</td>
      <td>1.189685</td>
      <td>-0.035760</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Larry Bird</td>
      <td>PF</td>
      <td>23</td>
      <td>BOS</td>
      <td>82</td>
      <td>10.4</td>
      <td>4.5</td>
      <td>1.7</td>
      <td>0.6</td>
      <td>21.3</td>
      <td>0.538</td>
      <td>98.0</td>
      <td>1.111960</td>
      <td>1.610954</td>
      <td>-0.488813</td>
      <td>1.732051</td>
      <td>0.950654</td>
      <td>0.389407</td>
      <td>1.419048</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Ron Brewer</td>
      <td>SG</td>
      <td>24</td>
      <td>POR</td>
      <td>82</td>
      <td>2.6</td>
      <td>2.6</td>
      <td>1.2</td>
      <td>0.6</td>
      <td>15.7</td>
      <td>0.503</td>
      <td>107.0</td>
      <td>0.226937</td>
      <td>-0.844652</td>
      <td>1.988893</td>
      <td>-0.550000</td>
      <td>-0.377608</td>
      <td>-0.210030</td>
      <td>-0.811107</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1980</td>
      <td>Bill Cartwright</td>
      <td>C</td>
      <td>22</td>
      <td>NYK</td>
      <td>82</td>
      <td>8.9</td>
      <td>2.0</td>
      <td>0.6</td>
      <td>1.2</td>
      <td>21.7</td>
      <td>0.608</td>
      <td>108.0</td>
      <td>1.011212</td>
      <td>-0.681442</td>
      <td>0.127057</td>
      <td>-0.957427</td>
      <td>-0.267118</td>
      <td>1.287452</td>
      <td>-1.403512</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1980</td>
      <td>Maurice Cheeks</td>
      <td>PG</td>
      <td>23</td>
      <td>PHI</td>
      <td>79</td>
      <td>3.5</td>
      <td>7.0</td>
      <td>2.3</td>
      <td>0.4</td>
      <td>11.4</td>
      <td>0.589</td>
      <td>101.0</td>
      <td>-1.445300</td>
      <td>0.380143</td>
      <td>1.015129</td>
      <td>0.853405</td>
      <td>0.443764</td>
      <td>1.653868</td>
      <td>1.251597</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7135</th>
      <td>2023</td>
      <td>Robert Williams</td>
      <td>C</td>
      <td>25</td>
      <td>BOS</td>
      <td>35</td>
      <td>8.3</td>
      <td>1.4</td>
      <td>0.6</td>
      <td>1.4</td>
      <td>8.0</td>
      <td>0.742</td>
      <td>107.0</td>
      <td>-1.045001</td>
      <td>-0.555064</td>
      <td>0.363521</td>
      <td>-0.413175</td>
      <td>-0.203454</td>
      <td>1.991505</td>
      <td>1.585472</td>
    </tr>
    <tr>
      <th>7136</th>
      <td>2023</td>
      <td>Zion Williamson</td>
      <td>PF</td>
      <td>22</td>
      <td>NOP</td>
      <td>29</td>
      <td>7.0</td>
      <td>4.6</td>
      <td>1.1</td>
      <td>0.6</td>
      <td>26.0</td>
      <td>0.652</td>
      <td>112.0</td>
      <td>1.310210</td>
      <td>0.853026</td>
      <td>-0.007336</td>
      <td>0.833476</td>
      <td>0.446832</td>
      <td>1.542391</td>
      <td>0.897235</td>
    </tr>
    <tr>
      <th>7137</th>
      <td>2023</td>
      <td>James Wiseman</td>
      <td>C</td>
      <td>21</td>
      <td>DET</td>
      <td>24</td>
      <td>8.1</td>
      <td>0.7</td>
      <td>0.2</td>
      <td>0.8</td>
      <td>12.7</td>
      <td>0.563</td>
      <td>118.0</td>
      <td>-0.211837</td>
      <td>-0.956661</td>
      <td>-0.554847</td>
      <td>-1.950571</td>
      <td>-0.300917</td>
      <td>-1.481261</td>
      <td>-1.937799</td>
    </tr>
    <tr>
      <th>7138</th>
      <td>2023</td>
      <td>Trae Young</td>
      <td>PG</td>
      <td>24</td>
      <td>ATL</td>
      <td>73</td>
      <td>3.0</td>
      <td>10.2</td>
      <td>1.1</td>
      <td>0.1</td>
      <td>26.2</td>
      <td>0.573</td>
      <td>119.0</td>
      <td>1.120768</td>
      <td>1.965644</td>
      <td>-1.289317</td>
      <td>-0.127685</td>
      <td>-0.748418</td>
      <td>0.051627</td>
      <td>-1.301938</td>
    </tr>
    <tr>
      <th>7139</th>
      <td>2023</td>
      <td>Ivica Zubac</td>
      <td>C</td>
      <td>25</td>
      <td>LAC</td>
      <td>76</td>
      <td>9.9</td>
      <td>1.0</td>
      <td>0.4</td>
      <td>1.3</td>
      <td>10.8</td>
      <td>0.661</td>
      <td>112.0</td>
      <td>-0.548648</td>
      <td>-0.784548</td>
      <td>0.210459</td>
      <td>-1.181873</td>
      <td>0.576251</td>
      <td>0.420030</td>
      <td>-0.016015</td>
    </tr>
  </tbody>
</table>
<p>7140 rows × 20 columns</p>
</div>




```python
starters["AZS"] = starters.loc[:, "PTSZ":"DRtgZ"].mean(axis=1)
starters.sort_values("AZS", ascending=False).head(50)["Player"].value_counts()
```




    Player
    Hakeem Olajuwon          8
    Michael Jordan           7
    David Robinson           6
    Dwyane Wade              5
    LeBron James             4
    James Harden             4
    Magic Johnson            3
    Kevin Durant             3
    Giannis Antetokounmpo    2
    Scottie Pippen           2
    Larry Bird               1
    Draymond Green           1
    Shaquille O'Neal         1
    Paul Millsap             1
    Dwight Howard            1
    Chris Paul               1
    Name: count, dtype: int64




```python
starters.sort_values("AZS", ascending=False).head(50)
```




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
      <th>Year</th>
      <th>Player</th>
      <th>Pos</th>
      <th>Age</th>
      <th>Tm</th>
      <th>G</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>...</th>
      <th>TS%</th>
      <th>DRtg</th>
      <th>PTSZ</th>
      <th>ASTZ</th>
      <th>BLKZ</th>
      <th>STLZ</th>
      <th>TRBZ</th>
      <th>TS%Z</th>
      <th>DRtgZ</th>
      <th>AZS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6175</th>
      <td>2019</td>
      <td>Giannis Antetokounmpo</td>
      <td>PF</td>
      <td>24</td>
      <td>MIL</td>
      <td>72</td>
      <td>12.5</td>
      <td>5.9</td>
      <td>1.3</td>
      <td>1.5</td>
      <td>...</td>
      <td>0.644</td>
      <td>99.0</td>
      <td>2.441825</td>
      <td>2.707217</td>
      <td>2.704507</td>
      <td>1.910410</td>
      <td>3.001977</td>
      <td>2.339537</td>
      <td>3.370416</td>
      <td>2.639413</td>
    </tr>
    <tr>
      <th>394</th>
      <td>1984</td>
      <td>Magic Johnson</td>
      <td>PG</td>
      <td>24</td>
      <td>LAL</td>
      <td>67</td>
      <td>7.3</td>
      <td>13.1</td>
      <td>2.2</td>
      <td>0.7</td>
      <td>...</td>
      <td>0.628</td>
      <td>105.0</td>
      <td>1.511602</td>
      <td>2.498379</td>
      <td>3.208445</td>
      <td>1.254227</td>
      <td>4.298256</td>
      <td>2.996295</td>
      <td>1.327788</td>
      <td>2.442142</td>
    </tr>
    <tr>
      <th>1226</th>
      <td>1990</td>
      <td>Hakeem Olajuwon</td>
      <td>C</td>
      <td>27</td>
      <td>HOU</td>
      <td>82</td>
      <td>14.0</td>
      <td>2.9</td>
      <td>2.1</td>
      <td>4.6</td>
      <td>...</td>
      <td>0.541</td>
      <td>93.0</td>
      <td>2.088100</td>
      <td>2.088095</td>
      <td>3.058527</td>
      <td>3.612712</td>
      <td>2.604576</td>
      <td>0.207789</td>
      <td>3.202919</td>
      <td>2.408960</td>
    </tr>
    <tr>
      <th>1542</th>
      <td>1992</td>
      <td>David Robinson</td>
      <td>C</td>
      <td>26</td>
      <td>SAS</td>
      <td>68</td>
      <td>12.2</td>
      <td>2.7</td>
      <td>2.3</td>
      <td>4.5</td>
      <td>...</td>
      <td>0.597</td>
      <td>94.0</td>
      <td>1.911430</td>
      <td>1.642017</td>
      <td>2.580521</td>
      <td>3.673266</td>
      <td>1.819023</td>
      <td>1.661048</td>
      <td>3.115735</td>
      <td>2.343291</td>
    </tr>
    <tr>
      <th>1373</th>
      <td>1991</td>
      <td>Hakeem Olajuwon</td>
      <td>C</td>
      <td>28</td>
      <td>HOU</td>
      <td>56</td>
      <td>13.8</td>
      <td>2.3</td>
      <td>2.2</td>
      <td>3.9</td>
      <td>...</td>
      <td>0.549</td>
      <td>93.0</td>
      <td>1.815636</td>
      <td>1.415487</td>
      <td>2.862640</td>
      <td>3.935226</td>
      <td>2.467379</td>
      <td>0.270221</td>
      <td>3.131034</td>
      <td>2.271089</td>
    </tr>
    <tr>
      <th>4520</th>
      <td>2009</td>
      <td>Dwyane Wade</td>
      <td>SG</td>
      <td>27</td>
      <td>MIA</td>
      <td>79</td>
      <td>5.0</td>
      <td>7.5</td>
      <td>2.2</td>
      <td>1.3</td>
      <td>...</td>
      <td>0.574</td>
      <td>105.0</td>
      <td>2.491936</td>
      <td>2.945833</td>
      <td>3.544931</td>
      <td>3.274785</td>
      <td>1.395539</td>
      <td>0.780597</td>
      <td>1.452109</td>
      <td>2.269390</td>
    </tr>
    <tr>
      <th>1390</th>
      <td>1991</td>
      <td>David Robinson</td>
      <td>C</td>
      <td>25</td>
      <td>SAS</td>
      <td>82</td>
      <td>13.0</td>
      <td>2.5</td>
      <td>1.5</td>
      <td>3.9</td>
      <td>...</td>
      <td>0.615</td>
      <td>96.0</td>
      <td>2.615953</td>
      <td>1.719641</td>
      <td>2.862640</td>
      <td>2.079757</td>
      <td>2.141460</td>
      <td>1.626064</td>
      <td>2.409910</td>
      <td>2.207918</td>
    </tr>
    <tr>
      <th>4433</th>
      <td>2009</td>
      <td>LeBron James</td>
      <td>SF</td>
      <td>24</td>
      <td>CLE</td>
      <td>81</td>
      <td>7.6</td>
      <td>7.2</td>
      <td>1.7</td>
      <td>1.1</td>
      <td>...</td>
      <td>0.591</td>
      <td>99.0</td>
      <td>2.407305</td>
      <td>3.355693</td>
      <td>1.775000</td>
      <td>1.964406</td>
      <td>1.661375</td>
      <td>1.360109</td>
      <td>2.804227</td>
      <td>2.189731</td>
    </tr>
    <tr>
      <th>5013</th>
      <td>2012</td>
      <td>LeBron James</td>
      <td>SF</td>
      <td>27</td>
      <td>MIA</td>
      <td>62</td>
      <td>7.9</td>
      <td>6.2</td>
      <td>1.9</td>
      <td>0.8</td>
      <td>...</td>
      <td>0.605</td>
      <td>97.0</td>
      <td>2.684309</td>
      <td>3.026839</td>
      <td>1.079623</td>
      <td>2.366757</td>
      <td>2.159972</td>
      <td>1.977304</td>
      <td>2.022859</td>
      <td>2.188238</td>
    </tr>
    <tr>
      <th>1075</th>
      <td>1989</td>
      <td>Hakeem Olajuwon</td>
      <td>C</td>
      <td>26</td>
      <td>HOU</td>
      <td>82</td>
      <td>13.5</td>
      <td>1.8</td>
      <td>2.6</td>
      <td>3.4</td>
      <td>...</td>
      <td>0.552</td>
      <td>95.0</td>
      <td>2.330149</td>
      <td>0.459302</td>
      <td>2.207276</td>
      <td>4.090904</td>
      <td>2.398385</td>
      <td>0.349693</td>
      <td>3.381106</td>
      <td>2.173831</td>
    </tr>
    <tr>
      <th>1047</th>
      <td>1989</td>
      <td>Michael Jordan</td>
      <td>SG</td>
      <td>25</td>
      <td>CHI</td>
      <td>81</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>2.9</td>
      <td>0.8</td>
      <td>...</td>
      <td>0.614</td>
      <td>103.0</td>
      <td>2.801831</td>
      <td>2.095634</td>
      <td>2.072827</td>
      <td>2.097729</td>
      <td>2.164874</td>
      <td>2.105943</td>
      <td>1.810635</td>
      <td>2.164210</td>
    </tr>
    <tr>
      <th>5143</th>
      <td>2013</td>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>24</td>
      <td>OKC</td>
      <td>81</td>
      <td>7.9</td>
      <td>4.6</td>
      <td>1.4</td>
      <td>1.3</td>
      <td>...</td>
      <td>0.647</td>
      <td>100.0</td>
      <td>3.185754</td>
      <td>1.922241</td>
      <td>2.788581</td>
      <td>1.040959</td>
      <td>2.071473</td>
      <td>2.332590</td>
      <td>1.625057</td>
      <td>2.138093</td>
    </tr>
    <tr>
      <th>910</th>
      <td>1988</td>
      <td>Michael Jordan</td>
      <td>SG</td>
      <td>24</td>
      <td>CHI</td>
      <td>82</td>
      <td>5.5</td>
      <td>5.9</td>
      <td>3.2</td>
      <td>1.6</td>
      <td>...</td>
      <td>0.603</td>
      <td>101.0</td>
      <td>2.771163</td>
      <td>0.788076</td>
      <td>3.670056</td>
      <td>2.287971</td>
      <td>1.154688</td>
      <td>1.772378</td>
      <td>2.396859</td>
      <td>2.120170</td>
    </tr>
    <tr>
      <th>6070</th>
      <td>2018</td>
      <td>James Harden</td>
      <td>SG</td>
      <td>28</td>
      <td>HOU</td>
      <td>72</td>
      <td>5.4</td>
      <td>8.8</td>
      <td>1.8</td>
      <td>0.7</td>
      <td>...</td>
      <td>0.619</td>
      <td>105.0</td>
      <td>2.559618</td>
      <td>3.499929</td>
      <td>1.746517</td>
      <td>1.631735</td>
      <td>1.625327</td>
      <td>1.856894</td>
      <td>1.897254</td>
      <td>2.116753</td>
    </tr>
    <tr>
      <th>6446</th>
      <td>2020</td>
      <td>James Harden</td>
      <td>SG</td>
      <td>30</td>
      <td>HOU</td>
      <td>68</td>
      <td>6.6</td>
      <td>7.5</td>
      <td>1.8</td>
      <td>0.9</td>
      <td>...</td>
      <td>0.626</td>
      <td>108.0</td>
      <td>2.929869</td>
      <td>2.158478</td>
      <td>2.524809</td>
      <td>2.156677</td>
      <td>2.358743</td>
      <td>1.451304</td>
      <td>1.180700</td>
      <td>2.108654</td>
    </tr>
    <tr>
      <th>4145</th>
      <td>2007</td>
      <td>Dwyane Wade</td>
      <td>SG</td>
      <td>25</td>
      <td>MIA</td>
      <td>51</td>
      <td>4.7</td>
      <td>7.5</td>
      <td>2.1</td>
      <td>1.2</td>
      <td>...</td>
      <td>0.583</td>
      <td>103.0</td>
      <td>1.657158</td>
      <td>2.373822</td>
      <td>4.437920</td>
      <td>2.510103</td>
      <td>0.735726</td>
      <td>1.164460</td>
      <td>1.685474</td>
      <td>2.080666</td>
    </tr>
    <tr>
      <th>4632</th>
      <td>2010</td>
      <td>LeBron James</td>
      <td>SF</td>
      <td>25</td>
      <td>CLE</td>
      <td>76</td>
      <td>7.3</td>
      <td>8.6</td>
      <td>1.6</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.604</td>
      <td>102.0</td>
      <td>2.620584</td>
      <td>4.253827</td>
      <td>1.645906</td>
      <td>1.366548</td>
      <td>1.413973</td>
      <td>1.591229</td>
      <td>1.573896</td>
      <td>2.066566</td>
    </tr>
    <tr>
      <th>1643</th>
      <td>1993</td>
      <td>Michael Jordan</td>
      <td>SG</td>
      <td>29</td>
      <td>CHI</td>
      <td>78</td>
      <td>6.7</td>
      <td>5.5</td>
      <td>2.8</td>
      <td>0.8</td>
      <td>...</td>
      <td>0.564</td>
      <td>102.0</td>
      <td>3.059072</td>
      <td>1.646388</td>
      <td>1.732230</td>
      <td>2.644322</td>
      <td>2.392998</td>
      <td>0.747148</td>
      <td>2.082079</td>
      <td>2.043462</td>
    </tr>
    <tr>
      <th>4903</th>
      <td>2011</td>
      <td>Dwyane Wade</td>
      <td>SG</td>
      <td>29</td>
      <td>MIA</td>
      <td>76</td>
      <td>6.4</td>
      <td>4.6</td>
      <td>1.5</td>
      <td>1.1</td>
      <td>...</td>
      <td>0.581</td>
      <td>102.0</td>
      <td>1.942102</td>
      <td>1.411546</td>
      <td>3.979886</td>
      <td>1.366730</td>
      <td>2.581723</td>
      <td>0.886416</td>
      <td>1.990513</td>
      <td>2.022702</td>
    </tr>
    <tr>
      <th>5527</th>
      <td>2015</td>
      <td>James Harden</td>
      <td>SG</td>
      <td>25</td>
      <td>HOU</td>
      <td>81</td>
      <td>5.7</td>
      <td>7.0</td>
      <td>1.9</td>
      <td>0.7</td>
      <td>...</td>
      <td>0.605</td>
      <td>103.0</td>
      <td>2.631189</td>
      <td>2.744850</td>
      <td>1.721105</td>
      <td>2.024944</td>
      <td>2.267481</td>
      <td>1.408014</td>
      <td>1.270008</td>
      <td>2.009656</td>
    </tr>
    <tr>
      <th>264</th>
      <td>1983</td>
      <td>Magic Johnson</td>
      <td>SG</td>
      <td>23</td>
      <td>LAL</td>
      <td>79</td>
      <td>8.6</td>
      <td>10.5</td>
      <td>2.2</td>
      <td>0.6</td>
      <td>...</td>
      <td>0.603</td>
      <td>103.0</td>
      <td>0.196677</td>
      <td>3.735996</td>
      <td>1.696707</td>
      <td>2.379708</td>
      <td>3.119342</td>
      <td>1.929865</td>
      <td>0.988573</td>
      <td>2.006695</td>
    </tr>
    <tr>
      <th>475</th>
      <td>1985</td>
      <td>Larry Bird</td>
      <td>SF</td>
      <td>28</td>
      <td>BOS</td>
      <td>80</td>
      <td>10.5</td>
      <td>6.6</td>
      <td>1.6</td>
      <td>1.2</td>
      <td>...</td>
      <td>0.585</td>
      <td>103.0</td>
      <td>1.400636</td>
      <td>2.728286</td>
      <td>2.064941</td>
      <td>1.615455</td>
      <td>3.587314</td>
      <td>0.696444</td>
      <td>1.860807</td>
      <td>1.993412</td>
    </tr>
    <tr>
      <th>5708</th>
      <td>2016</td>
      <td>Draymond Green</td>
      <td>PF</td>
      <td>25</td>
      <td>GSW</td>
      <td>81</td>
      <td>9.5</td>
      <td>7.4</td>
      <td>1.5</td>
      <td>1.4</td>
      <td>...</td>
      <td>0.587</td>
      <td>100.0</td>
      <td>0.466784</td>
      <td>4.547467</td>
      <td>1.510925</td>
      <td>2.507193</td>
      <td>1.654965</td>
      <td>1.343399</td>
      <td>1.824548</td>
      <td>1.979326</td>
    </tr>
    <tr>
      <th>523</th>
      <td>1985</td>
      <td>Michael Jordan</td>
      <td>SG</td>
      <td>21</td>
      <td>CHI</td>
      <td>82</td>
      <td>6.5</td>
      <td>5.9</td>
      <td>2.4</td>
      <td>0.8</td>
      <td>...</td>
      <td>0.592</td>
      <td>107.0</td>
      <td>2.258514</td>
      <td>1.222625</td>
      <td>1.967015</td>
      <td>2.638653</td>
      <td>2.664604</td>
      <td>1.774225</td>
      <td>1.238959</td>
      <td>1.966371</td>
    </tr>
    <tr>
      <th>5869</th>
      <td>2017</td>
      <td>Kevin Durant</td>
      <td>PF</td>
      <td>28</td>
      <td>GSW</td>
      <td>62</td>
      <td>8.3</td>
      <td>4.8</td>
      <td>1.1</td>
      <td>1.6</td>
      <td>...</td>
      <td>0.651</td>
      <td>101.0</td>
      <td>2.418286</td>
      <td>2.142968</td>
      <td>1.845411</td>
      <td>0.956850</td>
      <td>1.167389</td>
      <td>2.821684</td>
      <td>2.406688</td>
      <td>1.965611</td>
    </tr>
    <tr>
      <th>1671</th>
      <td>1993</td>
      <td>Hakeem Olajuwon</td>
      <td>C</td>
      <td>30</td>
      <td>HOU</td>
      <td>82</td>
      <td>13.0</td>
      <td>3.5</td>
      <td>1.8</td>
      <td>4.2</td>
      <td>...</td>
      <td>0.577</td>
      <td>96.0</td>
      <td>1.987313</td>
      <td>1.826284</td>
      <td>2.464271</td>
      <td>2.743900</td>
      <td>1.662472</td>
      <td>0.747548</td>
      <td>2.207370</td>
      <td>1.948451</td>
    </tr>
    <tr>
      <th>1851</th>
      <td>1994</td>
      <td>David Robinson</td>
      <td>C</td>
      <td>28</td>
      <td>SAS</td>
      <td>80</td>
      <td>10.7</td>
      <td>4.8</td>
      <td>1.7</td>
      <td>3.3</td>
      <td>...</td>
      <td>0.577</td>
      <td>98.0</td>
      <td>2.488472</td>
      <td>2.756669</td>
      <td>1.862808</td>
      <td>2.665801</td>
      <td>1.062791</td>
      <td>1.134571</td>
      <td>1.509894</td>
      <td>1.925858</td>
    </tr>
    <tr>
      <th>5087</th>
      <td>2012</td>
      <td>Dwyane Wade</td>
      <td>SG</td>
      <td>30</td>
      <td>MIA</td>
      <td>49</td>
      <td>4.8</td>
      <td>4.6</td>
      <td>1.7</td>
      <td>1.3</td>
      <td>...</td>
      <td>0.559</td>
      <td>99.0</td>
      <td>1.711407</td>
      <td>1.291344</td>
      <td>4.181539</td>
      <td>1.837743</td>
      <td>1.680120</td>
      <td>0.837880</td>
      <td>1.940165</td>
      <td>1.925743</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>1991</td>
      <td>Michael Jordan</td>
      <td>SG</td>
      <td>27</td>
      <td>CHI</td>
      <td>82</td>
      <td>6.0</td>
      <td>5.5</td>
      <td>2.7</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.605</td>
      <td>102.0</td>
      <td>2.761916</td>
      <td>1.299861</td>
      <td>2.092766</td>
      <td>2.278040</td>
      <td>1.561274</td>
      <td>1.506825</td>
      <td>1.958481</td>
      <td>1.922738</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>1995</td>
      <td>Scottie Pippen</td>
      <td>SF</td>
      <td>29</td>
      <td>CHI</td>
      <td>79</td>
      <td>8.1</td>
      <td>5.2</td>
      <td>2.9</td>
      <td>1.1</td>
      <td>...</td>
      <td>0.559</td>
      <td>98.0</td>
      <td>1.196176</td>
      <td>1.982946</td>
      <td>1.499350</td>
      <td>3.494062</td>
      <td>1.908802</td>
      <td>0.343549</td>
      <td>3.032231</td>
      <td>1.922445</td>
    </tr>
    <tr>
      <th>2193</th>
      <td>1996</td>
      <td>David Robinson</td>
      <td>C</td>
      <td>30</td>
      <td>SAS</td>
      <td>82</td>
      <td>12.2</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>3.3</td>
      <td>...</td>
      <td>0.589</td>
      <td>96.0</td>
      <td>1.995263</td>
      <td>1.723213</td>
      <td>1.744107</td>
      <td>2.319480</td>
      <td>2.182741</td>
      <td>1.325609</td>
      <td>2.160247</td>
      <td>1.921523</td>
    </tr>
    <tr>
      <th>5185</th>
      <td>2013</td>
      <td>LeBron James</td>
      <td>PF</td>
      <td>28</td>
      <td>MIA</td>
      <td>76</td>
      <td>8.0</td>
      <td>7.3</td>
      <td>1.7</td>
      <td>0.9</td>
      <td>...</td>
      <td>0.640</td>
      <td>101.0</td>
      <td>2.606896</td>
      <td>3.869625</td>
      <td>0.089287</td>
      <td>2.587323</td>
      <td>0.296576</td>
      <td>2.862683</td>
      <td>1.134923</td>
      <td>1.921044</td>
    </tr>
    <tr>
      <th>5830</th>
      <td>2017</td>
      <td>Giannis Antetokounmpo</td>
      <td>SF</td>
      <td>22</td>
      <td>MIL</td>
      <td>80</td>
      <td>8.8</td>
      <td>5.4</td>
      <td>1.6</td>
      <td>1.9</td>
      <td>...</td>
      <td>0.599</td>
      <td>104.0</td>
      <td>1.498380</td>
      <td>1.932192</td>
      <td>3.862543</td>
      <td>1.087653</td>
      <td>2.381146</td>
      <td>1.191782</td>
      <td>1.490426</td>
      <td>1.920589</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>1995</td>
      <td>David Robinson</td>
      <td>C</td>
      <td>29</td>
      <td>SAS</td>
      <td>81</td>
      <td>10.8</td>
      <td>2.9</td>
      <td>1.7</td>
      <td>3.2</td>
      <td>...</td>
      <td>0.602</td>
      <td>99.0</td>
      <td>2.312544</td>
      <td>1.712055</td>
      <td>1.889226</td>
      <td>2.820335</td>
      <td>1.384206</td>
      <td>1.241882</td>
      <td>1.998628</td>
      <td>1.908411</td>
    </tr>
    <tr>
      <th>4714</th>
      <td>2010</td>
      <td>Dwyane Wade</td>
      <td>SG</td>
      <td>28</td>
      <td>MIA</td>
      <td>77</td>
      <td>4.8</td>
      <td>6.5</td>
      <td>1.8</td>
      <td>1.1</td>
      <td>...</td>
      <td>0.562</td>
      <td>103.0</td>
      <td>2.079634</td>
      <td>2.557133</td>
      <td>3.121246</td>
      <td>1.913888</td>
      <td>1.112348</td>
      <td>0.730670</td>
      <td>1.837930</td>
      <td>1.907550</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>1995</td>
      <td>Hakeem Olajuwon</td>
      <td>C</td>
      <td>32</td>
      <td>HOU</td>
      <td>72</td>
      <td>10.8</td>
      <td>3.5</td>
      <td>1.8</td>
      <td>3.4</td>
      <td>...</td>
      <td>0.563</td>
      <td>100.0</td>
      <td>2.341404</td>
      <td>2.330870</td>
      <td>2.102525</td>
      <td>3.086404</td>
      <td>1.384206</td>
      <td>0.366278</td>
      <td>1.732651</td>
      <td>1.906334</td>
    </tr>
    <tr>
      <th>129</th>
      <td>1982</td>
      <td>Magic Johnson</td>
      <td>SG</td>
      <td>22</td>
      <td>LAL</td>
      <td>78</td>
      <td>9.6</td>
      <td>9.5</td>
      <td>2.7</td>
      <td>0.4</td>
      <td>...</td>
      <td>0.590</td>
      <td>102.0</td>
      <td>0.271301</td>
      <td>3.221363</td>
      <td>0.434898</td>
      <td>2.671911</td>
      <td>3.182120</td>
      <td>1.860997</td>
      <td>1.700397</td>
      <td>1.906141</td>
    </tr>
    <tr>
      <th>6048</th>
      <td>2018</td>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>29</td>
      <td>GSW</td>
      <td>68</td>
      <td>6.8</td>
      <td>5.4</td>
      <td>0.7</td>
      <td>1.8</td>
      <td>...</td>
      <td>0.640</td>
      <td>107.0</td>
      <td>2.802401</td>
      <td>2.407672</td>
      <td>3.764864</td>
      <td>-0.917405</td>
      <td>2.136200</td>
      <td>2.445771</td>
      <td>0.615098</td>
      <td>1.893514</td>
    </tr>
    <tr>
      <th>937</th>
      <td>1988</td>
      <td>Hakeem Olajuwon</td>
      <td>C</td>
      <td>25</td>
      <td>HOU</td>
      <td>79</td>
      <td>12.1</td>
      <td>2.1</td>
      <td>2.1</td>
      <td>2.7</td>
      <td>...</td>
      <td>0.555</td>
      <td>98.0</td>
      <td>2.036733</td>
      <td>0.559635</td>
      <td>1.566208</td>
      <td>3.873582</td>
      <td>2.122248</td>
      <td>0.415850</td>
      <td>2.652949</td>
      <td>1.889601</td>
    </tr>
    <tr>
      <th>1246</th>
      <td>1990</td>
      <td>David Robinson</td>
      <td>C</td>
      <td>24</td>
      <td>SAS</td>
      <td>82</td>
      <td>12.0</td>
      <td>2.0</td>
      <td>1.7</td>
      <td>3.9</td>
      <td>...</td>
      <td>0.597</td>
      <td>97.0</td>
      <td>2.088100</td>
      <td>0.828429</td>
      <td>2.391165</td>
      <td>2.600058</td>
      <td>1.803501</td>
      <td>1.220824</td>
      <td>2.239439</td>
      <td>1.881645</td>
    </tr>
    <tr>
      <th>2850</th>
      <td>2000</td>
      <td>Shaquille O'Neal</td>
      <td>C</td>
      <td>27</td>
      <td>LAL</td>
      <td>79</td>
      <td>13.6</td>
      <td>3.8</td>
      <td>0.5</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.578</td>
      <td>95.0</td>
      <td>3.146170</td>
      <td>3.224506</td>
      <td>1.641668</td>
      <td>-0.413745</td>
      <td>2.464607</td>
      <td>1.399367</td>
      <td>1.645805</td>
      <td>1.872625</td>
    </tr>
    <tr>
      <th>5760</th>
      <td>2016</td>
      <td>Paul Millsap</td>
      <td>PF</td>
      <td>30</td>
      <td>ATL</td>
      <td>81</td>
      <td>9.0</td>
      <td>3.3</td>
      <td>1.8</td>
      <td>1.7</td>
      <td>...</td>
      <td>0.556</td>
      <td>96.0</td>
      <td>1.180652</td>
      <td>1.169158</td>
      <td>2.141423</td>
      <td>3.545619</td>
      <td>1.406249</td>
      <td>0.483534</td>
      <td>3.141783</td>
      <td>1.866917</td>
    </tr>
    <tr>
      <th>1497</th>
      <td>1992</td>
      <td>Michael Jordan</td>
      <td>SG</td>
      <td>28</td>
      <td>CHI</td>
      <td>80</td>
      <td>6.4</td>
      <td>6.1</td>
      <td>2.3</td>
      <td>0.9</td>
      <td>...</td>
      <td>0.579</td>
      <td>102.0</td>
      <td>2.589103</td>
      <td>1.928577</td>
      <td>1.507557</td>
      <td>1.804918</td>
      <td>1.846627</td>
      <td>1.060416</td>
      <td>2.249170</td>
      <td>1.855195</td>
    </tr>
    <tr>
      <th>1832</th>
      <td>1994</td>
      <td>Hakeem Olajuwon</td>
      <td>C</td>
      <td>31</td>
      <td>HOU</td>
      <td>80</td>
      <td>11.9</td>
      <td>3.6</td>
      <td>1.6</td>
      <td>3.7</td>
      <td>...</td>
      <td>0.565</td>
      <td>95.0</td>
      <td>2.126249</td>
      <td>1.682836</td>
      <td>2.267400</td>
      <td>2.381869</td>
      <td>1.480177</td>
      <td>0.818151</td>
      <td>2.218887</td>
      <td>1.853653</td>
    </tr>
    <tr>
      <th>4823</th>
      <td>2011</td>
      <td>Dwight Howard</td>
      <td>C</td>
      <td>25</td>
      <td>ORL</td>
      <td>78</td>
      <td>14.1</td>
      <td>1.4</td>
      <td>1.4</td>
      <td>2.4</td>
      <td>...</td>
      <td>0.616</td>
      <td>94.0</td>
      <td>2.166355</td>
      <td>-0.033404</td>
      <td>1.858552</td>
      <td>2.604231</td>
      <td>2.734338</td>
      <td>1.151755</td>
      <td>2.485251</td>
      <td>1.852440</td>
    </tr>
    <tr>
      <th>6636</th>
      <td>2021</td>
      <td>James Harden</td>
      <td>SG</td>
      <td>31</td>
      <td>BRK</td>
      <td>36</td>
      <td>8.5</td>
      <td>10.9</td>
      <td>1.3</td>
      <td>0.8</td>
      <td>...</td>
      <td>0.619</td>
      <td>111.0</td>
      <td>1.202159</td>
      <td>3.085050</td>
      <td>2.083474</td>
      <td>1.034118</td>
      <td>2.865507</td>
      <td>1.312919</td>
      <td>1.359604</td>
      <td>1.848976</td>
    </tr>
    <tr>
      <th>1840</th>
      <td>1994</td>
      <td>Scottie Pippen</td>
      <td>SF</td>
      <td>28</td>
      <td>CHI</td>
      <td>72</td>
      <td>8.7</td>
      <td>5.6</td>
      <td>2.9</td>
      <td>0.8</td>
      <td>...</td>
      <td>0.544</td>
      <td>97.0</td>
      <td>1.118565</td>
      <td>2.480838</td>
      <td>0.901057</td>
      <td>3.866708</td>
      <td>1.950348</td>
      <td>0.502911</td>
      <td>2.093832</td>
      <td>1.844894</td>
    </tr>
    <tr>
      <th>4484</th>
      <td>2009</td>
      <td>Chris Paul</td>
      <td>PG</td>
      <td>23</td>
      <td>NOH</td>
      <td>78</td>
      <td>5.5</td>
      <td>11.0</td>
      <td>2.8</td>
      <td>0.1</td>
      <td>...</td>
      <td>0.599</td>
      <td>103.0</td>
      <td>2.071209</td>
      <td>2.586308</td>
      <td>-0.661802</td>
      <td>3.507347</td>
      <td>2.375233</td>
      <td>1.427052</td>
      <td>1.545453</td>
      <td>1.835829</td>
    </tr>
    <tr>
      <th>775</th>
      <td>1987</td>
      <td>Michael Jordan</td>
      <td>SG</td>
      <td>23</td>
      <td>CHI</td>
      <td>82</td>
      <td>5.2</td>
      <td>4.6</td>
      <td>2.9</td>
      <td>1.5</td>
      <td>...</td>
      <td>0.562</td>
      <td>104.0</td>
      <td>3.043472</td>
      <td>0.682469</td>
      <td>2.618009</td>
      <td>2.180671</td>
      <td>1.407999</td>
      <td>0.842779</td>
      <td>2.025245</td>
      <td>1.828663</td>
    </tr>
    <tr>
      <th>797</th>
      <td>1987</td>
      <td>Hakeem Olajuwon</td>
      <td>C</td>
      <td>24</td>
      <td>HOU</td>
      <td>75</td>
      <td>11.4</td>
      <td>2.9</td>
      <td>1.9</td>
      <td>3.4</td>
      <td>...</td>
      <td>0.554</td>
      <td>99.0</td>
      <td>1.951886</td>
      <td>1.373929</td>
      <td>2.298050</td>
      <td>3.311323</td>
      <td>1.495413</td>
      <td>0.050663</td>
      <td>2.262391</td>
      <td>1.820522</td>
    </tr>
  </tbody>
</table>
<p>50 rows × 21 columns</p>
</div>




```python
starters["AZS"] = starters.loc[:, "PTSZ":"DRtgZ"].mean(axis=1)
starters.sort_values("AZS", ascending=False).head(50)["Pos"].value_counts()
```




    Pos
    SG    18
    C     16
    SF     9
    PF     5
    PG     2
    Name: count, dtype: int64




```python
starters[starters["Player"] == "Tim Duncan"]
```




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
      <th>Year</th>
      <th>Player</th>
      <th>Pos</th>
      <th>Age</th>
      <th>Tm</th>
      <th>G</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>...</th>
      <th>TS%</th>
      <th>DRtg</th>
      <th>PTSZ</th>
      <th>ASTZ</th>
      <th>BLKZ</th>
      <th>STLZ</th>
      <th>TRBZ</th>
      <th>TS%Z</th>
      <th>DRtgZ</th>
      <th>AZS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2453</th>
      <td>1998</td>
      <td>Tim Duncan</td>
      <td>PF</td>
      <td>21</td>
      <td>SAS</td>
      <td>82</td>
      <td>11.9</td>
      <td>2.7</td>
      <td>0.7</td>
      <td>2.5</td>
      <td>...</td>
      <td>0.577</td>
      <td>95.0</td>
      <td>1.483496</td>
      <td>0.494271</td>
      <td>3.346789</td>
      <td>-0.951822</td>
      <td>1.693402</td>
      <td>1.347073</td>
      <td>2.131796</td>
      <td>1.363572</td>
    </tr>
    <tr>
      <th>2631</th>
      <td>1999</td>
      <td>Tim Duncan</td>
      <td>PF</td>
      <td>22</td>
      <td>SAS</td>
      <td>50</td>
      <td>11.4</td>
      <td>2.4</td>
      <td>0.9</td>
      <td>2.5</td>
      <td>...</td>
      <td>0.541</td>
      <td>91.0</td>
      <td>1.702221</td>
      <td>0.539850</td>
      <td>2.572285</td>
      <td>0.204376</td>
      <td>1.526528</td>
      <td>0.572049</td>
      <td>2.463326</td>
      <td>1.368662</td>
    </tr>
    <tr>
      <th>2785</th>
      <td>2000</td>
      <td>Tim Duncan</td>
      <td>PF</td>
      <td>23</td>
      <td>SAS</td>
      <td>74</td>
      <td>12.4</td>
      <td>3.2</td>
      <td>0.9</td>
      <td>2.2</td>
      <td>...</td>
      <td>0.555</td>
      <td>95.0</td>
      <td>1.556041</td>
      <td>0.936240</td>
      <td>2.404671</td>
      <td>0.082052</td>
      <td>2.427935</td>
      <td>1.035298</td>
      <td>1.850501</td>
      <td>1.470391</td>
    </tr>
    <tr>
      <th>2942</th>
      <td>2001</td>
      <td>Tim Duncan</td>
      <td>PF</td>
      <td>24</td>
      <td>SAS</td>
      <td>82</td>
      <td>12.2</td>
      <td>3.0</td>
      <td>0.9</td>
      <td>2.3</td>
      <td>...</td>
      <td>0.536</td>
      <td>94.0</td>
      <td>1.160053</td>
      <td>0.525143</td>
      <td>2.405774</td>
      <td>0.163332</td>
      <td>1.836740</td>
      <td>0.392457</td>
      <td>2.266244</td>
      <td>1.249963</td>
    </tr>
    <tr>
      <th>3122</th>
      <td>2002</td>
      <td>Tim Duncan</td>
      <td>PF</td>
      <td>25</td>
      <td>SAS</td>
      <td>82</td>
      <td>12.7</td>
      <td>3.7</td>
      <td>0.7</td>
      <td>2.5</td>
      <td>...</td>
      <td>0.576</td>
      <td>96.0</td>
      <td>2.068310</td>
      <td>1.158709</td>
      <td>2.224742</td>
      <td>-0.585076</td>
      <td>2.118375</td>
      <td>1.525105</td>
      <td>2.418871</td>
      <td>1.561291</td>
    </tr>
    <tr>
      <th>3296</th>
      <td>2003</td>
      <td>Tim Duncan</td>
      <td>PF</td>
      <td>26</td>
      <td>SAS</td>
      <td>81</td>
      <td>12.9</td>
      <td>3.9</td>
      <td>0.7</td>
      <td>2.9</td>
      <td>...</td>
      <td>0.564</td>
      <td>94.0</td>
      <td>1.518056</td>
      <td>1.120155</td>
      <td>2.908816</td>
      <td>-0.816273</td>
      <td>2.174333</td>
      <td>0.962062</td>
      <td>2.322421</td>
      <td>1.455653</td>
    </tr>
    <tr>
      <th>3469</th>
      <td>2004</td>
      <td>Tim Duncan</td>
      <td>PF</td>
      <td>27</td>
      <td>SAS</td>
      <td>69</td>
      <td>12.4</td>
      <td>3.1</td>
      <td>0.9</td>
      <td>2.7</td>
      <td>...</td>
      <td>0.534</td>
      <td>89.0</td>
      <td>1.620563</td>
      <td>0.750950</td>
      <td>2.197057</td>
      <td>-0.149214</td>
      <td>1.950215</td>
      <td>0.302300</td>
      <td>2.475447</td>
      <td>1.306760</td>
    </tr>
    <tr>
      <th>3657</th>
      <td>2005</td>
      <td>Tim Duncan</td>
      <td>PF</td>
      <td>28</td>
      <td>SAS</td>
      <td>66</td>
      <td>11.1</td>
      <td>2.7</td>
      <td>0.7</td>
      <td>2.6</td>
      <td>...</td>
      <td>0.540</td>
      <td>93.0</td>
      <td>0.990199</td>
      <td>0.354353</td>
      <td>2.704459</td>
      <td>-0.635176</td>
      <td>1.372306</td>
      <td>0.348982</td>
      <td>3.246079</td>
      <td>1.197315</td>
    </tr>
    <tr>
      <th>3846</th>
      <td>2006</td>
      <td>Tim Duncan</td>
      <td>PF</td>
      <td>29</td>
      <td>SAS</td>
      <td>80</td>
      <td>11.0</td>
      <td>3.2</td>
      <td>0.9</td>
      <td>2.0</td>
      <td>...</td>
      <td>0.523</td>
      <td>94.0</td>
      <td>0.867979</td>
      <td>0.867637</td>
      <td>1.901084</td>
      <td>0.402492</td>
      <td>1.430736</td>
      <td>-0.442048</td>
      <td>2.621645</td>
      <td>1.092789</td>
    </tr>
    <tr>
      <th>4027</th>
      <td>2007</td>
      <td>Tim Duncan</td>
      <td>C</td>
      <td>30</td>
      <td>SAS</td>
      <td>80</td>
      <td>10.6</td>
      <td>3.4</td>
      <td>0.8</td>
      <td>2.4</td>
      <td>...</td>
      <td>0.579</td>
      <td>94.0</td>
      <td>1.447906</td>
      <td>2.145793</td>
      <td>1.524193</td>
      <td>0.631916</td>
      <td>1.198386</td>
      <td>0.419141</td>
      <td>2.428417</td>
      <td>1.399393</td>
    </tr>
    <tr>
      <th>4200</th>
      <td>2008</td>
      <td>Tim Duncan</td>
      <td>C</td>
      <td>31</td>
      <td>SAS</td>
      <td>78</td>
      <td>11.3</td>
      <td>2.8</td>
      <td>0.7</td>
      <td>1.9</td>
      <td>...</td>
      <td>0.546</td>
      <td>97.0</td>
      <td>1.236234</td>
      <td>1.378393</td>
      <td>0.756641</td>
      <td>0.214998</td>
      <td>1.112401</td>
      <td>-0.141223</td>
      <td>1.898496</td>
      <td>0.922277</td>
    </tr>
    <tr>
      <th>4393</th>
      <td>2009</td>
      <td>Tim Duncan</td>
      <td>C</td>
      <td>32</td>
      <td>SAS</td>
      <td>75</td>
      <td>10.7</td>
      <td>3.5</td>
      <td>0.5</td>
      <td>1.7</td>
      <td>...</td>
      <td>0.549</td>
      <td>100.0</td>
      <td>1.461687</td>
      <td>2.317873</td>
      <td>0.763298</td>
      <td>-0.478615</td>
      <td>1.170702</td>
      <td>-0.268730</td>
      <td>1.640409</td>
      <td>0.943803</td>
    </tr>
    <tr>
      <th>4588</th>
      <td>2010</td>
      <td>Tim Duncan</td>
      <td>C</td>
      <td>33</td>
      <td>SAS</td>
      <td>78</td>
      <td>10.1</td>
      <td>3.2</td>
      <td>0.6</td>
      <td>1.5</td>
      <td>...</td>
      <td>0.560</td>
      <td>101.0</td>
      <td>1.258842</td>
      <td>1.972374</td>
      <td>0.225631</td>
      <td>0.097872</td>
      <td>0.855748</td>
      <td>-0.274936</td>
      <td>0.980815</td>
      <td>0.730907</td>
    </tr>
    <tr>
      <th>4783</th>
      <td>2011</td>
      <td>Tim Duncan</td>
      <td>C</td>
      <td>34</td>
      <td>SAS</td>
      <td>76</td>
      <td>8.9</td>
      <td>2.7</td>
      <td>0.7</td>
      <td>1.9</td>
      <td>...</td>
      <td>0.537</td>
      <td>100.0</td>
      <td>0.416695</td>
      <td>1.414082</td>
      <td>1.065993</td>
      <td>0.181691</td>
      <td>0.561908</td>
      <td>-0.266066</td>
      <td>1.065107</td>
      <td>0.634201</td>
    </tr>
    <tr>
      <th>4969</th>
      <td>2012</td>
      <td>Tim Duncan</td>
      <td>C</td>
      <td>35</td>
      <td>SAS</td>
      <td>58</td>
      <td>9.0</td>
      <td>2.3</td>
      <td>0.7</td>
      <td>1.5</td>
      <td>...</td>
      <td>0.531</td>
      <td>99.0</td>
      <td>0.946940</td>
      <td>1.099411</td>
      <td>0.451227</td>
      <td>-0.069192</td>
      <td>0.536687</td>
      <td>-0.214539</td>
      <td>0.753140</td>
      <td>0.500525</td>
    </tr>
    <tr>
      <th>5142</th>
      <td>2013</td>
      <td>Tim Duncan</td>
      <td>C</td>
      <td>36</td>
      <td>SAS</td>
      <td>69</td>
      <td>9.9</td>
      <td>2.7</td>
      <td>0.7</td>
      <td>2.7</td>
      <td>...</td>
      <td>0.554</td>
      <td>95.0</td>
      <td>1.389275</td>
      <td>1.025729</td>
      <td>2.183548</td>
      <td>-0.134553</td>
      <td>0.619252</td>
      <td>0.174248</td>
      <td>2.236699</td>
      <td>1.070600</td>
    </tr>
    <tr>
      <th>5322</th>
      <td>2014</td>
      <td>Tim Duncan</td>
      <td>C</td>
      <td>37</td>
      <td>SAS</td>
      <td>74</td>
      <td>9.7</td>
      <td>3.0</td>
      <td>0.6</td>
      <td>1.9</td>
      <td>...</td>
      <td>0.535</td>
      <td>98.0</td>
      <td>0.706751</td>
      <td>1.233477</td>
      <td>1.416378</td>
      <td>-0.407244</td>
      <td>0.630963</td>
      <td>-0.262826</td>
      <td>1.691763</td>
      <td>0.715609</td>
    </tr>
    <tr>
      <th>5499</th>
      <td>2015</td>
      <td>Tim Duncan</td>
      <td>C</td>
      <td>38</td>
      <td>SAS</td>
      <td>77</td>
      <td>9.1</td>
      <td>3.0</td>
      <td>0.8</td>
      <td>2.0</td>
      <td>...</td>
      <td>0.560</td>
      <td>97.0</td>
      <td>0.504343</td>
      <td>1.577033</td>
      <td>1.521492</td>
      <td>0.464832</td>
      <td>0.435711</td>
      <td>0.054093</td>
      <td>1.706324</td>
      <td>0.894832</td>
    </tr>
    <tr>
      <th>5685</th>
      <td>2016</td>
      <td>Tim Duncan</td>
      <td>C</td>
      <td>39</td>
      <td>SAS</td>
      <td>61</td>
      <td>7.3</td>
      <td>2.7</td>
      <td>0.8</td>
      <td>1.3</td>
      <td>...</td>
      <td>0.523</td>
      <td>96.0</td>
      <td>-0.691852</td>
      <td>1.064539</td>
      <td>-0.030651</td>
      <td>0.118940</td>
      <td>-0.434970</td>
      <td>-1.056478</td>
      <td>1.976689</td>
      <td>0.135174</td>
    </tr>
  </tbody>
</table>
<p>19 rows × 21 columns</p>
</div>



The first two charts above show the top 50 seasons since 1985 in terms of average z-score and which players are most represented in the top 50 list respectively. A few things to take note of are that the best season to date is Giannis Antetokounpo's 2019 season and the player that shows up most on the top 50 seasons list is Hakeem Olajuwon. On the list of the most represented players are many familiar names to the GOAT conversation; Michael Jordan shows up in second with 7 appearances; LeBron James is tied in 5th place with James Harden with 4 appearances; Shaquille O'Neal has two appearances giving him a 3-way tie with Scottie Pippen and Giannis Antetokounpo. There are also some surprises on this list. Namely Shawn Marion, Paul Milsap, and Draymond Green sneaking on the list with one appearance each. Finally, there are some notable absences from this list. Frequently mentioned all-time greats Magic Johnson and Stephen Curry do not make the top 50 even once. There is only one top 50 season logged by a point guard, Chris Paul in 2009. 5-time champion Kobe Bryant also does not make an appearance. I suspect that the metric favors well-rounded players above all else. Specialists, players with deficiencies, and players who perform about average in some areas are overlooked using this metric.

The ultimate goal of this project is to be able to compare the careers of NBA players who did not play at the same time and may not play the same position. To do this, we need to take the average of all seasons played by each player. The chart below shows the best 10 players of all time according to their average z-score and the graphs display the top 10 players' career trajectory by year, age, and NBA experience. These graphs show a few interesting trends.


```python
temp = starters.drop(columns=["Pos", "Tm"])
topten = temp.groupby("Player").mean().sort_values("AZS", ascending=False).head(10)
topten
```




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
      <th>Year</th>
      <th>Age</th>
      <th>G</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>PTS</th>
      <th>TS%</th>
      <th>DRtg</th>
      <th>PTSZ</th>
      <th>ASTZ</th>
      <th>BLKZ</th>
      <th>STLZ</th>
      <th>TRBZ</th>
      <th>TS%Z</th>
      <th>DRtgZ</th>
      <th>AZS</th>
    </tr>
    <tr>
      <th>Player</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Giannis Antetokounmpo</th>
      <td>2019.000000</td>
      <td>24.000000</td>
      <td>71.333333</td>
      <td>10.411111</td>
      <td>5.111111</td>
      <td>1.177778</td>
      <td>1.288889</td>
      <td>25.077778</td>
      <td>0.604778</td>
      <td>103.777778</td>
      <td>1.744614</td>
      <td>1.523868</td>
      <td>2.124588</td>
      <td>0.739359</td>
      <td>2.666348</td>
      <td>1.166727</td>
      <td>1.888342</td>
      <td>1.693407</td>
    </tr>
    <tr>
      <th>Magic Johnson</th>
      <td>1986.500000</td>
      <td>26.500000</td>
      <td>76.000000</td>
      <td>7.160000</td>
      <td>11.920000</td>
      <td>1.830000</td>
      <td>0.380000</td>
      <td>19.780000</td>
      <td>0.612100</td>
      <td>105.200000</td>
      <td>1.423602</td>
      <td>2.483323</td>
      <td>1.053628</td>
      <td>0.599309</td>
      <td>2.880157</td>
      <td>2.012159</td>
      <td>1.093152</td>
      <td>1.649333</td>
    </tr>
    <tr>
      <th>Michael Jordan</th>
      <td>1993.153846</td>
      <td>29.153846</td>
      <td>79.769231</td>
      <td>6.253846</td>
      <td>5.300000</td>
      <td>2.346154</td>
      <td>0.807692</td>
      <td>30.169231</td>
      <td>0.566615</td>
      <td>102.846154</td>
      <td>2.615980</td>
      <td>1.249304</td>
      <td>1.485442</td>
      <td>1.834788</td>
      <td>1.710245</td>
      <td>0.825819</td>
      <td>1.673023</td>
      <td>1.627800</td>
    </tr>
    <tr>
      <th>James Harden</th>
      <td>2018.785714</td>
      <td>28.785714</td>
      <td>62.928571</td>
      <td>6.671429</td>
      <td>8.914286</td>
      <td>1.542857</td>
      <td>0.614286</td>
      <td>26.664286</td>
      <td>0.607071</td>
      <td>108.642857</td>
      <td>1.897346</td>
      <td>2.466099</td>
      <td>1.283494</td>
      <td>1.239006</td>
      <td>2.028288</td>
      <td>1.331793</td>
      <td>0.898166</td>
      <td>1.592028</td>
    </tr>
    <tr>
      <th>David Robinson</th>
      <td>1996.461538</td>
      <td>30.461538</td>
      <td>75.461538</td>
      <td>10.584615</td>
      <td>2.453846</td>
      <td>1.400000</td>
      <td>2.969231</td>
      <td>20.707692</td>
      <td>0.577769</td>
      <td>95.076923</td>
      <td>1.516678</td>
      <td>1.190398</td>
      <td>1.566084</td>
      <td>1.989557</td>
      <td>1.231306</td>
      <td>1.081073</td>
      <td>2.134115</td>
      <td>1.529888</td>
    </tr>
    <tr>
      <th>LeBron James</th>
      <td>2013.500000</td>
      <td>28.500000</td>
      <td>71.050000</td>
      <td>7.540000</td>
      <td>7.340000</td>
      <td>1.515000</td>
      <td>0.745000</td>
      <td>27.200000</td>
      <td>0.589350</td>
      <td>104.500000</td>
      <td>2.116010</td>
      <td>3.030000</td>
      <td>0.656565</td>
      <td>1.300018</td>
      <td>1.381772</td>
      <td>1.072515</td>
      <td>1.051202</td>
      <td>1.515440</td>
    </tr>
    <tr>
      <th>Hakeem Olajuwon</th>
      <td>1993.500000</td>
      <td>30.500000</td>
      <td>68.777778</td>
      <td>10.838889</td>
      <td>2.405556</td>
      <td>1.716667</td>
      <td>2.994444</td>
      <td>21.005556</td>
      <td>0.545944</td>
      <td>98.000000</td>
      <td>1.534505</td>
      <td>1.079104</td>
      <td>1.548591</td>
      <td>2.873457</td>
      <td>1.301346</td>
      <td>0.218726</td>
      <td>1.697768</td>
      <td>1.464785</td>
    </tr>
    <tr>
      <th>Larry Bird</th>
      <td>1985.750000</td>
      <td>28.750000</td>
      <td>74.250000</td>
      <td>9.975000</td>
      <td>6.400000</td>
      <td>1.700000</td>
      <td>0.841667</td>
      <td>24.133333</td>
      <td>0.562000</td>
      <td>101.583333</td>
      <td>1.426098</td>
      <td>2.850564</td>
      <td>0.407759</td>
      <td>1.777972</td>
      <td>1.805053</td>
      <td>0.429544</td>
      <td>1.553503</td>
      <td>1.464356</td>
    </tr>
    <tr>
      <th>Dwyane Wade</th>
      <td>2010.500000</td>
      <td>28.500000</td>
      <td>65.357143</td>
      <td>4.742857</td>
      <td>5.614286</td>
      <td>1.642857</td>
      <td>0.864286</td>
      <td>23.064286</td>
      <td>0.556714</td>
      <td>104.214286</td>
      <td>1.527151</td>
      <td>1.642880</td>
      <td>2.679780</td>
      <td>1.528820</td>
      <td>1.056177</td>
      <td>0.487822</td>
      <td>1.129448</td>
      <td>1.436011</td>
    </tr>
    <tr>
      <th>Kevin Durant</th>
      <td>2015.687500</td>
      <td>26.687500</td>
      <td>64.062500</td>
      <td>7.050000</td>
      <td>4.487500</td>
      <td>1.018750</td>
      <td>1.156250</td>
      <td>27.387500</td>
      <td>0.626437</td>
      <td>107.000000</td>
      <td>2.316395</td>
      <td>1.357493</td>
      <td>2.003975</td>
      <td>0.157210</td>
      <td>1.159962</td>
      <td>1.908261</td>
      <td>0.629483</td>
      <td>1.361825</td>
    </tr>
  </tbody>
</table>
</div>



The chart above shows the top 10 players in the three-point era (since 1980) by average z-score. On the list are a lot of your typical names like Michael Jordan, LeBron James, Magic Johnson, and Larry Bird. But there are also some surprising names like Giannis Antetokounpo ranking as the best player in the three-point era. 


```python
fig, ax = plt.subplots(figsize=(12,8))
for entry in topten.iterrows():
    player = starters[starters["Player"] == entry[0]]
    line, = plt.plot(player["Year"], player["AZS"], '-', label=entry[0])
plt.xlabel('Year')
plt.ylabel("Average Z-Score")
plt.title('Top 5 NBA Players Average Z-Score Over The Years')
plt.grid(True)
plt.legend()
plt.show()
```


    
![png](GOATdebate_files/GOATdebate_20_0.png)
    


This graph tracks the top 10 players from the start of the three-point era to today. There are two interesting things that this graph shows. The first is that basketball around the turn of the millennium was about to lose some of its biggest stars: David Robinson, Michael Jordan, and Hakeem Olajuwon. They'd been declining for the last few years and would retire in 2002 and 2003. Luckily for the NBA, the 2003 draft is argued to be the greatest draft class ever giving rise to two top 10 players, Dwyane Wade and LeBron James. In the decades to come, they would be joined by several more stars that have defined a generation of basketball, but in 2024, the basketball landscape looks similar to how it did in the early 2000s. The stars that had dominated the past 20 years or so of basketball are all in their twilight years, except Giannis Antetokounpo, who is still just 29 years old. Giannis should still have a few more years of performing at a high level left, but time will tell if any younger players can make their way onto this list.


```python
starters[starters["Player"] == "Kevin Durant"]
```




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
      <th>Year</th>
      <th>Player</th>
      <th>Pos</th>
      <th>Age</th>
      <th>Tm</th>
      <th>G</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>...</th>
      <th>DRtg</th>
      <th>PTSZ</th>
      <th>ASTZ</th>
      <th>BLKZ</th>
      <th>STLZ</th>
      <th>TRBZ</th>
      <th>TS%Z</th>
      <th>DRtgZ</th>
      <th>AZS</th>
      <th>Experience</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4202</th>
      <td>2008</td>
      <td>Kevin Durant</td>
      <td>SG</td>
      <td>19</td>
      <td>SEA</td>
      <td>80</td>
      <td>4.4</td>
      <td>2.4</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>...</td>
      <td>110.0</td>
      <td>0.726914</td>
      <td>-0.500826</td>
      <td>3.328921</td>
      <td>-0.105099</td>
      <td>6.264179e-01</td>
      <td>-0.397289</td>
      <td>-0.343978</td>
      <td>0.476437</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4394</th>
      <td>2009</td>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>20</td>
      <td>OKC</td>
      <td>74</td>
      <td>6.5</td>
      <td>2.8</td>
      <td>1.3</td>
      <td>0.7</td>
      <td>...</td>
      <td>109.0</td>
      <td>1.892262</td>
      <td>0.257669</td>
      <td>0.450794</td>
      <td>0.821037</td>
      <td>9.485695e-01</td>
      <td>0.958064</td>
      <td>0.029518</td>
      <td>0.765416</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4589</th>
      <td>2010</td>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>21</td>
      <td>OKC</td>
      <td>82</td>
      <td>7.6</td>
      <td>2.8</td>
      <td>1.4</td>
      <td>1.0</td>
      <td>...</td>
      <td>104.0</td>
      <td>2.688725</td>
      <td>0.325481</td>
      <td>1.645906</td>
      <td>0.888919</td>
      <td>1.607031e+00</td>
      <td>1.677027</td>
      <td>1.041875</td>
      <td>1.410709</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4785</th>
      <td>2011</td>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>22</td>
      <td>OKC</td>
      <td>78</td>
      <td>6.8</td>
      <td>2.7</td>
      <td>1.1</td>
      <td>1.0</td>
      <td>...</td>
      <td>107.0</td>
      <td>2.292525</td>
      <td>0.097138</td>
      <td>1.025324</td>
      <td>0.295769</td>
      <td>1.181573e+00</td>
      <td>1.427475</td>
      <td>0.208168</td>
      <td>0.932568</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4970</th>
      <td>2012</td>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>23</td>
      <td>OKC</td>
      <td>66</td>
      <td>8.0</td>
      <td>3.5</td>
      <td>1.3</td>
      <td>1.2</td>
      <td>...</td>
      <td>101.0</td>
      <td>2.846449</td>
      <td>0.893345</td>
      <td>2.455426</td>
      <td>0.717662</td>
      <td>2.228302e+00</td>
      <td>2.100928</td>
      <td>1.014929</td>
      <td>1.751006</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5143</th>
      <td>2013</td>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>24</td>
      <td>OKC</td>
      <td>81</td>
      <td>7.9</td>
      <td>4.6</td>
      <td>1.4</td>
      <td>1.3</td>
      <td>...</td>
      <td>100.0</td>
      <td>3.185754</td>
      <td>1.922241</td>
      <td>2.788581</td>
      <td>1.040959</td>
      <td>2.071473e+00</td>
      <td>2.332590</td>
      <td>1.625057</td>
      <td>2.138093</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>5324</th>
      <td>2014</td>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>25</td>
      <td>OKC</td>
      <td>81</td>
      <td>7.4</td>
      <td>5.5</td>
      <td>1.3</td>
      <td>0.7</td>
      <td>...</td>
      <td>104.0</td>
      <td>3.331636</td>
      <td>2.588583</td>
      <td>0.972864</td>
      <td>0.632097</td>
      <td>1.680617e+00</td>
      <td>2.463552</td>
      <td>0.850874</td>
      <td>1.788603</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>5501</th>
      <td>2015</td>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>26</td>
      <td>OKC</td>
      <td>27</td>
      <td>6.6</td>
      <td>4.1</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>...</td>
      <td>105.0</td>
      <td>2.415218</td>
      <td>1.267848</td>
      <td>2.188121</td>
      <td>-0.106336</td>
      <td>1.306896e+00</td>
      <td>2.513154</td>
      <td>0.429500</td>
      <td>1.430629</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>5687</th>
      <td>2016</td>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>27</td>
      <td>OKC</td>
      <td>72</td>
      <td>8.2</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>...</td>
      <td>104.0</td>
      <td>2.397269</td>
      <td>1.719524</td>
      <td>3.039459</td>
      <td>-0.076898</td>
      <td>2.050499e+00</td>
      <td>2.438464</td>
      <td>0.876759</td>
      <td>1.777868</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>5869</th>
      <td>2017</td>
      <td>Kevin Durant</td>
      <td>PF</td>
      <td>28</td>
      <td>GSW</td>
      <td>62</td>
      <td>8.3</td>
      <td>4.8</td>
      <td>1.1</td>
      <td>1.6</td>
      <td>...</td>
      <td>101.0</td>
      <td>2.418286</td>
      <td>2.142968</td>
      <td>1.845411</td>
      <td>0.956850</td>
      <td>1.167389e+00</td>
      <td>2.821684</td>
      <td>2.406688</td>
      <td>1.965611</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>6048</th>
      <td>2018</td>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>29</td>
      <td>GSW</td>
      <td>68</td>
      <td>6.8</td>
      <td>5.4</td>
      <td>0.7</td>
      <td>1.8</td>
      <td>...</td>
      <td>107.0</td>
      <td>2.802401</td>
      <td>2.407672</td>
      <td>3.764864</td>
      <td>-0.917405</td>
      <td>2.136200e+00</td>
      <td>2.445771</td>
      <td>0.615098</td>
      <td>1.893514</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>6228</th>
      <td>2019</td>
      <td>Kevin Durant</td>
      <td>PF</td>
      <td>30</td>
      <td>GSW</td>
      <td>78</td>
      <td>6.4</td>
      <td>5.9</td>
      <td>0.7</td>
      <td>1.1</td>
      <td>...</td>
      <td>110.0</td>
      <td>2.129183</td>
      <td>2.707217</td>
      <td>1.509145</td>
      <td>-0.242165</td>
      <td>4.370969e-16</td>
      <td>1.911774</td>
      <td>0.167429</td>
      <td>1.168940</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>6610</th>
      <td>2021</td>
      <td>Kevin Durant</td>
      <td>PF</td>
      <td>32</td>
      <td>BRK</td>
      <td>35</td>
      <td>7.1</td>
      <td>5.6</td>
      <td>0.7</td>
      <td>1.3</td>
      <td>...</td>
      <td>112.0</td>
      <td>2.044600</td>
      <td>1.442052</td>
      <td>1.910822</td>
      <td>-0.539102</td>
      <td>5.290162e-01</td>
      <td>1.953191</td>
      <td>0.123767</td>
      <td>1.066335</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>6811</th>
      <td>2022</td>
      <td>Kevin Durant</td>
      <td>PF</td>
      <td>33</td>
      <td>BRK</td>
      <td>55</td>
      <td>7.4</td>
      <td>6.4</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>...</td>
      <td>112.0</td>
      <td>2.363009</td>
      <td>2.121725</td>
      <td>0.413518</td>
      <td>-0.054504</td>
      <td>4.414068e-01</td>
      <td>1.701911</td>
      <td>-0.154111</td>
      <td>0.976137</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>6999</th>
      <td>2023</td>
      <td>Kevin Durant</td>
      <td>PF</td>
      <td>34</td>
      <td>TOT</td>
      <td>47</td>
      <td>6.7</td>
      <td>5.0</td>
      <td>0.7</td>
      <td>1.4</td>
      <td>...</td>
      <td>113.0</td>
      <td>1.723997</td>
      <td>1.078915</td>
      <td>2.222839</td>
      <td>-0.574173</td>
      <td>2.920029e-01</td>
      <td>2.139730</td>
      <td>0.590074</td>
      <td>1.067627</td>
      <td>15.5</td>
    </tr>
    <tr>
      <th>7000</th>
      <td>2023</td>
      <td>Kevin Durant</td>
      <td>PF</td>
      <td>34</td>
      <td>BRK</td>
      <td>39</td>
      <td>6.7</td>
      <td>5.3</td>
      <td>0.8</td>
      <td>1.5</td>
      <td>...</td>
      <td>113.0</td>
      <td>1.804085</td>
      <td>1.248331</td>
      <td>2.501611</td>
      <td>-0.222260</td>
      <td>2.920029e-01</td>
      <td>2.044156</td>
      <td>0.590074</td>
      <td>1.179714</td>
      <td>15.5</td>
    </tr>
  </tbody>
</table>
<p>16 rows × 22 columns</p>
</div>




```python
starters = starters.drop(index=7000) #deleting Kevin Durant's 39 game stint with BKN because it is included in his total stats for the year
fig, ax = plt.subplots(figsize=(12,8))
for entry in topten.iterrows():
    player = starters[starters["Player"] == entry[0]]
    line, = plt.plot(player["Age"], player["AZS"], '-', label=entry[0])
plt.legend()
plt.xlabel('Age')
plt.ylabel("Average Z-Score")
plt.title('Top 5 NBA Players Average Z-Score By Age')
```




    Text(0.5, 1.0, 'Top 5 NBA Players Average Z-Score By Age')




    
![png](GOATdebate_files/GOATdebate_23_1.png)
    



```python
starters["Experience"] = starters.groupby("Player")["Year"].rank()
```


```python
fig, ax = plt.subplots(figsize=(12,8))
for entry in topten.iterrows():
    player = starters[starters["Player"] == entry[0]]
    line, = plt.plot(player["Experience"], player["AZS"], '-', label=entry[0])
plt.legend()
plt.xlabel('Experience')
plt.ylabel("Average Z-Score")
plt.title('Top 5 NBA Players Average Z-Score By Experience')
```




    Text(0.5, 1.0, 'Top 5 NBA Players Average Z-Score By Experience')




    
![png](GOATdebate_files/GOATdebate_25_1.png)
    


These two graphs are interesting because they show that the NBA's greatest players all have a similar career trajectory, peaking somewhere between their 5th and 8th season, or in their mid to late 20s, before declining in the latter half of their career. For some, like Dwyane Wade and Hakeem Olajuwon, this decline can come fast. For others, like LeBron James, they can have long-lived careers while still being more productive than their peers.


```python
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta
```

I thought the best way to view the Z-score data would be to create spider graphs for each of the top 10 players. The spider graphs give a sense of how these players achieved their average z-score and allow you to compare the variables between players more easily. 


```python
N = 7
theta = radar_factory(N, frame='polygon')
data = []
data.append(["PTSZ", "ASTZ", "BLKZ", "STLZ", "TRBZ", "TS%Z", "DRtgZ"])
for entry in topten.iterrows():
    data.append((entry[0], entry[1].loc["PTSZ":"DRtgZ"].values))
spoke_labels = data.pop(0)

fig, axs = plt.subplots(2, 5, figsize=(12, 6), subplot_kw=dict(projection='radar'))
count = 0
for (title, case_data), ax in zip(data, axs.flatten()):
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                    horizontalalignment='center', verticalalignment='center')
    ax.plot(theta, case_data)
    ax.fill(theta, case_data, alpha=0.25, label='_nolegend_')
    ax.set_varlabels(spoke_labels)
    ax.set_ylim(0, 3)
    count+=1
plt.tight_layout()
fig.suptitle("Spider Graphs for Top 5 NBA Players")
plt.show()
```


    
![png](GOATdebate_files/GOATdebate_29_0.png)
    



```python
#Extract MVP and All-Star Game Status from Basketball-Reference
asg = pd.DataFrame()
mvp = pd.DataFrame(columns=["Player", "MVP", "Year"])
for year in np.arange(1985, 2023):
    req = requests.get(f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html")
    if (req.status_code != 200):
        print(str(req.status_code) + f" on https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html")
    soup = bs(req.content)
    y = soup.find(string="Most Valuable Player")
    mvp.loc[len(mvp.index)] = [y.parent.parent.find('a').text, 1, year]
    print(y.parent.parent.find('a').text)
    time.sleep(4)
    req = requests.get(f"https://www.basketball-reference.com/allstar/NBA_{year}.html")
    if (req.status_code != 200):
        print(str(req.status_code) + f" on https://www.basketball-reference.com/allstar/NBA_{year}.html")
    soup = bs(req.content)
    tables = soup.findAll('table')
    if tables:
        result = pd.read_html(str(tables[0]))[0]
        team1 = pd.read_html(str(tables[1]))[0]
        team2 = pd.read_html(str(tables[2]))[0]
        df = pd.concat([team1, team2], ignore_index=True)
        df["Year"] = year
        if asg.shape[0] == 0:
            asg = df
        else:
            asg = pd.concat([asg, df], ignore_index=True)
        #print(result.iloc[0][('Unnamed: 0_level_0', 'Unnamed: 0_level_1')] + ": " + str(result.iloc[0][('Scoring', 'T')]) + "\n" +
        #result.iloc[1][('Unnamed: 0_level_0', 'Unnamed: 0_level_1')] + ": " + str(result.iloc[1][('Scoring', 'T')]))
    time.sleep(4)
asg = asg[(asg["Totals", "FG"] != "FG") & (asg["Totals", "FG"] != "Totals") & (asg['Unnamed: 0_level_0', 'Starters'] != "Team Totals")]
asg['ASG'] = 1
asg = asg.drop(columns=df.columns.difference([('ASG', ''), ('Unnamed: 0_level_0', 'Starters'), ('Year', '')]))
asg.columns = ["Player", "Year", "ASG"]
starters = starters.merge(mvp, "left")
starters = starters.merge(asg, "left")
starters["ASG"] = starters["ASG"].fillna(0)
starters["MVP"] = starters["MVP"].fillna(0)

#
x = starters.loc[:, "PTSZ":"DRtgZ"]
y1 = starters["ASG"]
y2 = starters["MVP"]
```

    Larry Bird
    Larry Bird
    Magic Johnson
    Michael Jordan
    Magic Johnson
    Magic Johnson
    Michael Jordan
    Michael Jordan
    Charles Barkley
    Hakeem Olajuwon
    David Robinson
    Michael Jordan
    Karl Malone
    Michael Jordan
    Karl Malone
    404 on https://www.basketball-reference.com/allstar/NBA_1999.html
    Shaquille O'Neal
    Allen Iverson
    Tim Duncan
    Tim Duncan
    Kevin Garnett
    Steve Nash
    Steve Nash
    Dirk Nowitzki
    Kobe Bryant
    LeBron James
    LeBron James
    Derrick Rose
    LeBron James
    LeBron James
    Kevin Durant
    Stephen Curry
    Stephen Curry
    Russell Westbrook
    James Harden
    Giannis Antetokounmpo
    Giannis Antetokounmpo
    Nikola Jokić
    Nikola Jokić


For the last part of this project, I wanted to see if these z-scores I've calculated can be used to classify seasons as either All-Star or MVP caliber. To do this, I first started with a logistic regression, which is useful for binary classifications like we are dealing with. I created two logistic regressions, one for the All-Stars and one for the MVP. 20% of the data was randomly selected and set aside for testing.


```python
X_train, X_test, y_train, y_test = train_test_split(x, y1, test_size=0.20)
logreg = LogisticRegression(random_state=16)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
conf = metrics.confusion_matrix(y_test, y_pred)
print(conf)

target_names = ['Not an All-Star', 'All-Star']
print(classification_report(y_test, y_pred, target_names=target_names))
```

    [[1109   33]
     [  85  108]]
                     precision    recall  f1-score   support
    
    Not an All-Star       0.93      0.97      0.95      1142
           All-Star       0.77      0.56      0.65       193
    
           accuracy                           0.91      1335
          macro avg       0.85      0.77      0.80      1335
       weighted avg       0.91      0.91      0.91      1335
    



```python
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
```


    
![png](GOATdebate_files/GOATdebate_33_0.png)
    


The data above is for the All-Star regression. I first want to detail a few definitions that will be useful in assessing the performance of the models. Precision is a measure of the quality of a model's predictions. In other words, if the model says a player is an All-Star, how often is the model correct? This metric exists for both the positive (All-Star) and negative (Not an All-Star) predictions. Recall is the percentage of observations belonging to each class that the model gets correct. In other words, of the All-Stars, how many did the model predict as All-Stars? The f1-score is the harmonic mean of the precision and recall of the model. It provides another measure of the model's accuracy. These metrics are provided by the classification report.

Overall, the model is very accurate, with an accuracy of 91% per the classification report, but this does not tell the whole story. The model is very good at handling non-All-Stars, with a precision and specificity of 0.93 and 0.97 respectively, thus it has a very high f1-score of 0.95. The model does not handle All-Star observations as well with significantly lower precision and recall scores of 0.77 and 0.56. The f1-score for All-Stars is 0.65, which while not great, is still acceptable. I suspect that these results arise from the fact that only around 15% of the players in the data set are all-stars.

Another metric is the area under the receiver operating characteristic curve or AUC-ROC. The ROC, or receiver operating characteristic plots the relationship between false positive and true positive rate. The area under this curve is 0.5 for a classifier that randomly guesses and 1 for a perfect classifier. I suspect that this metric is high because of the scarcity of All-Stars in the dataset.


```python
X_train, X_test, y_train, y_test = train_test_split(x, y2, test_size=0.20)
logreg = LogisticRegression(random_state=16)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

target_names = ['Not an MVP', 'MVP']
print(classification_report(y_test, y_pred, target_names=target_names))
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
```

    [[1325    1]
     [   7    2]]
                  precision    recall  f1-score   support
    
      Not an MVP       0.99      1.00      1.00      1326
             MVP       0.67      0.22      0.33         9
    
        accuracy                           0.99      1335
       macro avg       0.83      0.61      0.67      1335
    weighted avg       0.99      0.99      0.99      1335
    



    
![png](GOATdebate_files/GOATdebate_35_1.png)
    


The data for the MVP logistic regression looks similar to the All-Star regression: very good f1-scores for non-MVPs and a high AUC score, but this time the model handles MVP observations even worse. I suspect this is because there are even fewer MVPs in the league than there are all-stars. The model's f1-score of 0.33 means the model performs worse than random guessing for MVPs.

After using the logistic regression for binary classification, I wanted to see if I could do any better with K-Nearest Neighbors classification. In this type of classification, predictions are determined by the labels of the nearest k, in this case 5, neighbors. Distance for this classifier is determined by normal Euclidean distance.


```python
X_train, X_test, y_train, y_test = train_test_split(x, y1, test_size=0.20)
classifier = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
target_names = ["Not and All-Star", "All-Star"]
print(classification_report(y_test, y_pred, target_names=target_names))
```

    [[1093   70]
     [  80   92]]
                      precision    recall  f1-score   support
    
    Not and All-Star       0.93      0.94      0.94      1163
            All-Star       0.57      0.53      0.55       172
    
            accuracy                           0.89      1335
           macro avg       0.75      0.74      0.74      1335
        weighted avg       0.88      0.89      0.89      1335
    



```python
X_train, X_test, y_train, y_test = train_test_split(x, y2, test_size=0.20)
classifier = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
target_names = ["Not and MVP", "MVP"]
print(classification_report(y_test, y_pred, target_names=target_names))
```

    [[1325    2]
     [   7    1]]
                  precision    recall  f1-score   support
    
     Not and MVP       0.99      1.00      1.00      1327
             MVP       0.33      0.12      0.18         8
    
        accuracy                           0.99      1335
       macro avg       0.66      0.56      0.59      1335
    weighted avg       0.99      0.99      0.99      1335
    


In both cases, the KNN classifiers performed worse than the logistic regression. These classifiers probably performed worse because of the scarcity of positive observations in both cases. MVPs do not significantly outperform other players enough so that they are isolated in space and the same goes for MVPs, so the likelihood of being surrounded by a non-MVP or non-All-Star is high just due to the negative observations outnumbering the positive ones.

Takeaways and Future Directions

The main takeaway from this project is that it is hard to get a single metric to accurately capture all of an NBA's productivity and impact. If my metric alone, average Z-score was used to determine who the greatest NBA players are of all time, the top 5 list would be Giannis Antetokounpo, Michael Jordan, James Harden, David Robinson, and LeBron James in that order. Not only is this not a conventional top 5 list, but the stats used in the calculation have a great effect on who comes out, which may introduce some bias. In the future, I would switch the statistics used in the calculation of the average z-score, perhaps relying less on traditional counting stats and more on advanced statistics and rate stats which can take into account a player's usage to see what effect that has on the outcome.

At the end of the project, the goal was to see if the Z-scores of NBA players could be used to predict whether that player was an All-Star or not and if it could predict if the player was an MVP or not. I was able to achieve a useful classifier using the logistic regression approach for binary classification for predicting if a player was an All-Star, but not if the player was an MVP. The KNN approach as the classifier didn't turn up any better results for either case. In the future, I would like to see if a team's starting 5 z-score information can be used to predict how many wins that team gets either through a linear regression or a neural network.
