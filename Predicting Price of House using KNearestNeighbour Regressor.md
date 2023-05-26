<font color=red>Import Pandas


```python
import pandas as pd
```

<font color=red>Import 'house_rental_data.csv.txt'


```python
house_rental_data = pd.read_csv('house_rental_data.csv.txt')
house_rental_data
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
      <th>Unnamed: 0</th>
      <th>Sqft</th>
      <th>Floor</th>
      <th>TotalFloor</th>
      <th>Bedroom</th>
      <th>Living.Room</th>
      <th>Bathroom</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1177.698</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>62000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2134.800</td>
      <td>5</td>
      <td>7</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>78000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1138.560</td>
      <td>5</td>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>58000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1458.780</td>
      <td>2</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>967.776</td>
      <td>11</td>
      <td>14</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>45000</td>
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
    </tr>
    <tr>
      <th>640</th>
      <td>644</td>
      <td>1359.156</td>
      <td>7</td>
      <td>15</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>641</th>
      <td>645</td>
      <td>377.148</td>
      <td>4</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>24800</td>
    </tr>
    <tr>
      <th>642</th>
      <td>646</td>
      <td>740.064</td>
      <td>13</td>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>643</th>
      <td>647</td>
      <td>1707.840</td>
      <td>3</td>
      <td>14</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>65000</td>
    </tr>
    <tr>
      <th>644</th>
      <td>648</td>
      <td>1376.946</td>
      <td>6</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>36000</td>
    </tr>
  </tbody>
</table>
<p>645 rows × 8 columns</p>
</div>



<font color=red>Remove 'Unnamed' column


```python
house_rental_data = house_rental_data.drop(house_rental_data.columns[0], axis = 1)
house_rental_data
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
      <th>Floor</th>
      <th>TotalFloor</th>
      <th>Bedroom</th>
      <th>Living.Room</th>
      <th>Bathroom</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>62000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>7</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>78000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>58000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>14</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>640</th>
      <td>7</td>
      <td>15</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>641</th>
      <td>4</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>24800</td>
    </tr>
    <tr>
      <th>642</th>
      <td>13</td>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>643</th>
      <td>3</td>
      <td>14</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>65000</td>
    </tr>
    <tr>
      <th>644</th>
      <td>6</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>36000</td>
    </tr>
  </tbody>
</table>
<p>645 rows × 6 columns</p>
</div>



<font color=red>Find 'Number of rows and columns'


```python
house_rental_data.shape
```




    (645, 6)



<font color=red>Data Preparation and Train-Test Split for Machine Learning

Data Splitting: Feature and Target Variable Separation
<br>Drop 'Price' Column and save as variable x and y
<br><font color=green>Press only one time shift+Enter, function will run otherwise it will through error if we press more that one.


```python
x = house_rental_data.drop(['Price'], axis=1)
y = house_rental_data['Price']
```

Splitting Data into Training and Testing Sets using scikit-learn's train_test_split
<br><font color=green>Note: The test_size parameter is set to 0.2, which means that 20% of the data will be allocated to the testing set, while the remaining 80% will be used for training. 
<br>The random_state parameter is set to 0 to ensure reproducibility of the split.


```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
```

Find 'Number of rows' and 'Number of columns' x_test


```python
x_test.shape
```




    (129, 5)



Training a K-Nearest Neighbors Regressor Model


```python
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(x_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsRegressor</label><div class="sk-toggleable__content"><pre>KNeighborsRegressor()</pre></div></div></div></div></div>



Generating Predictions using the K-Nearest Neighbors Regressor Model


```python
y_pred = knr.predict(x_test)
y_pred
```




    array([ 83777.6,  35800. ,  55000. ,  44999.8,  39600. ,  84777.6,
            79400. ,  42599.8,  37600. ,  31000. ,  44999.8,  30600. ,
            46600. ,  49600. ,  63400. ,  67200. ,  62555. ,  36400. ,
            51100. ,  53377.6,  35600. , 119000. ,  74400. , 151600. ,
            51100. ,  61400. , 106000. ,  38000. ,  86600. ,  57799.8,
            58200. ,  36400. ,  52000. ,  84777.6,  37600. ,  26720. ,
            80400. ,  67200. ,  73400. ,  57200. ,  45599.8,  80400. ,
            58917.6,  51100. ,  98999.8,  45424. ,  54800. ,  32920. ,
            87000. ,  58600. , 151800. ,  55400. ,  30600. ,  65400. ,
            47799.8,  67200. ,  51500. ,  49600. ,  76800. ,  84600. ,
            89600. ,  53377.6,  43378. ,  22600. ,  90800. ,  66000. ,
            47000. ,  35560. ,  31000. ,  38460. ,  43880. ,  80599.8,
            44799.8,  58600. ,  84777.6,  28320. ,  49199.8,  73999.8,
            85400. ,  51800. ,  41812.8,  41000. ,  76199.8,  43880. ,
            57200. ,  64000. ,  52200. ,  52199.8,  79399.6,  22480. ,
            61400. ,  74400. ,  55400. ,  43000. ,  51100. ,  39199.8,
            58917.6,  79400. ,  46800. ,  56600. ,  34559.8,  66400. ,
            46199.8,  76800. ,  84777.6,  38260. ,  32260. ,  95400. ,
            56500. ,  47800. ,  31760. ,  67000. ,  63077.6,  51100. ,
            92800. ,  51100. ,  55000. ,  68600. ,  77399.8,  68999.8,
            25000. ,  67200. ,  37600. ,  49519.8,  42560. ,  53200. ,
            76000. ,  39260. ,  44999.8])



Making Single Data Point Prediction using the K-Nearest Neighbors Regressor Model 
<br>Also import numpy


```python
import numpy as np
input_data = (2,1,3,5,4)
convert_to_array = np.asarray(input_data)
re_shape = convert_to_array.reshape(1,-1)
prediction = knr.predict(re_shape)
print(prediction)
```

    [62600.]
    

    C:\ProgramData\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but KNeighborsRegressor was fitted with feature names
      warnings.warn(
    

Assessing the Accuracy of the K-Nearest Neighbors Regressor Model with the score() Function


```python
knr.score(x_test, y_test)
```




    0.3768979233327673


