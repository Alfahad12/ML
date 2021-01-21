# ML

Alfahad Mallick
Task #1
Prediction using supervised ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df2="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df3=pd.read_csv(df2)
df3
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25
10	7.7	85
11	5.9	62
12	4.5	41
13	3.3	42
14	1.1	17
15	8.9	95
16	2.5	30
17	1.9	24
18	6.1	67
19	7.4	69
20	2.7	30
21	4.8	54
22	3.8	35
23	6.9	76
24	7.8	86
df3.head()
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
df3.tail()
Hours	Scores
20	2.7	30
21	4.8	54
22	3.8	35
23	6.9	76
24	7.8	86
df3.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25 entries, 0 to 24
Data columns (total 2 columns):
Hours     25 non-null float64
Scores    25 non-null int64
dtypes: float64(1), int64(1)
memory usage: 464.0 bytes
df3.describe()
Hours	Scores
count	25.000000	25.000000
mean	5.012000	51.480000
std	2.525094	25.286887
min	1.100000	17.000000
25%	2.700000	30.000000
50%	4.800000	47.000000
75%	7.400000	75.000000
max	9.200000	95.000000
df3.isnull().sum()
Hours     0
Scores    0
dtype: int64
plt.scatter(df3['Hours'],df3['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours vs Score')
Text(0.5, 1.0, 'Hours vs Score')

x=df3.iloc[:,:-1].values
y=df3.iloc[:,1].values
3
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
regr
regr=LinearRegression()
regr
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
x_train,y_train
regr.fit(x_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
l=regr.coef_*x+regr.intercept_
l
array([[26.8422321 ],
       [52.29250548],
       [33.69422878],
       [85.57363222],
       [36.63079879],
       [17.05366541],
       [92.4256289 ],
       [56.20793216],
       [83.61591888],
       [28.79994544],
       [77.74277887],
       [60.12335883],
       [46.41936547],
       [34.67308545],
       [13.13823874],
       [89.48905889],
       [26.8422321 ],
       [20.96909209],
       [62.08107217],
       [74.80620886],
       [28.79994544],
       [49.35593548],
       [39.56736879],
       [69.91192552],
       [78.72163554]])
plt.scatter(x,y)
<matplotlib.collections.PathCollection at 0x5253e50>

o
plt.plot(x,l)
plt.show()

plt.scatter(x,y)
plt.plot(x,l)
plt.show()

regr.intercept_
2.370815382341881
)
print(regr.coef_)
[9.78856669]
y_predct
y_predct=regr.predict(x_test)
y_predct
array([17.05366541, 33.69422878, 74.80620886, 26.8422321 , 60.12335883,
       39.56736879, 20.96909209, 78.72163554])
:
pd.DataFrame({'Actual':y_test,'Predicted':y_predct})
Actual	Predicted
0	20	17.053665
1	27	33.694229
2	69	74.806209
3	30	26.842232
4	62	60.123359
5	35	39.567369
6	24	20.969092
7	86	78.721636
plt.scatter(x_test,y_test)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Actual values')
plt.scatter(x_test,y_test)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Actual values')
Text(0.5, 1.0, 'Actual values')

Predicted
plt.scatter(x_test,y_predct)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Predicted values')
Text(0.5, 1.0, 'Predicted values')

)
print(regr.predict([[9.25]]))
[92.91505723]
The predicted score if a student studies 9.25 hrs/day is 92.91505723
------------------------------------------THANK YOU-------------------------------------------------
​
