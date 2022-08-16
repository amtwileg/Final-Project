import pandas_datareader as pdr

df = pdr.get_data_yahoo('GOOG')
df.head()
df.tail()
df.describe()
