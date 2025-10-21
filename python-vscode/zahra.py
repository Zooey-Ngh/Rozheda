import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
print('matplotlib version: ',mpl.__version__)

df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')
print('data read into a panda dataframe')
# print(df_can.head())
print(df_can.shape)
df_can.set_index('Country',inplace=True)
print(df_can.head())
years = list(map(str,range(1980,2014)))

df_continent = df_can.groupby('Continent', axis = 0).sum()
print(df_continent.head())

# df_continent['Total'].plot(kind = 'pie', \
# figsize=(5,6), autopct = '%1.1f%%', startangle = 90,shadow =True)
# plt.title('Immigration to Canada by Continent [1980-2014]')
# plt.axis('equal')
# plt.legend(labels = df_continent.index, loc = 'upper left')
# plt.show()

# color_list = ['gold','yellowgreen','lightcoral','lightskyblue','lightgreen','pink']
# explode_list = [0.1,0,0,0.1,0.1]

# df_continent['Total'].plot(kind = 'pie', figsize=(10,6),autopct = '%1.1f%%',startangle = 90,shadow = True,
#                        labels = None,pctdistance = 1.12)    
# plt.title('Immigration to Canada by Continent [1980-2014]', y=1.12, fontsize = 15)
# plt.axis('equal')
# plt.legend(labels = df_continent.index, loc = 'upper left', fontsize = 7)
# plt.show()

df_japan = df_can.loc[['Japan'],years].transpose()
print(df_japan.head())
# df_japan.plot(kind= 'box', figsize=(5,6))
# plt.title('Box plot of Japanese Immigrants from 1980-2014')
# plt.ylabel('Number of Immigrants')
# plt.show()
df_japan.describe()
df_CI = df_can.loc[['China', 'India'], years].transpose()
print(df_CI.head())

# df_CI.plot(kind = 'box', figsize=(10,7))
# plt.title('Box plots of Immigrants from China and India 1980-2014')
# plt.ylabel('Number of Immigrants')
# plt.show()
# 
df_top15 =df_can.sort_values(['Total'], ascending=False, axis=0).head(15)
print(df_top15)
years_80s = list(map(str, range (1980,1990)))
years_90s = list(map(str,range(1990,2000)))
years_00s= list(map(str, range(2000,2010)))
df_80s = df_can.loc[:, years_80s].sum(axis=1)
df_90s = df_can.loc[:,years_90s].sum(axis=1)
df_00s = df_can.loc[:,years_00s].sum(axis=1)
new_df = pd.DataFrame({'1980s': df_80s, '1990s':df_90s, '2000s':df_00s})
print(new_df.head())

# new_df.plot(kind='box', figsize=(15,8))
# plt.title('Immigrants from top 15 countries for decades 80,90,2000')
# plt.show()
print(new_df.describe())

# new_df.plot(kind='box', figsize=(10,6))
# plt.title('Immigrants from 15countries for decades 80s,90s,2000s')
# plt.show()
new_df = new_df.reset_index()
print(new_df[new_df['2000s']> 209611.5])

df_tot = pd.DataFrame(df_can[years].sum(axis=0))
df_tot.index = map(int,df_tot.index)
df_tot.reset_index(inplace=True)
df_tot.columns = ['year', 'total']
print(df_tot.head())

# df_tot.plot(kind='scatter', x = 'year', y = 'total', figsize=(10,6), color = 'darkblue')
# plt.title('Total Immigration to Canada from 1980 -2013')
# plt.xlabel('year')
# plt.ylabel('Number of Immigrants')
# plt.show()
x = df_tot['year']
y = df_tot['total']
fit = np.polyfit(x,y,deg = 1)
print(fit)