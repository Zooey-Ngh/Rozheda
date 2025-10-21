
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px 

mpl.style.use('ggplot')
print('Matplotlib verssion:',mpl.__version__)

df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')
print('Data read into a pandas dataframe!')


print(df_can.shape)

df_can.set_index('Country',inplace=True)

years = list(map(str,range(1980,2014)))

df_total = df_can[years].sum(axis=0)
# print(df_total.head())
# fig,ax = plt.subplots()
# ax.plot(df_total)

# ax.set_title('Immigrants between 1980 to 2013')
# ax.set_xlabel('years')
# ax.set_ylabel('Total Immigrants')
# plt.show()

# # fig,ax = plt.subplots()
# # df_total.index = df_total.index.map(int)

# # ax.plot(df_total)
# # plt.show()
# df_can.reset_index(inplace= True)
# haiti = df_can[df_can['Country']=='Haiti']
# print(haiti)
# haiti = haiti[years].T
# haiti.index = haiti.index.map(int)
# fig,ax = plt.subplots()
# ax.plot(haiti)
# ax.set_title('Immigration from Haiti between 1980 to 2013')
# ax.set_xlabel('years')
# ax.set_ylabel('Number of immigrants')
# ax.legend(['Immigrants'])
# plt.show()

df_can.sort_values(['Total'], ascending=False, axis=0,inplace=True)
df_top5 = df_can.head()
print(df_top5)

df_bar_5 = df_top5.reset_index()
label = list(df_bar_5.Country)
print(label)
label[2]= 'UK'
fig,ax = plt.subplots()
ax.bar(label,df_bar_5['Total'], label = label)
plt.show()