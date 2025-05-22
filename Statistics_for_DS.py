#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.DataFrame({'movie':['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
                  'rating': [9,7,6,9,5,7,8,9,5]})


# In[4]:


df.mean()


# In[5]:


sns.displot(df, kde=True, rug=True)
plt.axvline(np.mean(df.rating), color='r', linestyle='-')
plt.axvline(np.median(df.rating), color='g', linestyle='-')
plt.axvline(df.rating.mode().values[0], color='y', linestyle='-')


# In[6]:


f, (ax_box, ax_hist) = plt.subplots(2, sharex=True,
                                    gridspec_kw={'height_ratios':(0.2, 1)})
mean = np.mean(df.rating)
median = np.median(df.rating)
mode = df.rating.mode().values[0]

sns.boxplot(data=df, x='rating', ax=ax_box)
ax_box.axvline(mean, color='r', linestyle = '-')
ax_box.axvline(median, color='g', linestyle = '-')
ax_box.axvline(mode, color='b', linestyle = '-')

sns.histplot(data=df, x='rating', ax=ax_hist, kde=True)
ax_hist.axvline(mean, color='r', linestyle='-', label='Mean')
ax_hist.axvline(median, color='g', linestyle='-', label='Median')
ax_hist.axvline(mode, color='b', linestyle='-', label='Mode')

ax_hist.legend()
ax_box.set(xlabel='')
plt.show()


# In[7]:


df.rating.var()


# In[8]:


df.rating.std()


# In[9]:


mean = df.groupby(['movie'])['rating'].mean()
std = df.groupby(['movie'])['rating'].std()


# In[10]:


mean


# In[11]:


std


# In[12]:


fig, ax = plt.subplots()
mean.plot.bar(yerr=std, ax=ax, capsize=4)


# In[13]:


df1 = pd.DataFrame({'pop_sample':range(20)})


# In[15]:


df1.sample(5).mean()


# In[16]:


df1.sample(10).mean()


# In[17]:


df1.mean()


# In[18]:


from scipy import stats
stats.sem(df1)


# In[19]:


df2 = sns.load_dataset('tips')
sns.set_theme(style='whitegrid')
ax = sns.boxplot(x='day', y='total_bill', data=df2)


# In[21]:


ax = sns.boxplot(x='day', y='total_bill', data=df2)
ax = sns.swarmplot(x='day', y='total_bill', data=df2, color='0.25')


# In[22]:


print(df2['total_bill'].quantile([0.05, 0.25, 0.5, 0.75]))


# In[23]:


print(df2['total_bill'].quantile(0.75) - df2['total_bill'].quantile(0.25))


# In[24]:


df3 = sns.load_dataset('iris')


# In[29]:


fig, ax = plt.subplots(figsize=(6,6))
ax = sns.heatmap(df3.corr(), vmin=-1, vmax=1,
                cmap=sns.diverging_palette(20, 220, as_cmap=True), ax=ax)
plt.tight_layout()
plt.show()


# In[30]:


a = [11, 12, 22, 11]
b = [7, 8, 9, 10]
c = [10, 11, 22, 23]
arr = np.array([a, b, c])


# In[31]:


cov_matrix = np.cov(arr, bias=True)


# In[32]:


cov_matrix


# In[33]:


sns.heatmap(cov_matrix, annot=True, fmt='g')
plt.show()


# In[34]:


df.skew()


# In[35]:


df.kurtosis()


# In[42]:


norm1 = np.arange(-5, 5, 0.001)
mean = 0.0
std = 1.0
pdf = stats.norm.pdf(norm1, mean, std)
plt.plot(norm, pdf)
plt.show


# In[43]:


import pylab
stats.probplot(df3.sepal_length, plot=pylab)


# In[44]:


sns.kdeplot(df3.sepal_length)


# In[46]:


from scipy.stats import binom
n = 6
p = 0.5
r_value = list(range(n+1))
dist = [binom.pmf(r, n, p) for r in r_value]
plt.bar(r_value, dist)
plt.show()


# In[47]:


s = np.random.poisson(5, 10000)
count, bins, ignored = plt.hist(s, 10, density=True)
plt.show()


# In[49]:


import statsmodels.stats.api as sms
sms.DescrStatsW(df3.sepal_length).tconfint_mean()


# In[55]:


fig, ax = plt.subplots()
ax2 = ax.twinx()
n, bins, patches = ax.hist(df3.sepal_length, bins=100)
n, bins, patches = ax2.hist(df3.sepal_length, cumulative=1, histtype='step',
                            bins=100, color='r')


# In[56]:


plt.hist(df3.sepal_length, cumulative=True, label='CDF', histtype='step',
        alpha=0.8, color='g')


# In[57]:


cdf = stats.norm.cdf(norm1)
plt.plot(norm1, cdf)
plt.show()


# In[59]:


ax = sns.displot(df3.sepal_length)


# In[ ]:




