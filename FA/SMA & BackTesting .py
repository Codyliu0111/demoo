#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")


# In[44]:


df = pd.read_csv("/Users/Cody/Desktop/sample_project/Algo/eurusd.csv", parse_dates = ["Date"], index_col = "Date")
df,
df.info


# In[45]:


df.plot(figsize = (12, 8), title = "EUR/USD", fontsize = 12)
plt.show()


# In[46]:


df["returns"] = np.log(df.div(df.shift(1)))


# In[47]:


df.dropna(inplace = True)


# In[48]:


df.returns.hist(bins = 100, figsize = (12, 8))
plt.title("EUR/USD returns")
plt.show()


# In[49]:


df.returns.sum()


# In[50]:


np.exp(df.returns.sum())


# In[51]:


df.returns.cumsum().apply(np.exp)
df["creturns"] = df.returns.cumsum().apply(np.exp)
df.creturns.plot(figsize = (12, 8), title = "EUR/USD - Buy and Hold", fontsize = 12)
plt.show()


# In[52]:


df.creturns.iloc[-1] , df.returns.sum()


# In[53]:


df.returns.mean() * 252 , # mean return


# In[54]:


df.returns.std() * np.sqrt(252) # risk


# In[56]:


df["cummax"] = df.creturns.cummax()
df[["creturns", "cummax"]].dropna().plot(figsize = (12, 8), title = "EUR/USD - max drawdown", fontsize = 12)
plt.show()


# In[ ]:





# ## SMA Strategy

# In[57]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")


# In[63]:


data = pd.read_csv("/Users/Cody/Desktop/sample_project/Algo/eurusd.csv", parse_dates = ["Date"], index_col = "Date")


# In[64]:


def run_strategy(SMA):
    data = df.copy()
    data["returns"] = np.log(data.price.div(data.price.shift(1)))
    data["SMA_S"] = data.price.rolling(int(SMA[0])).mean()
    data["SMA_L"] = data.price.rolling(int(SMA[1])).mean()
    data.dropna(inplace = True)
    
    data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
    data["strategy"] = data.position.shift(1) * data["returns"]
    data.dropna(inplace = True)
    
    
    return data[["returns", "strategy"]].sum().apply(np.exp)[-1]


# In[65]:


sma_s = 50
sma_l = 200


# In[66]:


data.price.rolling(50).mean()


# In[67]:


data["SMA_S"] = data.price.rolling(sma_s).mean()
data["SMA_L"] = data.price.rolling(sma_l).mean()


# In[68]:


data


# In[69]:


data.plot(figsize = (12, 8), title = "EUR/USD - SMA{} | SMA{}".format(sma_s, sma_l), fontsize = 12)
plt.legend(fontsize = 12)
plt.show()


# In[70]:


data.dropna(inplace = True)


# In[71]:


data.loc["2016"].plot(figsize = (12, 8), title = "EUR/USD - SMA{} | SMA{}".format(sma_s, sma_l), fontsize = 12)
plt.legend(fontsize = 12)
plt.show()


# In[72]:


data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1 )


# In[73]:


data.loc[:, ["SMA_S", "SMA_L", "position"]].plot(figsize = (12, 8), fontsize = 12, secondary_y = "position",
                                                title = "EUR/USD - SMA{} | SMA{}".format(sma_s, sma_l))
plt.show()


# In[74]:


data.loc["2016", ["SMA_S", "SMA_L", "position"]].plot(figsize = (12, 8), fontsize = 12, secondary_y = "position",
                                                     title = "EUR/USD - SMA{} | SMA{}".format(sma_s, sma_l))
plt.show()


# ## Vectorized Strategy Backtesting

# In[75]:


data


# In[77]:


data["returns"] = np.log(data.price.div(data.price.shift(1)))
data["strategy"] = data.position.shift(1) * data["returns"]


# In[80]:


data.dropna(inplace = True)
data


# In[81]:


data[["returns", "strategy"]].sum() # absolute performance


# In[82]:


data[["returns", "strategy"]].sum().apply(np.exp) # absolute performance


# In[83]:


data[["returns", "strategy"]].mean() * 252 # annualized return


# In[84]:


data[["returns", "strategy"]].std() * np.sqrt(252) # annualized risk


# In[85]:


data["creturns"] = data["returns"].cumsum().apply(np.exp)
data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)


# In[86]:


data[["creturns", "cstrategy"]].plot(figsize = (12, 8), title = "EUR/USD - SMA{} | SMA{}".format(sma_s, sma_l), fontsize = 12)
plt.legend(fontsize = 12)
plt.show()


# In[ ]:




