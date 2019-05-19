#!/usr/bin/env python
# coding: utf-8

# ## Importin Packages & Libraries

# In[1]:


#!pip install kaggle
#!pip install dask
#!pip install dask_ml


# In[2]:


import shutil
from dask import dataframe as ddf
import kaggle
import zipfile

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
 
import warnings
from scipy.stats import skew
from dask_ml.preprocessing import DummyEncoder
from sklearn.preprocessing import MinMaxScaler

from dask.distributed import Client, progress

from dask_ml.linear_model import LinearRegression
from dask_ml.xgboost import XGBRegressor

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from dask_ml.metrics import mean_squared_error as mse

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import parallel_backend

import time


# ## Reading the Data

# In[3]:


start = time.time()
start


# In[4]:


kaggle.api.authenticate()
kaggle.api.dataset_download_files("marklvl/bike-sharing-dataset")


# In[5]:


zip_ref = zipfile.ZipFile("bike-sharing-dataset.zip", 'r')
zip_ref.extractall("data")
zip_ref.close()
zip_ref = zipfile.ZipFile("data/bike-sharing-dataset.zip", 'r')
zip_ref.extractall()
zip_ref.close()


# In[6]:


shutil.move("day.csv", "data/day.csv")
shutil.move("hour.csv", "data/hour.csv")
shutil.move("Readme.txt", "data/Readme.txt")
shutil.move("Bike-Sharing-Dataset.zip", "data/Bike-Sharing-Dataset.zip")


# In[7]:


day = ddf.read_csv("data/day.csv", parse_dates = True)
hour = ddf.read_csv("data/hour.csv", parse_dates = True)


# In[8]:


day.head()


# In[9]:


hour.head()


# In[10]:


#To get the number of NAs in the original dataset
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 6))
sns.heatmap(
    day.isnull().compute(), yticklabels=False, cbar=False, cmap="viridis", ax=ax1
).set_title("Hour dataset")
sns.heatmap(
    hour.isnull().compute(), yticklabels=False, cbar=False, cmap="viridis", ax=ax2
).set_title("Day dataset")

print(
    "   There are {} null values in the Hour dataset.           There are {} null values in the Day dataset.".format(
        hour.isnull().sum().sum().compute(), day.isnull().sum().sum().compute()
    )
)


# ## EDA

# ### Variable Names and Encoding

# In[11]:


day.rename(
    columns={
        "dteday": "datetime",
        "holiday": "is_holiday",
        "workingday": "is_workingday",
        "weathersit": "weather_condition",
        "hum": "humidity",
        "mnth": "month",
        "cnt": "total_count",
        "hr": "hour",
        "yr": "year",
    }
).compute()


hour.rename(
    columns={
        "dteday": "datetime",
        "holiday": "is_holiday",
        "workingday": "is_workingday",
        "weathersit": "weather_condition",
        "hum": "humidity",
        "mnth": "month",
        "cnt": "total_count",
        "hr": "hour",
        "yr": "year",
    }
).compute()


# In[12]:


hour.head()


# ## Frequencies

# In[13]:


plt.figure(figsize=(15, 6))
sns.barplot(
    "yr",
    "cnt",
    hue="season",
    data=day.compute(),
    palette=["#b4f724", "#f7f725", "#af7018", "#68d4e8"],
    ci=None,
)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Year")
plt.ylabel("Number of bikes rented per hour")
plt.title("Seasonal number of bikes rented per day")
plt.xticks(ticks=(0, 1), labels=("2011", "2012"))
plt.grid(which="major", axis="y")


# In[14]:


plt.figure(figsize=(15, 6))
sns.barplot(
    "season",
    "cnt",
    hue="weathersit",
    data=hour.compute(),
    palette="rainbow",
    ci=None,
)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Season")
plt.ylabel("Hourly number of bikes rented")
plt.title("Number of bikes rented per hour by weather condition and season")
plt.xticks(ticks=(0, 1, 2, 3, 3.5))
plt.grid(which="major", axis="y")

plt.figure(figsize=(14, 6))
sns.barplot(
    "season",
    "cnt",
    data=hour.compute(),
    palette=["#b4f724", "#f7f725", "#af7018", "#68d4e8"],
    ci=None,
)
plt.xlabel("Season")
plt.ylabel("Hourly number of bikes rented")
plt.title("Number of bikes rented per hour by weather condition and season")
plt.xticks(ticks=(0, 1, 2, 3, 3.5))
plt.grid(which="major", axis="y")


# In[15]:


sns.catplot(
    x="mnth",
    y="cnt",
    kind="point",
    hue="workingday",
    data=hour.compute(),
    ci=None,
    palette="Set1",
    aspect=2.3,
    legend=False,
)
plt.legend(("Weekend", "Workday"), loc="upper right", bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Month")
plt.ylabel("Hourly number of bikes rented")
plt.title("Number of bikes rented per hour by type of day")
plt.axhline(hour.cnt.mean().compute(), ls="--", color="#a5a5a5")
plt.text(0.5, hour.cnt.mean().compute() - 10, "Average", color="#a5a5a5")


# In[16]:


sns.catplot(
    x="hr",
    y="cnt",
    kind="point",
    hue="workingday",
    data=hour.compute(),
    ci=None,
    palette="Set1",
    aspect=2.3,
    legend=False,
)
plt.legend(("Weekend", "Workday"), loc="upper right", bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Hour")
plt.ylabel("Hourly number of bikes rented")
plt.title("Number of bikes rented per hour by type of day and month")
plt.axhline(hour.cnt.mean().compute(), ls="--", color="#a5a5a5")
plt.text(0.5, hour.cnt.mean().compute() - 30, "Average", color="#a5a5a5")


# ## Boxplot (Outliers)

# Box plots are useful for getting an idea of the distribution of numerical features and detect potential outliers. Therefore, we use different box plots to see the different distributions of `total_count`, `casual` and `registrered`. It is clear that each variable has outliers, we deal with them in the Chapter Preprocessing.

# In[17]:


plt.subplots(figsize=(15, 6))
sns.boxplot(
    data=hour[["cnt", "casual", "registered"]].compute(),
    palette=["#2ecc71", "#b4f724", "#f7f725"],
)
plt.grid(which="major", axis="y")
plt.xlabel("Rental type")
plt.ylabel("Number of bikes rented")
plt.title("Boxplots of count variables")


# Additionally, we created box plot for the weather variables with the same purpose of the previous ones.

# In[18]:


plt.subplots(figsize=(15, 6))
sns.boxplot(
    data=hour[["windspeed", "temp", "atemp", "hum"]].compute(),
    palette=["#e74c3c", "#9b59b6", "#3498db", "#95a5a6"],
)
plt.grid(which="major", axis="y")
plt.xlabel("Weather variables")
plt.ylabel("Number of bikes rented")
plt.title("Boxplots of weather variables")


# ## Relationships between variables

# Another important step in the EDA, before feature engineering, is to see what the relationship is between potential predicting variables and the target variable. This gives us an idea of what features might be important for including in the model.

# In[19]:


plt.subplots(figsize=(15, 6))
corr = hour.compute()[
    [
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "holiday",
        "workingday",
        "casual",
        "registered",
    ]
].corrwith(hour["cnt"].compute())
sns.barplot(
    x=corr.index,
    y=corr,
    palette=[
        "#9b59b6",
        "#3498db",
        "#95a5a6",
        "#e74c3c",
        "#34495e",
        "#2be5dc",
        "#b4f724",
        "#f7f725",
    ],
)
plt.axhline(-0, 0, color="black")
plt.grid(which="major", axis="y")
plt.ylabel("Correlation")
plt.title("Correlations of variables with target variable")


# In[20]:


plt.figure(figsize=(15, 6))
sns.heatmap(hour.corr().compute(), cmap="RdBu_r", vmax=1, annot=True)
plt.title("Correlation Matrix")


# Clearly, from above heatmap, we can se that the dataset has multicolinearity. `temp` and `atemp` are highly correlated.
# Will need to drop one of them.

# In[21]:


# Visualize the relationship among all continous variables using pairplots
NumericFeatureList = ["temp", "atemp", "hum", "windspeed"]
sns.pairplot(hour.compute(), hue="yr", vars=NumericFeatureList, height=3)


# As we can see looking at the EDA and inspecting the datasets, the `hour` dataset holds the same information than `day` and with a lot more detail. Therefore we have decided to continue the analysis using onlt `hour` as our data.

# ## Preprocessing

# ### Handling Outliers

# #### Clipping Outliers

# Since the outliers remain after the normalization, we can now proceed if it is viable to remove them or just clip them to any range of interquartiles.

# In[22]:


plt.subplots(figsize=(15, 6))
sns.boxplot(data=hour[["windspeed", "hum"]].compute(), palette=["#e74c3c", "#95a5a6"])
plt.grid(which="major", axis="y")
plt.xlabel("Weather variables")
plt.ylabel("Number of bikes rented")
plt.title("Boxplots of weather variables")


# In[23]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
sns.countplot(hour.windspeed.compute(), ax=ax1)
sns.pointplot(x="windspeed", y="cnt", data=hour.compute(), ci=None, ax=ax2)
ax1.set_xticklabels([])
ax2.set_xticklabels([])


# As we can see above, the outliers do seem to skew the data a little bit, and the percentage of rows that contain outliers seems to be small enough to remove them safely. We now proceed to remove those outliers. We will first try to clip them to the .98 IQR for `windspeed` and to the 0.01 IQR for `humidity` since they seem to the thresholds to remove the outliers.

# In[24]:


plt.subplots(figsize=(15, 6))
sns.boxplot(data=hour[["windspeed", "hum"]].compute(), palette=["#e74c3c", "#95a5a6"])
plt.grid(which="major", axis="y")
plt.xlabel("Weather variables")
plt.ylabel("Number of bikes rented")
plt.title("Boxplots of weather variables")


# Now the outliers have been removed by forcing their values inside the respective IQR for each variable.

# In[25]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
sns.countplot(hour.windspeed.compute(), ax=ax1)
sns.pointplot(x="windspeed", y="cnt", data=hour.compute(), ci=False, ax=ax2)
ax1.set_xticklabels([])
ax2.set_xticklabels([])


# ## Dropping Variables

# We decided to drop the columns `casual` and `registered` since we are only interested in the `total_count` and they are so correlated that can cause overfitting.
# 
# `temp` is also extremely correlated with `atemp`, but since we believe `atemp` has much more effect on the target variable, we decided to keep that one.
# 
# **Update: we get better scores with `temp` so we will leave it.
# 
# 
# Besides having `temp` embedded, `atemp` also comprises `humidity` and `windspeed`, but since they are not as correlated in the matrix we decided to leave them.
# 
# ><font size=2, color=black>_Apparent temperature is the temperature equivalent perceived by humans, caused by the combined effects of air temperature, relative humidity and wind speed. The measure is most commonly applied to the perceived outdoor temperature._</font>

# We expect that the vatiables `average_activity` and `is_commute` created above will emprise all the information conveyed in `hour`, but since scores are higher with it we decided to leave it.

# We will also drop all variables used to compute our `atemp_comp` variable.

# In[26]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

sns.lineplot(
    x="hr",
    y="cnt",
    hue="workingday",
    data=hour.compute(),
    ci=None,
    palette="Set1",
    ax=ax1,
)

sns.lineplot(x="hr", y="cnt", hue="weekday", data=hour.compute(), ci=None, ax=ax2)


# In[27]:


del hour["casual"]
del hour["registered"]


# ## One Hot Encoder

# First we check the data to see which variables will be dummified, in this case they are: `season`, `year`, `month`, `weather_condition` and `average_activity`.

# In[28]:


cat_variables = hour.dtypes[hour.dtypes == "object"].index
cat_variables


# In[29]:


ddf_hour = hour.categorize()


# In[30]:


ddf_hour.head()


# Now we proceed to dummify the selected variables to be able to be used in our models.

# In[31]:


hour = ddf.get_dummies(ddf_hour, columns=cat_variables)
print("The dataset now contains {} columns.".format(ddf_hour.shape[1]))


# ## Skewness

# Now we will check if there is any skewness in our target variable, if so we will proceed to take the log in order to make it normally distributed.

# In[32]:


plt.subplots(figsize=(15, 6))
sns.distplot(ddf_hour.cnt.compute(), color="red")
plt.title("Distribution of Total Count")


# In[33]:


ddf_hour.cnt = np.log1p(ddf_hour.cnt)


# As we can see, now the distribution of the target variable looks much more like a normally distributed one. This will help since several models we will use in our analysis do not perform well with skewed data.

# In[34]:


plt.subplots(figsize=(15, 6))
sns.distplot(ddf_hour.cnt.compute())
plt.title("Distribution of Log-transformed Total Count")


# ## Train/Test Split

# We will now proceed to split the dataset into `train` and `test`. Since this is a time series dataset, we have to account for the order on which events happened, therefore we will train the models with the whole first year and the first three quarters of the second year, and use the last quarter as test data to evaluate the models.
# 
# The fourth quarter begins in October 1st, therefore we set the cutoff point for our split on 2012/10/01.

# In[35]:


train = ddf_hour[: 15212 - 1]
test = ddf_hour[15212 - 1 :]


# In[36]:


train.head()


# In[37]:


del train['dteday']
del test['dteday']


# We divide the datasets into `X_train`, `X_test`, `y_train` and `y_test` to fit the models.

# In[38]:


X_train = train
X_test  = test
y_train = train['cnt']
y_test  = test['cnt']


# In[39]:


X_train.head()


# In[40]:


del X_train['cnt']
del X_test['cnt']


# In[41]:


X = ddf_hour
y = ddf_hour["cnt"]


# In[42]:


del X['dteday']
del X['cnt']


# ## Modeling

# We will now proceed to the modeling part, where we will make use of various models starting with the most simple ones and gradually increasing its complexity. We will include first a time series analysis,  linear models and then move to non-linear models.

# ### Linear Models

# First, we need to create the client

# In[43]:


client = Client()
client


# #### Linear Regression

# In[44]:


from scikitplot.metrics import plot_calibration_curve
from scikitplot.plotters import plot_learning_curve
from scikitplot.estimators import plot_feature_importances


# In[45]:


lr = LinearRegression()
with joblib.parallel_backend('dask'):
    lr_model = lr.fit(X_train.values,y_train.values)
    y_pred_lr = lr.predict(X_test.values)


# In[46]:


mse(y_test.values, y_pred_lr)


# In[47]:


r2_score(y_test.values.compute(), y_pred_lr.compute())


# ### Non Linear Models

# #### Random Forest Regressor

# In[48]:


from sklearn.model_selection import cross_val_score, GridSearchCV


# In[49]:


rf = RandomForestRegressor(
    random_state=42,
    n_estimators=500,
    min_samples_split=10,
    min_samples_leaf=2,
    max_depth=None,
)
with joblib.parallel_backend('dask'):
    rf_model = rf.fit(X_train.values,y_train.values)
    y_pred_rf = rf.predict(X_test.values)


# In[50]:


rf_score = round(r2_score(y_test, y_pred_rf), 4)
print("Random Forest Regressor Regression R2: {}".format(rf_score))


# ### XGBoost Regressor

# In[51]:


xgbr = XGBRegressor(max_depth=3, n_estimators=1000)
xgbr_model = xgbr.fit(X_train.values, y_train.values)
y_pred_xgbr = xgbr.predict(X_test.values)


# In[52]:


r2_score(y_test.values.compute(), y_pred_xgbr.compute())


# ## Conclusion

# After all the analysis we can say that we can condidently predict the number of bikes rented on a given day with an cross-validated accuracy of more than 91%.
# 
# In this particular case we argue that cross validation is not as necesary since we are dealing with time series data, therefore picking random rows to divide into `train` and  `test` is not the most accurate way to evaluate our models. Nevertheless even with cross validation the scores we got are equally good.
# 
# Curiously enough the first two appeared to be the least correlated to the target variable in the exploratory analysis. Contrary to our belief, the variables we created do not have as much predictive power as we thought.

# In[53]:


end = time.time()


# In[54]:


Time = end - start
print(Time)

