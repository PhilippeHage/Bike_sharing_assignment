#!/usr/bin/env python
# coding: utf-8

# ## Importin Packages & Libraries

# In[1]:


#!pip install kaggle
#!pip install dask
#!pip install dask_ml


# In[2]:


# Importing the data
import shutil
from dask import dataframe as ddf
import kaggle
import zipfile

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import dask.array as da

# Feature Engineering 
import warnings
from scipy.stats import skew
from dask_ml.preprocessing import DummyEncoder
from sklearn.preprocessing import MinMaxScaler

# Start a client
from dask.distributed import Client, progress

# Dask models
from dask_ml.linear_model import LinearRegression
from dask_ml.xgboost import XGBRegressor

# Sklearn models
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Evaluation metrics
from sklearn.metrics import r2_score
from dask_ml.metrics import mean_squared_error as mse

# Hyperparameter optimization
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import parallel_backend


# ## Reading the Data

# In[3]:


kaggle.api.authenticate()
kaggle.api.dataset_download_files("marklvl/bike-sharing-dataset")


# In[4]:


zip_ref = zipfile.ZipFile("bike-sharing-dataset.zip", 'r')
zip_ref.extractall("data")
zip_ref.close()
zip_ref = zipfile.ZipFile("data/bike-sharing-dataset.zip", 'r')
zip_ref.extractall()
zip_ref.close()


# In[5]:


shutil.move("day.csv", "data/day.csv")
shutil.move("hour.csv", "data/hour.csv")
shutil.move("Readme.txt", "data/Readme.txt")
shutil.move("Bike-Sharing-Dataset.zip", "data/Bike-Sharing-Dataset.zip")


# In[6]:


day = ddf.read_csv("data/day.csv", parse_dates = True)
hour = ddf.read_csv("data/hour.csv", parse_dates = True)


# In[7]:


day.head()


# In[8]:


hour.head()


# In[9]:


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

# In[10]:


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


# In[11]:


hour.head()


# ## Frequencies

# In[12]:


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


# In[13]:


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


# In[14]:


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


# In[15]:


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

# In[16]:


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

# In[17]:


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

# In[18]:


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


# In[19]:


plt.figure(figsize=(15, 6))
sns.heatmap(hour.corr().compute(), cmap="RdBu_r", vmax=1, annot=True)
plt.title("Correlation Matrix")


# Clearly, from above heatmap, we can se that the dataset has multicolinearity. `temp` and `atemp` are highly correlated.
# Will need to drop one of them.

# In[20]:


# Visualize the relationship among all continous variables using pairplots
NumericFeatureList = ["temp", "atemp", "hum", "windspeed"]
sns.pairplot(hour.compute(), hue="yr", vars=NumericFeatureList, height=3)


# As we can see looking at the EDA and inspecting the datasets, the `hour` dataset holds the same information than `day` and with a lot more detail. Therefore we have decided to continue the analysis using onlt `hour` as our data.

# ## Feature Engineering

# ### Variable Creation

# #### Average Activity

# As seen in the EDA, rentals increase and decrease depending on the time of the day. Hence, we decided to bin the `hour` variable into three categories: `sleep_hours`, `work_hours` and `free_hours`.
# 
# According to a study made by Fortune.com, the average hours where citizens of Washington D.C. are asleep are from 11:30pm to 7:20am.
# 
# For `sleep_hours` we stretched that range a little more on the lower side because we think that even if people are not asleep per se, they might be home already since 9:00pm.
# 
# As for `work_hours` we used the regular work schedule in the US, consisting of 8 hours, from 9:00 to 17:00 during the working days.
# 
# And lastly `free_hours` consists of anything in between the last two during working days and everything but `sleep_hours` during the weekends.
# 
# _Source: http://fortune.com/2015/07/07/cities-sleep-patterns-health/_

# #### Commute Schedule

# We realized that during working days, most people could use the bike rentals to commute to work, therefore we created a new variable `is_commute`.
# 
# We took into consideration the rush commute hours in Washington D.C. accordint to a study by TripSavvy.com where they explain that the most common commute times in the city are from 6 a.m. to 9:30 a.m. and 3:30 p.m. to 6:30 p.m.
# 
# We used these timeframes to create the new variable taking into account the day of the week as well, since rush hour happens mostly from Monday to Friday.
# 
# _Source: https://www.tripsavvy.com/driving-times-from-dc-1040439_

# #### Computed Apparent Temperature

# We also saw that one of the most influencial factors for outdoor activities is the Wind Chill factor, meaning the temperature felt by the body as a result of wind speed and actual measured temperature.
# 
# The perceived temperature due to wind chill is lower than the actual air temperature for all temperature values where the formula used is valid (-50°F to 50°F).
# 
# Therefore we proceed to calculate the `wind_chill` variable given the formula:
# <br/>
# <br/>
# <br/>
# $$Wind Chill Temperature = 35.74 + 0.6215×Temp - 35.75×Wind^{0.16} + 0.4275×Temp×Wind^{0.16}$$
# <br/>
# <br/>
# Note that all of the values are in Farenheit and miles per hour, so first of all we have to convert our values to those metrics.
# 
# _Source: http://mentalfloss.com/article/26730/how-wind-chill-calculated_

# In order to do this, since our values are normalized we have to undo this, luckily we have the min and max values used for each normalization in the dataset description, being:
# - min = -8, max = +39 for `temp`
# - min = -16, max = +50 for `atemp`
# - min = 0, max = +67 for `windspeed`
# 
# We will do this by using `MinMaxScaler` to set the range of values using the minimum and maximum above.

# Now we convert those values to the desired ones for the formula using:
# <br/>
# <br/>
# <br/>
# $$(°C × 9/5) + 32 = °F$$
# 
# $$ kmh / 1.609 = mph $$
# <br/>
# <br/>

# We can now proceed to calculating the `wind_chill` variable for our data.

# On the other hand we have the heat index effect, which is the opposite of the wind chill effect. Meaning that it estimates the temperature felt by the body as a result of air temperature and relative humidity. Heat index is often referred to as humiture.
# 
# We will now create the variable `heat_index` using the following formula:
# <br/>
# <br/>
# <br/>
# $$HeatIndex = -42.379 + 2.04901523*Temp + 10.14333127*Hum - 0.22475541*Temp*Hum - 6.83783*10^{-3}*Temp^2-5.481717*10^{-2}*Hum^2+1.22874*10^{-3}*Temp^2*Hum + 8.5282*10^{-4}*Temp*Hum^2 - 1.99*10^{-6}*Temp^2*Hum^2$$
# <br/>
# <br/>
# <br/>
# _Source: https://weather.com/safety/heat/news/heat-index-feels-like-temperature-summer_

# Now that we have `wind_chill` and `heat_index` we can proceed to calculate our version of apparent temperature or the temperature actually felt by the human body. Since the wind chill effect only happens when **[temperature < 35°F and windspeed > 5 mph]**, and the heat index effect only happens when **[temperature > 80°F and humidity > 40%]** we will create a new variable `atemp_x` taking into consideration this conditions and projecting the value of `atemp` otherwise.

# <table><tr>
# <td> <img src="https://d26tpo4cm8sb6k.cloudfront.net/img/wind-chill-old.png" alt="Wind Chill Chart" style="width: 500px;"/> </td>
# <td> <img src="https://climate.ncsu.edu/images/climate/heat_index_stull.jpg" alt="Heat Index Chart" style="width: 500px;"/> </td>
# </tr></table>

# Now that we have our own apparent temperature that takes into consideration wind chill and the heat index, we will proceed to normalize it and drop the temporary variables in the next steps.

# ## Preprocessing

# ### Handling Outliers

# #### Clipping Outliers

# Since the outliers remain after the normalization, we can now proceed if it is viable to remove them or just clip them to any range of interquartiles.

# In[21]:


plt.subplots(figsize=(15, 6))
sns.boxplot(data=hour[["windspeed", "hum"]].compute(), palette=["#e74c3c", "#95a5a6"])
plt.grid(which="major", axis="y")
plt.xlabel("Weather variables")
plt.ylabel("Number of bikes rented")
plt.title("Boxplots of weather variables")


# In[22]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
sns.countplot(hour.windspeed.compute(), ax=ax1)
sns.pointplot(x="windspeed", y="cnt", data=hour.compute(), ci=None, ax=ax2)
ax1.set_xticklabels([])
ax2.set_xticklabels([])


# As we can see above, the outliers do seem to skew the data a little bit, and the percentage of rows that contain outliers seems to be small enough to remove them safely. We now proceed to remove those outliers. We will first try to clip them to the .98 IQR for `windspeed` and to the 0.01 IQR for `humidity` since they seem to the thresholds to remove the outliers.

# In[23]:


plt.subplots(figsize=(15, 6))
sns.boxplot(data=hour[["windspeed", "hum"]].compute(), palette=["#e74c3c", "#95a5a6"])
plt.grid(which="major", axis="y")
plt.xlabel("Weather variables")
plt.ylabel("Number of bikes rented")
plt.title("Boxplots of weather variables")


# Now the outliers have been removed by forcing their values inside the respective IQR for each variable.

# In[24]:


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

# In[25]:


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


# In[26]:


del hour["casual"]
del hour["registered"]


# ## One Hot Encoder

# First we check the data to see which variables will be dummified, in this case they are: `season`, `year`, `month`, `weather_condition` and `average_activity`.

# In[27]:


cat_variables = hour.dtypes[hour.dtypes == "object"].index
cat_variables


# In[28]:


ddf_hour = hour.categorize()


# In[29]:


ddf_hour.head()


# Now we proceed to dummify the selected variables to be able to be used in our models.

# In[30]:


hour = ddf.get_dummies(ddf_hour, columns=cat_variables)
print("The dataset now contains {} columns.".format(ddf_hour.shape[1]))


# ## Skewness

# Now we will check if there is any skewness in our target variable, if so we will proceed to take the log in order to make it normally distributed.

# In[31]:


plt.subplots(figsize=(15, 6))
sns.distplot(ddf_hour.cnt.compute(), color="red")
plt.title("Distribution of Total Count")


# In[32]:


ddf_hour.cnt = np.log1p(ddf_hour.cnt)


# As we can see, now the distribution of the target variable looks much more like a normally distributed one. This will help since several models we will use in our analysis do not perform well with skewed data.

# In[33]:


plt.subplots(figsize=(15, 6))
sns.distplot(ddf_hour.cnt.compute())
plt.title("Distribution of Log-transformed Total Count")


# ## Train/Test Split

# We will now proceed to split the dataset into `train` and `test`. Since this is a time series dataset, we have to account for the order on which events happened, therefore we will train the models with the whole first year and the first three quarters of the second year, and use the last quarter as test data to evaluate the models.
# 
# The fourth quarter begins in October 1st, therefore we set the cutoff point for our split on 2012/10/01.

# In[34]:


train = ddf_hour[: 15212 - 1]
test = ddf_hour[15212 - 1 :]


# In[35]:


train.head()


# In[36]:


del train['dteday']
del test['dteday']


# We divide the datasets into `X_train`, `X_test`, `y_train` and `y_test` to fit the models.

# In[37]:


X_train = train
X_test  = test
y_train = train['cnt']
y_test  = test['cnt']


# In[38]:


X_train.head()


# In[39]:


del X_train['cnt']
del X_test['cnt']


# In[40]:


X = ddf_hour
y = ddf_hour["cnt"]


# In[41]:


del X['dteday']
del X['cnt']


# ## Modeling

# We will now proceed to the modeling part, where we will make use of various models starting with the most simple ones and gradually increasing its complexity. We will include first a time series analysis,  linear models and then move to non-linear models.

# ### Linear Models

# First, we need to create the client

# In[42]:


client = Client()
client


# #### Linear Regression

# In[43]:


from scikitplot.metrics import plot_calibration_curve
from scikitplot.plotters import plot_learning_curve
from scikitplot.estimators import plot_feature_importances


# In[44]:


lr = LinearRegression()
lr_model = lr.fit(X_train.values, y_train.values)
y_pred_lr = lr.predict(X_test.values)


# In[45]:


mse(y_test.values, y_pred_lr)


# In[46]:


r2_score(y_test.values.compute(), y_pred_lr.compute())


# In[47]:


def lr_plot():
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 4))
    plot_learning_curve(lr, title="Learning Curve", X=X.compute(), y=y.compute(), ax=ax1)
    sns.scatterplot(y_pred_lr.compute(), y_test.compute(), ax=ax2)
    ax2.set_title("Regression Curve")
    ax2.set_ylabel("Actual Values")
    ax2.set_xlabel("Predicted Values")
    fig.suptitle("Linear Regression", size=15)
    X_plot = np.linspace(0, 8, 2)
    Y_plot = X_plot
    ax2.plot(X_plot, Y_plot, color="r")
    ax2.text(0, 6, "R2:")
    ax2.text(0.5, 6, lr_score)


lr_plot()


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
rf_model = rf.fit(X_train.values, y_train.values)
y_pred_rf = rf.predict(X_test.values)


# In[50]:


rf_score = round(r2_score(y_test, y_pred_rf), 4)
print("Random Forest Regressor Regression R2: {}".format(rf_score))


# In[51]:


def rf_plot():
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 4))
    plot_learning_curve(rf, title="Learning Curve", X=X, y=y, ax=ax1)
    sns.scatterplot(y_pred_rf, y_test, ax=ax2)
    ax2.set_title("Regression Curve")
    ax2.set_ylabel("Actual Values")
    ax2.set_xlabel("Predicted Values")
    fig.suptitle("Random Forest Regressor", size=15)
    X_plot = np.linspace(0, 8, 2)
    Y_plot = X_plot
    ax2.plot(X_plot, Y_plot, color="r")
    ax2.text(0, 6, "R2:")
    ax2.text(0.5, 6, rf_score)


rf_plot()


# ### XGBoost Regressor

# In[52]:


xgbr = XGBRegressor(max_depth=3, n_estimators=1000)
xgbr_model = xgbr.fit(X_train.values, y_train.values)
y_pred_xgbr = xgbr.predict(X_test.values)


# In[56]:


xgbr_score = round(xgbr.score(X_test.compute(), y_test.compute()), 4)
print("XGBoost Regressor R2: {}".format(xgbr_score))


# In[54]:


print(
    "XGBoost Regressor CV R2: {}".format(round(cross_val_score(xgbr, X, y).mean(), 4))
)


# In[ ]:


def xgbr_plot():
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 4))
    plot_learning_curve(xgbr, title="Learning Curve", X=X, y=y, ax=ax1)
    sns.scatterplot(y_pred_xgbr, y_test, ax=ax2)
    ax2.set_title("Regression Curve")
    ax2.set_ylabel("Actual Values")
    ax2.set_xlabel("Predicted Values")
    fig.suptitle("XGBoost Regressor", size=15)
    X_plot = np.linspace(0, 8, 2)
    Y_plot = X_plot
    ax2.plot(X_plot, Y_plot, color="r")
    ax2.text(0, 6, "R2:")
    ax2.text(0.5, 6, xgbr_score)


xgbr_plot()


# ## Conclusion
