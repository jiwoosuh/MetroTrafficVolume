import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy.linalg as LA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import math
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
#%%
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
def calACF(yt, lag):
    yt_mean = np.mean(yt)
    yt_var = np.var(yt)
    acf = np.zeros(lag + 1)
    T = len(yt)
    for k in range(lag + 1):
        numerator = 0
        denominator = 0
        for t in range(k, T):
            numerator += (yt[t] - yt_mean) * (yt[t - k] - yt_mean)
        denominator = T * yt_var
        acf[k] = numerator / denominator
    return acf
def plotACF(yt, lag):
    acf = calACF(yt, lag)
    significance = 1.96 / np.sqrt(len(yt))
    acfneg = acf[::-1]
    acfpos = acf[1:]
    acf2 = np.concatenate((acfneg, acfpos))
    plt.stem(np.arange(-lag, lag + 1), acf2, markerfmt='ro', basefmt='b')
    plt.axhspan(-significance, significance, alpha=0.2, color='blue')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.title(f'Autocorrelation Function of {yt}')
    plt.show()
def ACF_PACF_Plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()
def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)
def rolling_mean(data, window_size):
    data = pd.DataFrame(data)
    rm = data.rolling(window_size).mean()
    return rm


def rolling_var(data, window_size):
    data = pd.DataFrame(data)
    rv = data.rolling(window_size).var()
    return rv

def plot_two_graphs(data1, data2):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    axs[0].plot(data1)
    axs[0].set_title("Rolling mean")
    axs[0].set_xlabel("Samples")
    axs[0].set_ylabel("Magnitude")
    axs[1].plot(data2)
    axs[1].set_title("Rolling variance")
    axs[1].set_xlabel("Samples")
    axs[1].set_ylabel("Magnitude")
    plt.tight_layout()
    plt.show()
#%%
url = 'https://raw.githubusercontent.com/jiwoosuh/MetroTrafficVolume/main/Metro_Interstate_Traffic_Volume.csv'
data = pd.read_csv(url, index_col='date_time', parse_dates = True)
data = data.loc['2012-10-03':]
group_sizes = data.groupby(data.index.date).size()
data.traffic_volume.plot()
plt.title("Traffic Volume from 2012-2018")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Volume")
plt.show()
# data = pd.read_csv("/Users/jiwoosuh/Desktop/Spring23/Time Series Analysis and Model/term project/Metro_Interstate_Traffic_Volume.csv", parse_dates=True)

# Specify the column that you want to remove duplicates from
# col_to_check = 'date_time'

# Loop over each row in the dataframe and remove duplicates in the specified column
# unique_values = set()
# duplicated_indexes = []
# for i, row in data.iterrows():
#     if row[col_to_check] in unique_values:
#         duplicated_indexes.append(i)
#     else:
#         unique_values.add(row[col_to_check])
#
# # Drop the duplicated rows and keep only the first occurrence
# data = data.drop(duplicated_indexes)
#
# # Save the new dataframe without duplicates
# #data.to_csv('data_without_duplicates.csv')
#
# #%%
# # set index
# data = data.set_index("date_time")
#
# data.index = pd.to_datetime(data.index)

#%%
# Data Cleaning
# Convert the date column to a pandas datetime object and set it as the index
url2= 'https://raw.githubusercontent.com/jiwoosuh/MetroTrafficVolume/main/data_without_duplicates.csv'
df = pd.read_csv(url2)
df['date_time'] = pd.to_datetime(df['date_time'])
df = df.set_index('date_time')

# Generate a new index with hourly frequency and reindex the DataFrame with it
new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
df = df.reindex(new_index)
df = df.loc['2012-10-03':]
df = df.interpolate()
df.isna().sum()
df = df.fillna(method='ffill')

#%%
plt.figure(figsize=(10,8))
df.traffic_volume.plot()
plt.title("Metro Traffic Voulme from 2013 to 2018")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.show()
# turns out 2015 has lots of missing data
################### DONE ###################
#%%
# decided to take 2017 only
df = df.loc['2017-01-01':'2018-01-01']
df = df.drop(["Unnamed: 0"], axis=1)


#%%
# Plot of the target variable
plt.figure(figsize=(10,8))
df.traffic_volume.plot()
plt.title("Metro Traffic Volume of 2017")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.tight_layout
plt.show()

#%% ACF PACF
import statsmodels.api as sm
ACF_PACF_Plot(df.traffic_volume, lags=50)
# oscilliate

df.describe()
#%% Correlation
corr = df.corr()
sns.heatmap(corr)
plt.tight_layout()
plt.title("Correlation Matrix")
plt.show()
#%%
# Decide to drop 2 columns of rain_1h and snow_1h as they don't have valid values
df = df.drop(["rain_1h", "snow_1h"], axis=1)
#%%
corr = df.corr()
sns.heatmap(corr)
plt.tight_layout()
plt.title("Correlation Matrix")
plt.show()

#%% Stationary
from statsmodels.tsa.stattools import kpss
print(ADF_Cal(df.traffic_volume))
print(kpss_test(df.traffic_volume))

#%% Rolling Mean and Var
rm = rolling_mean(df.traffic_volume, window_size=100)
rv = rolling_var(df.traffic_volume, window_size=100)
plot_two_graphs(rm, rv)

#%%
def seasonal_diff(data, period):
    data = pd.DataFrame(data)
    diff_data = data.diff(periods=period)
    return diff_data[period:]
df2 = df.copy()
df24 = seasonal_diff(df2.traffic_volume, 24)

df241 = seasonal_diff(df24.traffic_volume, 1)

df168 = seasonal_diff(df2.traffic_volume, 168)

df1681 = seasonal_diff(df168.traffic_volume, 1)
ACF_PACF_Plot(df24.traffic_volume,100)
ACF_PACF_Plot(df241.traffic_volume,100)
ACF_PACF_Plot(df168.traffic_volume,100)
ACF_PACF_Plot(df1681.traffic_volume,100)
print("ADF for the transformed dataset:", ADF_Cal(df1681.traffic_volume))
print("KPSS for the transformed dataset:", kpss_test(df1681.traffic_volume))

rm = rolling_mean(df1681.traffic_volume, window_size=100)
rv = rolling_var(df1681.traffic_volume, window_size=100)
plot_two_graphs(rm, rv)

#%%
plt.figure(figsize=(10,8))
df1681.traffic_volume.plot()
plt.title("Differenced with 168 Traffic Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.tight_layout
plt.show()


#%%
# split the data
col_df = list(df.columns.values)
for col in col_df:
    print(col, ':', str(df[col].unique()))
X = df[['holiday', 'temp', 'clouds_all', 'weather_main', 'weather_description']]
y = df['traffic_volume']
X_encoded = pd.get_dummies(X, columns=['holiday', 'weather_main', 'weather_description'], drop_first=True)
# X = df[['temp', 'rain_1h', 'snow_1h', 'clouds_all']]
# y = df['traffic_volume']
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

print("Size of training set:", len(X_train_std))
print("Size of test set:", len(X_test_std))


#%% STL

# plt.figure(figsize=(30,20))
val = pd.Series(df.traffic_volume, name = 'values')
STL = STL(val, period=168)
res = STL.fit()
fig = res.plot()
plt.show()
T = res.trend
S = res.seasonal
R = res.resid
Ft = max(0, 1-(np.var(R)/np.var(T+R)))
print("The strength of trend for this data set is", Ft)
Fs = max(0, 1-(np.var(R)/np.var(S+R)))
print("The strength of seasonality for this data set is", Fs)

plt.figure()
plt.plot(T, label = 'trend')
plt.plot(S, label = 'Seasonal')
plt.plot(R, label = 'residuals')
plt.title("Dataset after STL decomposition")
plt.xlabel('Date')
plt.ylabel('Values')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#%% 168 diff
# df_wo7 = seasonal_diff(df.traffic_volume, 168)
# df_wo71 = seasonal_diff(df_wo7.traffic_volume, 1)
train_data = df1681.iloc[:int(0.8 * len(df1681)), :]
test_data = df1681.iloc[int(0.8 * len(df1681)):, :]

#%%
# Fit the Holt-Winters method to the training data
model = ExponentialSmoothing(y_train, seasonal='mul', seasonal_periods=168).fit()
# model = SimpleExpSmoothing(y_train).fit()

# Make predictions on the testing data
predictions = model.predict(start=y_test.index[0], end=y_test.index[-1])

# Calculate the root mean squared error (RMSE) of the predictions
rmse = mean_squared_error(y_test, predictions, squared=False)

# Print the MSE and the predicted values
print(f"RMSE: {rmse}")
print(predictions)
#%%
# Fit the Holt-Winters method to the training data
model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=168).fit()
# model = SimpleExpSmoothing(y_train).fit()

# Make predictions on the testing data
predictions = model.predict(start=test_data.index[0], end=test_data.index[-1])

# Calculate the root mean squared error (RMSE) of the predictions
rmse = mean_squared_error(test_data, predictions, squared=False)

# Print the MSE and the predicted values
print(f"RMSE: {rmse}")
print(predictions)
#%%
plt.figure(figsize=(10,10))
plt.plot(train_data, label='Training data')
plt.plot(test_data, label='Testing data')
plt.plot(predictions, label='Predicted data')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Volume")
plt.title("Predictions with Holt-Winters method")
plt.show()

#%%
def average_model(train, test):
    y_hat_avg = pd.Series(data=train.mean(), index=test.index)
    return y_hat_avg

def naive_model(train, test):
    y_hat_naive = pd.Series(train.iloc[-1], index=test.index)
    y_hat_naive.name = 'naive_forecast'
    return y_hat_naive

def drift_model(train, test):
    y_hat_drift = pd.Series(train.iloc[-1] + (test.index - train.index[-1]) / (train.index[-1] - train.index[0]) * (train.iloc[-1] - train.iloc[0]), index=test.index)
    y_hat_drift.name = 'drift_forecast'
    return y_hat_drift

def ses_forecast(train, test, alpha):
    model_ses = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)
    predictions = model_ses.forecast(len(test))
    result = pd.Series(predictions, index=test.index)
    return result

y_avg = average_model(y_train, y_test)
y_naive = naive_model(y_train, y_test)
y_drift = drift_model(y_train, y_test)
y_ses = ses_forecast(y_train, y_test, 0.5)
rmse_avg = np.sqrt(mean_squared_error(y_test, y_avg))
rmse_naive = np.sqrt(mean_squared_error(y_test, y_naive))
rmse_drift = np.sqrt(mean_squared_error(y_test, y_drift))
rmse_ses = np.sqrt(mean_squared_error(y_test, y_ses))

plt.figure(figsize=(15, 10))
plt.plot(y_train, label='Training data')
plt.plot(y_test, label='Testing data')
plt.plot(y_avg, label='Average')
plt.plot(y_naive, label='Naive')
plt.plot(y_drift, label='Drift')
plt.plot(y_ses, label='SES')
plt.legend()
plt.title("Base Model")
plt.show()

print(f"RMSE for Average Model: {rmse_avg}")
print(f"RMSE for Naive Model: {rmse_naive}")
print(f"RMSE for Drift Model: {rmse_drift}")
print(f"RMSE for SES Model: {rmse_ses}")
# average is the best
#%%
y_avg = average_model(train_data.traffic_volume, test_data.traffic_volume)
y_naive = naive_model(train_data.traffic_volume, test_data.traffic_volume)
y_drift = drift_model(train_data.traffic_volume, test_data.traffic_volume)
y_ses = ses_forecast(train_data.traffic_volume, test_data.traffic_volume, 0.5)
rmse_avg = np.sqrt(mean_squared_error(test_data, y_avg))
rmse_naive = np.sqrt(mean_squared_error(test_data, y_naive))
rmse_drift = np.sqrt(mean_squared_error(test_data, y_drift))
rmse_ses = np.sqrt(mean_squared_error(test_data, y_ses))
# avg is the best
plt.figure(figsize=(15, 10))
plt.plot(train_data.traffic_volume, label='Training data')
plt.plot(test_data.traffic_volume, label='Testing data')
plt.plot(y_avg, label='Average')
plt.plot(y_naive, label='Naive')
plt.plot(y_drift, label='Drift')
plt.plot(y_ses, label='SES')
plt.legend()
plt.title("Base Model")
plt.show()
print(f"RMSE for Average Model: {rmse_avg}")
print(f"RMSE for Naive Model: {rmse_naive}")
print(f"RMSE for Drift Model: {rmse_drift}")
print(f"RMSE for SES Model: {rmse_ses}")

#%% 12 Multi Linear Regression
#XM = np.asmatrix(X_encoded)
X_encoded_std = scaler.fit_transform(X_encoded)
XM = np.asmatrix(X_encoded_std)
H = XM.T @ XM
s, d, v = np.linalg.svd(H)
print("SingularValues = ",d)
print("Condition Number=", LA.cond(XM))
#%%
X_train_std1 = sm.add_constant(X_train_std)
X_test_std1 = sm.add_constant(X_test_std, has_constant='add')
olsmodel = sm.OLS(y_train, X_train_std1).fit()
print(olsmodel.summary())
y_train_pred = olsmodel.predict(X_train_std1)
residual = y_train - y_train_pred
y_test_pred = olsmodel.predict(X_test_std1)
forecast = y_test - y_test_pred
rmse = mean_squared_error(y_test, y_test_pred, squared=False)
# Print the RMSE and the predicted values
print(f"Mean of residual_errors: {residual.mean()}")
print(f"Mean of forecast_errors: {forecast.mean()}")
print(f"RMSE: {rmse}")
residual_f = y_train - y_train_pred
forecast_f = y_test - y_test_pred
rmse = mean_squared_error(y_test, y_test_pred, squared=False)
# Print the RMSE and the predicted values
print(f"Mean of residual_errors: {residual.mean()}")
print(f"Mean of forecast_errors: {forecast.mean()}")
print(f"RMSE: {rmse}")
# ACF plot
ACF_PACF_Plot(residual_f,20)
#

#%%
# Backward stepwise selection
X_train_std_df = pd.DataFrame(X_train_std, index=X_train.index, columns=X_train.columns)
features = X_train.columns.tolist()
eliminate = []
model_results = []
model = sm.OLS(y_train, sm.add_constant(X_train_std_df)).fit()
pvalues = model.pvalues
max_pvalue = pvalues.drop('const').max()
prev_bic = model.bic
prev_adjr2 = model.rsquared_adj
while max_pvalue >= 0.05 and len(features) > 1:
    f = pvalues.drop('const').idxmax()
    eliminate.append(f)
    features.remove(f)
    X_train_sub = X_train_std_df[features]
    model = sm.OLS(y_train, sm.add_constant(X_train_sub)).fit()
    pvalues = model.pvalues
    max_pvalue = pvalues.drop('const').max()
    bic = model.bic
    adjr2 = model.rsquared_adj
    model_results.append({'AIC': model.aic, 'BIC': bic, 'Adj R^2': adjr2, 'Prev. BIC': prev_bic, 'Prev. Adj R^2': prev_adjr2})
    prev_bic = bic
    prev_adjr2 = adjr2

model_results = pd.DataFrame(model_results).sort_values(by="AIC", ascending=True)
print(model.summary())
print(f"Features to keep: {features}")
print(f"Features to eliminate: {eliminate}")


#%%
# calculate VIF for each feature to check for multicollinearity
# create a dataframe with your features
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_df = pd.DataFrame(X_train_std_df[features], columns=features)

# calculate VIF for each feature
vif = pd.DataFrame()
vif["feature"] = vif_df.columns
vif["VIF"] = [variance_inflation_factor(vif_df.values, i) for i in range(len(vif_df.columns))]

# print the results
print(vif)
# every value is between 1-10
#%% Final OLS model
X_train_std_new = X_train_std_df[features]
X_test_std_df = pd.DataFrame(X_test_std, index=X_test.index, columns=X_test.columns)
X_test_std_new = X_test_std_df[features]
olsmodel = sm.OLS(y_train, sm.add_constant(X_train_std_new)).fit()
print(olsmodel.summary())
y_train_pred = olsmodel.predict(sm.add_constant(X_train_std_new))
residual = y_train - y_train_pred
y_test_pred = olsmodel.predict(sm.add_constant((X_test_std_new), has_constant='add'))
forecast = y_test - y_test_pred
rmse = mean_squared_error(y_test, y_test_pred, squared=False)
# Print the RMSE and the predicted values
print(f"Mean of residual_errors: {residual.mean()}")
print(f"Mean of forecast_errors: {forecast.mean()}")
print(f"RMSE: {rmse}")
# ACF plot
ACF_PACF_Plot(residual,100)
# residual mean and var
residual.mean()
residual.var()
#%%
# Q value
lags = 140
import scipy.stats as stats
re = sm.tsa.stattools.acf(residual,lags, qstat=True)[1:]
Q = len(y_train)*np.sum(np.square(re[lags:]))
DOF = lags - 1
alfa = 0.01
chi_critical = stats.chi2.ppf(1-alfa, DOF)
if Q < chi_critical:
    print("The residual is white")
else:
    print("The residual is NOT white")
# print(f"Q value:{Q}")
# print(f"Chi-Critical:{chi_critical}")
t_values = olsmodel.tvalues
p_values = olsmodel.pvalues
t_test = pd.DataFrame({'t-values': t_values, 'p-values': p_values})
print(f'T-Test: {t_test}')
f_test = olsmodel.f_test(np.identity(len(olsmodel.params)))
f_value = f_test.fvalue
p_value = f_test.pvalue
print(f'F-Test. f-values: {f_value}, p-values: {p_value}')



#%% ARMA
def Cal_GPAC(acf, K, J):
    M = np.zeros((J, K-1))
    ry2 = np.concatenate((acf[::-1], acf[1:]))
    zero = len(acf)-1  # find ry(0) of acf
    for j in range(J):
        for k in range(1, K):
            den = np.array([ry2[zero-j-n:zero-j+k-n] for n in range(k)])
            num = np.copy(den).T
            num[-1] = ry2[zero+j+1:zero+j+k+1]
            num = num.T
            M[j, k-1] = np.linalg.det(num)/np.linalg.det(den)

    fig, ax = plt.subplots(figsize=(10, 10))
    heatmap = sns.heatmap(M, cmap='Pastel1', annot=True, linewidths=0.5, fmt='.3f')
    ax.set_title(f'GPAC table', fontsize=16)
    # ax.set_xlabel('AR lags', fontsize=14)
    # ax.set_ylabel('MA lags', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    heatmap.set_xticklabels([i+1 for i in range(K-1)])
    plt.show()
    return M

train_data_acf = sm.tsa.stattools.acf(train_data, nlags=100)
# ACF_PACF_Plot(y_train,100)
Cal_GPAC(train_data_acf, 15, 15)

#%%
# ACF_PACF_Plot(y_train,100)
# y_train_diff1 = seasonal_diff(y_train, 1)
# ACF_PACF_Plot(y_train_diff1,100)
# y_train_diff24 = seasonal_diff(y_train, 24)
# ACF_PACF_Plot(y_train_diff24,100)
# y_train_diff168 = seasonal_diff(y_train, 168)
# ACF_PACF_Plot(y_train_diff168,100)
# # y_train_diff124 = seasonal_diff(y_train_diff1, 24)
# # ACF_PACF_Plot(y_train_diff124,100)
# # y_train_diff1168 = seasonal_diff(y_train_diff1, 168)
# # ACF_PACF_Plot(y_train_diff1168,100)
# y_train_diff241 = seasonal_diff(y_train_diff24, 1)
# ACF_PACF_Plot(y_train_diff241,100)
# y_train_diff1681 = seasonal_diff(y_train_diff168, 1)
# ACF_PACF_Plot(y_train_diff1681,100)
# y_train_diff1681.plot()
# plt.show()

ACF_PACF_Plot(train_data,400)

#%% SARIMA
# model = sm.tsa.SARIMAX(train_data, order=(21,0,1), seasonal_order=(2,0,1,168)).fit()
# # coefficients = model.params.round(3)
# # print("Estimated coefficients:", coefficients)
# print(model.summary())

#%% SARIMA BEST**
model = sm.tsa.SARIMAX(train_data, order=(0,0,0), seasonal_order=(0,0,1,168)).fit()
coefficients = model.params.round(3)
print("Estimated coefficients:", coefficients)
print(model.summary())
SARIMA_y_model_hat = model.predict(start=1, end=len(train_data)-1)
plt.figure(figsize=(10, 6))
plt.plot(train_data[1:], label='y')
plt.plot(SARIMA_y_model_hat, label='1-step')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('1-Step Ahead Prediction with SARIMA')
plt.legend()
plt.show()

SARIMA_y_model_hat_h = model.forecast(steps=len(test_data))
plt.figure(figsize=(10, 6))
# plt.plot(np.arange(len(train_data), len(train_data)+len(train_data)-1), test_data, color='blue', label='Test')
# plt.plot(np.arange(len(train_data), len(train_data)+len(train_data)-1), y_model_hat_h, color='green', label='h-step')
plt.plot(test_data[1:], label='y')
plt.plot(SARIMA_y_model_hat_h, label='h-step')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('h-step Ahead Prediction with SARIMA')
plt.legend()
plt.show()
# print(model.summary())

SARIMA_model_rmse = mean_squared_error(test_data, SARIMA_y_model_hat_h, squared=False)
print(f"RMSE for SARIMA: {SARIMA_model_rmse}")
residual = train_data.traffic_volume[1:] - SARIMA_y_model_hat
forecast = test_data.traffic_volume[1:] - SARIMA_y_model_hat_h
ACF_PACF_Plot(residual,200)
print(f"Mean of Residual: {residual.mean()}")
print(f"Mean of Forecast Error: {forecast.mean()}")
print(f"Variance of Residual Error: {residual.var()}")
print(f"Variance of Forecast Error: {forecast.var()}")

# Q value
lags = 140
import scipy.stats as stats
re = acf(residual,lags)[1:]
Q = len(train_data)*np.sum(np.square(re[lags:]))
DOF = lags - 1
alfa = 0.01
chi_critical = stats.chi2.ppf(1-alfa, DOF)
if Q < chi_critical:
    print("The residual is white")
else:
    print("The residual is NOT white")

den = np.r_[1, 0]
num = np.r_[1, coefficients[0]]
num_roots = np.roots(num)
den_roots = np.roots(den)
print("Zeros of numerator: ", num_roots)
print("Poles of denominator: ", den_roots)
# Check for zero/pole cancellation
if np.isin(num_roots, den_roots).any():
    print("There is zero/pole cancellation")
else:
    print("There is no zero/pole cancellation")

#%%
# model = sm.tsa.SARIMAX(train_data, order=(1,0,0), seasonal_order=(0,1,2,168)).fit()
# coefficients = model.params.round(3)
# print("Estimated coefficients:", coefficients)
# print(model.summary())
## Process finished with exit code 137 (interrupted by signal 9: SIGKILL)

#%%
model = sm.tsa.ARIMA(train_data, order=(1,0,4)).fit()
coefficients = model.params.round(3)
print("Estimated coefficients:", coefficients)
print(model.summary())

#%% ARIMA BEST
model_arima = sm.tsa.ARIMA(train_data, order=(2,0,6)).fit()
coefficients = model_arima.params.round(3)
print("Estimated coefficients:", coefficients)
print(model_arima.summary())

ARIMA_y_model_hat = model_arima.predict(start=1, end=len(train_data)-1)
plt.figure(figsize=(10, 6))
plt.plot(train_data[1:], label='y')
plt.plot(ARIMA_y_model_hat, label='1-step')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('1-Step Ahead Prediction with ARIMA')
plt.legend()
plt.show()

ARIMA_y_model_hat_h = model_arima.forecast(steps=len(test_data))
plt.figure(figsize=(10, 6))
# plt.plot(np.arange(len(train_data), len(train_data)+len(train_data)-1), test_data, color='blue', label='Test')
# plt.plot(np.arange(len(train_data), len(train_data)+len(train_data)-1), y_model_hat_h, color='green', label='h-step')
plt.plot(test_data[1:], label='y')
plt.plot(ARIMA_y_model_hat_h, label='h-step')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('h-step Ahead Prediction with ARIMA')
plt.legend()
plt.show()
# print(model.summary())

ARIMA_model_rmse = mean_squared_error(test_data, ARIMA_y_model_hat_h, squared=False)
print(f"RMSE for ARIMA: {ARIMA_model_rmse}")
residual = train_data.traffic_volume[1:] - ARIMA_y_model_hat
ACF_PACF_Plot(residual,100)

