# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:30:25 2024

@author: HP
"""
import pandas as pd
walmart=pd.read_csv('Walmart_Footfalls_Raw.csv')

import numpy as np

walmart['t']=np.arange(1,160)

walmart['t_square']=walmart['t']*walmart['t']
walmart['log_footfalls']=np.log(walmart['Footfalls'])
walmart.columns

p=walmart['Month'][0]

p[0:3]

walmart['months']=0

for i in range(159):
    p=walmart['Month'][i]
    walmart['months'][i]=p[0:3]
    
month_dummies=pd.DataFrame(pd.get_dummies(walmart['months']))

walmart1=pd.concat([walmart,month_dummies],axis=1)

walmart1.Footfalls.plot()

train=walmart1.head(147)
test=walmart1.tail(12)

import statsmodels.formula.api as smf

linear_model=smf.ols('Footfalls ~ t',data=train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_linear=np.sqrt(np.mean((np.array(test['Footfalls'])-np.array(pred_linear))**2))
rmse_linear

exp=smf.ols('log_footfalls ~ t',data=train).fit()
pred_exp=pd.Series(exp.predict(pd.DataFrame(test['t'])))
rmse_exp=np.sqrt(np.mean((np.array(test['Footfalls'])-np.exp(pred_exp))**2))
rmse_exp


quad=smf.ols('Footfalls ~ t + t_square',data=train).fit()
pred_quad=pd.Series(quad.predict(pd.DataFrame(test[['t','t_square']])))
rmse_quad=np.sqrt(np.mean((np.array(test['Footfalls'])-np.array(pred_quad))**2))
rmse_quad



add_sea=smf.ols('Footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea=pd.Series(add_sea.predict(pd.DataFrame(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])))
rmse_add_sea=np.sqrt(np.mean((np.array(test['Footfalls'])-np.array(pred_add_sea))**2))
rmse_add_sea


mul_sea=smf.ols('log_footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_mul_sea=pd.Series(mul_sea.predict(test))
rmse_mul_sea=np.sqrt(np.mean((np.array(test['Footfalls'])-np.array(pred_mul_sea))**2))
rmse_mul_sea

add_sea_quad=smf.ols('Footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea_quad=pd.Series(add_sea_quad.predict(pd.DataFrame(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']])))
rmse_add_sea_quad=np.sqrt(np.mean((np.array(test['Footfalls'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

mult_add_sea=smf.ols('log_footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_mult_add_sea=pd.Series(mult_add_sea.predict(test))
rmse_mult_add_sea=np.sqrt(np.mean((np.array(test['Footfalls'])-np.array(pred_mult_add_sea))**2))
rmse_mult_add_sea

data={'MODEL':pd.Series(['rmse_linear','rmse_exp','rmse_quad','rmse_add_sea','rmse_mul_sea','rmse_add_sea_quad','rmse_mult_add_sea']),'RMSE_Vlaues':pd.Series([rmse_linear,rmse_exp,rmse_quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad,rmse_mult_add_sea])}

table_rmse=pd.DataFrame(data)
table_rmse

predict_data=pd.read_excel('Predict_new.xlsx')

model_full=smf.ols('Footfalls ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_new=pd.Series(model_full.predict(predict_data))
pred_new

predict_data['forecasted__footfalls']=pd.Series(pred_new)

full_res=walmart1.Footfalls - model_full.predict(walmart1)

import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_res,lags=12)

tsa_plots.plot_pacf(full_res,lags=12)

from statsmodels.tsa.ar_model import AutoReg
model_ar=AutoReg(full_res, lags=[1])
model_fit=model_ar.fit()

print('Coefficients: %s '% model_fit.params)

pred_res=model_fit.predict(start=len(full_res),end=len(full_res)+len(predict_data)-1,dynamic=False)
pred_res.reset_index(drop=True,inplace=True)

final_pred=pred_new+pred_res
final_pred
