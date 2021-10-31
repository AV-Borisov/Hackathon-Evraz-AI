import pandas as pd
import numpy as np
import lightgbm as lgb
from feature_engineering import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge

if __name__ == '__main__':
    
    folder = 'data_task1/'
    
    ss = pd.read_csv(folder + 'sample_submission.csv')
    tables = ['sip', 'chugun', 'produv', 'lom', 'plavki', 'chronom', 'target']
    
    # читаем тренировочные таблицы
    sip = pd.read_csv(folder + 'sip_train.csv')
    chugun = pd.read_csv(folder + 'chugun_train.csv')
    produv = pd.read_csv(folder + 'produv_train.csv')
    lom = pd.read_csv(folder + 'lom_train.csv')
    plavki = pd.read_csv(folder + 'plavki_train.csv')
    chronom = pd.read_csv(folder + 'chronom_train.csv')
    gas = pd.read_csv(folder + 'gas_train.csv')
    target = pd.read_csv(folder + 'target_train.csv')
    
    # Готовим датасет
    chronom_features = prepare_chronom_features(chronom)
    sip_features = prepare_sip_features(sip)
    chugun_features = prepare_chugun_features(chugun)
    produv_features = prepare_produv_features(produv, chronom)
    lom_features = prepare_lom_features(lom)
    gas_features = prepare_gas_features(gas, chronom)
    
    X = pd.concat([chronom_features, chugun_features, produv_features, lom_features, gas_features], axis=1)
    y = target.set_index('NPLV')
    
    f_columns = X.columns
    X.columns = [f'feature_{i}' for i in range(X.shape[1])]
    
    scaler = StandardScaler()
    X_ = scaler.fit_transform(X.loc[:, X.isna().sum() == 0].clip(0, 10000))
    
    
    # Обучение
    regT = lgb.LGBMRegressor(num_leaves=7, n_estimators=100)
    #regT = LinearRegression()
    regT.fit(X, np.log(y['TST'].clip(1550, 1700)))
    
    regC = Ridge(50)
    regC.fit(X_,  np.log(1 + y['C'].fillna(0.04).clip(0.02, 0.08)))
    
    # inference 
    
    ss = pd.read_csv(folder + 'sample_submission.csv')
    
    # читаем тестовые таблицы
    
    sip = pd.read_csv(folder + 'sip_test.csv')
    chugun = pd.read_csv(folder + 'chugun_test.csv')
    produv = pd.read_csv(folder + 'produv_test.csv')
    lom = pd.read_csv(folder + 'lom_test.csv')
    plavki = pd.read_csv(folder + 'plavki_test.csv')
    chronom = pd.read_csv(folder + 'chronom_test.csv')
    gas = pd.read_csv(folder + 'gas_test.csv')
    
    chronom_features = prepare_chronom_features(chronom)
    sip_features = prepare_sip_features(sip)
    chugun_features = prepare_chugun_features(chugun)
    produv_features = prepare_produv_features(produv, chronom)
    lom_features = prepare_lom_features(lom)
    gas_features = prepare_gas_features(gas, chronom)
    
    X_test = pd.concat([chronom_features, chugun_features, produv_features, lom_features, gas_features], axis=1)
    X_test.columns = [f'feature_{i}' for i in range(X_test.shape[1])]
    
    X_test = X_test[X.columns]
    
    ss['TST'] = np.exp(regT.predict(X_test))
    ss['C'] = regC.predict(scaler.transform(X_test.clip(0, 10000).loc[:, X.isna().sum() == 0]))
    
    ss.to_csv('submission.csv', index=False)
    