import pandas as pd
import numpy as np


# Генерим фичи с таблицы chronom
def prepare_chronom_features(chronom):

    chronom['VR_NACH'] = pd.to_datetime(chronom['VR_NACH'])
    chronom['VR_KON'] = pd.to_datetime(chronom['VR_KON'])

    temp = chronom[chronom['NOP'] == 'Продувка']
    chronom = chronom.merge(temp[['VR_NACH', 'NPLV']].rename(columns={'VR_NACH': 'nach'}), on='NPLV')
    #chronom = chronom[chronom['VR_NACH'] <= chronom['nach']]
    chronom_features = pd.DataFrame(index=temp['NPLV'])
    chronom_features['duration'] = (temp['VR_KON'] - temp['VR_NACH']).dt.seconds.values
    chronom_features['O2_sum'] = chronom.groupby('NPLV')['O2'].sum().values
    chronom_features.head()
    return chronom_features


# Генерим фичи с таблицы sap
def prepare_sip_features(sip):

    sip_features = sip.pivot_table(index='NPLV', columns='NMSYP', values='VSSYP', aggfunc=['mean', 'sum', 'count', 'std'])
    sip_features = sip.groupby(['NPLV'])[['VSSYP']].mean()
    #sip_features.columns = ['_'.join(x) for x in sip_features.columns]
    return sip_features

def prepare_chugun_features(chugun):
    chugun_features = chugun.drop(columns='DATA_ZAMERA').set_index('NPLV')
    return chugun_features

# produv

def prepare_produv_features(produv, chronom):
    temp = chronom[chronom['NOP'] == 'Продувка']
    produv['SEC'] = pd.to_datetime(produv['SEC'])
    produv = produv.merge(temp, on='NPLV')
    produv = produv[(produv['SEC'] < produv['VR_KON'])]
    #produv = produv[(produv['SEC'] > produv['VR_NACH']) & (produv['SEC'] < produv['VR_KON'])]

    produv_features = produv.groupby('NPLV')[['RAS', 'POL']].agg(['mean', 'max', 'median', 'sum', 'count'])
    produv_features.columns = ['_'.join(x) for x in produv_features.columns]
    return produv_features

# lom

def prepare_lom_features(lom):
    lom = lom[lom['NML'] != 'НБ  ']
    lom_features = lom.pivot_table(index='NPLV', columns='NML', values='VES', aggfunc=['sum']).fillna(0)
    #lom['VES'] = lom.groupby('NPLV')['VES'].mean()
    
    lom_features.columns = ['_'.join(x) for x in lom_features.columns]
    return lom_features


# gas

def prepare_gas_features(gas, chronom):
    features_cols = gas.columns[2: ]
    temp = chronom[chronom['NOP'] == 'Продувка'][['NPLV', 'VR_KON']]
    gas['Time'] = pd.to_datetime(gas['Time'])
    gas = gas.merge(temp, on='NPLV')
    gas = gas[gas['Time'] <= gas['VR_KON']]
    
    gas_features = gas.groupby('NPLV')[features_cols].agg(['mean', 'sum', 'max', 'min'])
    gas_features.columns = ['_'.join(x) for x in gas_features.columns]
    return gas_features
