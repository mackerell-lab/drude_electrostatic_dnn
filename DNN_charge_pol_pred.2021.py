#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Script related to: 
Deep Neural Network model to predict the electrostatic parameters in the polarizable classical Drude oscillator force field
Anmol Kumar, Poonam Pandey, Payal Chatterjee and Alexander D. MacKerell Jr.

"""

import numpy as np
import pandas as pd
from tensorflow import keras
from collections import OrderedDict 

def load_train_charge():
    charge_fea_train=pd.read_pickle('dgenff_dataset.2021/train_charge_feature.pkl')
    charge_target_train=pd.read_pickle('dgenff_dataset.2021/train_charge_target.pkl')
    train_charge_dataset=charge_fea_train.iloc[:,1:].values
    train_charge_target=charge_target_train.iloc[:,1].values
    train_charge_molid=np.array(charge_fea_train.index)
    train_charge_atomid=charge_fea_train.iloc[:,0].values
    return train_charge_molid,train_charge_atomid,train_charge_dataset,train_charge_target

def load_test_charge():
    charge_fea_test=pd.read_pickle('dgenff_dataset.2021/test_charge_feature.pkl')
    charge_target_test=pd.read_pickle('dgenff_dataset.2021/test_charge_target.pkl')
    test_charge_dataset=charge_fea_test.iloc[:,1:].values
    test_charge_target=charge_target_test.iloc[:,1].values
    test_charge_molid=np.array(charge_fea_test.index)
    test_charge_atomid=charge_fea_test.iloc[:,0].values
    return test_charge_molid,test_charge_atomid,test_charge_dataset,test_charge_target

def load_train_pol():
    alphathole_fea_train=pd.read_pickle('dgenff_dataset.2021/train_alphathole_feature.pkl')
    alphathole_target_train=pd.read_pickle('dgenff_dataset.2021/train_alphathole_target.pkl')
    train_alphathole_dataset=alphathole_fea_train.iloc[:,1:].values
    train_alpha_target=alphathole_target_train.iloc[:,1].values
    train_thole_target=alphathole_target_train.iloc[:,2].values
    train_alphathole_molid=np.array(alphathole_fea_train.index)
    train_alphathole_atomid=alphathole_fea_train.iloc[:,0].values
    return train_alphathole_molid,train_alphathole_atomid,train_alphathole_dataset,train_alpha_target,train_thole_target

def load_test_pol():
    alphathole_fea_test=pd.read_pickle('dgenff_dataset.2021/test_alphathole_feature.pkl')
    alphathole_target_test=pd.read_pickle('dgenff_dataset.2021/test_alphathole_target.pkl')
    test_alphathole_dataset=alphathole_fea_test.iloc[:,1:].values
    test_alpha_target=alphathole_target_test.iloc[:,1].values
    test_thole_target=alphathole_target_test.iloc[:,2].values
    test_alphathole_molid=np.array(alphathole_fea_test.index)
    test_alphathole_atomid=alphathole_fea_test.iloc[:,0].values
    return test_alphathole_molid,test_alphathole_atomid,test_alphathole_dataset,test_alpha_target,test_thole_target

def DNN_model(input_shape=[1]):
    activation_func="relu"
    optimizer=keras.optimizers.Adam(lr=0.0005)
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(keras.layers.Dense(1024, activation=activation_func,kernel_initializer='he_normal',kernel_constraint=keras.constraints.MaxNorm(3)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(512, activation=activation_func,kernel_initializer='he_normal',kernel_constraint=keras.constraints.MaxNorm(3)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

if __name__ == "__main__":
    pred_charge_dict=OrderedDict()
    pred_alphathole_dict=OrderedDict()
    pred_charge_dict['MOLID'],pred_charge_dict['ATOMID'],test_charge_features,test_charge=load_test_charge()
    pred_alphathole_dict['MOLID'],pred_alphathole_dict['ATOMID'],test_alphathole_features,test_alpha,test_thole=load_test_pol()
    pred_charges = pd.DataFrame(pred_charge_dict)
    pred_charges['QM-CHARGE']=test_charge
    pred_alphathole = pd.DataFrame(pred_alphathole_dict)
    pred_alphathole['QM-ALPHA']=test_alpha
    pred_alphathole['QM-THOLE']=test_thole
    
    train_charge_molid,train_charge_atomid,train_charge_features,train_charge=load_train_charge()
    train_alphathole_molid,train_alphathole_atomid,train_alphathole_features,train_alpha_target,train_thole_target=load_train_pol()
    
    model_charge=DNN_model(input_shape=[len(train_charge_features[0])])
    model_charge.load_weights('dgenff_dnn_model/dnn.charge.h5')
    model_alpha=DNN_model(input_shape=[len(train_alphathole_features[0])])
    model_alpha.load_weights('dgenff_dnn_model/dnn.alpha.h5')
    model_thole=DNN_model(input_shape=[len(train_alphathole_features[0])])
    model_thole.load_weights('dgenff_dnn_model/dnn.thole.h5')
    
    Pred_test_charge = model_charge.predict(test_charge_features)
    Pred_test_alpha = model_alpha.predict(test_alphathole_features)
    Pred_test_thole = model_thole.predict(test_alphathole_features)
    
    pred_charges['ML-CHARGE']=Pred_test_charge.flatten()
    pred_alphathole['ML-ALPHA']=Pred_test_alpha.flatten()
    pred_alphathole['ML-THOLE']=Pred_test_thole.flatten()
    pred_charges.set_index("MOLID", inplace = True) 
    pred_alphathole.set_index("MOLID", inplace = True) 

    pred_charges.to_csv('dnn_predicted_charges.csv',float_format='%0.3f')
    pred_alphathole.to_csv('dnn_predicted_alphathole.csv',float_format='%0.3f')

        
        
        
        
