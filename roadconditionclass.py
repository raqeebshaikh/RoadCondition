

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class prepare_modelinput():
    start_stop_var = None
    add_lat_var = pd.DataFrame()
    remove_nan_var = pd.DataFrame()
    shift_pred_var = pd.DataFrame()
    prepare_modelinput_var = None
    train_input = None
    prediction_out = np.array([])
    lattitude = None
    longitude = None
    
    def __init__(self,gps,gyroscope,model_path,timestep):
        self.gps = gps
        self.gyroscope = gyroscope
        self.model_path = model_path
        self.timestep = timestep

        
    def get_index_col(self,clm):
        val = self.gps.columns.to_list().index(clm)
        return val

    def start_stop(self):
        '''
        Dataframe of GPS which has LAT,LON
        Return {'time':,'latlon':}
        '''
        if self.start_stop_var != None:
            
            return self.start_stop_var
        time = []
        lat = self.gps['Latitude'].unique()
        self.lattitude = lat
        lon = self.gps['Longitude'].unique()
        self.longitude = lon
        latlon = list(zip(lat,lon))
        for val in latlon:
    #         print(val)
            latlo = self.gps[(self.gps['Latitude'] == val[0]) | (self.gps['Longitude'] == val[1])].iloc[0][self.get_index_col('Timestamp')]
            lonlo = self.gps[(self.gps['Latitude'] == val[0]) | (self.gps['Longitude'] == val[1])].iloc[-1][self.get_index_col('Timestamp')]
            time.append((latlo,lonlo))
        self.start_stop_var = {'time':time,'latlon':latlon}
        return {'time':time,'latlon':latlon}

    def middle_vec(self,v1,v2,points,plot=False,detail=False):
        '''
            Between two vectors it calculates addisiional vector depending on vectors required (points)

            Takes lattitude and longitude array and step len(X) 
                eg -    x = np.array([[18.5346199,73.93349]])
                        y = np.array([[18.5300897,73.9324609]])

            Returns      - middle_vec(x,y,4000)
            Outpus Shape - (4000,2

            Attribute---------------
            1) v1 - vector shape(1,2)
            2) v2 - vector shape(1,2)
            3) points - Total vectors to generate between two vectors
            4) plot - plot the points Default - False
        '''
        eucledian_distance = np.linalg.norm(v2-v1)
        total_coordinate = np.linspace(0,eucledian_distance,points)
        vector_ratio = total_coordinate/eucledian_distance
        p3 = (v2-v1).T
        p2 = vector_ratio.reshape(-1,1)
        out = p3*p2.T
        final = v1+out.T

        if detail == True:
            print('Eculedian Distance {0:.5f}'.format(eucledian_distance))
            print('Total Vector LineSpace {}'.format(total_coordinate.shape))
        if plot==True:
            p0 ,p1 = np.squeeze(np.dstack(np.concatenate((v1,v2))))
            plt.scatter(final[:,0],final[:,1])
            plt.scatter(p0,p1,c='red')

        return final

    def add_lat(self,detail=False):
        '''
        Take in Gyroscope DataFrame and Time,Lattitude and Longitude Dict from Function (start_stop) 
        Return  -
                Gyroscope DataFrame with aded Lat,Lon colum divides with eculedian distance .
                ['Timestamp', 'Milliseconds', 'X', 'Y', 'Z', 'Lat', 'Lon']
        '''
        
        if self.add_lat_var.empty == False:
#             print('add_lat from chase')
            return self.add_lat_var

        self.gyroscope['Lat'],self.gyroscope['Lon'] = [np.nan,np.nan]

        first = 0
        latval = 0
        lonval = 1
        lp = 0
        for val in self.start_stop()['time']:
            lp +=1


            filter_data = self.gyroscope.loc[self.gyroscope.loc[self.gyroscope['Timestamp'] == val[0]].index[first]:self.gyroscope.loc[self.gyroscope['Timestamp'] == val[1]].index[-1]]

    #         lon = np.expand_dims(np.array(time_lat['latlon'][lonval]),axis=0)
            try:
                lon = np.expand_dims(np.array(self.start_stop()['latlon'][lonval]),axis=0)
                lat = np.expand_dims(np.array(self.start_stop()['latlon'][latval]),axis=0)
            except:

                lat = np.expand_dims(np.array(self.start_stop()['latlon'][latval-1]),axis=0)
                lon = np.expand_dims(np.array(self.start_stop()['latlon'][lonval-1]),axis=0)
            latval += 1
            lonval += 1

            allvec = self.middle_vec(lat,lon,len(filter_data))

            self.gyroscope.loc[self.gyroscope.loc[self.gyroscope['Timestamp'] == val[0]].index[first]:self.gyroscope.loc[self.gyroscope['Timestamp'] == val[1]].index[-1],'Lat']= allvec[:,0]
            self.gyroscope.loc[self.gyroscope.loc[self.gyroscope['Timestamp'] == val[0]].index[first]:self.gyroscope.loc[self.gyroscope['Timestamp'] == val[1]].index[-1],'Lon']= allvec[:,1]
            first = -1


        if detail == True:
            print('Len filter_data : {}'.format(len(filter_data)))
            print('Euclideal allvec: {}'.format(allvec.shape))
            print('Out DataFrame Details:','='*25)
            print(self.gyroscope.isna().sum())
            print('gyro Dataframe len : {}'.format(len(self.gyroscope)))
            print('Unique : Lattitude - {},Longitude - {}'.format(len(self.gyroscope['Lat'].unique()),len(self.gyroscope['Lat'].unique())))
            print('='*50)
            pass
        self.add_lat_var = self.gyroscope
        return self.gyroscope
    
    def remove_nan(self,detail=False):
        '''
        Remove nan values of Lat,Lon 
        Return - Clean DataFrame without nan values
        '''
        if self.remove_nan_var.empty == False:
#             print('remove_nan from chase')
            return self.remove_nan_var
        self.add_lat().dropna(inplace=True)

        if detail == True:
            print('Dataframe len : {}'.format(len(self.add_lat())))
            print('Output DataFrame Details:','='*25)
            print(self.add_lat().isna().sum())
            print('='*50)
        
        self.remove_nan_var = self.add_lat()
        return self.add_lat()

    def preprocess_modelinput(self):
        '''
        Takes Clean data with columns [X,Y,Z,LAT,LON]
        and generate sample to input to model of shape(-1,timestep,features)
        eg-   (12827,100,3)

        Input  - preprocess_modelinput(cleandata,100,2,5)
        Output - data of shape (-1, 100, 3) 

        Directely Feed this output to model.predict(out)

        Sequence - After remove_nan()
        '''
        
        X = []
        x = 0
        y = self.timestep
        while y <= len(self.remove_nan()):
            X.append(self.remove_nan().iloc[x:y,[self.remove_nan().columns.to_list().index('X'),self.remove_nan().columns.to_list().index('Y'),self.remove_nan().columns.to_list().index('Z')]].values)

            x += 1
            y += 1
    #     data.shift(-len(X)).dropna(inplace=True)
        self.train_input = np.array(X)
        return np.array(X)
# ----
    def make_prediction(self):
        if self.prediction_out.shape[0] != 0:
            return self.prediction_out
        model = pickle.load( open( self.model_path, "rb" ))
        self.prediction_out = model.predict(self.preprocess_modelinput())
        return model.predict(self.preprocess_modelinput())
    
    # lenshift = len(cleandata)-inp_data.shape[0]
    def shift_pred(self):
        if self.shift_pred_var.empty == False:
#             print('add_lat from chase')
            return self.shift_pred_var
        shiftlen = len(self.make_prediction())
        prediction =np.argmax(self.make_prediction(), axis=1).reshape(-1,1)
        data_shift = self.remove_nan_var[['Lat','Lon']].shift(-((len(self.remove_nan_var)-shiftlen))).dropna()

        data_shift['Condition'] = prediction
        print(prediction.shape)
        print(len(data_shift))
        self.shift_pred_var = data_shift
        return self.shift_pred_var

    def clean_adujst_accuracy(self,window,min_wind=5):
        data = self.shift_pred()['Condition'].rolling(window,min_wind,True).max()
        self.shift_pred()['CleanCondition'] = data

        return self.shift_pred()

