import json
import numpy as np
import pandas as pd
from math import sqrt
from numpy import split
from numpy import array
from numpy import concatenate
from pandas import read_csv
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense

from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM,Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from flask import Flask,request,jsonify


app = Flask(__name__)
@app.route('/ts_model', methods=['POST'])
def prediction():
    n,op= 24,24
    data = request.get_json()
    test = data['eto']


    model = load_model("C:\\AISCR\\Aiscr_bilstm_24-24-20230702T130124Z-001\\Aiscr_bilstm_24-24")
    test = np.array(test)
    print(test)
    test = test.reshape((-1,1))

    scaler = MinMaxScaler()
    test = scaler.fit_transform(test)

    test = test.reshape((1,24,1))

    print(test)
    y_pred = model.predict(test)
    y_pred = scaler.inverse_transform(y_pred)
    y_pred = y_pred.flatten().tolist()


    print(y_pred)

    return jsonify({'ETo': y_pred })



if __name__ == '__main__':
    app.run()

