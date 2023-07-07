import pyeto
import pyeto.fao
from apscheduler.schedulers.background import BackgroundScheduler

from datetime import datetime, timedelta
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

import math
from pyeto._check import (
    check_day_hours as _check_day_hours,
    check_doy as _check_doy,
    check_latitude_rad as _check_latitude_rad,
    check_sol_dec_rad as _check_sol_dec_rad,
    check_sunset_hour_angle_rad as _check_sunset_hour_angle_rad,
)

#: Solar constant [ MJ m-2 min-1]
SOLAR_CONSTANT = 0.0820

# Stefan Boltzmann constant [MJ K-4 m-2 day-1]
STEFAN_BOLTZMANN_CONSTANT = 0.000000004903  #
"""Stefan Boltzmann constant [MJ K-4 m-2 day-1]"""

from flask import Flask,request,jsonify
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
app = Flask(__name__)

scheduler = BackgroundScheduler(daemon=True)


@app.route('/calculate_eto', methods=['POST'])
def calculate_eto():
    # Get the input parameters from the request


    # Initialize the Firebase app
    cred = credentials.Certificate('user-details-3840f-firebase-adminsdk-pbwo6-00a5793239.json')
    firebase_admin.initialize_app(cred)

    # Create a Firestore client
    db = firestore.client()

    # Retrieve data from a Firestore collection
    collection_ref = db.collection('hw')
    docs = collection_ref.get()
    query = collection_ref.order_by('Date', direction=firestore.Query.DESCENDING).limit(1)
    docs = query.get()
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    print(data)
    data = data[-1]
    svp = pyeto.fao.svp_from_t(data['Tavg'])
    svpmax = pyeto.fao.svp_from_t(data['Tmax'])
    svpmin = pyeto.fao.svp_from_t(data['Tmin'])
    avp = pyeto.avp_from_rhmean(svpmin, svpmax,data['RHavg'])
    NR = (data["NR"] * 24 * 60 * 60) / 1000000
    T = pyeto.celsius2kelvin(data["Tavg"])
    WS = data["Wind_Spd"]
    del_svp = pyeto.fao.delta_svp(data["Tavg"])
    psy = pyeto.fao.psy_const(pyeto.fao.atm_pressure(33))
    time = data['Date']

    # Calculate ETo using PyETo
    c = np.where(NR > 0, 0.24, 0.96)
    shf = np.where(NR > 0, 0.1 * NR, 0.5 * NR)
    a1 = (0.408 * (NR - shf) * del_svp / (del_svp + (psy * (1 + c * WS))))
    a2 = (37 * WS / T * (svp - avp) * psy / (del_svp + (psy * (1 + c * WS))))
    eto = a1+a2

    new_collection_ref = db.collection('eto-hourly')
    new_doc_ref = new_collection_ref.document()
    new_doc_ref.set({
        'time': datetime.now(),
        'date': time,
        'eto': eto
    })


    return jsonify({'date':time,
                    'eto': eto})

@app.route('/ts_model', methods=['POST'])
def prediction():
    cred = credentials.Certificate('user-details-3840f-firebase-adminsdk-pbwo6-00a5793239.json')
    firebase_admin.initialize_app(cred)

    # Create a Firestore client
    db = firestore.client()

    # Retrieve data from a Firestore collection
    collection_ref = db.collection('ts')
    query = collection_ref.order_by('Date', direction=firestore.Query.DESCENDING).limit(24)
    snapshots = query.stream()
    data = [snapshot.to_dict() for snapshot in snapshots]  # Extract the data from the DocumentSnapshot objects
    data = sorted(data, key=lambda x: x['Date'])  # Replace 'field_name' with the actual field name to sort by

    date =  []
    eto  = []
    print(data)
    for elements in data:
       date.append(elements['Date'])
       eto.append(elements['ETo'])

    print(date)


    new_date = []

    for dt_str in date:
        dt = datetime.strptime(dt_str, "%d-%m-%Y %H:%M")
        updated_dt = dt + timedelta(hours=24)
        new_date.append(updated_dt.strftime("%d-%m-%Y %H:%M"))
    print(new_date)
    #data = request.get_json()
    test = eto


    model = load_model("Aiscr_bilstm_24-24-20230702T130124Z-001/Aiscr_bilstm_24-24")
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

    joined_dict = {new_date[i]: y_pred[i] for i in range(len(new_date))}
    for key, value in joined_dict.items():
        doc_ref = db.collection('ts').document()
        doc_ref.set({
            'Date': key,
            'ETo': value
        })
    print(joined_dict)

    return jsonify(joined_dict)




if __name__ == '__main__':
    # Start the scheduler
    scheduler.start()

    # Add the tasks to the scheduler with desired schedules
    scheduler.add_job(calculate_eto, 'interval', hours=1)
    scheduler.add_job(prediction, 'interval', hours=24)

    # Run the Flask application

    app.run(debug=True)

