#Net Radiation using temperature
import pyeto
from dateutil import tz
import pyeto.fao
import os
import filelock
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import math
from pyeto._check import (
    check_day_hours as _check_day_hours,
    check_doy as _check_doy,
    check_latitude_rad as _check_latitude_rad,
    check_sol_dec_rad as _check_sol_dec_rad,
    check_sunset_hour_angle_rad as _check_sunset_hour_angle_rad,
)
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import load_model
from flask import Flask, request, jsonify

from firebase_admin import credentials, firestore, initialize_app

app = Flask(__name__)
app.config['TIMEOUT'] = 60

scheduler = BackgroundScheduler(daemon=True)

# Initialize the Firebase app
cred = credentials.Certificate('user-details-3840f-firebase-adminsdk-pbwo6-00a5793239.json')
initialize_app(cred)

# Create a Firestore client
db = firestore.client()

last_execution_time = None

# File paths and names
prediction_lock_file = 'prediction.lock'
calculate_eto_lock_file = 'calculate_eto.lock'

# Check if the prediction lock file already exists
if not os.path.exists(prediction_lock_file):
    # Create an empty prediction lock file
    open(prediction_lock_file, 'w').close()

# Check if the calculate_eto lock file already exists
if not os.path.exists(calculate_eto_lock_file):
    # Create an empty calculate_eto lock file
    open(calculate_eto_lock_file, 'w').close()


@app.route('/calculate_eto')
def calculate_eto_route():
    print(f"calculate_eto_route - Current time: {datetime.now()}")

    global last_execution_time
    with app.app_context():
        # Check if the function has already run within the last hour
        if last_execution_time is not None and datetime.utcnow() - last_execution_time < timedelta(hours=1):
            print("calculate_eto_route - Already executed within the last hour")
            return jsonify({'error': 'Already executed within the last hour'})

        # Acquire a lock
        lock_file_path = 'calculate_eto.lock'
        lock = filelock.FileLock(lock_file_path)

        try:
            # Try to acquire the lock
            with lock.acquire(timeout=0):
                # Retrieve data from Firestore collection
                collection_ref = db.collection('hw')
                query = collection_ref.order_by('Date', direction=firestore.Query.DESCENDING).limit(1)
                docs = query.get()
                data = [doc.to_dict() for doc in docs]

                if not data:
                    return jsonify({'error': 'No data available'})

                data = data[-1]
                svp = pyeto.fao.svp_from_t(data['Tavg'])
                svpmax = pyeto.fao.svp_from_t(data['Tmax'])
                svpmin = pyeto.fao.svp_from_t(data['Tmin'])
                avp = pyeto.avp_from_rhmean(svpmin, svpmax,data['RHavg'])
            
                new_date = pd.to_datetime(data['Date'])
                doy = new_date.timetuple().tm_yday
                sol_dec = pyeto.sol_dec(doy)
                sunset_hour_angle = pyeto.sunset_hour_angle(latitude, sol_dec)
                day_light_hours = pyeto.daylight_hours(sunset_hour_angle)
                ird = pyeto.inv_rel_dist_earth_sun(doy)
                latitude = pyeto.deg2rad(11.285278)
                et_rad =pyeto.et_rad(latitude, sol_dec, sunset_hour_angle, ird)
                altitude = 60
                cs_rad = pyeto.cs_rad(altitude,et_rad)
                del_svp = pyeto.fao.delta_svp(data["Tavg"])
                psy = pyeto.fao.psy_const(pyeto.fao.atm_pressure(33))
                coastal = False
                sol_from_t =pyeto.sol_rad_from_t(et_rad,cs_rad, data['Tmin'], data['Tmax'], coastal)
                shortwave = pyeto.net_in_sol_rad(sol_from_t, albedo=0.23)
                longwave =pyeto.net_out_lw_rad(data['Tmax'], data['Tmin'],sol_from_t,cs_rad, avp)
                NR = pyeto.net_rad(shortwave, longwave)
                #NR = (data["NR"] * 24 * 60 * 60) / 1000000
                T = pyeto.celsius2kelvin(data["Tavg"])
                WS = data["Wind_Spd"]
                #del_svp = pyeto.fao.delta_svp(data["Tavg"])
                #psy = pyeto.fao.psy_const(pyeto.fao.atm_pressure(33))
                time = data['Date']
            
                # Calculate ETo using PyETo
                c = np.where(NR > 0, 0.24, 0.96)
                shf = np.where(NR > 0, 0.1 * NR, 0.5 * NR)
                a1 = (0.408 * (NR - shf) * del_svp / (del_svp + (psy * (1 + c * WS))))
                a2 = (37 * WS / T * (svp - avp) * psy / (del_svp + (psy * (1 + c * WS))))
                eto = a1+a2

                source_timezone = tz.gettz('UTC')

                # Set the destination timezone to IST
                destination_timezone = tz.gettz('Asia/Kolkata')

                # Get the current time in UTC
                utc_time = datetime.utcnow()

                # Convert the UTC time to IST
                ist_time = utc_time.replace(tzinfo=source_timezone).astimezone(destination_timezone)

                # Store ETo in Firestore collection
                new_collection_ref = db.collection('eto-hourly')
                new_doc_ref = new_collection_ref.document()
                new_doc_ref.set({
                    'timestamp': ist_time,
                    'date': time,
                    'eto': eto
                })
                last_execution_time = datetime.utcnow()

                return jsonify({'timestamp': ist_time, 'date': time, 'eto': eto})

        except filelock.Timeout:
            return jsonify({'error': 'Another process is already running'})

        finally:
            # Release the lock
            lock.release()


@app.route('/ts_model')
def prediction():
    print(f"prediction - Current time: {datetime.now()}")

    with app.app_context():
        # Acquire a lock
        lock_file_path = 'prediction.lock'
        lock = filelock.FileLock(lock_file_path)

        try:
            # Try to acquire the lock
            with lock.acquire(timeout=0):
                # Retrieve data from Firestore collection
                collection_ref = db.collection('ts')
                query = collection_ref.order_by('Index', direction=firestore.Query.DESCENDING).limit(24)
                snapshots = query.stream()
                data = [snapshot.to_dict() for snapshot in snapshots]
                data = sorted(data, key=lambda x: x['Index'])

                if not data:
                    return jsonify({'error': 'No data available'})

                index = [element['Index'] for element in data]
                date = [element['Date'] for element in data]
                eto = [element['ETo'] for element in data]
                new_index = index[-1] + 1

                dt_str = date[-1]
                dt = datetime.strptime(dt_str, "%d-%m-%Y %H:%M")
                updated_dt = dt + timedelta(hours=1)
                new_date = updated_dt.strftime("%d-%m-%Y %H:%M")

                # Load the trained model
                model = load_model("Aiscr_bilstm_24-24-20230702T130124Z-001//Aiscr_bilstm_24-24")

                # Preprocess the input data
                test = np.array(eto)
                test = test.reshape((-1, 1))
                scaler = MinMaxScaler()
                test = scaler.fit_transform(test)
                test = test.reshape((1, 24, 1))

                # Make predictions using the model
                y_pred = model.predict(test)
                y_pred = scaler.inverse_transform(y_pred)
                y_pred = float(y_pred[0][0])

                source_timezone = tz.gettz('UTC')
                destination_timezone = tz.gettz('Asia/Kolkata')
                utc_time = datetime.utcnow()
                ist_time = utc_time.replace(tzinfo=source_timezone).astimezone(destination_timezone)

                # Store predictions in Firestore collection
                doc_ref = db.collection('ts').document()
                doc_ref.set({
                    'Index': new_index,
                    'Timestamp': ist_time,
                    'Date': new_date,
                    'ETo': y_pred
                })

                return jsonify({'timestamp': ist_time, 'date': new_date, 'eto': y_pred, 'index': new_index})

        except filelock.Timeout:
            return jsonify({'error': 'Another process is already running'})


@app.route('/health')
def health_check():
    return 'OK'


if __name__ == '__main__':
    # Start the scheduler
    scheduler.start()

    # Add the tasks to the scheduler with desired schedules
    scheduler.add_job(calculate_eto_route, 'interval', hours=1, next_run_time=datetime.now())
    scheduler.add_job(prediction, 'interval', hours=1, next_run_time=datetime.now())

    # Run the Flask application
    app.run(debug=True)
