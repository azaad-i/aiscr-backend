import pyeto
from dateutil import tz
import pyeto.fao
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from keras.models import load_model
from flask import Flask, request, jsonify
import tensorflow as tf
from firebase_admin import credentials, firestore, initialize_app

app = Flask(__name__)

scheduler = BackgroundScheduler(daemon=True)

# Initialize the Firebase app
cred = credentials.Certificate('user-details-3840f-firebase-adminsdk-pbwo6-00a5793239.json')
initialize_app(cred)

# Create a Firestore client
db = firestore.client()

@app.route('/calculate_eto', methods=['GET'])
def calculate_eto():
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
    avp = pyeto.avp_from_rhmean(svpmin, svpmax, data['RHavg'])
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
    eto = a1 + a2

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

    return jsonify({'timestamp': ist_time,'date': time, 'eto': eto})


@app.route('/ts_model', methods=['GET'])
def prediction():
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
    new_index =int(index[-1])+1
    #for dt in index:
    #    new_index.append(dt+24)
    dt = datetime.strptime(date[-1], "%d-%m-%Y %H:%M")
    updated_dt = dt + timedelta(hours=1)
    new_date = updated_dt.strftime("%d-%m-%Y %H:%M")
    '''
    for dt_str in date:
        dt = datetime.strptime(dt_str, "%d-%m-%Y %H:%M")
        updated_dt = dt + timedelta(hours=1)
        new_date.append(updated_dt.strftime("%d-%m-%Y %H:%M"))
    '''
    # Load the trained model
    #model = load_model("bilstm_24-1(str) (1).h5")
    interpreter = tf.lite.Interpreter(model_path="converted_model.tflite(24-1)")
    interpreter.allocate_tensors()
    test = np.array(eto)
    test = test.reshape((-1, 1))
    scaler = MinMaxScaler()
    test_scaled = scaler.fit_transform(test)
    test_scaled = test_scaled.reshape((1, 24, 1))

    test_scaled = test_scaled.astype(np.float32)

    # Perform inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], test_scaled)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-process the output data
    y_pred = scaler.inverse_transform(output_data)
    y_pred = y_pred.flatten().tolist()

    '''

    # Preprocess the input data
    test = np.array(eto)
    test = test.reshape((-1, 1))
    scaler = MinMaxScaler()
    test = scaler.fit_transform(test)
    test = test.reshape((1, 24, 1))

    # Make predictions using the model
    y_pred = model.predict(test)
    y_pred = scaler.inverse_transform(y_pred)
    y_pred = y_pred.flatten().tolist()
    '''
    source_timezone = tz.gettz('UTC')

    # Set the destination timezone to IST
    destination_timezone = tz.gettz('Asia/Kolkata')

    # Get the current time in UTC
    utc_time = datetime.utcnow()

    # Convert the UTC time to IST
    ist_time = utc_time.replace(tzinfo=source_timezone).astimezone(destination_timezone)
    # Prepare the results dictionary
    joined_dict = {new_date: y_pred }
    # Store predictions in Firestore collection
    for key, value in joined_dict.items():
        doc_ref = db.collection('ts').document()
        doc_ref.set({
            'Index':new_index,
            'Date': key,
            'ETo': value[-1],
            'Timestamp':ist_time
        })

    return jsonify({'Index':new_index,'timestamp': ist_time,'date': key, 'eto': value[-1]})

@app.route('/health')
def health_check():
    return 'OK'


if __name__ == '__main__':
    # Start the scheduler
    scheduler.start()

    # Add the tasks to the scheduler with desired schedules
    scheduler.add_job(calculate_eto, 'interval', minutes=1, start_date=datetime.now())
    scheduler.add_job(prediction, 'interval', minutes=1, start_date=datetime.now())


    # Run the Flask application
    app.run(debug=True)
