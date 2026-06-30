import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import logging
import paho.mqtt.client as mqtt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier


logging.basicConfig(level=logging.INFO, format='%(asctime)s - [AI NODE] - %(message)s')


MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
TOPIC_TELEMETRY = "rased/+/telemetry"      
TOPIC_ML_RESPONSE = "rased/ml_response" 

DATA_CSV_PATH = r"C:\Users\A7med\Downloads\smoke_detection_iot.csv.zip"
MODEL_PATH = "smoke_rf_model.pkl"
SCALER_PATH = "smoke_scaler.pkl"

FEATURES = [
    'Temperature[C]', 'Humidity[%]', 'Pressure[hPa]', 
    'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 
    'PM1.0', 'PM2.5'
]
def remove_outliers_iqr(data, columns):
    df_out = data.copy()
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.5 * IQR 
        upper_bound = Q3 + 2.5 * IQR
        df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]
    return df_out


def train_and_save_model():
    logging.info(f"Model not found. Starting training process using {DATA_CSV_PATH}...")
    
    if not os.path.exists(DATA_CSV_PATH):
        logging.error(f"Dataset '{DATA_CSV_PATH}' not found! Please check the path.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(DATA_CSV_PATH)
    except Exception as e:
        logging.error(f"Error reading the CSV file: {e}")
        sys.exit(1)

    if df is not None:
        logging.info("Cleaning data and removing outliers...")
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        
        cols_to_clean = ['Temperature[C]', 'Humidity[%]', 'Pressure[hPa]']
        df_cleaned = remove_outliers_iqr(df, cols_to_clean) 
        
        X = df_cleaned[FEATURES]
        y = df_cleaned['Fire Alarm']
        
        logging.info("Training Model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=15, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        logging.info("Training complete! Model and Scaler saved successfully.")


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info(f"Connected securely to MQTT Broker ({MQTT_BROKER}).")
        client.subscribe(TOPIC_TELEMETRY)
        logging.info(f"Listening for live sensor telemetry on: {TOPIC_TELEMETRY}")
    else:
        logging.error(f"Failed to connect. Return code: {rc}")
def on_message(client, userdata, msg):
    try:
        raw_data = msg.payload.decode()
        payload = json.loads(raw_data)
        tel = payload.get("telemetry")
        
        if not tel:
            return

        live_features_df = pd.DataFrame([[
            tel.get("temp", 0.0),
            tel.get("hum", 0.0),
            tel.get("pres", 0.0),
            tel.get("tvoc", 0.0),
            tel.get("eco2", 0.0),
            tel.get("H", 0.0),
            tel.get("C", 0.0),
            tel.get("pm1_0", 0.0),
            tel.get("pm2_5", 0.0)
        ]], columns=FEATURES)

        scaled_features = global_scaler.transform(live_features_df)
        confidence = global_model.predict_proba(scaled_features)[0][1] * 100

        if confidence >= 5.0:
            print(f"[ALARM] -> FIRE DETECTED! (Conf: {confidence:.1f}%) | Temp: {tel.get('temp')}C")
            client.publish(TOPIC_ML_RESPONSE, "fire")
        else:
            print(f"[SAFE] -> Normal. (Fire Prob: {confidence:.1f}%) | Temp: {tel.get('temp')}C")
            client.publish(TOPIC_ML_RESPONSE, "normal")

    except json.JSONDecodeError:
        pass 
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        train_and_save_model()
    
    logging.info("Loading Model & Scaler for live inference...")
    global_model = joblib.load(MODEL_PATH)
    global_scaler = joblib.load(SCALER_PATH)
    
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever() 
    except KeyboardInterrupt:
        logging.info("Shutting down AI Node...")
        client.disconnect()
