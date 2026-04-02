import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import logging

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

"""
CONFIGURACIÓN DE LOGS
"""
today = datetime.today().strftime("%Y-%m-%d")

log_file = f"logs/update_PRED_{today}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

"""
CONFIGURACIÓN:
Datos necesarios para la extracción de información.
"""

#documentos donde se tiene la información histórica
csv_LME = "global_file_2021-03-15_to_2026-03-14.csv"
csv_PRED = "predict_2026-03-22.csv"
#información del modelo de la red neuroonal
model = load_model("modelo_lstm3.keras")
window = 63
days = 5 

"""
PREDICCIÓN DEL VALOR DEL LME: 
Predice el valor del LME utilizando la información histórica con la
que se cuenta en el archivo csv_LME. La variable que predice y la información que 
se alimenta al modelo depende de la arquitectura de la red neuronal generada. 
En este caso toma la información de las variables CASH_OFFER y 3_MONTH y devuelve una predicción 
para el valor inmediato posterior de CASH_OFFER. Se estiman varios días para asegurar la continuidad
dentro del archivo histórico. 
"""
def predict_last_n_days(df, model, window, days):

    #Escalar información de entreda al modelo
    scaler = MinMaxScaler()
    scaler.fit(df)
    data_scaled = scaler.transform(df)

    results = []

    #Ciclo que genera n predicciones para n días empezando del día mas reciente - n días. 
    #Recorre hasta el último día.
    for i in range(days):
        # Índice del punto de corte
        end_idx = len(df) - days + i

        # Tomar ventana histórica REAL
        window_data = data_scaled[end_idx - window:end_idx]

        # Asegurar tamaño correcto. Si el tamaño de los datos tomados para window es menor
        #que el window necesario salta la iteración. 
        if len(window_data) < window:
            continue

        X = np.expand_dims(window_data, axis=0)

        # Predicción
        pred_scaled = model.predict(X, verbose=0)

        dummy = np.zeros((1, data_scaled.shape[1]))
        dummy[0, 0] = pred_scaled[0, 0]

        pred = scaler.inverse_transform(dummy)[0, 0]

        # Fecha objetivo (día siguiente al corte)
        pred_date = df.index[end_idx]

        # Valor real (si existe)
        real_value = df.iloc[end_idx]["cash_offer"]

        results.append((pred_date, pred, real_value))

    # Última predicción (futuro)
    last_window = data_scaled[-window:]
    X = np.expand_dims(last_window, axis=0)

    pred_scaled = model.predict(X, verbose=0)

    dummy = np.zeros((1, data_scaled.shape[1]))
    dummy[0, 0] = pred_scaled[0, 0]

    future_pred = scaler.inverse_transform(dummy)[0, 0]

    future_date = df.index[-1] + pd.Timedelta(days=1)

    results.append((future_date, future_pred, np.nan))

    return results

"""
SCRIPT PRINCIPAL
"""

def main():
    try:
        logging.info("Inicio del programa")
        #Obtención de información de entrada para la red nueronal 
        df_global = pd.read_csv(csv_LME, parse_dates=["date"], index_col="date")
        #Considerar únicamente datos reales 
        df_global = df_global[df_global["interpolated_LME"] == False]

        #Variables de entrada al modelo
        features = ["cash_offer", "3-month_price"]
        df_global = df_global[features].copy()

        logging.info("Calculando predicciones")
        #Calculo de predecciones
        results = predict_last_n_days(df_global, model, window, days)
        df_pred = pd.DataFrame(results, columns=["date", "prediction", "real"])

        #Cálculo del error porcentual vs el dato real
        df_pred["error_pct"] = (
        (df_pred["prediction"] - df_pred["real"]) / df_pred["real"]
        ) * 100
        df_pred = df_pred.set_index("date")

        logging.info(f"Predicciones calculadas: \n{(df_pred)}\n")

        logging.info("Actualizando base de datos")
        #Lectura de csv con la base de datos 
        if os.path.exists(csv_PRED):
            df_hist = pd.read_csv(csv_PRED, index_col=0, parse_dates=True)
        else:
            df_hist = pd.DataFrame(columns=["prediction", "real", "error_pct"])

        # unir datos nuevos con los ya existentes
        df_hist = pd.concat([df_hist, df_pred])

        # eliminar duplicados
        df_hist = df_hist[~df_hist.index.duplicated(keep="last")]

        # ordenar información
        df_hist = df_hist.sort_index()

        #Completar rango de fechas completas
        full_range = pd.date_range(start=df_hist.index.min(), end=df_hist.index.max(), freq="D")

        #reindexar con el rango completo de fechas 
        df_hist = df_hist.reindex(full_range)
        df_hist.index.name = "date"

        # forward fill (solo con las predicciones, los datos reales que queden vacios indican días no hábiles)
        df_hist["prediction"] = df_hist["prediction"].ffill()

        df_hist.to_csv(csv_PRED)
        logging.info("Base actualizada correctamente\n")

    except Exception as e:
        logging.error(f"Error en ejecución: {e}")


if __name__ == "__main__":
    main()

