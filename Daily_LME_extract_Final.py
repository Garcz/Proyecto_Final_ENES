import pandas as pd
import json
import requests
import time
import logging

from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

"""
CONFIGURACIÓN DE LOGS
"""
today = datetime.today().strftime("%Y-%m-%d")

log_file = f"logs/update_{today}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

"""
CONFIGURACIÓN:
Datos necesarios para la extracción de información.
"""

# URL del metal
metal_url = "https://www.lme.com/en/metals/non-ferrous/lme-aluminium#Price+graphs"
# datasource del gráfico Official Prices
datasource_official = "dddbc815-1a81-4f35-beed-6a193f4c946a"
# datasource del gráfico Closing Prices
datasource_closing = "37cf78d1-222d-4618-b9f4-c5f13ff421e6"

# token de consulta de banxico
BANXICO_TOKEN = "8597637d49292ac0004ebf2efdc0bfec10d2a1794f28f3c60c7be2ff65446961"
# serie para consulta del tipo de cambio USD-MXN (fix)
SERIE = "SF43718"

# archivo que contiene la base de datos
csv_file = "global_file_2021-03-15_to_2026-03-14.csv"

"""
INICIAR SELENIUM:
Utiliza chrome, abrirá una ventana nueva, una vez iniciado el 
navegador no debe ser cerrada hasta que termine de correr el programa. 
Este código no funciona en google colab ya que la página del LME detecta 
el intento de scrapping.
"""

def start_driver():

    options = Options()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)

    return driver

"""
DESCARGAR DATA LME:
Extrae la información para el metal seleccionado del periodo de tiempo
seleccionado. Se extra la información de las gráficas:
"official price" y el "closing price".
"""

def download_lme_data(driver, datasource_id, start_date, end_date):

    script = f"""
    var callback = arguments[arguments.length - 1];

    fetch("https://www.lme.com/api/trading-data/chart-data?datasourceId={datasource_id}&startDate={start_date}&endDate={end_date}")
      .then(response => response.json())
      .then(data => callback(JSON.stringify(data)))
      .catch(error => callback("ERROR"));
    """

    result = driver.execute_async_script(script)

    data = json.loads(result)

    labels = data["Labels"]

    rows = {}

    for dataset in data["Datasets"]:

        row = dataset["RowTitle"].lower().replace(" ", "-")
        label = dataset["Label"].lower()

        column_name = f"{row}_{label}"

        rows[column_name] = dataset["Data"]

    df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(labels, dayfirst=True)

    return df

"""
DESCARGAR DATA Tipo de cambio DOF (BANXICO):
Extrae la información para el tipo de cambio del 
periodo de tiempo seleccionado.  
"""

def get_fx_rate(start_date, end_date):

    url = (
        "https://www.banxico.org.mx/SieAPIRest/service/v1/series/"
        f"{SERIE}/datos/"
        f"{start_date}/{end_date}"
    )

    headers = {
        "Bmx-Token": BANXICO_TOKEN
    }

    response = requests.get(url, headers=headers)

    data = response.json()

    datos = data["bmx"]["series"][0]["datos"]

    df_fx = pd.DataFrame(datos)

    # convertir fecha
    df_fx["date"] = pd.to_datetime(df_fx["fecha"], dayfirst=True)

    # limpiar valores no disponibles
    df_fx = df_fx[df_fx["dato"] != "N/E"]

    # convertir a número
    df_fx["usd_mxn_fix"] = df_fx["dato"].astype(float)

    # columnas finales
    df_fx = df_fx[["date", "usd_mxn_fix"]]

    return df_fx


"""
SCRIPT PRINCIPAL
"""
def main():
    try:
        logging.info("Inicio del programa")

        """
        FECHAS DE EXTRACCIÓN:
        Determina cual es la información más reciente que contiene la base de datos 
        y determina el periodo para extraer información. Se extraerán 5 dias. Desde 
        cuatro días previo al dato más reciente hasta un día posterior del último dato. 
        """

        df_global = pd.read_csv(csv_file)
        df_global["date"] = pd.to_datetime(df_global["date"], dayfirst=False, errors="coerce")
        last_date_db = df_global["date"].max()
        start_date = last_date_db + timedelta(days=-5)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")

        #Arrancar buscador en selenium
        logging.info("Arranque del buscador")
        driver = start_driver()
        driver.get(metal_url)
        time.sleep(5)

        #Extracción de información del LME
        logging.info("Inicia extracción de datos del LME")
        df_official = download_lme_data(
            driver,
            datasource_official,
            start_date,
            end_date
        )

        df_closing = download_lme_data(
            driver,
            datasource_closing,
            start_date,
            end_date
        )
        logging.info(f"Datos extraidos LME: \n{(df_official[["date","cash_offer"]])}")

        #Unión de los datos del closing price y official price en un solo df
        df_new = pd.merge(
            df_official,
            df_closing,
            on="date",
            how="outer"
        )

        if df_new.empty:
            print("No hay nuevos datos del LME.")
            exit()

        #Extracción de información del tipo de cambio del DOF
        logging.info("Inicia extracción de datos del tipo de cambio")
        usd_mxn = get_fx_rate(start_date, end_date)
        logging.info(f"Datos extraidos tipo de cambio: \n{usd_mxn}")

        #Unión de toda la información en un solo df
        logging.info("Tratamiento de datos")
        df_new = df_new.merge(
            usd_mxn,
            on="date",
            how="left"
        )

        #rellenar fechas faltantes para tener continuidad dentro de la serie de datos
        full_dates = pd.date_range(
            start=df_new["date"].min(),
            end=df_new["date"].max(),
            freq="D"
        )

        df_new = df_new.set_index("date").reindex(full_dates)
        df_new.index.name = "date"

        #Crea indicadores de los datos que serán imputados 
        df_new["interpolated_LME"] = df_new["cash_bid"].isna()
        df_new["interpolated_MXN"] = df_new["usd_mxn_fix"].isna()

        #rellenar información faltannte
        df_new = df_new.ffill()
        df_new = df_new.reset_index()

        #aseguar información se almacene como datos numéricos 
        cols_to_convert = df_new.columns.difference(
            ["date","interpolated_LME", "interpolated_MXN"]
        )
        df_new[cols_to_convert] = df_new[cols_to_convert].apply(
            pd.to_numeric, errors="coerce"
        )

        #cáclulo del valor en MXN del precio del metal para "cash offer"
        if "cash_offer" in df_new.columns:
            df_new["cash_offer_mxn"] = (
                df_new["cash_offer"] * df_new["usd_mxn_fix"]
            ).round(4)

        #Actualización del df que contiene la serie de datos global
        #Elimiación de la información duplicada. Prioriza la información original
        #Sobre la identada y la nueva sobre la más antigua 
        
        df_all = pd.concat([df_global, df_new], ignore_index=True)

        df_all["real_count"] = (
            (~df_all["interpolated_LME"]).astype(int)
            + (~df_all["interpolated_MXN"]).astype(int)
        )

        df_all["is_new"] = [0]*len(df_global) + [1]*len(df_new)

        df_global = (
            df_all
            .sort_values(["date","real_count","is_new"], ascending=[True,False,False])
            .drop_duplicates("date", keep="first")
            .drop(columns=["real_count","is_new"])
        )
        
        #Ordenar nuevamente el df
        df_global = df_global.sort_values("date")
        df_global = df_global.ffill()

        #Actualizar el archivo que contiene la serie de datos
        logging.info("Actualización de base de datos")
        df_global.to_csv(csv_file, index=False)
        logging.info("Base actualizada correctamente\n")

    except Exception as e:
        logging.error(f"Error en ejecución: {e}")

if __name__ == "__main__":
    main()

