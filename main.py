from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from typing import Union, List  # Import Union to handle both single and list values
from enum import Enum  # Enum for dropdown options
import uvicorn
import pickle
from fastapi.middleware.cors import CORSMiddleware

# Load the pre-trained model
catboost_model_tuned = pickle.load(open("backend/15loocv.pkl", "rb"))

# The columns used during model training
X_columns = ['umur', 'jenis_kelamin', 'lokasi', 'status_pernikahan', 
             'jenis_kopi', 'harga_kopi', 'rasa_kopi_yang_disuka', 
             'preferensi_penyajian', 'waktu_pembelian']

app = FastAPI()

# Middleware to allow requests from a frontend app (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be configured to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enum for coffee types
class JenisKopiEnum(str, Enum):
    kopi_hitam = "Kopi Hitam"
    latte = "Latte"
    kopi_susu = "Kopi Susu"
    espresso = "Espresso"
    cappuccino = "Cappuccino"
    americano = "Americano"
    mocha = "Mocha"

# Model input data class, using Union to accept both single values and lists
class InputData(BaseModel):
    umur: Union[List[int], int]
    jenis_kelamin: Union[List[str], str]
    lokasi: Union[List[str], str]
    status_pernikahan: Union[List[str], str]
    jenis_kopi: Union[List[JenisKopiEnum], JenisKopiEnum]
    harga_kopi: Union[List[int], int]
    rasa_kopi_yang_disuka: Union[List[str], str]
    preferensi_penyajian: Union[List[str], str]
    waktu_pembelian: Union[List[str], str]
@app.get("/")
def index():
    return {'message': 'Selamat Datang Di Kopicast'}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Ensure all inputs are converted to lists
        umur_list = [data.umur] if isinstance(data.umur, int) else data.umur
        jenis_kelamin_list = [data.jenis_kelamin] if isinstance(data.jenis_kelamin, str) else data.jenis_kelamin
        lokasi_list = [data.lokasi] if isinstance(data.lokasi, str) else data.lokasi
        status_pernikahan_list = [data.status_pernikahan] if isinstance(data.status_pernikahan, str) else data.status_pernikahan
        jenis_kopi_list = [data.jenis_kopi] if isinstance(data.jenis_kopi, str) else data.jenis_kopi
        harga_kopi_list = [data.harga_kopi] if isinstance(data.harga_kopi, int) else data.harga_kopi
        rasa_kopi_yang_disuka_list = [data.rasa_kopi_yang_disuka] if isinstance(data.rasa_kopi_yang_disuka, str) else data.rasa_kopi_yang_disuka
        preferensi_penyajian_list = [data.preferensi_penyajian] if isinstance(data.preferensi_penyajian, str) else data.preferensi_penyajian
        waktu_pembelian_list = [data.waktu_pembelian] if isinstance(data.waktu_pembelian, str) else data.waktu_pembelian

        predictions = []
        total_prediksi = 0

        # Iterate over all combinations of input fields
        for umur in umur_list:
            for jenis_kelamin in jenis_kelamin_list:
                for lokasi in lokasi_list:
                    for status_pernikahan in status_pernikahan_list:
                        for jenis_kopi in jenis_kopi_list:
                            for harga_kopi in harga_kopi_list:
                                for rasa_kopi_yang_disuka in rasa_kopi_yang_disuka_list:
                                    for preferensi_penyajian in preferensi_penyajian_list:
                                        for waktu_pembelian in waktu_pembelian_list:
                                            # Prepare the data for prediction
                                            data_input = {
                                                'umur': [umur],
                                                'jenis_kelamin': [jenis_kelamin],
                                                'lokasi': [lokasi],
                                                'status_pernikahan': [status_pernikahan],
                                                'jenis_kopi': [jenis_kopi],
                                                'harga_kopi': [harga_kopi],
                                                'rasa_kopi_yang_disuka': [rasa_kopi_yang_disuka],
                                                'preferensi_penyajian': [preferensi_penyajian],
                                                'waktu_pembelian': [waktu_pembelian],
                                            }

                                            df_input = pd.DataFrame(data_input)
                                            df_input = df_input.reindex(columns=X_columns, fill_value=0)

                                            # Predict using the pre-trained CatBoost model
                                            y_pred_catboost = catboost_model_tuned.predict(df_input)

                                            # Add the prediction to the total
                                            total_prediksi += y_pred_catboost[0]

                                            # Append each prediction
                                            predictions.append({
                                                "umur": umur,
                                                "jenis_kelamin": jenis_kelamin,
                                                "lokasi": lokasi,
                                                "status_pernikahan": status_pernikahan,
                                                "jenis_kopi": jenis_kopi,
                                                "harga_kopi": harga_kopi,
                                                "rasa_kopi_yang_disuka": rasa_kopi_yang_disuka,
                                                "preferensi_penyajian": preferensi_penyajian,
                                                "waktu_pembelian": waktu_pembelian,
                                                "prediksi_frekuensi_pembelian": y_pred_catboost[0]
                                            })

        # Return all predictions and the total sum of predictions
        return {
            "predictions": predictions,
            "total_prediksi_frekuensi_pembelian": total_prediksi
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
