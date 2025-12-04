from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("model_xgboost.pkl")
scaler = joblib.load("scaler.pkl")

REQUIRED_FIELDS = [
    "gender", "age", "academic_pressure", "work_pressure", "cgpa",
    "study_satisfaction", "job_satisfaction", "work_study_hours",
    "financial_stress", "family_history_of_mental_illness",
    "depression", "sleep_score"
]

def validate_input(data):
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    return missing

def generate_recommendation(data):
    saran = []

    ap = data["academic_pressure"]
    if ap >= 4:
        saran.append(
            "Tekanan akademik Anda cukup tinggi. Cobalah memecah tugas besar menjadi bagian kecil agar tidak terasa berat."
        )
    elif ap <= 2:
        saran.append("Tekanan akademik Anda ringan. Pertahankan ritme belajar yang stabil ini.")

    ss = data["study_satisfaction"]
    if ss <= 2:
        saran.append(
            "Anda kurang puas dengan proses belajar. Cobalah mencari metode belajar baru yang lebih cocok."
        )
    elif ss >= 4:
        saran.append("Kepuasan belajar Anda sangat baik. Pertahankan motivasi positif ini.")

    fs = data["financial_stress"]
    if fs >= 4:
        saran.append(
            "Anda mengalami stres finansial tinggi. Mencatat pengeluaran harian bisa membantu mengontrol kondisi."
        )
    elif fs == 1:
        saran.append("Stres finansial Anda sangat rendah. Pertahankan kebiasaan keuangan Anda.")

    sl = data["sleep_score"]
    if sl == 1:
        saran.append("Kualitas tidur rendah. Kurangi layar sebelum tidur agar istirahat lebih nyenyak.")
    elif sl == 2:
        saran.append("Kualitas tidur Anda belum optimal. Menata jadwal tidur dapat membantu.")
    elif sl >= 3:
        saran.append("Kualitas tidur Anda cukup baik. Pertahankan pola istirahat ini.")

    wh = data["work_study_hours"]
    if wh >= 9:
        saran.append(
            "Jam belajar/kerja Anda sangat panjang. Pastikan memberi waktu istirahat agar tidak burnout."
        )
    elif wh <= 4:
        saran.append(
            "Waktu belajar Anda cukup ringan. Tetap jaga keseimbangan agar produktivitas tetap stabil."
        )

    dp = data["depression"]
    if dp == 1:
        saran.append(
            "Anda menunjukkan gejala depresi. Tidak apa-apa mencari dukungan dari teman, keluarga, atau profesional."
        )

    if len(saran) == 0:
        saran.append("Secara umum Anda stabil. Pertahankan pola hidup sehat Anda ya.")

    return saran

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API Kesehatan Mental Mahasiswa - XGBoost"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        missing = validate_input(data)
        if missing:
            return jsonify({
                "status": "error",
                "message": "Field tidak lengkap",
                "missing_fields": missing
            }), 400
        input_list = [
            data["gender"],
            data["age"],
            data["academic_pressure"],
            data["work_pressure"],
            data["cgpa"],
            data["study_satisfaction"],
            data["job_satisfaction"],
            data["work_study_hours"],
            data["financial_stress"],
            data["family_history_of_mental_illness"],
            data["depression"],
            data["sleep_score"]
        ]

        input_array = np.array(input_list).reshape(1, -1)

        # Scaling
        scaled = scaler.transform(input_array)

        # Prediksi
        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0]
        confidence = float(max(proba))

        pred_label = {0: "Indikasi Tinggi", 1: "Indikasi Sedang", 2: "Indikasi Rendah"}.get(pred, "Tidak Diketahui")

        rekomendasi = generate_recommendation(data)

        return jsonify({
            "status": "success",
            "prediction_raw": int(pred),
            "prediction_label": pred_label,
            "confidence": confidence,
            "rekomendasi": rekomendasi
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Terjadi kesalahan pada server"
        }), 500


#Run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)