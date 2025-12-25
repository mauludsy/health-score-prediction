import joblib
import pandas as pd
import numpy as np
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS


warnings.filterwarnings("ignore")

# ===== Inisialisasi Flask =====
app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS agar bisa diakses dari frontend

# ===== 1. Load Model, Scaler, dan PolynomialFeatures =====
model_path = "PolyLReg_4features.pkl"
scaler_path = "HealthScaler.pkl"
poly_path = "PolyFeatures.pkl"

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    poly = joblib.load(poly_path)
    print("✅ Model, Scaler, dan PolynomialFeatures berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model/scaler/poly: {e}")
    raise

# ===== 2. Endpoint Utama =====
@app.route("/", methods=["GET"])
def home():
    return "✅ API Prediksi Skor Kesehatan - Polynomial Regression", 200

# ===== 3. Endpoint Prediksi =====
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"📩 Data diterima dari frontend: {data}")

        # ===== Validasi kolom wajib =====
        required_fields = ["BMI", "Exercise_Frequency", "Sleep_Hours", "Smoking_Status"]
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({"error": f"Field '{field}' wajib disertakan dan tidak boleh kosong."}), 400
            try:
                data[field] = float(data[field])
            except ValueError:
                return jsonify({"error": f"Nilai '{field}' harus berupa angka yang valid."}), 400

        # ===== Buat DataFrame sesuai kolom model =====
        input_df = pd.DataFrame([{
            "BMI": data["BMI"],
            "Exercise_Frequency": data["Exercise_Frequency"],
            "Sleep_Hours": data["Sleep_Hours"],
            "Smoking_Status": data["Smoking_Status"]
        }])
        print(f"📊 DataFrame untuk scaling: \n{input_df}")

        # ===== Scaling =====
        input_scaled = scaler.transform(input_df)

        # ===== Polynomial Transform =====
        input_poly = poly.transform(input_scaled)
        print(f"🔧 Data setelah scaling dan transformasi polinomial:\n{input_poly}")

        # ===== Prediksi =====
        prediction = model.predict(input_poly)[0]
        health_score = np.clip(prediction, 0, 100)  # Batasi skor antara 0-100
        print(f"🎯 Hasil prediksi: {health_score}")

        # # ===== Klasifikasi Kategori =====
        # if health_score >= 80:
        #     kategori = "Sangat Baik"
        # elif health_score >= 60:
        #     kategori = "Baik"
        # elif health_score >= 40:
        #     kategori = "Cukup"
        # else:
        #     kategori = "Buruk"

        # print(f"🏷️ Kategori Skor Kesehatan: {kategori}")

        # ===== Kirim hasil ke frontend =====
        return jsonify({
            "BMI": data["BMI"],
            "Exercise_Frequency": data["Exercise_Frequency"],
            "Sleep_Hours": data["Sleep_Hours"],
            "Smoking_Status": data["Smoking_Status"],
            "health_score": round(health_score, 2),
            # "kategori": kategori,
            "message": "Prediksi berhasil dihitung ✅"
        })

    except Exception as e:
        print(f"❌ Terjadi kesalahan di endpoint /predict: {e}")
        return jsonify({
            "error": str(e),
            "message": "Terjadi kesalahan internal server. Mohon coba lagi."
        }), 500

# ===== 4. Jalankan Server =====
if __name__ == "__main__":
    app.run(debug=True, port=5000)

#python app.py 