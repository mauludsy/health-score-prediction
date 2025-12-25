import argparse
import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(
        description="Prediksi Skor Kesehatan Menggunakan Polynomial Regression (4 fitur)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Masukkan 4 fitur: BMI, Exercise_Frequency, Sleep_Hours, Smoking_Status"
    )
    args = parser.parse_args()

    # ===== Parsing input =====
    try:
        values = list(map(float, args.data.strip().split()))
        if len(values) != 4:
            print("❌ Input harus terdiri dari 4 nilai.")
            return
        input_df = pd.DataFrame([values], columns=['BMI', 'Exercise_Frequency', 'Sleep_Hours', 'Smoking_Status'])
        print("\n📊 Data input:")
        print(input_df)
    except ValueError:
        print("❌ Input tidak valid. Pastikan semua fitur berupa angka.")
        return

    # ===== Load Scaler, PolynomialFeatures, dan Model =====
    try:
        scaler = joblib.load("HealthScaler.pkl")
        poly = joblib.load("PolyFeatures.pkl")
        model = joblib.load("PolyLReg_4features.pkl")
        print("\n✅ Model, Scaler, dan PolynomialFeatures berhasil dimuat.")
    except Exception as e:
        print(f"❌ Error memuat model/scaler/poly: {e}")
        return

    # ===== Scaling & Polynomial Transform =====
    try:
        input_scaled = scaler.transform(input_df)
        input_poly = poly.transform(input_scaled)  # <- wajib agar jumlah fitur sesuai
        print("\n🔧 Data setelah scaling dan transformasi polinomial:")
        print(input_poly)
    except Exception as e:
        print(f"❌ Error saat transformasi data: {e}")
        return

    # ===== Prediksi =====
    try:
        prediction = model.predict(input_poly)[0]
        prediction = np.clip(prediction, 0, 100)  # Batasi skor 0-100
        print(f"\n🎯 Prediksi Health Score: {prediction:.2f}")
    except Exception as e:
        print(f"❌ Error saat prediksi: {e}")
        return

    # ===== Klasifikasi =====
    if prediction >= 80:
        kategori = "Sangat Baik"
    elif prediction >= 60:
        kategori = "Baik"
    elif prediction >= 40:
        kategori = "Cukup"
    else:
        kategori = "Buruk"

    print("\n🏷️ Kategori Skor Kesehatan:", kategori)
    print("\n✅ Prediksi selesai.")

if __name__ == "__main__":
    main()


#  python predict_health.py --data "23.5, 4, 7, 1" (sangat baik)
#  python predict_health.py --data "19.8, 95.7, 4.3, 0" (buruk)