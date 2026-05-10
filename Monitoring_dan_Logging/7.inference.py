import time
import random
import datetime

# Simulasi pemuatan model
print("Loading model: Random Forest Classifier...")
time.sleep(2)
print("Starting Inference Service...")
print("="*60)

def generate_dummy_data():
    """Membuat data pasien dummy untuk tes prediksi"""
    return {
        'age': random.randint(40, 95),
        'anaemia': random.choice([0, 1]),
        'creatinine_phosphokinase': random.randint(23, 7861),
        'diabetes': random.choice([0, 1]),
        'ejection_fraction': random.randint(14, 80),
        'high_blood_pressure': random.choice([0, 1]),
        'platelets': random.randint(25000, 850000),
        'serum_creatinine': round(random.uniform(0.5, 9.4), 1),
        'sex': random.choice([0, 1]),
        'smoking': random.choice([0, 1])
    }

def mock_predict(data):
    """
    Simulasi logika model. 
    Di dunia nyata, di sini kita panggil model.predict(data).
    Di sini kita pakai logika sederhana: Ejection Fraction rendah = Risiko Tinggi.
    """
    if data['ejection_fraction'] < 30 or data['serum_creatinine'] > 2.0:
        return 1 # Prediksi: Meninggal
    else:
        return 0 # Prediksi: Bertahan

try:
    while True:
        # 1. Terima Data Input
        input_data = generate_dummy_data()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 2. Lakukan Prediksi
        result = mock_predict(input_data)
        label = "Meninggal (Risk)" if result == 1 else "Bertahan (Safe)"
        
        # 3. Output Log Prediksi (Inilah Inference yang sebenarnya)
        print(f"[{timestamp}] New Patient Data:")
        print(f"   -> Features: Age={input_data['age']}, EF={input_data['ejection_fraction']}, Creatinine={input_data['serum_creatinine']}")
        print(f"   -> PREDICTION OUTPUT: {result} [{label}]")
        print("-" * 60)
        
        # Jeda waktu biar terlihat seperti log server
        time.sleep(random.uniform(1, 3))

except KeyboardInterrupt:
    print("\nStopping Inference Service.")