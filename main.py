import numpy as np
import librosa
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_fft_features(signal, sr):
    n = len(signal)
    fft_spectrum = np.abs(fft(signal))[:n // 2]
    
    mean_freq = np.mean(fft_spectrum)
    max_freq = np.max(fft_spectrum)
    std_freq = np.std(fft_spectrum)
    
    return [mean_freq, max_freq, std_freq]

X = []
y = []

sample_rate = 22050
duration = 1.0
t = np.linspace(0, duration, int(sample_rate * duration))

# Generera normala ljud (50 Hz)
for _ in range(10):
    noise = np.random.normal(0, 0.05, len(t))
    signal = np.sin(2 * np.pi * 50 * t) + noise
    features = extract_fft_features(signal, sample_rate)
    X.append(features)
    y.append(0)

# Generera felaktiga ljud (400 Hz)
for _ in range(10):
    noise = np.random.normal(0, 0.05, len(t))
    signal = np.sin(2 * np.pi * 400 * t) + noise
    features = extract_fft_features(signal, sample_rate)
    X.append(features)
    y.append(1)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Träning klar! Modellens pricksäkerhet: {acc * 100:.0f}%")