# Signalbehandling och Klassificering av Ljuddata

Detta är ett mindre Python-projekt inom Data Science som demonstrerar grundläggande signalbehandling av ljudsignaler (`.wav`) och klassificering med maskininlärning.

## Vad projektet gör
Projektet genererar syntetiska ljudsignaler (normalt maskinljud och felaktigt maskinljud), tillämpar **Fast Fourier Transform (FFT)** för att omvandla signalerna till frekvensdomänen, extraherar tre statistiska features och tränar en `RandomForestClassifier` för att identifiera fel.

## Beroenden
Följande Python-bibliotek behövs:
* `numpy`
* `scipy`
* `librosa`
* `scikit-learn`
* `soundfile`

Alla beroenden finns angivna i `requirements.txt`.

## Installation och förberedelse
1. Klona detta repository.
2. Installera nödvändiga bibliotek:
   ```bash
   pip install -r requirements.txt