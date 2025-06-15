from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import torch
import joblib
from torch import nn

app = Flask(__name__)

# Загрузка модели и препроцессора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
    nn.Linear(21, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 1)
).to(device)
model.load_state_dict(torch.load('model/best_model.pt', map_location=device))
model.eval()

preprocessor = joblib.load('model/preprocessor.pkl')
scaler_y = joblib.load('model/scaler_y.pkl')
station_mean = joblib.load('model/station_mean.pkl')  # Словарь со средними по станциям метро


def load_options(filename):
    path = f"options/{filename}"
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


# Загрузка вариантов выбора
stations = load_options('stations.txt')
regions = load_options('regions.txt')
apartment_types = load_options('apartment_types.txt')
renovations = load_options('renovations.txt')

# Словари отображения для русского интерфейса
region_display = {
    "Moscow" : "Москва",
    "Moscow region" : "Московская обл"
}

renovation_display = {
    "Without renovation": "Без ремонта",
    "Cosmetic": "Косметический",
    "European-style renovation": "Евроремонт",
    "Designer": "Дизайнерский"
}

apartment_display = {
    "New building": "Новостройка",
    "Secondary": "Вторичное"
}


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            region_rus = request.form['Region']
            renovation_rus = request.form['Renovation']
            apartment_rus = request.form['Apartment type']

            region_code = [k for k, v in region_display.items() if v == region_rus][0]
            renovation_code = [k for k, v in renovation_display.items() if v == renovation_rus][0]
            apartment_code = [k for k, v in apartment_display.items() if v == apartment_rus][0]

            # 3. Сбор данных
            data = {
                'Area': float(request.form['Area']),
                'Living area': float(request.form['Living area']),
                'Number of rooms': int(request.form['Number of rooms']),
                'Floor': int(request.form['Floor']),
                'Number of floors': int(request.form['Number of floors']),
                'Minutes to metro': float(request.form['Minutes to metro']),
                'Region': region_code,
                'Metro station': request.form['Metro station'],
                'Renovation': renovation_code,
                'Apartment type': apartment_code,
            }

            df = pd.DataFrame([data])

            # Feature engineering
            df['rel_floor'] = df['Floor'] / df['Number of floors']
            df['is_first'] = (df['Floor'] == 1).astype(int)
            df['is_last'] = (df['Floor'] == df['Number of floors']).astype(int)
            df['living_ratio'] = df['Living area'] / df['Area']
            df['kitchen_ratio'] = 0.0  # по умолчанию
            df['station_te'] = station_mean.get(df['Metro station'].iloc[0], station_mean.mean())

            # Удаляем ненужные колонки
            df.drop(['Metro station', 'Floor'], axis=1, inplace=True)
            if 'Kitchen area' in df.columns:
                df.drop('Kitchen area', axis=1, inplace=True)

            # Преобразование признаков
            X = preprocessor.transform(df)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

            # Предсказание
            with torch.no_grad():
                y_scaled = model(X_tensor).cpu().numpy().flatten()
            y_pred = np.expm1(y_scaled * scaler_y.scale_ + scaler_y.mean_)

            return render_template(
                'index.html',
                prediction=f"{y_pred[0]:,.0f}",
                stations=stations,
                regions=list(region_display.values()),
                renovations=list(renovation_display.values()),
                apartment_types=list(apartment_display.values())
            )
        except Exception as e:
            return f"Ошибка обработки формы: {e}"

    return render_template(
        'index.html',
        stations=stations,
        regions=list(region_display.values()),
        renovations=list(renovation_display.values()),
        apartment_types=list(apartment_display.values())
    )


if __name__ == '__main__':
    app.run(debug=True)
