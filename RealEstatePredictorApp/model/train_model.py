import os
import random
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
import joblib

SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 0) Задаём device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Загрузка и обрезка выбросов
df = pd.read_csv('../data/moscow_dataset.csv')
df = df[df['Price'] < 300e6].copy()

# 2) Сохраняем целевую переменную и сразу делаем log1p+стандартизацию
y_raw = df['Price'].values.reshape(-1,1)
y_log = np.log1p(y_raw)
scaler_y = StandardScaler().fit(y_log)
y_scaled = scaler_y.transform(y_log)
joblib.dump(scaler_y, '../model/scaler_y.pkl')

# 3) Удаляем 'Price' из df, чтобы дальше не мешался
df.drop('Price', axis=1, inplace=True)

# 4) Feature engineering (rel_floor, living_ratio и т.д.)
df['rel_floor']    = df['Floor'] / df['Number of floors']
df['is_first']     = (df['Floor'] == 1).astype(int)
df['is_last']      = (df['Floor'] == df['Number of floors']).astype(int)
df['living_ratio'] = df['Living area'] / df['Area']
df['kitchen_ratio']= df['Kitchen area'] / df['Area']

# 5) Target-encoding станции метро
df['Price_log0'] = y_log.flatten()
station_mean = df.groupby('Metro station')['Price_log0'].mean()
df['station_te'] = df['Metro station'].map(station_mean)
joblib.dump(station_mean, '../model/station_mean.pkl')


# 6) Удаляем больше не нужные колонки
df.drop(['Metro station','Kitchen area','Floor','Price_log0'], axis=1, inplace=True)

# 7) Готовим препроцессор
num_feats = ['Minutes to metro','Number of rooms','Area','Living area','Number of floors',
             'rel_floor','living_ratio','kitchen_ratio','station_te']
cat_feats = ['Apartment type','Region','Renovation','is_first','is_last']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)
]).fit(df)
X = preprocessor.transform(df)
joblib.dump(preprocessor, '../model/preprocessor.pkl')

# 8) Разбиваем на train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y_scaled, test_size=0.3, random_state=0
)

# Конвертация в тензоры
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32).to(device)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).to(device)

# Дальше — DataLoader и остальная часть без изменений
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)


# 9) DataLoader
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=256)

# 10) Модель
model = nn.Sequential(
    nn.Linear(X_train_t.shape[1], 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 32),                nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(32, 16),                nn.ReLU(),
    nn.Linear(16, 1)
).to(device)

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# 11) Обучение с ранним стопом
best_val = float('inf')
patience = 10
wait = 0

for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_losses = [criterion(model(xb), yb).item() for xb, yb in val_loader]
    val_loss = np.mean(val_losses)

    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        torch.save(model.state_dict(), '../model/best_model.pt')
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break


# 12) Оценка MAE в рублях
model.load_state_dict(torch.load('../model/best_model.pt'))
model.eval()
with torch.no_grad():
    pred_log = model(X_val_t).cpu().numpy().flatten()
# обратное масштабирование и снятие log1p
y_std, y_mean = scaler_y.scale_[0], scaler_y.mean_[0]
pred_price   = np.expm1(pred_log * y_std + y_mean)
true_price   = np.expm1(y_val.flatten()  * y_std + y_mean)

val_df = pd.DataFrame({'true': true_price, 'pred': pred_price})
print("MAE до 20 млн:", mean_absolute_error(
    val_df[val_df['true'] < 20_000_000]['true'],
    val_df[val_df['true'] < 20_000_000]['pred']
))
print("MAE 20–50 млн:", mean_absolute_error(
    val_df[(val_df['true'] >= 20_000_000) & (val_df['true'] < 50_000_000)]['true'],
    val_df[(val_df['true'] >= 20_000_000) & (val_df['true'] < 50_000_000)]['pred']
))
print("MAE > 50 млн:", mean_absolute_error(
    val_df[val_df['true'] >= 50_000_000]['true'],
    val_df[val_df['true'] >= 50_000_000]['pred']
))

weights = 1 / (true_price + 1e-8) # чем дороже, тем меньше вес
weighted_mae = np.average(np.abs(true_price - pred_price), weights=weights)
print(f"Weighted MAE: {weighted_mae:.0f} ₽")

log_true = np.log1p(true_price)
log_pred = np.log1p(pred_price)
log_mae = mean_absolute_error(log_true, log_pred)
print(f"Log-space MAE: {log_mae:.3f}")


