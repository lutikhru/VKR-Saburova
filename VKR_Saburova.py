import tkinter as tk
from math import sqrt
from tkinter import ttk, messagebox
from datetime import datetime
from meteostat import Point, Hourly
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def fetch_weather_data():
    try:
        # Преобразование введенных дат в формат datetime
        start = datetime.strptime(start_date.get(), '%Y-%m-%d')
        end = datetime.strptime(end_date.get(), '%Y-%m-%d')
        
        # Установка локации для Тетуан, Марокко
        location = Point(35.5889, -5.3626)
        
        # Получение погодных данных
        data = Hourly(location, start, end)
        data = data.fetch()
        
        if data.empty:
            messagebox.showerror("Ошибка", "Не удалось получить данные о погоде.")
            return
        
        # Добавление столбца времени
        data['time'] = data.index

        # Удаление столбцов с отсутствующими значениями
        df_cleaned = data.dropna(axis=1)
        
        # Сохранение данных в CSV файл
        df_cleaned.to_csv('cleaned_weather_data.csv', index=False)
        
        messagebox.showinfo("Успех", "Данные о погоде успешно получены и сохранены.")
    except Exception as e:
        messagebox.showerror("Ошибка", str(e))

def predict_consumption():
    try:
        # Загрузка данных
        data = pd.read_csv('powerconsumption.csv')
        new_data = pd.read_csv('cleaned_weather_data.csv')

        # Создание копии new_data для переименования столбцов
        new_data_renamed = new_data.copy()
        new_data_renamed = new_data_renamed.drop(columns = ['dwpt', 'wdir', 'prcp', 'pres', 'coco'], axis = 1)
        new_data_renamed.rename(columns={'temp': 'Temperature', 'rhum': 'Humidity', 'wspd': 'WindSpeed'}, inplace=True)

        # Подготовка данных
        features = data[['Temperature', 'Humidity', 'WindSpeed']]
        targets = data[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

        # Масштабирование признаков
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Обучение моделей XGBoost для каждой зоны отдельно
        models = {}
        predictions = {}

        for zone in targets.columns:
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            model.fit(X_train_scaled, y_train[zone])
            models[zone] = model
            predictions[zone] = model.predict(X_test_scaled)

        # Подготовка новых данных для предсказания
        new_features = new_data_renamed[['Temperature', 'Humidity', 'WindSpeed']]
        new_features_scaled = scaler.transform(new_features)

        new_predictions = {}
        for zone in targets.columns:
            new_predictions[zone] = models[zone].predict(new_features_scaled)

        # Создание DataFrame с предсказаниями
        new_predictions_df = pd.DataFrame(new_predictions)
        new_predictions_df.columns = ['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']

        new_data_with_predictions = pd.concat([new_data_renamed, new_predictions_df], axis=1)

        # Сохранение данных с предсказаниями в новый файл
        new_data_with_predictions.to_csv('predicted_power_consumption.csv', index=False)

        for zone in targets.columns:
            mae = mean_absolute_error(y_test[zone], predictions[zone]) / 1000
            rmse = sqrt(mean_squared_error(y_test[zone], predictions[zone])) / 1000
            print("MSE for on test set: {:.4f}".format(rmse), f'for {zone}')
            print("MAE for on test set: {:.4f}".format(mae), f'for {zone}')
        
        messagebox.showinfo("Прогнозирование завершено", "Прогнозы сохранены в файл 'predicted_power_consumption.csv'.")
    except Exception as e:
        messagebox.showerror("Ошибка", str(e))
        print("Произошла ошибка:", e)

# Создание главного окна
root = tk.Tk()
root.title("Прогнозирование потребления электричества")

# Создание фреймов
frame1 = ttk.Frame(root, padding="10")
frame1.grid(row=0, column=0, sticky=(tk.W, tk.E))

frame2 = ttk.Frame(root, padding="10")
frame2.grid(row=1, column=0, sticky=(tk.W, tk.E))

# Виджеты для выбора даты
ttk.Label(frame1, text="Дата начала (ГГГГ-ММ-ДД):").grid(row=0, column=0)
start_date = ttk.Entry(frame1)
start_date.grid(row=0, column=1)

ttk.Label(frame1, text="Дата окончания (ГГГГ-ММ-ДД):").grid(row=1, column=0)
end_date = ttk.Entry(frame1)
end_date.grid(row=1, column=1)

# Кнопка для получения данных о погоде
ttk.Button(frame1, text="Получить данные о погоде", command=fetch_weather_data).grid(row=2, column=1)

# Кнопка для прогнозирования потребления электричества
ttk.Button(frame2, text="Прогнозировать потребление", command=predict_consumption).grid(row=0, column=1)

# Запуск главного цикла приложения
root.mainloop()
