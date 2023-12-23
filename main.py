import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()
root.title("Regressors")


def read_data_frame():
    path_to_df = os.path.join('DataSet', 'day.csv')
    return pd.read_csv(path_to_df)


df = read_data_frame()
print(df.dtypes)
Data = df[["season",
           "mnth", "holiday", "weekday", "workingday",
           "weathersit", "temp", "atemp", "hum", "windspeed",
           ]]
Data = Data.astype(float)
print(Data.dtypes)
print(Data["season"].unique())
print(Data["holiday"].unique())
print(Data["weekday"].unique())
print(Data["workingday"].unique())
print(Data["weathersit"].unique())

Answ = df["cnt"]
Answ.value_counts()
print(Answ.value_counts())
trainData_scaled = MinMaxScaler().fit_transform(Data)
trainX, testX, trainY, testY = train_test_split(trainData_scaled, Answ, random_state=1)

label1 = ttk.Label(root, text="R2 = ")
label2 = ttk.Label(root, text="MSE = ")
label3 = ttk.Label(root, text="MAE = ")


def update_label_text(label, text):
    label.config(text=text)


def lin_regression(trainX, testX, trainY, testY):
    LR = LinearRegression()
    classifier_LR = LR.fit(trainX, trainY)
    predict_LR = LR.predict(testX)
    r2 = "R2 = " + str(r2_score(testY, predict_LR))
    mse = "MSE = " + str(mean_squared_error(testY, predict_LR))
    mae = "MAE = " + str(mean_absolute_error(testY, predict_LR))
    update_label_text(label1, r2)
    update_label_text(label2, mse)
    update_label_text(label3, mae)


def polynom_features(trainX, testX, trainY, testY):
    PF = PolynomialFeatures(degree=1)
    polyTrain = PF.fit_transform(trainX)
    polyTest = PF.fit_transform(testX)

    polyRegression = LinearRegression()
    polyRegression.fit(polyTrain, trainY)
    predict_polyRegression = polyRegression.predict(polyTest)
    r2 = "R2 = " + str(r2_score(testY, predict_polyRegression))
    mse = "MSE = " + str(mean_squared_error(testY, predict_polyRegression))
    mae = "MAE = " + str(mean_absolute_error(testY, predict_polyRegression))
    update_label_text(label1, r2)
    update_label_text(label2, mse)
    update_label_text(label3, mae)


def forest_regression(trainX, testX, trainY, testY):
    RF = RandomForestRegressor(n_estimators=127)
    RF.fit(trainX, trainY)
    predict_RF = RF.predict(testX)
    r2 = "R2 = " + str(r2_score(testY, predict_RF))
    mse = "MSE = " + str(mean_squared_error(testY, predict_RF))
    mae = "MAE = " + str(mean_absolute_error(testY, predict_RF))
    update_label_text(label1, r2)
    update_label_text(label2, mse)
    update_label_text(label3, mae)

    plot_graph(RF)


def gradient_regression(trainX, testX, trainY, testY):
    reg = GradientBoostingRegressor(random_state=0)
    reg.fit(trainX, trainY)
    predict_reg = reg.predict(testX)
    r2 = "R2 = " + str(r2_score(testY, predict_reg))
    mse = "MSE = " + str(mean_squared_error(testY, predict_reg))
    mae = "MAE = " + str(mean_absolute_error(testY, predict_reg))
    update_label_text(label1, r2)
    update_label_text(label2, mse)
    update_label_text(label3, mae)


def plot_graph(RF):
    for widget in frame_graph.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(7, 5), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)

    # Построение графика в соответствии с вашим кодом
    pd.DataFrame(RF.feature_importances_, index=Data.columns,
                 columns=["Importance"]).sort_values("Importance").plot.bar(ax=ax)

    # Вставка графика в Tkinter окно
    canvas = FigureCanvasTkAgg(fig, master=frame_graph)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


# Создание кнопок
button1 = ttk.Button(root, text="LinearRegression", command=lambda: lin_regression(trainX, testX, trainY, testY))
button2 = ttk.Button(root, text="PolynomialFeatures", command=lambda: polynom_features(trainX, testX, trainY, testY))
button3 = ttk.Button(root, text="RandomForestRegressor", command=lambda: forest_regression(trainX, testX, trainY, testY))
button4 = ttk.Button(root, text="GradientBoostingRegressor", command=lambda: gradient_regression(trainX, testX, trainY, testY))

# Размещение кнопок и меток в окне
button1.grid(row=0, column=0, padx=5, pady=5)
button2.grid(row=0, column=1, padx=5, pady=5)
button3.grid(row=0, column=2, padx=5, pady=5)
button4.grid(row=0, column=3, padx=5, pady=5)

label1.grid(row=1, column=0, padx=5, pady=5)
label2.grid(row=2, column=0, padx=5, pady=5)
label3.grid(row=3, column=0, padx=5, pady=5)

# Создание контейнера для графика
frame_graph = ttk.Frame(root)
frame_graph.grid(row=5, column=0, columnspan=3, padx=50, pady=50)

# Запуск главного цикла
root.mainloop()


def print_df_info(data):
    print(data)
    print(data.describe())
    print(data.info())


if __name__ == '__main__':
    print("end.")
