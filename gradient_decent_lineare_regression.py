# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 07:26:06 2021

@author: dea40349
"""

# GRADIEN DESCENT am Beispiel Lineare Regression
# Datensatz "Umsatz Speiseeis in Abhängigkeit von der Temperatur

# Variablen:
# X: Temmperatur, y: Umsatz
# m: Steigung der gesuchten Regressionsgeraden
# t: Y-Achsenabschnitt
# alpha: Lernrate
# max_iter: Maximale Anzahl Iterationen
# max_error: Maximaler Fehler (bricht ab, wenn eines der beiden Werte erreicht)
# d_m: jeweils aktueller Wert für m eingesetzt in die 1. Ableitung der Fehlerfunktion
# d_t: jeweils aktueller Wert für t eingesetzt in die 1. Ableitung der Fehlerfunktion
# N: Anzahl der Beobachtungen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/troescherw/datasets/master/speiseeis_umsatz.csv"

data = pd.read_csv(url)
X = data.Temperatur
y = data.Umsatz


def gradient_descent(X, y, alpha=0.001, max_iter=1000, max_error=0.0001):
    m = t = 0
    N = len(y)
    i = 0
    d_m = 1000
    while i<max_iter and np.absolute(d_m)>max_error: 
        y_pred = m*X + t  # Vorhergesagte Y-Werte
        d_m = (-2/N) * np.sum(X * (y - y_pred))  # eingesetzt in part. Ableitung für m
        d_t = (-2/N) * np.sum(y - y_pred)  # eingetzt in part. Ableitung für t
        
        m = m - alpha * d_m  # Berechne neues m
        t = t - alpha * d_t  # Berechne neues t
        
    return np.round(m,4), np.round(t,4)
    
m, t = gradient_descent(X,y,  0.001, 100000)
print("Ergebnisse:")
print(f"m={m}, t={t}")

# Plot der Regressionsgerade:

y_hat_1 = np.min(X) * m + t
y_hat_2 = np.max(X) * m + t
plt.plot([np.min(X), np.max(X)], [y_hat_1, y_hat_2], color="r")
plt.scatter(X,y)
plt.show()    

# Vergleich mit Ergebnis von sklearn.linear_model.LinearRegression
import statsmodels.formula.api as smf
model = smf.ols("Umsatz~Temperatur", data = data).fit()
print(model.summary())
