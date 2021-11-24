import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import pprint
import csv
import itertools as it
import cbpro
import time
import sys

ticker = sys.argv[1]
action = sys.argv[2]
increment = sys.argv[3]
size = sys.argv[4]
rounder = sys.argv[5]
bids = 0
asks = 0
counting = 0
buyer = size

public_client = cbpro.PublicClient()
public_client.get_products()
# Get the order book at the default level.
key = "32 chars"
passphrase = "11 chars"
b64secret = "88 chars"
bids = 0
auth_client = cbpro.AuthenticatedClient(key, b64secret, passphrase)
testing = public_client.get_product_historic_rates(ticker+'-USD')

y = testing

#hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = pd.DataFrame(y)
print(hist)
hist = hist.set_index(0)
hist.index = pd.to_datetime(hist.index, unit='s')
target_col = 1

def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data
train, test = train_test_split(hist, test_size=0.2)

def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [CAD]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
line_plot(train[target_col], test[target_col], 'training', 'test', title='')

def normalise_zero_base(df):
    return df / df.iloc[0] - 1

def normalise_min_max(df):
    return (df - df.min()) / (data.max() - df.min())

def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test

def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

np.random.seed(42)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 302
epochs = 20
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'
train, test, X_train, X_test, y_train, y_test = prepare_data(
    hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)
model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)
history = model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
targets = test[target_col][window_len:]
preds = model.predict(X_test).squeeze()
predicate = mean_absolute_error(preds, y_test)
preds = test[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
line_plot(targets, preds, 'actual', 'prediction', lw=3)


def pretty(d, indent=0):
   for key in d:
      print('\t' * indent + str(key))

def yrange(n):
    i = 0
    while i < n:
        yield i
        i += 1


#pretty(list(y))
avg = [float(sum(col))/len(col) for col in zip(*y)]
print(avg[1])

testing = public_client.get_product_order_book(ticker+'-USD')
print(ticker+'-USD')

for key, value in testing.items():
    if(key == "bids"):
        print("BIDS "+value[0][0])
        bids = value[0][0]
    if(key == "asks"):
        print("ASKS "+value[0][0])
        asks = value[0][0]

average = (avg[1] - float(asks)) / avg[1]
print(average)
difference = (float(asks) - float(bids)) / float(asks)
print(asks)
print(bids)
print("difference")
print(difference)
print("predicate")
print(predicate)

account = auth_client.get_accounts()
#print(account)
defi = "none"
r = 0
g = 0
dictionary = 0
while r < 115:
    r = r + 1
    for c in account[r].items():
        if str(c[1]) == ticker:
            print(str(c[1]))
            g = r
        if g == r and str(c[0]) == "balance":
            dictionary = float(c[1])

r = 0
h = 0
usd = 0
while r < 115:
    r = r + 1
    for c in account[r].items():
        if str(c[1]) == "USD":
            print(str(c[1]))
            h = r
        if h == r and str(c[0]) == "balance":
            usd = float(c[1])

print(dictionary)

decrement = (float(increment))

calc = (float(asks) * float(3)) / 100
print("calc")
print(calc)
print(float(calc) * float(asks))
size = calc

#65000 * 0.003
#x * 100

if dictionary > 0:
    if (predicate <= 0.02 and (dictionary > 0)) and (difference <= 0.001 and (dictionary > 0)) and (average <= 0.001 and (dictionary > 0)):
        size = dictionary
        action = "sell"
        print(action)
        increment = float(increment) + (float(increment) * -1)
        auth_client.cancel_all(product_id=ticker + '-USD')
        erc = (auth_client.place_limit_order(product_id=ticker+'-USD',side=action,price=round(float((float(bids)+float(increment))),int(rounder)),size=size))
        print(erc)

    if (predicate <= 0.02 and (dictionary > 0)) and (difference <= 0.001 and (dictionary > 0)):
        size = dictionary
        action = "sell"
        print(action)
        increment = float(increment) + (float(increment) * -1)
        auth_client.cancel_all(product_id=ticker + '-USD')
        erc = (auth_client.place_limit_order(product_id=ticker+'-USD',side=action,price=round(float((float(bids)+float(increment))),int(rounder)),size=size))
        print(erc)

if usd >= 3:
    if (predicate >= 0.02):
        size = buyer
        action = "buy"
        print(action)
        auth_client.cancel_all(product_id=ticker + '-USD')
        erc = (auth_client.place_limit_order(product_id=ticker+'-USD',side=action,price=round(float((float(bids)+float(decrement))),int(rounder)),size=size))
        print(erc)

    if (predicate >= 0.02) and (average >= 0.001) and (difference >= 0.001):
        size = buyer
        action = "buy"
        print(action)
        auth_client.cancel_all(product_id=ticker + '-USD')
        erc = (auth_client.place_limit_order(product_id=ticker+'-USD',side=action,price=round(float((float(bids)+float(decrement))),int(rounder)),size=size))
        print(erc)
