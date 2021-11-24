# CoinBase
Coinbase Trading Bots

LSTM Machine Learned Trading Bot for CoinBase Pro

This script takes the past 300 minutes of prices and places the average value mean into the predicate variable.

You may need to install packages with pip or pip3 to get it to work.

pip3 install keras
pip3 install matplotlib
pip3 install numpy
pip3 install pandas
pip3 install seaborn
pip3 install scikit
pip3 install cbpro

You will need to program certain command line variables 

For example and you can change this:

python3 price.py "MKR" "buy" "0.000001" "0.001" "4"

the variables conform to the following:

ticker = sys.argv[1]
action = sys.argv[2]
increment = sys.argv[3]
size = sys.argv[4]
rounder = sys.argv[5]

You will also want to tweak these variable conditions according to your need:

    if (predicate <= 0.02 and (dictionary > 0)) and (difference <= 0.001 and (dictionary > 0)) and (average <= 0.001 and (dictionary > 0)):

Just ask for help if you need it I should respond.  This should work in Linux and Windows and Mac.
