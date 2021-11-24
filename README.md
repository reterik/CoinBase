# CoinBase
Coinbase Trading Bots

LSTM Machine Learned Trading Bot for CoinBase

You may need to install packages with pip or pip3 to get it to work.

You will need to program certain command line variables 

For example and you can change this:

python3 price.py "MKR" "buy" "0.000001" "0.001" "4"

the variables conform to the following:

ticker = sys.argv[1]
action = sys.argv[2]
increment = sys.argv[3]
size = sys.argv[4]
rounder = sys.argv[5]

