import yfinance as yf

ticker = "AMZN"
df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
print(df.head())
