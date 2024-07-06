# STOCK PRICE PREDICTION DOCUMENT
A project that apply some machine learning models in predicting stock.

<img src="https://img.shields.io/github/stars/kuroneko2002/stock-pred"/> ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/kuroneko2002/stock-pred)

## 💻 Technical Stack

- [Python](https://www.python.org) - A programming language that lets you work quickly and integrate systems more effectively.
- [Scikit-learn](https://scikit-learn.org/stable/) - Machine Learning in Python.
- [Plotly Dash](https://dash.plotly.com) - Dash is the original low-code framework for rapidly building data apps in Python.

## 💽 Setup project

### ⚙️ Download package
```console
pip install scikit-learn yfinance keras xgboost dash plotly 
```

### 💎 Update latest data for csv file
There are two ways to get/update data:
- Open getHistory.ipynb and run the only cell in it.
- Uncomment "get_all_csv()" in main.py for the first run.

### 📦 Run main.py to train model
```python
python main.py
```

### 📦 Run appUI.py to view app browser UI
```python
python appUI.py
```
