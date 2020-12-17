import numpy as np
import pandas as pd

data = pd.read_excel('normal.xlsx', 'Sheet1', index_col=0)  # xlsxファイル読み込み
data.to_csv('data.csv', encoding='utf-8')                   # csvに変換
df = pd.read_csv('data.csv', index_col=0)                   # 1行目・1列目をヘッダーとしてcsv読み込み
a = df.values                                               # ヘッダーを除いた値読み込み
np.savetxt('normal.csv', a, delimiter=',')                  # csvファイルとして保存