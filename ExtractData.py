import pandas as pd
data = pd.read_excel('Donnees_COVID.xlsx')
data = data.dropna(axis=0)
data = data.drop([64])
data = data.drop([66])

print(data)

