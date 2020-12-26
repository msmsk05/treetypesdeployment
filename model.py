import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('final.csv')

X=dataset.drop("Cover_Type", inplace=True)
y=dataset.Cover_Type


from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier()
classifier.fit(X, y)

pickle.dump(classifier, open('model.pkl','wb'))

model=pickle.load(open("model.pkl", "rb"))

print(model.predict([[2788, 2, 3555,221, 2984]]))