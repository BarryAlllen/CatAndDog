import os

import matplotlib.pyplot
import pandas as pd

filenames = os.listdir('H:\\Machine Learning\\CNN\\CNNProjects\\data\\train')
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append("dog")
    else:
        categories.append("cat")

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

print(df.head())
print(df.tail())
df['category'].value_counts().plot.bar()
matplotlib.pyplot.show()