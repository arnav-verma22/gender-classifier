import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv("gender-classifier-DFE-791531.csv", encoding="ISO-8859-1")

for i in df.columns:
    print(i, len(df[i].unique()))

columns = ['_golden', '_last_judgment_at', 'profile_yn', 'profile_yn:confidence', 'created', 'gender_gold',
           'profile_yn_gold', 'profileimage', 'tweet_coord', 'tweet_created', 'tweet_location', 'user_timezone']

x = df.drop(columns=columns)


for j in x.columns:
    print(j, x[j].isnull().values.any())

from sklearn.impute import SimpleImputer
mv = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value= 0)
mv = mv.fit([x['gender:confidence']])
[x['gender:confidence']] = mv.transform([x['gender:confidence']])

mv = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value= 'other')
mv = mv.fit([x['gender']])
[x['gender']] = mv.transform([x['gender']])

colums_to_encoded = ['_unit_state', 'link_color', 'sidebar_color']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in colums_to_encoded:
    x[i] = le.fit_transform(x[i])

colums_to_scale = ['_trusted_judgments', 'link_color', 'sidebar_color',
                   'fav_number', 'retweet_count', 'tweet_count']

y = x['gender']
x = x.drop(columns=['gender', 'description', 'text', 'name'])



from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x = sc_X.fit_transform(x)

y = pd.get_dummies(y)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=8, activation='tanh'))
ann.add(tf.keras.layers.Dense(units=7, activation='tanh'))
ann.add(tf.keras.layers.Dense(units=5, activation='softmax'))
ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
ann.fit(xtrain, ytrain, batch_size = 32, epochs = 100)

nn_prediction = ann.predict(xtest)

from sklearn.metrics import confusion_matrix, accuracy_score
print(accuracy_score(ytest, nn_prediction))
