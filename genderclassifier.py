import pandas as pd
import numpy as np

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

colums_to_encoded = ['_unit_state', 'gender', 'link_color', 'sidebar_color']

for i in colums_to_encoded:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    x[i] = le.fit_transform(x[i])

colums_to_scale = ['_trusted_judgments', 'link_color', 'sidebar_color',
                   'fav_number', 'retweet_count', 'tweet_count']

y = x['gender']
x.drop(x['gender'])
