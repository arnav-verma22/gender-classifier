import pandas as pd

df = pd.read_csv("gender-classifier-DFE-791531.csv", encoding="ISO-8859-1")

for i in df.columns:
    print(i, len(df[i].unique()))

columns = ['_golden', '_last_judgment_at', 'profile_yn', 'profile_yn:confidence', 'created', 'gender_gold',
           'profile_yn_gold', 'profileimage', 'tweet_coord', 'tweet_created', 'tweet_location', 'user_timezone']

x = df.drop(columns=columns)
y = df['gender']

for j in x.columns:
    print(j, x[j].isnull().values.any())

from sklearn.impute import SimpleImputer
mv = SimpleImputer(missing_values = 0, strategy = 'mean', axis = 0)
mv = mv.fit(x[:, 1:3])
x[:, 1:3] = mv.transform(x[:, 1:3])

colums_to_encoded = ['_unit_state', 'gender', 'link_color', 'sidebar_color']

for i in colums_to_encoded:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    x[i] = le.fit_transform(x[i])