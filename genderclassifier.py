import pandas as pd

df = pd.read_csv("gender-classifier-DFE-791531.csv", encoding="ISO-8859-1")

for i in df.columns:
    print(i, len(df[i].unique()))

columns = ['_golden', '_last_judgment_at', 'profile_yn', 'profile_yn:confidence', 'created', 'gender_gold',
           'profile_yn_gold', 'profileimage', 'tweet_coord', 'tweet_created', 'tweet_location', 'user_timezone']