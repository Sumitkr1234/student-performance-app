import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

df = pd.read_csv("student-mat.csv")
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

df['performance'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
df.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)

X = df[['age', 'studytime', 'failures', 'absences', 'goout', 'freetime', 'Dalc', 'Walc']]
y = df['performance']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
