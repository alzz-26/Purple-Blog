import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

df = pd.read_csv('backend/data/synthetic_data.csv')

feature_cols = [c for c in df.columns if c not in ['content', 'likes', 'comments']]
X = df[feature_cols]

y = df[['likes', 'comments']]

model = MultiOutputRegressor(HistGradientBoostingRegressor())
model.fit(X, y)

joblib.dump(model, 'backend/model/engagement_model.pkl')
print("Model trained and saved as engagement_model.pkl")
