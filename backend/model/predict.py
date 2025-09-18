import sys
import json
import pandas as pd
from textblob import TextBlob
import joblib

model = joblib.load('engagement_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

input_data = json.loads(sys.stdin.read())
content = input_data.get('content', '')
hour = input_data.get('hour_of_day', 12)
dow = input_data.get('day_of_week', 0)

sent = TextBlob(content).sentiment
val, ar = sent.polarity, abs(sent.subjectivity)
tfidf_vec = vectorizer.transform([content]).toarray()[0]

features = [val, ar, hour, dow] + list(tfidf_vec)
df = pd.DataFrame([features])

pred = model.predict(df)[0]

print(json.dumps({'predictedLikes': int(pred[0]), 'predictedComments': int(pred[1])}))
