import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("--- Starting Model Training (with Emotional Features) ---")

# 1. Load and Clean Data
df = pd.read_csv('twitter_dataset.csv', encoding='utf-8')
df['description'] = df['description'].astype(str).fillna('')
numerical_cols = ['likes', 'replies', 'views', 'followers']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df = df[df['views'] > 0].copy()
print(f"Cleaned dataset rows: {len(df)}")

if df.empty:
    print("Error: No data remaining. Cannot train model.")
    exit()

# 2. Add Emotional Features using VADER
analyzer = SentimentIntensityAnalyzer()
def get_sentiment_scores(sentence):
    scores = analyzer.polarity_scores(sentence)
    return scores['pos'], scores['neg'], scores['neu'], scores['compound']

sentiment_scores = df['description'].apply(get_sentiment_scores)
df[['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_compound']] = pd.DataFrame(sentiment_scores.tolist(), index=df.index)
print("Emotional features created successfully.")


# 3. Feature Engineering (reverting to ratios)
df['likes_per_view'] = df['likes'] / df['views']
df['replies_per_view'] = df['replies'] / df['views']
df['log_likes_per_view'] = np.log1p(df['likes_per_view'])
df['log_replies_per_view'] = np.log1p(df['replies_per_view'])

# 4. Define Features (X) and Targets (y)
feature_columns = [
    'description', 
    'followers', 
    'sentiment_pos', 
    'sentiment_neg', 
    'sentiment_compound'
]
X = df[feature_columns]
y = df[['log_likes_per_view', 'log_replies_per_view']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_with_views = X_test.copy()
X_test_with_views['views'] = df.loc[X_test.index, 'views']
# Save y_test
joblib.dump((X_test_with_views, y_test), 'test_data_with_views.pkl') 

print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows).")

# 5. Define Preprocessor and Model
numeric_features_for_model = [
    'followers', 
    'sentiment_pos', 
    'sentiment_neg', 
    'sentiment_compound'
]
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=1000, stop_words='english'), 'description'),
        ('numeric', 'passthrough', numeric_features_for_model)
    ],
    remainder='drop'
)
hgb_regressor = HistGradientBoostingRegressor(random_state=42)
multi_output_model = MultiOutputRegressor(hgb_regressor)

# 6. Process data and train model
print("Preprocessing data and training the model...")
X_train_transformed = preprocessor.fit_transform(X_train)
X_train_dense = X_train_transformed.toarray()
multi_output_model.fit(X_train_dense, y_train)
print("Model training complete.")

# 7. Save the preprocessor and model
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(multi_output_model, 'engagement_model.pkl')
print("Preprocessor and model saved successfully.")