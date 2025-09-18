import joblib
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the preprocessor and the model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('engagement_model.pkl')
analyzer = SentimentIntensityAnalyzer()

def predict_engagement(description, followers, views):
    
    # Get sentiment scores for the input text 
    scores = analyzer.polarity_scores(description)
    sentiment_pos = scores['pos']
    sentiment_neg = scores['neg']
    sentiment_compound = scores['compound']

    # Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        'description': [description],
        'followers': [followers],
        'sentiment_pos': [sentiment_pos],
        'sentiment_neg': [sentiment_neg],
        'sentiment_compound': [sentiment_compound]
    })

    # Transform data, convert to dense, and predict log-ratios
    input_transformed = preprocessor.transform(input_data)
    input_dense = input_transformed.toarray()
    predicted_log_ratios = model.predict(input_dense)

    # Inverse transform to get final counts
    predicted_ratios = np.expm1(predicted_log_ratios)
    predicted_likes_count = predicted_ratios[0, 0] * views
    predicted_replies_count = predicted_ratios[0, 1] * views
    
    prediction_result = {
        'predicted_likes': round(predicted_likes_count),
        'predicted_replies': round(predicted_replies_count)
    }
    return prediction_result