import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("Starting model evaluation...")

# Load the preprocessor and the model separately
try:
    preprocessor = joblib.load('preprocessor.pkl')
    model = joblib.load('engagement_model.pkl')
    # Load the correct test data file 
    X_test_with_views, y_test_log_ratios = joblib.load('test_data_with_views.pkl')
    print("Preprocessor, model, and test data loaded successfully.")
except FileNotFoundError:
    print("Error: Could not find necessary .pkl files. Please run train.py first.")
    exit()

# Transform the test data and convert to dense array
X_test_transformed = preprocessor.transform(X_test_with_views)
X_test_dense = X_test_transformed.toarray()

# Make predictions on the dense test data
predicted_log_ratios = model.predict(X_test_dense)

# Inverse transform predictions to get actual counts
predicted_ratios = np.expm1(predicted_log_ratios)
predicted_likes_ratio = predicted_ratios[:, 0]
predicted_replies_ratio = predicted_ratios[:, 1]
predicted_likes_count = predicted_likes_ratio * X_test_with_views['views'].values
predicted_replies_count = predicted_replies_ratio * X_test_with_views['views'].values
true_ratios = np.expm1(y_test_log_ratios)
true_likes_count = true_ratios['log_likes_per_view'].values * X_test_with_views['views'].values
true_replies_count = true_ratios['log_replies_per_view'].values * X_test_with_views['views'].values

def order_of_magnitude_accuracy(y_true, y_pred):
    correct = 0
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val == 0 and pred_val == 0:
            correct += 1; continue
        if true_val <= 0 or pred_val <= 0:
            continue
        true_order = np.floor(np.log10(true_val))
        pred_order = np.floor(np.log10(pred_val))
        if true_order == pred_order:
            correct += 1
    return (correct / len(y_true)) * 100 if len(y_true) > 0 else 0

oom_likes = order_of_magnitude_accuracy(true_likes_count, predicted_likes_count)
mae_likes = mean_absolute_error(true_likes_count, predicted_likes_count)
rmse_likes = np.sqrt(mean_squared_error(true_likes_count, predicted_likes_count))
r2_likes = r2_score(true_likes_count, predicted_likes_count)
oom_replies = order_of_magnitude_accuracy(true_replies_count, predicted_replies_count)
mae_replies = mean_absolute_error(true_replies_count, predicted_replies_count)
rmse_replies = np.sqrt(mean_squared_error(true_replies_count, predicted_replies_count))
r2_replies = r2_score(true_replies_count, predicted_replies_count)

print("\n--- Model Performance Metrics ---")
print("---------------------------------------------------------")
header = f"{'Metric':<35} | {'Likes':<10} | {'Replies':<10}"
print(header)
print("---------------------------------------------------------")
print(f"{'Order-of-Magnitude Accuracy (%)':<35} | {oom_likes:<10.2f} | {oom_replies:<10.2f}")
print(f"{'Mean Absolute Error (MAE)':<35} | {mae_likes:<10.2f} | {mae_replies:<10.2f}")
print(f"{'Root Mean Squared Error (RMSE)':<35} | {rmse_likes:<10.2f} | {rmse_replies:<10.2f}")
print(f"{'Coefficient of Determination (R²)':<35} | {r2_likes:<10.2f} | {r2_replies:<10.2f}")
print("---------------------------------------------------------")