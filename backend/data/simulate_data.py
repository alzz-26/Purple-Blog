import csv
import random
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

NUM_SAMPLES = 500
OUTPUT_CSV = 'backend/data/synthetic_data.csv'
VECTOR_PATH = 'backend/model/vectorizer.pkl'

# Content templates by topic & tone
BLOG_TEMPLATES = {
    'technology': {
        'enthusiastic': [
            "Breakthrough in {} is transforming the industry rapidly.",
            "Latest trends in {} show incredible potential and growth.",
            "Exciting innovations in {} change how we interact with technology."
        ],
        'neutral': [
            "{} is a steadily growing field with many developments.",
            "The state of {} reflects global technological progress.",
            "Research in {} continues with diverse approaches and results."
        ],
        'critical': [
            "{} faces many challenges and ethical questions today.",
            "Risks involved in deploying {} often get overlooked.",
            "Critical evaluations of {} reveal limitations and issues."
        ]
    },
    'lifestyle': {
        'enthusiastic': [
            "Exploring {} leads to improved well-being and happiness.",
            "New approaches in {} inspire healthier and balanced lives.",
            "Incorporating {} into daily routines yields great benefits."
        ],
        'neutral': [
            "{} is common in modern lifestyles across demographics.",
            "Practices related to {} vary widely across cultures.",
            "Experts study {} with focus on consistency and effects."
        ],
        'critical': [
            "{} trends sometimes promote unrealistic expectations.",
            "Commercial interests heavily influence perceptions of {}.",
            "Some effects of {} are questioned by healthcare professionals."
        ]
    },
    'sports': {
        'enthusiastic': [
            "Records continue to be broken in {} competitions worldwide.",
            "Trainings in {} have become more advanced and specialized.",
            "{} fans show undying passion and support for their teams."
        ],
        'neutral': [
            "{} is an organized sport with debated rules and formats.",
            "Studies on {} explore injury prevention and performance.",
            "Competitions in {} are held at local and international levels."
        ],
        'critical': [
            "Funding problems limit development programs in {} heavily.",
            "Some {} practices risk athlete health and safety.",
            "Doping scandals in {} tarnish the reputation and fair play."
        ]
    },
    'health': {
        'enthusiastic': [
            "Advances in {} improve patient outcomes significantly.",
            "{} awareness campaigns lead to better community health.",
            "Innovative {} treatments offer hope to many patients."
        ],
        'neutral': [
            "{} research involves complex methodologies and long trials.",
            "Public health officials monitor {} statistics regularly.",
            "Healthcare policies concerning {} vary among regions."
        ],
        'critical': [
            "{} related misinformation spreads rapidly online.",
            "Financial barriers hinder access to {} treatments.",
            "Overuse of {} medications could lead to adverse effects."
        ]
    },
    'finance': {
        'enthusiastic': [
            "{} innovations create opportunities for new investors.",
            "Financial strategies in {} are evolving quickly.",
            "Sustainable {} investing is gaining popularity globally."
        ],
        'neutral': [
            "{} market trends fluctuate based on numerous factors.",
            "Regulations in {} aim to protect consumers and markets.",
            "Data analysis in {} supports better decision-making."
        ],
        'critical': [
            "{} scams damage investor confidence regularly.",
            "Complexities in {} financial products confuse many clients.",
            "Market manipulations pose serious risks in {} sectors."
        ]
    }
}

def generate_content():
    topic = random.choice(list(BLOG_TEMPLATES.keys()))
    tone = random.choice(list(BLOG_TEMPLATES[topic].keys()))
    template = random.choice(BLOG_TEMPLATES[topic][tone])

    topic_terms = {
        'technology': ['artificial intelligence', 'blockchain', 'quantum computing', 'cybersecurity'],
        'lifestyle': ['mindfulness', 'sustainability', 'work-life balance', 'minimalism'],
        'sports': ['basketball', 'marathon running', 'soccer', 'tennis'],
        'health': ['mental health', 'nutrition science', 'chronic diseases', 'preventive care'],
        'finance': ['stock markets', 'personal budgeting', 'retirement planning', 'cryptocurrency']
    }

    term = random.choice(topic_terms[topic])
    content = template.format(term)
    return content

def random_datetime(start, end):
    delta = end - start
    rand_secs = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=rand_secs)

def simulate_engagement(text_weight, time_bonus):
    base_likes, base_comments = 20, 5
    likes = max(0, base_likes + text_weight*15 + time_bonus + np.random.normal(0,5))
    comments = max(0, base_comments + text_weight*7 + np.random.normal(0,2))
    return int(likes), int(comments)

def main():
    samples = []
    now = datetime.now()
    start = now - timedelta(days=60)

    for _ in range(NUM_SAMPLES):
        content = generate_content()
        pub_time = random_datetime(start, now)
        samples.append((content, pub_time))

    contents = [c for c, _ in samples]
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(contents).toarray()
    joblib.dump(vectorizer, VECTOR_PATH)

    header = ['content', 'sentiment_valence', 'arousal', 'hour_of_day', 'day_of_week'] \
             + [f'tfidf_{w}' for w in vectorizer.get_feature_names_out()] \
             + ['likes', 'comments']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, (content, pub_time) in enumerate(samples):
            sent = TextBlob(content).sentiment
            val, ar = round(sent.polarity, 3), round(abs(sent.subjectivity), 3)
            time_bonus = 20 if 18 <= pub_time.hour <= 22 else 0
            text_weight = val + ar
            likes, comments = simulate_engagement(text_weight, time_bonus)

            row = [content, val, ar, pub_time.hour, pub_time.weekday()]
            row += list(tfidf_matrix[idx])
            row += [likes, comments]
            writer.writerow(row)

    print(f"Generated {NUM_SAMPLES} samples for training in {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
