from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. Training messages
messages = [
    "hey are we meeting today for cricket",
    "win cash prize and free recharge now",
    "please send the school notes",
    "congratulations you won a brand new car click link",
    "call me when you reach home",
    "exclusive offer claim your flat 50 percent discount today",
    "mom said dinner is ready",
    "urgent claim your lottery ticket reward",
]

# 2. Labels: 0 means normal message, 1 means spam
labels = [0, 1, 0, 1, 0, 1, 0, 1]

# 3. Build a pipeline: convert words into counts, then apply Naive Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB())

# 4. Train the model
model.fit(messages, labels)

# 5. Test with new messages the model has never seen
new_messages = [
    "hey call me after tuition",
    "claim your free recharge bonus today",
]

predictions = model.predict(new_messages)

for text, pred in zip(new_messages, predictions):
    verdict = "Spam" if pred == 1 else "Normal message"
    print(f"Message: '{text}' -> Result: {verdict}")
