# SMS Spam Classifier using Naive Bayes

A simple text classification project that identifies whether an incoming message is spam or a normal message using natural language processing techniques and the Naive Bayes algorithm.

# How it works

1. Text messages are cleaned and transformed into word counts using CountVectorizer.
2. The Multinomial Naive Bayes algorithm calculates the probability of a message belonging to either class based on the individual words present.
3. The trained model predicts whether unseen text is normal or spam.

# Tech stack

* Python
* Scikit-Learn

# Getting started

# Prerequisites

Install the required library:

```bash
pip install scikit-learn
git clone [https://github.com/your-username/sms-spam-classifier.git](https://github.com/your-username/sms-spam-classifier.git)
cd sms-spam-classifier
python spam_detector.py
