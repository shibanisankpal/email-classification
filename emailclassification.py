import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

# Function to preprocess the text data
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(w) for w in words if not w in stop_words]
    return ' '.join(words)

# Load the custom dataset
def load_data():
    # Custom dataset with sample emails
    data = {
        'subject': [
            'Hello',
            'Congratulations! You have won a vacation.',
            'Meeting reminder',
            'Important: Account security compromised.',
            'Review the attached document.',
            'URGENT: Limited time offer. Buy now and get 50% off.',
            'Parent-teacher meeting reminder.',
            'Discount sale: Get up to 70% off on selected items.',
            'Invitation: Join us for a data science webinar.',
            'Payment confirmation',
            'Participate in our survey and get a $100 gift card.',
            'Don\'t miss out on our exclusive deals.',
            'Your package has been shipped.',
            'Important notice: Account password reset.',
            'Last chance to register for our event.',
        ],
        'body': [
            'This is a non-spam email.',
            'You have been selected as the lucky winner of a free vacation.',
            'This email is to remind you of the team meeting tomorrow at 2 PM.',
            'Your account security has been compromised. Take immediate action.',
            'Please review the attached document for more information.',
            'Don\'t miss out on this limited-time offer. Buy now and get 50% off your purchase.',
            'This is a reminder for the parent-teacher meeting scheduled for next week.',
            'Get up to 70% off on selected items during the discount sale.',
            'You are invited to join us for an informative webinar on data science.',
            'Your transaction has been confirmed. Thank you for your payment.',
            'Get a $100 gift card by participating in our survey. Your opinion matters to us.',
            'Don\'t miss out on our exclusive deals and discounts this Black Friday.',
            'Your package with order number 123456 has been shipped. Track your order here.',
            'Important notice: Your account password has been reset for security reasons.',
            'This is your last chance to register for our upcoming event. Secure your spot now.',
        ],
        'label': [
            'non-spam',
            'spam',
            'non-spam',
            'spam',
            'non-spam',
            'spam',
            'non-spam',
            'non-spam',
            'non-spam',
            'non-spam',
            'non-spam',
            'spam',
            'non-spam',
            'spam',
            'non-spam',
        ]
    }
    df = pd.DataFrame(data)
    return df

# Main function
def main():
    st.title("Email Classifier")
    st.write("This app classifies emails as spam or not spam using a custom dataset.")

    # Load the dataset
    df = load_data()

    # Preprocess the text data
    df['subject'] = df['subject'].apply(preprocess_text)
    df['body'] = df['body'].apply(preprocess_text)
    df['email'] = df['subject'] + ' ' + df['body']

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['email'])
    y = df['label']

    # Train an AdaBoost classifier
    classifier = AdaBoostClassifier()
    classifier.fit(X, y)

    # User input
    user_input_subject = st.text_input("Enter the email subject:")
    user_input_body = st.text_area("Enter the email body:")
    if st.button("Classify"):
        if user_input_subject and user_input_body:
            preprocessed_subject = preprocess_text(user_input_subject)
            preprocessed_body = preprocess_text(user_input_body)
            preprocessed_email = preprocessed_subject + ' ' + preprocessed_body
            vectorized_email = vectorizer.transform([preprocessed_email])
            prediction = classifier.predict(vectorized_email)
            st.write("Predicted Label:", prediction[0])
        else:
            st.write("Please enter both the email subject and body.")

    # Evaluate the model on the entire dataset
    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y, y_pred)
    st.write("Accuracy:", accuracy)

if __name__ == '__main__':
    main()
