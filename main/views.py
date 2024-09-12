from django.shortcuts import render
from django.http import HttpResponse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

def preprocess_text(text):
    """Preprocess the text: tokenize, remove stopwords, and apply stemming."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return filtered_words

def sentence_scoring(text, sentences, tfidf_matrix):
    """Score each sentence based on TF-IDF scores and other features."""
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        score = 0
        for word in word_tokenize(sentence.lower()):
            if word in tfidf_matrix.vocabulary_:
                score += tfidf_matrix.idf_[tfidf_matrix.vocabulary_[word]]
        # Additional scoring criteria (optional)
        score += (len(sentence.split()) / len(sentences))  # Give preference to longer sentences
        sentence_scores.append((sentence, score))
    return sentence_scores

def extractive_summary(text, num_sentences=3):
    """Generate an extractive summary from the text."""
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text  # If text is too short, return it as is

    # Preprocess text for TF-IDF
    filtered_words = preprocess_text(text)
    clean_text = ' '.join(filtered_words)

    # Calculate TF-IDF scores
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([clean_text])

    # Score sentences
    sentence_scores = sentence_scoring(text, sentences, vectorizer)

    # Sort sentences by score
    ranked_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

    # Select the top sentences for the summary
    summary_sentences = [sentence for sentence, score in ranked_sentences[:num_sentences]]
    summary = ' '.join(summary_sentences)

    return summary

def summary(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        num_sentences = int(request.POST.get('num_sentences', 3))
        if text:
            summarized_text = extractive_summary(text, num_sentences)
            text_block = {
                'text': text,
                'summarized_text': summarized_text
            }
        else:
            text_block = {
                'text': '',
                'summarized_text': ''
            }
    else:
        text_block = {
            'text': '',
            'summarized_text': ''
        }

    return render(request, 'summarize.html', text_block)
