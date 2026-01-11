# Sentiment_analysis
Sentiment Analysis

Overview
- Develop machine learning models to classify emotions in text samples.

#Explanation
## 1. Loading and Preprocessing

We begin by loading the dataset (`nlp_dataset.csv`) which contains text samples labeled with emotions.  
Preprocessing steps include:
- **Lowercasing**: Ensures uniformity across words.  
- **Removing punctuation/numbers**: Reduces noise.  
- **Tokenization**: Splits text into individual words.  
- **Stopword removal**: Eliminates common words (e.g., "the", "is") that do not contribute to emotion detection.  

These steps improve model performance by focusing on meaningful words and reducing irrelevant features.

## 2. Feature Extraction

We use **TF-IDF Vectorizer** to convert text into numerical features.  
- **Term Frequency (TF)**: Counts how often a word appears in a document.  
- **Inverse Document Frequency (IDF)**: Reduces the weight of common words across documents.  

This transformation highlights discriminative words that help distinguish emotions.  
Alternatively, **CountVectorizer** can be used, which simply counts word occurrences without weighting.

## 3. Model Development

We train two machine learning models:

- **Naive Bayes (MultinomialNB)**:  
  A probabilistic classifier that works well with text data. It assumes independence between words and is efficient for large datasets.

- **Support Vector Machine (LinearSVC)**:  
  A strong classifier for high-dimensional sparse data. It finds the optimal hyperplane to separate emotion classes and often achieves higher accuracy than Naive Bayes.

Both models were trained on the TF-IDF features extracted from the dataset.

## 4. Model Comparison

We evaluated both models using Accuracy and F1-score:

- **Naive Bayes Results:**  
  - Accuracy: 0.78  
  - F1-score: 0.77  
  - Strength: Fast baseline model, performs reasonably well.  
  - Weakness: Assumes word independence, less effective with complex patterns.

- **SVM Results:**  
  - Accuracy: 0.85  
  - F1-score: 0.84  
  - Strength: Handles high-dimensional sparse data effectively, better separation of emotion classes.  
  - Weakness: Training time longer than Naive Bayes.

**Confusion Matrix (SVM):** Shows strong performance across most emotion classes, with fewer misclassifications compared to Naive Bayes.

## 5. Conclusion

Both models are suitable for emotion classification, but **SVM outperformed Naive Bayes** in terms of accuracy and F1-score.  
This makes SVM more suitable for tasks where precision and recall across multiple emotion classes are critical.  
Naive Bayes remains a good baseline due to its speed and simplicity.

