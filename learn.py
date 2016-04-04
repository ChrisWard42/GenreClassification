"""Module to execute machine learning algorithms and generate statistics on book data"""

import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
import matplotlib.pyplot as plt

debug = True

genres = ['Adventure', 'Biography', 'History', 'Mystery-Horror', 'Poetry', 'Romance', 'Science Fiction']
# book_directory = "/Users/nikolaivogler/manybooks_sanitized"  # Nikolai
book_directory = "Z:\\dev\\projects\\cs175\\manybooks_sanitized"  # Chris

# seed random number generator for testing
np.random.seed(0)
rs = np.random.RandomState()

'''
Below code borrowed from: http://stackoverflow.com/questions/26126442/combining-text-stemming-and-removal-of-punctuation-in-nltk-and-scikit-learn
'''


def tokenize(text):
    import nltk
    from nltk.stem.porter import PorterStemmer
    from string import punctuation
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    # tokens = [t for t in tokens if t not in punctuation]
    filtered_tokens = []
    for token in tokens:
        if all(c.isalpha() or c.isspace() for c in token):
            filtered_tokens.append(token)
    # TODO: remove proper nouns (prior to converting to lowercase in tokenize (I think))
    stems = stem_tokens(filtered_tokens, stemmer)
    return stems


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


'''
Some code modified from http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#example-hetero-feature-union-py
This was for our attempted implementation of FeatureUnions to get stylistic information about our books.
However, this process failed because of the long amount of time (days) needed for Python to process all of our texts
and the amount of memory required to store all of the information that caused multiple memory errors on our computers.
'''


class BookStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def fit(self, x, y=None):
        return self

    def transform(self, books):
        from nltk.tokenize import sent_tokenize
        all_sentences = [sent_tokenize(book) for book in self.data]
        return [{'length': len(self.data[i]),
                'num_sentences': len(all_sentences[i]),
                'avg_sentence_length': sum([len(sent) for sent in all_sentences[i]]) / len(all_sentences[i])}
                for i in range(len(self.data))]


class GetBookData(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def _fetch_book_data(self):
        '''
        Use sklearn's load_files to
        :param genres:
        :return:
        '''
        book_data = joblib.load("book_data.pkl")
        return book_data
        # from sklearn.datasets import load_files
        # genres = ['Adventure', 'Biography', 'History', 'Mystery-Horror', 'Poetry', 'Romance', 'Science Fiction']
        # return load_files(book_directory, categories=genres, load_content=True, encoding='ISO-8859-1', shuffle=True)

    def fit(self, x, y=None):
        return self

    def transform(self):
        return self._fetch_book_data().data


'''
More code borrowed from a scikit-learn tutorial on visualizing confusion matrices:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
'''


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


'''
Learning Functions
'''


def extract_text_features(train_data, test_data):
    """
    Extracts text features from each document by converting into a sparse
    representation of the counts of their terms, and then weights the terms
    using tf-idf representation and return all forms of the counts.

    :param train_data: <NDArray> data used to train classifier
    :param test_data: <NDArray> data used to test classification accuracy
    :return: <NPArray> Counts of term frequency in training data,
             <NPArray> tf-idf matrix representation for training data,
             <NPArray> Counts of term frequency in test data,
             <NPArray> tf-idf matrix representation for test data,
             <CountVectorizer> CountVectorizer object
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from nltk.corpus import stopwords

    # create X_train_counts & X_test_counts
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data)
    X_test_counts = count_vect.transform(test_data)

    # create X_train_tfidf & X_test_tfidf
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    return X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf, count_vect


def fetch_book_data(genres=genres):
    """
    Uses sklearn's load_files to generate a data set representation of all of the books separated
    by genre.

    :param genres: <List<String>> List of genres to separate into
    :return: <Bunch> Dict-like object containing data, filenames, raw text, etc.
    """
    from sklearn.datasets import load_files
    # TODO: might want to shuffle these so that they can be used in GRADIENT DESCENT (i.i.d)
    return load_files(book_directory, categories=genres, load_content=False, encoding='ISO-8859-1', shuffle=True)


def fit_and_predict_multinomialNB(X_train, Y_train, X_test):
    """
    DEPRECATED FUNCTION
    """
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB().fit(X_train, Y_train)
    return clf, clf.predict(X_test)


def fit_and_predict_proba_multinomialNB(X_train, Y_train, X_test):
    """
    DEPRECATED FUNCTION
    """
    from sklearn.naive_bayes import MultinomialNB

    Y_hat = MultinomialNB().fit(X_train, Y_train).predict_proba(X_test)
    labels = np.unique(Y_train)  # get Y target labels to tag our probabilities with
    proba = Y_hat.tolist()
    return [zip(i, labels) for i in proba]


def fit_and_predict_LR(X_train, Y_train, X_test):
    """
    Fits the model using logistic regression and returns the classifier and the
    predictions it generates.

    :param X_train: <NDArray> tf-idf matrix representation for training data
    :param Y_train: <NDArray> matrix representation for targets
    :param X_test: <NDArray> tf-idf matrix representation for test data
    :return: <LogisticRegression> Object representing LR classifier,
             <NDArray> Predictions on test data
    """
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(verbose=2).fit(X_train, Y_train)
    return clf, clf.predict(X_test)


def fit_and_predict_proba_LR(X_train, Y_train, X_test):
    """
    Fits the model using logistic regression and returns a list of lists containing
    the prediction probabilities for each class in the data.

    :param X_train: <NDArray> tf-idf matrix representation for training data
    :param Y_train: <NDArray> matrix representation for targets
    :param X_test: <NDArray> tf-idf matrix representation for test data
    :return: <List<List>> Prediction probabilities for each class
    """
    from sklearn.linear_model import LogisticRegression
    Y_hat = LogisticRegression(verbose=2).fit(X_train, Y_train).predict_proba(X_test)
    labels = np.unique(Y_train)  # get Y target labels to tag our probabilities with
    proba = Y_hat.tolist()
    return [zip(i, labels) for i in proba]


def show_n_most_informative_features(classifier, vectorizer, categories, n):
    """
    Function adapted from http://scikit-learn.org/stable/datasets/twenty_newsgroups.html

    Prints the n most informative features from classifier's predictions.

    :param classifier: <Classifier> Object representing classifier fitted to data
    :param vectorizer: <CountVectorizer> CountVectorizer object
    :param categories: <List<String>> Genres categorization divisions
    :param n: <Int> Number of features to report
    :return:
    """
    feature_names = np.asarray(vectorizer.get_feature_names())
    print("{} Most Informative Features:".format(n))
    for i, category in enumerate(categories):
        top_n = np.argsort(classifier.coef_[i])[-n:]
        print("%s: %s" % (category, " ".join(feature_names[top_n])))


def show_n_least_informative_features(classifier, vectorizer, categories, n):
    """
    Function adapted from http://scikit-learn.org/stable/datasets/twenty_newsgroups.html

    Prints the n least informative features from classifier's predictions.

    :param classifier: <Classifier> Object representing classifier fitted to data
    :param vectorizer: <CountVectorizer> CountVectorizer object
    :param categories: <List<String>> Genres categorization divisions
    :param n: <Int> Number of features to report
    :return:
    """
    feature_names = np.asarray(vectorizer.get_feature_names())
    print("{} Least Informative Features:".format(n))
    for i, category in enumerate(categories):
        bottom_n = np.argsort(classifier.coef_[i])[:n]
        print("%s: %s" % (category, " ".join(feature_names[bottom_n])))


if __name__ == '__main__':
    # Fetch the book data from the genre directories (CHOOSE ONE METHOD)
    if debug: print("Fetching book data...")
    # book_data = joblib.load('book_data.pkl')  # Load data from pickle file
    book_data = fetch_book_data(genres)  # Load data from folders

    # Split train and test data
    if debug: print("Splitting into train and test...")
    # Y_test = joblib.load('Y_test.pkl')
    X_train, X_test, Y_train, Y_test = train_test_split(book_data.data, book_data.target, test_size=0.25, random_state=rs)

    # Extract count vector from train and test data and get tf-idf matrix
    if debug: print("Extracting text features...")
    # X_test_tfidf = joblib.load('X_test_tfidf.pkl')
    # count_vect = joblib.load('count_vect.pkl')
    X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf, count_vect = extract_text_features(X_train, X_test)

    # Get class predictions using argmax (SELECT ONE MODEL AND ONE METHOD OF LOADING)
    if debug: print("Fitting and predicting model...")
    lr_clf, Yhat_LR = fit_and_predict_LR(X_train_tfidf, Y_train, X_test_tfidf)
    # lr_clf = joblib.load('lr_clf.pkl')
    # OLD
    # mnb_clf, mnb_clf_fit, Yhat_MNB = fit_and_predict_multinomialNB(X_train_tfidf, Y_train, X_test_tfidf)

    # Make predictions using the model
    if debug: print("Predicting on model...")
    # Yhat_LR = joblib.load('Yhat_LR.pkl')
    Yhat_LR = lr_clf.predict(X_test_tfidf)

    # Ensure model persistence; save to file using pickle
    if debug: print("Dumping dem pickles yo...")
    joblib.dump(lr_clf, 'lr_clf.pkl', compress=9)
    # joblib.dump(X_test_tfidf, 'X_test_tfidf.pkl', compress=9)
    # joblib.dump(book_data, 'book_data.pkl', compress=9)
    # joblib.dump(count_vect, 'count_vect.pkl', compress=9)
    # joblib.dump(Y_test, 'Y_test.pkl', compress=9)
    # joblib.dump(Yhat_LR, 'Yhat_LR.pkl', compress=9)

    # Checking baseline accuracy of the model
    if debug: print("Checking baseline accuracy...")
    print('Baseline accuracy with logistic regression: {:.4f}'.format(np.mean(Yhat_LR == Y_test)))

    # OLD
    # print('Accuracy with Multinomial naive Bayes: {:.4f}'.format(np.mean(Yhat_MNB == Y_test)))

    # Generating Classification Report
    if debug: print("Generating classification report...")
    print(classification_report(Y_test, Yhat_LR, target_names=book_data.target_names))

    # Generating Confusion Matrix
    if debug: print("Generating confusion matrix...")
    print(confusion_matrix(Y_test, Yhat_LR))

    # Fitting and predicting probability model
    if debug: print("Fitting and predicting probability model...")
    predicted_proba_LR = fit_and_predict_proba_LR(X_train_tfidf, Y_train, X_test_tfidf)
    # OLD
    # predicted_proba_multNB = fit_and_predict_proba_multinomialNB(X_train_tfidf, Y_train, X_test_tfidf)

    # Print Probability Model results
    if debug: print("Printing probability model results...")
    with open('predicted_proba_LR_baseline', 'w+') as f:
        f.write('\n'.join(str(list(predicted_proba_LR))))

    # Print out most/least informative features
    n = 25
    # print('Multinomial Naive Bayes')
    # show_n_most_informative_features(mnb_clf, count_vect, book_data.target, n)
    # show_n_least_informative_features(mnb_clf, count_vect, book_data.target, n)
    print('Logistic Regression')
    show_n_most_informative_features(lr_clf, count_vect, genres, n)
    show_n_least_informative_features(lr_clf, count_vect, genres, n)

    # Plotting Confusion Matrix
    cm = confusion_matrix(Y_test, Yhat_LR)
    plt.figure()
    plot_confusion_matrix(cm, book_data.target_names)

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, book_data.target_names, title='Normalized confusion matrix')

    plt.show()


    '''
    The following code blocks represent code that we attempted to run, but could not get working to add to our current
    feature extraction methods. This was very unfortunate because we wanted to build on our features using FeatureUnions
    and other methods such as POS tag frequency features. The following code fragments are therefore unused due to time
    constraints because of having to recreate our dataset and computational infeasibility (memory errors with huge text
    files, POS tagging is slow, ngram tokenization is EXTREMELY slow, etc.)

    from sklearn.grid_search import GridSearchCV

    parameter grid for grid search of pipeline
    params = {
        'tfidf__tokenizer': [None], # tokenize, tokenize_and_stem
        'tfidf__analyzer': ['char'],
        'tfidf__max_features': [5000, 10000, 25000, 50000],
        # 'tfidf__min_df': ,
        # 'tfidf__max_df': ,
        'tfidf__ngram_range': [(2, 2), (3, 3), (4, 4)], # only when analyzer='char'
        'features__transformer_weights': [{'tfidf': 1.0, 'book_stats': 0.75}, {'tfidf': 0.75, 'book_stats': 1.0}],
        'learner__penalty': ['l1', 'l2'],
        'learner__dual': [True, False],
        'learner__C': [0.1, 1.0, 10.0]
    }

    # more advanced features
    baseline_plus_stats_features = FeatureUnion(
                            [('book_stats', Pipeline([
                                ('book_data', GetBookData()),
                                ('get_book_stats', BookStats()),
                                ('dict', DictVectorizer())
                            ])),
                            ('baseline_tfidf', baseline_features)
                            ]  # add comma

                            # weight components in FeatureUnion
                            # transformer_weights={
                            #     #’pos_freq’: 1.0
                            # }
                            )

    pipeline = Pipeline([('extract', baseline_plus_stats_features),
                         ('clf', LogisticRegression())
                        ])

    fitted_pipeline = pipeline.fit(X_train, Y_train)
    Yhat_LR = fitted_pipeline.predict(X_test)
    print('1.5 Accuracy with logistic regression: {:.4f}'.format(np.mean(Yhat_LR == Y_test)))

    #
    # X_train_bpsf = baseline_plus_stats_features.fit_transform(X_train)
    # X_test_bpsf = baseline_plus_stats_features.transform(X_test)

    joblib.dump(X_train_bpsf, '{}.pkl'.format('X_train_bpsf'))
    joblib.dump(X_test_bpsf, '{}.pkl'.format('X_test_bpsf'))

    joblib.dump(baseline_plus_stats_features, '{}.pkl'.format('baseline_plus_stats_features'))

    lr_clf, Yhat_LR = fit_and_predict_LR(X_train_bpsf, Y_train, X_test_bpsf)
    print('2. Accuracy with logistic regression: {:.4f}'.format(np.mean(Yhat_LR == Y_test)))

    # attempt at char n-grams feature extraction because we read in a paper (Fine-grained Genre Classification using Structural Learning Algorithms) that these were most effective
    #  at determining genre/style for their experiments
    trigram_features = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(3, 3), max_features=50000)
    joblib.dump(trigram_features, '{}.pkl'.format('trigram_features'))

    X_train_tgf = trigram_features.fit_transform(X_train)
    X_test_tgf = trigram_features.transform(X_test)

    joblib.dump(X_train_tgf, '{}.pkl'.format('X_train_tgf'))
    joblib.dump(X_test_tgf, '{}.pkl'.format('X_test_tgf'))

    lr_clf, Yhat_LR = fit_and_predict_LR(X_train_tgf, Y_train, X_test_tgf)
    print('3. Accuracy with logistic regression: {:.4f}'.format(np.mean(Yhat_LR == Y_test)))

    quadgram_features = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(4, 4), max_features=50000)
    joblib.dump(quadgram_features, '{}.pkl'.format('quadgram_features'))

    X_train_qgf = quadgram_features.fit_transform(X_train)
    X_test_qgf = quadgram_features.transform(X_test)

    joblib.dump(X_train_qgf, '{}.pkl'.format('X_train_qgf'))
    joblib.dump(X_test_qgf, '{}.pkl'.format('X_test_qgf'))

    lr_clf, Yhat_LR = fit_and_predict_LR(X_train_qgf, Y_train, X_test_qgf)
    print('4. Accuracy with logistic regression: {:.4f}'.format(np.mean(Yhat_LR == Y_test)))

    # Attempted implementation of Pipeline with FeatureUnion and feature weights
    # This would have been really nice to set up and optimize with a GridSearch, but it required so much computational
    # power and time

    pipeline = Pipeline([
                    ('fetch_book_data', fetch_book_data()),
                    ('features',
                        FeatureUnion(transformer_list=[
                            ('book_stats', FeatureUnion([
                                ('get_book_stats', BookStats()),
                                ('dict', DictVectorizer())
                            ]),
                            ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english')))
                            )
                            #(‘pronoun_counter’, PronounCountTransformer()),
                            #(‘pos_freq_extractor’, POSFreqTransformer())
                            ],

                            # weight components in FeatureUnion
                            transformer_weights={
                                #’pos_freq’: 1.0
                            }
                        )),
                    ('learner', LogisticRegression(n_jobs=-1, random_state=rs))
    ])
    '''
