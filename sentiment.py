import sys
from collections import Counter
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
import collections
import sklearn.naive_bayes
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
import nltk
import random
import numpy as np
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    

    # YOUR CODE HERE
    train_sentence = train_pos + train_neg
    final_words_list=[]
    pos_sentences = train_pos
    neg_sentences = train_neg

    one_per_pos_cnt = len(pos_sentences) * 0.01
    one_per_neg_cnt = len(neg_sentences) * 0.01
    
    words_list = [item for sublist in train_sentence for item in sublist]
	

    words_pos_count = {}
    words_neg_count = {}

    for word in words_list:
        words_pos_count[word] = 0
        words_neg_count[word] = 0

    #Remove stop words
    words_list = [word for word in words_list if word not in stopwords]
    words_list = list(set(words_list))

    for sentence in pos_sentences:
        for word in list(set(sentence)):
            words_pos_count[word]+=1

    for sentence in neg_sentences:
        for word in list(set(sentence)):
            words_neg_count[word]+=1
    
    #print words_list
    #Is in at least 1% of the positive texts or 1% of the negative texts
    #Is in at least twice as many postive texts as negative texts, or vice-versa.
    for word in words_list:
        one_per = words_pos_count[word] >= one_per_pos_cnt or words_neg_count[word] >= one_per_neg_cnt
        two_times_diff = words_pos_count[word] >= 2*words_neg_count[word] or words_neg_count[word] >= 2*words_pos_count[word]
        if (one_per)and(two_times_diff):
            final_words_list.append(word)

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []
    new=[]

    for sentence in train_pos:
        for word in final_words_list:
            if word in sentence:
                new.append(1)
            else:
                new.append(0)
        train_pos_vec.append(new)
        new=[]
    #print train_pos_vec[1:10]

    for sentence in train_neg:
        for word in final_words_list:
            if word in sentence:
                new.append(1)
            else:
                new.append(0)
        train_neg_vec.append(new)
        new=[]

    for sentence in test_pos:
        for word in final_words_list:
            if word in sentence:
                new.append(1)
            else:
                new.append(0)
        test_pos_vec.append(new)
        new=[]

    for sentence in test_neg:
        for word in final_words_list:
            if word in sentence:
                new.append(1)
            else:
                new.append(0)
        test_neg_vec.append(new)
        new=[]


    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    labeled_train_pos = []
    labeled_train_neg = []
    labeled_test_pos = []
    labeled_test_neg = []

    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []

    for i in range(len(train_pos)):
        labeled_train_pos.append(LabeledSentence(train_pos[i], tags=['train_pos_%d' % i]))
    for i in range(len(train_neg)):
        labeled_train_neg.append(LabeledSentence(train_neg[i], tags=['train_neg_%d' % i]))
    for i in range(len(test_pos)):
        labeled_test_pos.append(LabeledSentence(test_pos[i], tags=['test_pos_%d' % i]))
    for i in range(len(test_neg)):
        labeled_test_neg.append(LabeledSentence(test_neg[i], tags=['test_neg_%d' % i]))

    

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    for i in range(len(train_pos)):
        train_pos_vec.append(list(model.docvecs['train_pos_%d' % i]))
    for i in range(len(train_neg)):
        train_neg_vec.append(list(model.docvecs['train_neg_%d' % i]))
    for i in range(len(test_pos)):
        test_pos_vec.append(list(model.docvecs['test_pos_%d' % i]))
    for i in range(len(test_neg)):
        test_neg_vec.append(list(model.docvecs['test_neg_%d' % i]))

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    
    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(X, Y)

    lr_model = LogisticRegression()
    lr_model.fit(X,Y)
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = GaussianNB()
    nb_model.fit(X, Y)

    lr_model = LogisticRegression()
    lr_model.fit(X,Y)

    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    prediction_pos = model.predict(test_pos_vec)
    prediction_neg = model.predict(test_neg_vec)

    for prediction in prediction_pos:
    	if prediction == "pos":
    		tp += 1
    	elif prediction == "neg":
    		fn += 1

    for prediction in prediction_neg:
    	if prediction == "pos":
    		fp += 1
    	elif prediction == "neg":
    		tn += 1

    accuracy = float((tp + tn))/float((tp + fp + tn + fn))
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)

if __name__ == "__main__":
    main()
