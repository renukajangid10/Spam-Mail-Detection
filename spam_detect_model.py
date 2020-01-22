import os
def makelist(folder):
    list_content=[]
    all_files = os.listdir(folder)
    for file in all_files:
        try :
            f = open(folder + file, 'r')
            list_content.append(f.read())
            f.close()
        except :
            pass
    return list_content
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

stop = stopwords.words('english')

def process(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence)]
from collections import Counter
def Features_Extraction(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(text).items() if not word in stop}
    else:
        return {word: True for word in text if not word in stop}
from nltk import NaiveBayesClassifier, classify
def training_Model (Features, samples):
    Size = int(len(Features) * samples)
    training , testing = Features[:Size], Features[Size:]
    #print ('Training = ' + str(len(training)) + ' emails')
    #print ('Testing = ' + str(len(testing)) + ' emails')
    classifier = NaiveBayesClassifier.train(training)
    return training, testing, classifier
import random
import pickle
if __name__ ==  '__main__':
    slist = makelist('C:/Users/SHREEG/enron2/spam/')
    hlist = makelist('C:/Users/SHREEG/enron2/ham/')
    final_list = [(mail, 'spam') for mail in slist]
    final_list += [(mail, 'ham') for mail in hlist]
    random.shuffle(final_list)
    print("Total Mails : " + str(len(final_list)) + " mail")
    pre = [(process(mail),label) for (mail,label) in final_list]
    features = [(Features_Extraction(mail, 'bow'), label) for (mail, label) in pre]
    #print("Total Features : " + str(len(features)))

    training, testing, classifier = training_Model(features, 0.8)

    f = open("spam_model.pkl","wb")
    pickle.dump(classifier,f)
    f.close()
