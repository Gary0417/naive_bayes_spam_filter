import csv
import random
import string


def import_dataset(file: str, delim: str = "," ):
    """
    Import data set.
    """

    data = list()

    with open(file, 'r') as csvfile:
        reader_variable = csv.reader(csvfile, delimiter = delim)
        for row in reader_variable:
            data.append(row)

    # separate header from the data        
    header = data[0]
    del data[0]

    return header, data

def clean_dataset(data:list):
    """
    Clean up the data set.
    """
    # row[1] are the SMS.
    for row in data:
        row[1] = str(row[1]).translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        row[1] = row[1].lower() # set all letters to lower case
        row[1] = row[1].split() # split the sentences to distinct words
       
    return data

def randomise_dataset(data:list, seed:int = 42):
    """
    Randomise the order of the data.
    """

    random.seed(seed)
    random.shuffle(data)

    return data

def split_dataset(data:list , ratio_train:float = 0.7):
    """
    Split the data into training data and test data.
    """

    train_data, test_data = list(), list()
    total_number_of_data = len(data)
    train_data = data[:int(total_number_of_data * ratio_train)]
    test_data = data[int(total_number_of_data * ratio_train):]

    return train_data, test_data

def separate_spam_ham(data:list):
    """
    Separate spam SMS from ham SMS.
    """

    spam_data, ham_data = list() ,list()

    for row in data:
        if int(row[0]) == 1:
            spam_data.append(row)
        else:
            ham_data.append(row)
    
    return spam_data, ham_data

def extract_sms(data:list):
    """
    Extract the SMS from the data (remove the label).
    """

    sms = list()

    for row in data:
        sms.append(row[1]) # row[1] are the SMS
 
    return sms

def generate_vocabulary(smss:list):
    """
    Extract all the words that appear in the SMS.
    """

    vocab = list()

    for sms in smss:
        for word in sms:
            vocab.append(word)
   
    return vocab

def remove_duplicates(vocab:list):
    """
    Remove duplicates from the vocabulary.
    """

    unique_words = list(dict.fromkeys(vocab))

    return unique_words

    