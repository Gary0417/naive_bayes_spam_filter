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

def calculate_probability(words:list, unique_words:list, smss:list, constants_dict:dict):
    """
    Calculate the probability of finding a word in the spam/ham SMS.
    """

    dictionary =dict()
    K = 1

    for word in unique_words:
        word_frequency = 0
        for sms in smss:
            if word in sms:
                word_frequency += 1
    
        total_words_in_emails = len(words)
        probability = (word_frequency + K) / (total_words_in_emails) + constants_dict['n_unique_words']  # Laplace smoothing
        dictionary[word] = probability
        
    return dictionary

def multiply_probabilities(prob_list:list) :        
    """
    Multiply all the word probabilities together.
    """

    total_prob = 1

    for prob in prob_list: 
         total_prob *= prob  

    return total_prob

def bayes_classifier(sms:list, spamicity:dict, hamicity:dict, constants_dict:dict):
    """
    Use Naive Bayes to classify an SMS as spam.
    """

    probs_WS = list() # probabilities of a word found in spam SMS.
    probs_WH = list() # probabilities of a word found in ham SMS.
    
    for word in sms:
        try:
            pr_WS = spamicity[word]
            
        except KeyError:
            pr_WS = 1 / (constants_dict['n_spam_words'] + constants_dict['n_unique_words'])  # Laplacian Smoothing
        
        probs_WS.append(pr_WS)

        try:
            pr_WH = hamicity[word]
            
        except KeyError:
            pr_WH = 1 / (constants_dict['n_ham_words'] + constants_dict['n_unique_words']) # Laplacian Smoothing
        
        probs_WH.append(pr_WH)

    total_pr_WS = multiply_probabilities(probs_WS)
    total_pr_WH = multiply_probabilities (probs_WH)

    final_classification = total_pr_WS * prob_spam / (total_pr_WS * prob_spam + total_pr_WH * prob_ham)

    if final_classification >= 0.5:
        return 1

    else:
        return 0

def calculate_metrics(test_data:list, spamicity:dict, hamicity:dict, constants_dict:dict):
    """
    Calculate metrics of the model (accuracy, precision, recall and F1).
    """

    n_test_data = len(test_data)
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for row in test_data:
        actual_result = int(row[0]) 
        predicted_result = bayes_classifier(row[1], spamicity, hamicity, constants_dict)

        if actual_result == 0 and predicted_result == 0:
            true_negative += 1

        elif actual_result == 0 and predicted_result == 1:
            false_positive += 1

        elif actual_result == 1 and predicted_result == 0:
            false_negative += 1

        elif actual_result == 1 and predicted_result == 1:
            true_positive += 1
    
    accuracy = (true_positive + true_negative) / n_test_data
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    F1 = 2 * ((precision * recall) / (precision + recall))
    
    print(f"Number of test data: {n_test_data}")
    print(f"True Positive: {true_positive}")
    print(f"True Negative: {true_negative}")
    print(f"False Positive: {false_positive}")
    print(f"False Negative: {false_negative}")
    print('')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {F1}")

if __name__ == "__main__":
    header, df = import_dataset('./data/spam.csv')
    cleaned_df = clean_dataset(df)
    randomised_df = randomise_dataset(cleaned_df)
    train_data, test_data = split_dataset(randomised_df)

    spam_data, ham_data = separate_spam_ham(train_data)
    spam_sms, ham_sms = extract_sms(spam_data), extract_sms(ham_data)
    spam_vocab, ham_vocab= generate_vocabulary(spam_sms), generate_vocabulary(ham_sms)
    unique_spam_vocab = remove_duplicates(spam_vocab)
    unique_ham_vocab = remove_duplicates(ham_vocab)
    all_unique_words = remove_duplicates(unique_ham_vocab + unique_spam_vocab)
    
    n_spam_words = len(spam_vocab)
    n_ham_words = len(ham_vocab)
    n_unique_words = len(all_unique_words)

    n_spam_sms = len(spam_sms)
    n_ham_sms = len(ham_sms)
    prob_spam = (n_spam_sms + 1) / (n_spam_sms + n_ham_sms + 2) # Laplacian Smoothing
    prob_ham = (n_ham_sms + 1) / (n_spam_sms + n_ham_sms + 2) # Laplacian Smoothing

    constants_dict = {'n_spam_words' : n_spam_words,
                      'n_ham_words' : n_ham_words,
                      'n_unique_words' : n_unique_words,
                      'prob_spam' : prob_spam,
                      'prob_ham' : prob_ham }
    
    spamicity = calculate_probability(spam_vocab, unique_spam_vocab, spam_sms, constants_dict)
    hamicity  = calculate_probability(ham_vocab, unique_ham_vocab, ham_sms, constants_dict)
    
    calculate_metrics(test_data, spamicity, hamicity, constants_dict)
    