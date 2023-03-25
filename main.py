from data_preprocessing import *
from model_building import * 

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
    
    calculate_metrics(test_data, spamicity, hamicity, constants_dict, prob_spam, prob_ham)
    