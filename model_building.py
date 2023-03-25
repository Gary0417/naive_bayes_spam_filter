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

def bayes_classifier(sms:list, spamicity:dict, hamicity:dict, constants_dict:dict, prob_spam, prob_ham):
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

def calculate_metrics(test_data:list, spamicity:dict, hamicity:dict, constants_dict:dict, prob_spam, prob_ham):
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
        predicted_result = bayes_classifier(row[1], spamicity, hamicity, constants_dict, prob_spam, prob_ham)

        if actual_result == 0 and predicted_result == 0:
            true_negative += 1

        elif actual_result == 0 and predicted_result == 1:
            false_positive += 1

        elif actual_result == 1 and predicted_result == 0:
            false_negative += 1

        elif actual_result == 1 and predicted_result == 1:
            true_positive += 1
    
    accuracy = round((true_positive + true_negative) / n_test_data, 2)
    precision = round(true_positive / (true_positive + false_positive), 2)
    recall = round(true_positive / (true_positive + false_negative), 2)
    F1 = round(2 * ((precision * recall) / (precision + recall)), 2)
    
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
