# Libraries
import re
import sys
import numpy as np
import string
import math
import random
from math import log
from collections import defaultdict
from decimal import Decimal
import matplotlib.pyplot as plt


# Split of data for "exercise 5"
# all_data = []
# training_data = []
# validation_data = []
# testing_data = []
# all_prob = defaultdict()       # probabilities for all possible combinations for the alphabet trigrams


'''
EXERCISE 1
'''
# Preprocess every line of the files
def preprocess_line(line):
    char_removed = re.sub('[^a-zA-Z0-9. ]', "", line)
    number_formated = re.sub('[0-9]', "0", char_removed)
    string_lower = number_formated.lower()
    added_hashes = "##" + string_lower + "#"

    return added_hashes


# Get input and output (optional) files
if len(sys.argv) < 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1]    # input argument: the training file
outfile = sys.argv[2]   # output argument: the model output


# Generate all possible combinations between alphabet characters to form trigrams
def data_combinations():
    all_prob = defaultdict()
    s = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "0", " ", ".", "#"]
    for i in s:
        for j in s:
            for k in s:
                all_prob[i+j+k] = 0
    return all_prob


# Process the training file
def process_file():
    all_data = read_file(infile)

    '''
    EXERCISE 3
    '''
    # use_base_model_ex3(data=all_data)

    '''
    EXERCISE 4
    '''
    # generate_from_LM(data=all_data)

    '''
    EXERCISE 5
    '''
    compute_perplexity_test_file(data=all_data)


'''
EXERCISE 3
'''
def use_base_model_ex3(data):
    tri_count, bi_count = count_n_grams(data=data)
    tri_probabilities = calculate_probabilities(tri_count=tri_count, bi_count=bi_count, alpha=1)
    write_all_probabilities(filename='model-all-prob.en', tri_probabilities=tri_probabilities)
    write_english_example(filename='model-ng-prob.en', tri_probabilities=tri_probabilities)


'''
EXERCISE 4
'''
def generate_from_LM(data):
    tri_count, bi_count = count_n_grams(data=data)
    tri_probabilities = calculate_probabilities(tri_count=tri_count, bi_count=bi_count, alpha=1)
    our_string = generate_string(probabilities=tri_probabilities, len=300)
    their_probabilities = read_model()
    their_string = generate_string(probabilities=their_probabilities, len=300)
    write_generated_string(filename='random_strings.txt', string1=our_string, string2=their_string)


'''
EXERCISE 5
'''
def compute_perplexity_test_file(data):
    en_data = read_file('training.en')
    es_data = read_file('training.es')
    de_data = read_file('training.de')
    test_file_data = read_test_file();
    tri_count, bi_count = count_n_grams(data=data)


    # example_perplexity = calculate_perplexity(data=["##abaab#"], tri_count=tri_count, bi_count=bi_count, alpha=1)
    # print (example_perplexity)

    tri_count, bi_count = count_n_grams(data=en_data)
    en_values, en_alpha_values = compute_k_fold_validation(data=en_data)
    en_avg_perplexity = np.sum(en_values) / len(en_values)
    en_best_alpha = np.sum(en_alpha_values) / len(en_alpha_values)
    # tri_probabilities = calculate_probabilities(tri_count=tri_count, bi_count=bi_count, alpha=1)
    en_perplexity_test_file = calculate_perplexity(data=test_file_data, tri_count=tri_count, bi_count=bi_count, alpha=1)
    # tri_probabilities = calculate_probabilities(tri_count=tri_count, bi_count=bi_count, alpha=en_best_alpha)
    en_perplexity_test_file_alpha = calculate_perplexity(data=test_file_data, tri_count=tri_count, bi_count=bi_count, alpha=en_best_alpha)
    print ("Perplexity EN: ", en_perplexity_test_file)
    print ("Perplexity EN (using best alpha): ", en_perplexity_test_file_alpha)
    print ("Perplexity values over the k-fold validation algorithm EN: ", en_values)
    print ("Best alpha from k-fold validation algorithm EN: ", en_best_alpha)

    tri_count, bi_count = count_n_grams(data=es_data)
    es_values, es_alpha_values = compute_k_fold_validation(data=es_data)
    es_avg_perplexity = np.sum(es_values) / len(es_values)
    es_best_alpha = np.sum(es_alpha_values) / len(es_alpha_values)
    # calculate_probabilities(tri_count=tri_count, bi_count=bi_count, alpha=1)
    es_perplexity_test_file = calculate_perplexity(data=test_file_data, tri_count=tri_count, bi_count=bi_count, alpha=1)
    # calculate_probabilities(tri_count=tri_count, bi_count=bi_count, alpha=es_best_alpha)
    es_perplexity_test_file_alpha = calculate_perplexity(data=test_file_data, tri_count=tri_count, bi_count=bi_count, alpha=es_best_alpha)
    print ("Perplexity ES: ", es_perplexity_test_file)
    print ("Perplexity ES (using best alpha): ", es_perplexity_test_file_alpha)
    print ("Perplexity values over the k-fold validation algorithm ES: ", es_values)
    print ("Best alpha from k-fold validation algorithm ES: ", es_best_alpha)

    tri_count, bi_count = count_n_grams(data=de_data)
    de_values, de_alpha_values = compute_k_fold_validation(data=de_data)
    de_avg_perplexity = np.sum(de_values) / len(de_values)
    de_best_alpha = np.sum(de_alpha_values) / len(de_alpha_values)
    # calculate_probabilities(tri_count=tri_count, bi_count=bi_count, alpha=1)
    de_perplexity_test_file = calculate_perplexity(data=test_file_data, tri_count=tri_count, bi_count=bi_count, alpha=1)
    # calculate_probabilities(tri_count=tri_count, bi_count=bi_count, alpha=de_best_alpha)
    de_perplexity_test_file_alpha = calculate_perplexity(data=test_file_data, tri_count=tri_count, bi_count=bi_count, alpha=de_best_alpha)
    print ("Perplexity DE: ", de_perplexity_test_file)
    print ("Perplexity DE (using best alpha): ", de_perplexity_test_file_alpha)
    print ("Perplexity values over the k-fold validation algorithm DE: ", de_values)
    print ("Best alpha from k-fold validation algorithm DE: ", de_best_alpha)

    # Grid with values from k-fold validation
    x_grid = np.arange(start=0, stop=10, step=1)
    plt.clf()
    plt.plot(x_grid, en_values, 'b-', label='English Model')
    plt.plot(x_grid, es_values, 'r-', label='Spanish Model')
    plt.plot(x_grid, de_values, 'g-', label='German Model')
    plt.legend()
    plt.show()

    # Bar grid for average perplexity after k-fold validation
    x_grid_prime = np.arange(start=0, stop=3, step=1)
    yy = [en_avg_perplexity, es_avg_perplexity, de_avg_perplexity]
    plt.clf()
    plt.bar(1, en_avg_perplexity, color="blue", label='English Model: ' + str('%.3f' % en_avg_perplexity))
    plt.bar(2, es_avg_perplexity, color="red", label='Spanish Model: ' + str('%.3f' % es_avg_perplexity))
    plt.bar(3, de_avg_perplexity, color="green", label='German Model: ' + str('%.3f' % de_avg_perplexity))
    plt.legend()
    plt.show()

    # Bar grid for average perplexity for testing file with alpha=1
    x_grid_prime = np.arange(start=0, stop=3, step=1)
    yy = [en_perplexity_test_file, es_perplexity_test_file, de_perplexity_test_file]
    plt.clf()
    plt.bar(1, en_perplexity_test_file, color="blue", label='English Model: ' + str('%.3f' % en_perplexity_test_file))
    plt.bar(2, es_perplexity_test_file, color="red", label='Spanish Model: ' + str('%.3f' % es_perplexity_test_file))
    plt.bar(3, de_perplexity_test_file, color="green", label='German Model: ' + str('%.3f' % de_perplexity_test_file))
    plt.legend()
    plt.show()

    # Bar grid for average perplexity for testing file with alpha=best_alpha
    x_grid_prime = np.arange(start=0, stop=3, step=1)
    yy = [en_perplexity_test_file_alpha, es_perplexity_test_file_alpha, de_perplexity_test_file_alpha]
    plt.clf()
    plt.bar(1, en_perplexity_test_file_alpha, color="blue", label='English Model: ' + str('%.3f' % en_perplexity_test_file_alpha))
    plt.bar(2, es_perplexity_test_file_alpha, color="red", label='Spanish Model: ' + str('%.3f' % es_perplexity_test_file_alpha))
    plt.bar(3, de_perplexity_test_file_alpha, color="green", label='German Model: ' + str('%.3f' % de_perplexity_test_file_alpha))
    plt.legend()
    plt.show()


# Read file
def read_file(filename):
    data = []
    with open(filename) as f:
        for line in f:
            line = preprocess_line(line)    # process the file
            data.append(line)               # store data

    return data


# Compute k-fold cross-validation
def compute_k_fold_validation(data):
    values = []
    alpha_values = []
    p = math.floor(len(data) / 10)    # 10% (validation set / testing set)
    r = len(data) - p
    for i in range(10):
        v = p * i
        training_data = data[:v] + data[v+p:]       # training_data
        validation_data = data[v:v+p]               # validation_data
        avg_perplexity, alpha_best = build_model(training_data=training_data, validation_data=validation_data)
        alpha_values.append(alpha_best)
        values.append(avg_perplexity)

    return values, alpha_values


# Split data into training, validation and test sets    (UNUSED)
def split_data():
    p = math.floor(len(all_data) / 10)  # 10% (validation set / testing set)
    r = len(all_data) - 2 * p           # 80% (training set)

    training_data.extend(all_data[:r])          # training_data
    validation_data.extend(all_data[r:r+p])     # validation_data
    testing_data.extend(all_data[r+p:r+p+p])    # testing_data


# Build language model
def build_model(training_data, validation_data):
    tri_count, bi_count = count_n_grams(data=training_data)
    avg_perplexity_min = 1000000
    alpha_best = 0.1
    for i in range(10):
        avg_perplexity = calculate_perplexity(data=validation_data, tri_count=tri_count, bi_count=bi_count, alpha=(i*0.1 + 0.1))
        if avg_perplexity < avg_perplexity_min:
            avg_perplexity_min = avg_perplexity
            alpha_best = i*0.1 + 0.1
    return avg_perplexity_min, alpha_best


# Count trigrams and bigrams
def count_n_grams(data):
    tri_count = defaultdict(int)        # count of all trigrams in input
    bi_count = defaultdict(int)         # count of all bigrams in input

    for i in range(len(data)):
        line = data[i]
        k = 0
        for j in range(len(line)-(2)):
            trigram = line[j:j+3]
            tri_count[trigram] += 1    # trigrams count

            bigram = line[j:j+2]
            bi_count[bigram] +=1       # bigrams count
            k+=1

        last_brigram = line[k:k+2]
        bi_count[last_brigram] += 1    # bigrams count

    return tri_count, bi_count


# Calculate perplexity
def calculate_perplexity(data, tri_count, bi_count, alpha):
    tri_probabilities = calculate_probabilities(tri_count=tri_count, bi_count=bi_count, alpha=alpha)
    vv = 30         # alphabet, number of characters (alphabet=26, #, ., 0, space)
    sum_perplexity = 0      # sum of perplexities from each line
    avg_perplexity = 0      # average perplexity
    for i in range(len(data)):
        line = data[i]
        trigrams = []
        entropy_prime = 0
        for j in range(len(line)-(2)):
            trigram = line[j:j+3]
            trigrams.append(trigram)
            entropy_prime -= np.log2(tri_probabilities[trigram])

        entropy = entropy_prime / len(trigrams)       # divide enntropy_prime by the number of trigrams
        perplexity = 2 ** entropy
        sum_perplexity += perplexity
    avg_perplexity = sum_perplexity / len(data)         # sum_perplexity divided by the length of data (row number)

    return avg_perplexity

# Calculate perplexity (OLD VERSION)
'''
def calculate_perplexity(data, tri_count, bi_count, alpha):
    vv = 30         # alphabet, number of characters (alphabet=26, #, ., 0, space)
    sum_perplexity = 0      # sum of perplexities from each line
    avg_perplexity = 0      # average perplexity
    for i in range(len(data)):
        line = data[i]
        trigrams = []
        # extract trigrams
        for j in range(len(line)-(2)):
            trigram = line[j:j+3]
            trigrams.append(trigram)

        entropy = 0
        entropy_prime = 0
        tri_probabilities = defaultdict(int) #the probabilities trigrams on each line
        for j in range(len(trigrams)):
            tri = trigrams[j]
            bi = tri[:2]

            prob = (tri_count[tri] + alpha) / (bi_count[bi] + (vv * alpha))
            entropy_prime -= np.log2(prob)

        entropy = entropy_prime / len(trigrams)       # divide enntropy_prime by the number of trigrams
        perplexity = 2 ** entropy
        sum_perplexity += perplexity
    avg_perplexity = sum_perplexity / len(data)         # sum_perplexity divided by the length of data (row number)

    return avg_perplexity
'''


# Calculate probabilities using smoothing with set alpha
def calculate_probabilities(tri_count, bi_count, alpha):
    vv = 30     # alphabet, number of characters (alphabet=26, #, ., 0, space)
    tri_probabilities = data_combinations()    # the probabilities for all trigrams
    for k,v in tri_count.items():
        bi = k[:2]
        tri_probabilities[k] = (tri_count[k] + alpha) / (bi_count[bi] + (vv * alpha))
        # all_prob[k] = tri_probabilities[k]
    for k,v in tri_probabilities.items():
        bi = k[:2]
        if (tri_probabilities[k] == 0):
            tri_probabilities[k] = alpha / (bi_count[bi] + (vv * alpha))
    return tri_probabilities

# Write english example for "exercise 3"
def write_english_example(filename, tri_probabilities):
    sorted_dict = sorted(tri_probabilities.items())
    with open(filename, 'w') as f:
        for i in range(len(sorted_dict)):
            t = sorted_dict[i][0]
            if (t[:2] == 'ng'):
                f.write(sorted_dict[i][0])
                f.write('\t')
                f.write(str('%.3e' % Decimal(sorted_dict[i][1])))
                f.write('\n')


# Write all probabilities to file
def write_all_probabilities(filename, tri_probabilities):
    sorted_dict = sorted(tri_probabilities.items())
    with open(filename, 'w') as f:
        for i in range(len(sorted_dict)):
            f.write(sorted_dict[i][0])
            f.write('\t')
            f.write(str('%.3e' % Decimal(sorted_dict[i][1])))
            f.write('\n')


# Generate random string based on probabilities of trigrams
def generate_string(probabilities, len):
    string = "##"
    actual_len = 3
    while actual_len < len:
        keys = [k for k,v in probabilities.items() if k[:2] == string[-2:]]
        values = [v for k,v in probabilities.items() if k[:2] == string[-2:]]

        if (keys):
            new_values = np.array(values)
            scaled_values = new_values / np.sum(new_values)
            tri = np.random.choice(keys, p=scaled_values)
            string += tri[-1]
        else:
            string += " "

        actual_len += 1
    string += "#"
    return string


# Write random generated strings to file
def write_generated_string(filename, string1, string2):
    with open(filename, 'w') as f:
        f.write(string1)
        f.write('\n\n')
        f.write(string2)


# Read given model
def read_model():
    tri_probabilities = defaultdict(int)
    with open('model-br.en') as f:
        for line in f:
            key = line[:3]
            value = float(line[4:])
            if (key not in tri_probabilities):
                tri_probabilities[key] = value

    return tri_probabilities


# Read given test file
def read_test_file():
    test_file_data = []
    with open('test') as f:
        for line in f:
            line = preprocess_line(line)    # process the file
            test_file_data.append(line)     # store all data (lines)

    return test_file_data


# Calculate perplexity for "exercise 0"
def calculate_perplexity_ex0():
    prob_arr = np.array([0.2, 0.7, 0.6, 0.25, 0.5, 0.1]);
    entropy = 0;
    for i in prob_arr:
        entropy -= i * np.log2(i)   # 2.598
    perplexity = 2 ** entropy       # 6.058


if __name__ == '__main__':
    # calculate_perplexity_ex0()
    process_file()
