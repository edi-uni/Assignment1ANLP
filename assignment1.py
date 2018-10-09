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
all_data = []
training_data = []
validation_data = []
testing_data = []


# Preprocess every line of the files ("exercise 1")
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

# Process the training file
def process_file():
    with open(infile) as f:
        for line in f:
            line = preprocess_line(line)    # process the file
            all_data.append(line)           # store all data (lines)
    '''
    EXERCISE 3
    '''
    # use_base_model_ex3(data=all_data)       # "exercise 3" - uses all data from training file

    '''
    EXERCISE 4
    '''
    # generate_random_strings(data=all_data)    # "exercise 4" - uses all data from training file

    '''
    EXERCISE 5
    '''
    compute_perplexity_test_file(data=all_data)

    # split_data()                            # split data into training, validation and test sets
    # build_model()                           # build language model



'''
EXERCISE 3
'''
def use_base_model_ex3(data):
    tri_count, bi_count = count_n_grams(data=data)
    probabilities = calculate_probabilities(tri_count=tri_count, bi_count=bi_count)
    write_english_example(tri_probabilities=probabilities)


'''
EXERCISE 4
'''
def generate_random_strings(data):
    tri_count, bi_count = count_n_grams(data=data)
    probabilities = calculate_probabilities(tri_count=tri_count, bi_count=bi_count)
    generate_string(probabilities=probabilities, len=302)
    print ('==========================================================================================')
    their_probabilities = read_model()
    generate_string(probabilities=their_probabilities, len=302)


'''
EXERCISE 5
'''
def compute_perplexity_test_file(data):
    en_data, es_data, de_data = read_all_files()
    test_file_data = read_test_file();
    tri_count, bi_count = count_n_grams(data=all_data)
    # avg_perplexity = calculate_perplexity(data=test_file_data, tri_count=tri_count, bi_count=bi_count)
    # print(avg_perplexity)
    example_perplexity = calculate_perplexity(data=["##abaab#"], tri_count=tri_count, bi_count=bi_count)
    print (example_perplexity)

    en_values = compute_k_fold_validation(data=en_data)
    en_avg_perplexity = np.sum(en_values) / len(en_values)
    print (en_values)
    es_values = compute_k_fold_validation(data=es_data)
    es_avg_perplexity = np.sum(es_values) / len(es_values)
    print (es_values)
    de_values = compute_k_fold_validation(data=de_data)
    de_avg_perplexity = np.sum(de_values) / len(de_values)
    print (de_values)

    x_grid = np.arange(start=0, stop=10, step=1)

    plt.clf()
    plt.plot(x_grid, en_values, 'b-')
    plt.plot(x_grid, es_values, 'r-')
    plt.plot(x_grid, de_values, 'g-')
    plt.show()

    x_grid_prime = np.arange(start=0, stop=3, step=1)
    yy = [en_avg_perplexity, es_avg_perplexity, de_avg_perplexity]
    plt.clf()
    plt.bar(x_grid_prime, yy, color="blue")
    plt.show()


def read_all_files():
    en_data = []
    es_data = []
    de_data = []
    with open('training.en') as f:
        for line in f:
            line = preprocess_line(line)
            en_data.append(line)

    with open('training.es') as f:
        for line in f:
            line = preprocess_line(line)
            es_data.append(line)

    with open('training.de') as f:
        for line in f:
            line = preprocess_line(line)
            de_data.append(line)

    return en_data, es_data, de_data


def compute_k_fold_validation(data):
    values = []
    p = math.floor(len(data) / 10)    # 10% (validation set / testing set)
    r = len(data) - p
    for i in range(10):
        v = p * i
        training_data = data[:v] + data[v+p:]       # training_data
        validation_data = data[v:v+p]               # validation_data
        avg_perplexity = build_model(training_data=training_data, validation_data=validation_data)
        values.append(avg_perplexity)

    return values



# Split data into training, validation and test sets
def split_data():
    p = math.floor(len(all_data) / 10)  # 10% (validation set / testing set)
    r = len(all_data) - 2 * p           # 80% (training set)

    training_data.extend(all_data[:r])          # training_data
    validation_data.extend(all_data[r:r+p])     # validation_data
    testing_data.extend(all_data[r+p:r+p+p])    # testing_data


# Build language model
def build_model(training_data, validation_data):
    tri_count, bi_count = count_n_grams(data=training_data)
    avg_perplexity = calculate_perplexity(data=validation_data, tri_count=tri_count, bi_count=bi_count)
    # print (avg_perplexity)
    return (avg_perplexity)


# Count trigrams and bigrams
def count_n_grams(data):
    tri_count = defaultdict(int)        # count of all trigrams in input
    bi_count = defaultdict(int)         # count of all bigrams in input

    # print ("LEN: ", len(data))
    # print (data)

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


def calculate_perplexity(data, tri_count, bi_count):
    alpha = 0.4     # smoothing parameter
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

        # print ("Perplexity: ", perplexity)
        sum_perplexity += perplexity

    avg_perplexity = sum_perplexity / len(data)         # sum_perplexity divided by the length of data (row number)
    # print ("AVG Perplexity: ", avg_perplexity)

    return avg_perplexity


# Calculate probabilities using smoothing with set alpha
def calculate_probabilities(tri_count, bi_count):
    alpha = 0.4 # the value if from Brants (Large language models and machine translation)
    vv = 30     # alphabet, number of characters (alphabet=26, #, ., 0, space)
    tri_probabilities = defaultdict(int)    # the probabilities for all trigrams
    for k,v in tri_count.items():
        bi = k[:2]
        tri_probabilities[k] = (tri_count[k] + alpha) / (bi_count[bi] + (vv * alpha))

    return tri_probabilities

# Write english example for "exercise 3"
def write_english_example(tri_probabilities):
    sorted_dict = sorted(tri_probabilities.items())
    with open('model_en.en', 'w') as f:
        for i in range(len(sorted_dict)):
            t = sorted_dict[i][0]
            if (t[:2] == 'ng'):
                f.write(sorted_dict[i][0])
                f.write('\t')
                f.write(str('%.3e' % Decimal(sorted_dict[i][1])))
                f.write('\n')


def generate_string(probabilities, len):
    string = "##"
    actual_len = 2
    alphabet = "abcdefghijklmnopqrstuvwxyz0#. "
    # alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x']
    # trigrams = probabilities.keys()
    while actual_len < len:
        keys = [k for k,v in probabilities.items() if k[:2] == string[-2:]]
        values = [v for k,v in probabilities.items() if k[:2] == string[-2:]]

        if (keys):
            new_values = np.array(values)
            scaled_values = new_values / np.sum(new_values)
            tri = np.random.choice(keys, p=scaled_values)
            string += tri[-1]
        else:
            rand_char = random.choice(alphabet)
            string += " "#rand_char

        actual_len += 1

    string = string[2:]

    print(string)

def read_model():
    tri_probabilities = defaultdict(int)
    with open('model-br.en') as f:
        for line in f:
            key = line[:3]
            value = float(line[4:])
            if (key not in tri_probabilities):
                tri_probabilities[key] = value
            # print (line)
            # print (line[:3])
            # print (float(line[4:]))

    # for k,v in trigrams.items():
    #     print(k, " - ", v)

    return tri_probabilities


def read_test_file():
    test_file_data = []
    with open('test') as f:
        for line in f:
            line = preprocess_line(line)    # process the file
            test_file_data.append(line)     # store all data (lines)
            # print (line)
            # print ('==========================================================================================')

    return test_file_data


def calculate_perplexity_ex0():
    prob_arr = np.array([0.2, 0.7, 0.6, 0.25, 0.5, 0.1]);
    entropy = 0;
    for i in prob_arr:
        entropy -= i * np.log2(i)   # 2.598
    perplexity = 2 ** entropy       # 6.058


if __name__ == '__main__':
    process_file()
    # write_trigrams()
    # calculate_perplexity_ex0()
