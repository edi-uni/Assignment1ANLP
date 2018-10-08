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
    generate_random_strings(data=all_data)

    # split_data()                            # split data into training, validation and test sets
    # build_model()                           # build language model


# EXERCISE 3
def use_base_model_ex3(data):
    tri_count, bi_count = count_n_grams(data=data)
    probabilities = calculate_probabilities(tri_count=tri_count, bi_count=bi_count)
    write_english_example(tri_probabilities=probabilities)


# EXERCISE 4
def generate_random_strings(data):
    tri_count, bi_count = count_n_grams(data=data)
    probabilities = calculate_probabilities(tri_count=tri_count, bi_count=bi_count)
    generate_our_string(probabilities=probabilities, len=299)


# Split data into training, validation and test sets
def split_data():
    p = math.floor(len(all_data) / 10)  # 10% (validation set / testing set)
    r = len(all_data) - 2 * p           # 80% (training set)

    training_data.extend(all_data[:r])          # training_data
    validation_data.extend(all_data[r:r+p])     # validation_data
    testing_data.extend(all_data[r+p:r+p+p])    # testing_data


# Build language model
def build_model():
    tri_count, bi_count = count_n_grams(data=training_data)
    # write_english_example(prob=prob_training_set)  ####out from here
    avg_perplexity = calculate_perplexity(data=validation_data, tri_count=tri_count, bi_count=bi_count)
    print (avg_perplexity)


# Count trigrams and bigrams
def count_n_grams(data):
    tri_count = defaultdict(int)        # count of all trigrams in input
    bi_count = defaultdict(int)         # count of all bigrams in input

    # print ("LEN: ", len(data))
    # print (data)

    for i in range(len(data)):
        line = data[i]
        k = 0
        for j in range(len(line)-(3)):
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

        print ("Perplexity: ", perplexity)
        sum_perplexity += perplexity

    avg_perplexity = sum_perplexity / len(data)         # sum_perplexity divided by the length of data (row number)
    print ("AVG Perplexity: ", avg_perplexity)

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


def generate_our_string(probabilities, len):
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

    string += "#"

    print(string)


if __name__ == '__main__':
    process_file()
    # write_trigrams()
    # calculate_perplexity_ex0()
