#Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
import numpy as np
import math
from random import random
from math import log
from collections import defaultdict
from decimal import Decimal


# tri_counts=defaultdict(int) #counts of all trigrams in input
# bi_counts=defaultdict(int) #counts of all bigrams in input
#
# tri_probabilities=defaultdict(int) #the probabilities for all trigrams

all_data = []
training_data = []
validation_data = []
testing_data = []



#this function currently does nothing.
def preprocess_line(line):
    # print (line)
    char_removed = re.sub('[^a-zA-Z0-9. ]', "", line)
    number_formated = re.sub('[0-9]', "0", char_removed)
    string_lower = number_formated.lower()
    added_hashes = "##" + string_lower + "#"
    # print (added_hashes)
    # print ('==========================================================================================')

    return added_hashes



#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.
if len(sys.argv) < 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1] #get input argument: the training file
outfile = sys.argv[2]

#This bit of code gives an example of how you might extract trigram counts
#from a file, line by line. If you plan to use or modify this code,
#please ensure you understand what it is actually doing, especially at the
#beginning and end of each line. Depending on how you write the rest of
#your program, you may need to modify this code.
def process_file():
    with open(infile) as f:
        for line in f:
            line = preprocess_line(line)    # process the file
            all_data.append(line)          # store all data (lines)
    split_data()
    use_mle()
    # print (len(all_lines))
    # for i in all_lines:
    #     print ('==========================================================================================')
    #     print (i)


def split_data():
    p = math.floor(len(all_data) / 10)  # 10% (validation set / testing set)
    r = len(all_data) - 2 * p           # 80% (training set)

    training_data.extend(all_data[:r])          # training_data
    validation_data.extend(all_data[r:r+p])     # validation_data
    testing_data.extend(all_data[r+p:r+p+p])    # testing_data

    # print ("Size %d, %d, %d" % (len(training_data), len(validation_data), len(testing_data)))

def use_mle():
    # tri_prob = compute_probabilities()
    tri_count, bi_count = count_n_grams()
    print ("TRI", len(tri_count))
    print ("BI", len(bi_count))
    # for k, v in tri_prob.items():
    #     print (k, '\t', str('%.3e' % Decimal(v)))
    # print (len(tri_prob))

    pp = calculate_perplexity(tri_count=tri_count, bi_count=bi_count)




def count_n_grams():
    tri_count=defaultdict(int) #counts of all trigrams in input
    bi_count=defaultdict(int) #counts of all bigrams in input
    tri_probabilities=defaultdict(int) #the probabilities for all trigrams

    count = 0
    # print (len(training_data))
    for i in range(len(training_data)):
        line = training_data[i]
        # print (training_data[i])
        count+=1
        k = 0
        for j in range(len(line)-(3)):
            trigram = line[j:j+3]
            tri_count[trigram] += 1    # trigrams count

            bigram = line[j:j+2]
            bi_count[bigram] +=1       # bigrams count
            k+=1

        last_brigram = line[k:k+2]
        bi_count[last_brigram] += 1    # bigrams count

    for k,v in tri_count.items():
        bi = k[:2]
        tri_probabilities[k] = v / bi_count[bi]    # trigrams probabilities
    print (len(tri_count))
    print (len(bi_count))
    print (count)

    return tri_count, bi_count

def calculate_perplexity(tri_count, bi_count):
    alpha = 0.4
    vv = 30
    count = 0
    pp = 0
    ee = 0
    for i in range(len(validation_data)):
        line = validation_data[i]
        trigrams = []
        print (trigrams)
        for j in range(len(line)-(3)):
            trigram = line[j:j+3]
            trigrams.append(trigram)

        entropy = 0
        print (len(trigrams))
        for i in range(len(trigrams)):
            tri = trigrams[i]
            bi = tri[:2]
            # print (tri_count[tri], "-", bi_count[bi], "-", len(bi_count))
            # print ((tri_count[tri] + alpha), (bi_count[bi] + (vv * alpha)))

            prob = (tri_count[tri] + alpha) / (bi_count[bi] + (vv * alpha))
            # print(prob)
            if (prob > 1):
                print ("SHIT")
            # print (prob, (-prob * np.log2(prob)))
            if ((-prob * np.log2(prob)) > 1):
                print ("SHIT2")
            entropy -= np.log2(prob)
            # print (entropy)
            count+=1

        entropy = entropy / len(trigrams)
        print ("Entropy: ", entropy)
        ee += entropy
        perplexity = 2 ** entropy

        print ("Perplexity: ", perplexity)
        pp += perplexity

    print (ee/100)
    print (pp/100)

    all_bigrams = 0
    for k,v in bi_count.items():
        all_bigrams += v

    # entropy = 0;
    # count = 0
    # for i in range(len(trigrams)):
    #     tri = trigrams[i]
    #     bi = tri[:2]
    #     # print (tri_count[tri], "-", bi_count[bi], "-", len(bi_count))
    #     # print ((tri_count[tri] + alpha), (bi_count[bi] + (vv * alpha)))
    #
    #     prob = (tri_count[tri] + alpha) / (bi_count[bi] + (vv * alpha))
    #     print(prob)
    #     if (prob > 1):
    #         print ("SHIT")
    #     print (prob, (-prob * np.log2(prob)))
    #     if ((-prob * np.log2(prob)) > 1):
    #         print ("SHIT2")
    #     entropy -= prob * np.log2(prob)
    #     # print (entropy)
    #     count+=1
    # print (count)

    # print ("Entropy: ", entropy)
    #
    # perplexity = 2 ** entropy
    #
    # print ("Perplexity: ", perplexity)


def write_trigrams():
    sorted_dict = sorted(tri_probabilities.items())
    with open(outfile, 'w') as f:
        for i in range(len(sorted_dict)):
            f.write(sorted_dict[i][0])
            f.write('\t')
            f.write(str('%.3e' % Decimal(sorted_dict[i][1])))
            f.write('\n')
    write_english_example(sorted_dict=sorted_dict)

def write_english_example(sorted_dict):
    with open('model_en.en', 'w') as f:
        for i in range(len(sorted_dict)):
            t = sorted_dict[i][0]
            if (t[:2] == 'ng'):
                f.write(sorted_dict[i][0])
                f.write('\t')
                f.write(str('%.3e' % Decimal(sorted_dict[i][1])))
                f.write('\n')

def calculate_perplexity_ex0():
    prob_arr = np.array([0.2, 0.7, 0.6, 0.25, 0.5, 0.1]);
    entropy = 0;
    for i in prob_arr:
        entropy -= i * np.log2(i)   # 2.598
    perplexity = 2 ** entropy       # 6.058




#Some example code that prints out the counts. For small input files
#the counts are easy to look at but for larger files you can redirect
#to an output file (see Lab 1).
# print("Trigram counts in ", infile, ", sorted alphabetically:")
# for trigram in sorted(tri_counts.keys()):
#     print(trigram, ": ", tri_counts[trigram])
# print("Trigram counts in ", infile, ", sorted numerically:")
# for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
#     print(tri_count[0], ": ", str(tri_count[1]))




if __name__ == '__main__':
    process_file()
    # write_trigrams()
    # calculate_perplexity_ex0()
