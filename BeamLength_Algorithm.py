from nltk.corpus import brown
from nltk import *
from numpy import *
import operator
print("\nImplementation using General Algorithm using variable beam length")
sents = brown.tagged_sents(tagset='universal')
print("Loaded Brown corpus with {} sentences".format(len(sents)))
tags = []
words = []
tag_words = []
start = '<s>'
end = '</s>'
training_sents = []
testing_sents = []

random_sents_indices = numpy.random.choice(len(sents), int(round(len(sents)*0.8)), replace=False)
training_sents = [sents[i] for i in random_sents_indices]
for i in range(len(sents)):
    if i not in random_sents_indices:
        testing_sents += [sents[i]]

print("\nSplit training and testing sentences")
print("Number of Training Sentences = {}".format(len(training_sents)))
print("Number of Testing Sentences = {}".format(len(testing_sents)))
for sent in training_sents:
    sent = [tuple([start, start])] + sent + [tuple([end, end])]
    tags += [t for (_, t) in sent]
    words += [w.lower() for (w, _) in sent]
    tag_words += [(w.lower(), t) for (w, t) in sent]
print("\nTraining HMM")
uniquetags = list(set(tags))
N = 2
tag_bigrams = list(ngrams(tags, N))
print("Applying Transition Smoothing using Witten-Bell")
transition_smoothed = {}
inditags = []
for tag in uniquetags:
    inditags = [nexttag for (prevtag, nexttag) in tag_bigrams if prevtag == tag]
    transition_smoothed[tag] = WittenBellProbDist(FreqDist(inditags), bins=1e5)


def calc_transition_prob(prevtag, nexttag):
    return float(transition_smoothed[prevtag].prob(nexttag))


print("Applying Emission Smoothing using Witten-Bell")
emission_smoothed = {}
indiwords = []
for tag in uniquetags:
    indiwords = [w.lower() for (w, t) in tag_words if t == tag]
    emission_smoothed[tag] = WittenBellProbDist(FreqDist(indiwords), bins=1e5)


def calc_emission_prob(word, tag):
    return float(emission_smoothed[tag].prob(word))

def firstElement(inp):
    return inp[0]


print("\nTesting HMM with test sentences")
K = int(input('Enter Beam Width(1 to 14):'))
print("Applying General Algorithm with variable beam width of {}".format(K))

count = 0
totaltags = 0

for n in range(len(testing_sents)):
    test_sent = testing_sents[n]
    mat = [[[0.00000,0] for i in range(len(test_sent))] for j in range(len(uniquetags))]
    for p in range(len(test_sent)):
        templist_float = []
        if p > 0:
            values_list = []
            for element in range(len(uniquetags)):
                values_list += [[mat[element][p - 1][0], element]]
            values_list.sort(key = firstElement,reverse=True)
            index_list = [ind for [val, ind] in values_list]
            index_list = index_list[K:]
            for element in index_list:
                mat[element][p-1][0] = 0.0
        for q in range(len(uniquetags)):
            templist_float = []
            if p == 0:
                mat[q][p] = [calc_transition_prob('<s>', uniquetags[q]) * float(calc_emission_prob(test_sent[0][0].lower(), uniquetags[q])), 0]

            elif p > 0:

                for m in range(len(uniquetags)):
                    templist_float += [float(mat[m][p-1][0]*calc_transition_prob(uniquetags[m], uniquetags[q]) * calc_emission_prob(test_sent[p][0], uniquetags[q]))]

                index, value = max(enumerate(templist_float), key=operator.itemgetter(1))
                mat[q][p] = [value, index]

    templist_float = []
    sentlength = len(test_sent)
    values_list = []
    for element in range(len(uniquetags)):
        values_list += [[mat[element][sentlength - 1][0], element]]
    values_list.sort(key=firstElement, reverse=True)
    index_list = [ind for [val, ind] in values_list]
    index_list = index_list[K:]
    for element in index_list:
        mat[element][sentlength - 1][0] = 0.0
    for m in range(len(uniquetags)):
        templist_float += [float(mat[m][sentlength-1][0]*calc_transition_prob(uniquetags[m], '</s>'))]

        index, value = max(enumerate(templist_float), key=operator.itemgetter(1))

    final_tags = [[None] for i in range(len(test_sent))]
    final_tags[sentlength-1] = uniquetags[index]

    for i in range(sentlength - 1):
        final_tags[sentlength - i - 2] = uniquetags[mat[index][sentlength - i - 1][1]]
        index = mat[index][sentlength - i - 1][1]

    test_tags = [t for (w,t) in test_sent]

    for i in range(len(test_sent)):
        if test_tags[i] == final_tags[i]:
            count = count + 1
    totaltags = totaltags + sentlength


print("\nAccuracy = {}".format(float(count)/float(totaltags)*100.0))
