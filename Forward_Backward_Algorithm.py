from nltk.corpus import brown
from nltk import *
from numpy import *
import operator
print("Implementation using Forward-Backward Algorithm")
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

print("\nTesting HMM with test sentences")
print("Applying Forward-Backward Algorithm(Kindly wait for 40 to 50 secs)")
count = 0
totaltags = 0


for n in range(len(testing_sents)):
    test_sent = testing_sents[n]
    sentlength = len(test_sent)
    mat = [[0.00000 for i in range(len(test_sent))] for j in range(len(uniquetags))]
    for p in range(len(test_sent)):
        templist_float = 0.0
        for q in range(len(uniquetags)):
            templist_float = 0.0
            if p == 0:
                mat[q][p] = calc_transition_prob('<s>', uniquetags[q]) * float(calc_emission_prob(test_sent[0][0].lower(), uniquetags[q]))

            elif p > 0:
                for m in range(len(uniquetags)):
                    templist_float += float(mat[m][p-1]*calc_transition_prob(uniquetags[m], uniquetags[q]) * calc_emission_prob(test_sent[p][0].lower(), uniquetags[q]))
                mat[q][p] = templist_float

    templist_float = 0.0
    for m in range(len(uniquetags)):
        templist_float += float(mat[m][sentlength-1]*calc_transition_prob(uniquetags[m], '</s>'))

    probsentfrwd = templist_float
    # print(probsentfrwd)
    bmat = [[0.00000 for i in range(sentlength)] for j in range(len(uniquetags))]
    for p in range(sentlength-1,-1,-1):
        templist_float = 0.0
        for q in range(len(uniquetags)):
            templist_float = 0.0
            if p == sentlength-1:
                bmat[q][p] = calc_transition_prob(uniquetags[q], end)
            elif p < sentlength-1:
                for m in range(len(uniquetags)):
                    templist_float += bmat[m][p+1]*calc_transition_prob(uniquetags[q], uniquetags[m])* calc_emission_prob(test_sent[p+1][0].lower(), uniquetags[m])
                bmat[q][p] = templist_float
    templist_float = 0.0
    for m in range(len(uniquetags)):
        templist_float += bmat[m][0]*calc_transition_prob(start, uniquetags[m])*calc_emission_prob(test_sent[0][0].lower(), uniquetags[m])
    probsentbkwd = templist_float
    # print(probsentbkwd)


    final_tags = [[None] for i in range(len(test_sent))]
    for i in range(len(test_sent)):
        templist_float = []
        for j in range(len(uniquetags)):
            templist_float += [mat[j][i]*bmat[j][i]]

        index, value = max(enumerate(templist_float), key=operator.itemgetter(1))
        final_tags[i] = uniquetags[index]
    test_tags = [t for (w,t) in test_sent]


    for s in range(len(test_sent)):
        if test_tags[s] == final_tags[s]:
            count = count + 1
    totaltags = totaltags + sentlength
print("\nAccuracy = {}".format(float(count)/float(totaltags)*100.0) )