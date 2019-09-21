from FileOp import *

def classify(words, words_prob, prime,c):
    prob = 1
    for word in words:
        if word in words_prob:
            prob *= words_prob[word]
        else:
            prob *= c
    return prob*prime

i = 0
for test_path in test_paths:
    print('### calculating dir:',test_path)
    file_paths = read_file_path(test_path)
    result = {}
    result[0] = 0  #spam
    result[1] = 0  #nonspam
    for file_path in file_paths:
        words = read_file_context(file_path)
        spam_prob = classify(words, spam_words_prob,0.5,c[i])
        nonspam_prob = classify(words,nonspam_words_prob,0.5,c[i])
        if spam_prob > nonspam_prob:
             result[0] += 1
        else:
             result[1] += 1
    total_files = result[0] + result[1]
    print(result[0],result[1],total_files)
    spam_part = float(result[0]) / total_files
    nonspam_part = 1 - spam_part
    i+=1
    print('### spam email part:',spam_part)
    print('### nonspam email part',nonspam_part)
    print('----------------------------------')
            
