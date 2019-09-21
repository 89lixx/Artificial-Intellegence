import os
nonspam_dir = './ex6DataEmails/nonspam-train'
spam_dir = './ex6DataEmails/spam-train'
test_paths = [spam_dir,nonspam_dir]
#read all the file paths below a directory
def read_file_path(file_dir):
    file_names = []
    for root, dirs, files in os.walk(file_dir):
        for name in files:
            file_names.append(root+'/'+name)
    return file_names

#read contexts in an email
#split them by space
def read_file_context(email_path):
    f = open(email_path, 'r')
    text1 = f.readlines()
    words = []
    text = ' '.join(text1)
    for i in text.split(' '):
        #almost words length less than 2 are not bad words
        if(len(i) <= 2):
            continue
        words.append(i)
    return words

#calculate words probability in all emails
def words_probability(file_dir):
    email_paths = read_file_path(file_dir)
    words = []
    for email_path in email_paths:
        words.append(read_file_context(email_path))
        #words means all context in all emails
        #it's struct is [[],[]...]
    prob = {} #dictionary
    count = 0
    occurences = 0
    for single_words in words:
        for word in single_words:
            #count += 1
            if word not in prob:
                prob[word.lower()] = 1
                continue
            prob[word.lower()] += 1
    for key in prob:
        count += 1
        occurences += prob[key]
        prob[key] = float(prob[key]) / len(file_dir)
        #zhi qian xie cuo le
        #chu de shi suo you de ci
        #ying gai chu yi wen jian de shu liang
        
    return prob,(float(occurences)/count)/len(file_dir)
'''
#this function is used to split words that user input
def split_words(bad_words):
    words = bad_words.split(' ')
    for i in range(len(words)):
        words[i] = words[i].lower()
    return words
'''

print('')
print('Loading context of all emails...')
spam_words_prob,c1 = words_probability(spam_dir)
nonspam_words_prob,c2 = words_probability(nonspam_dir)
c = [c1,c2]
print('done')
print('')
