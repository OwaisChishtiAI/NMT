# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 23:53:08 2018
Task: Arabic-English NMT Preprocessing.
@author: Syed Owais Chishti
"""

import re

lines = open("ara.txt", encoding='utf-8', errors='ignore').read().split('\n')
lines = lines[:-1]




eng = []
id2line = {}
for line in lines:
    _line = line.replace('"', "")
    _line = _line.split('\t')
    eng.append(_line[0])
    id2line[_line[0]] = _line[-1]



questions = []
answers = []
for conversation in id2line.items():
    for i in range(len(conversation) - 1):
        questions.append(conversation[i])
        answers.append(conversation[i+1])
        
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text
    



clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
    
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1



#mapping to integers
questionsword2int = {}
word_number = 0
for word, c in word2count.items():
    questionsword2int[word] = word_number
    word_number += 1

answersword2int = {}
word_number = 0
for word, c in word2count.items():
    answersword2int[word] = word_number
    word_number += 1        
        
        
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionsword2int[token] = len(questionsword2int) + 1
    
for token in tokens:
    answersword2int[token] = len(answersword2int) + 1
    

answersints2word = {w_i:w for w, w_i in answersword2int.items()}

for i in range(len(answers)):
    answers[i] += ' <EOS>'


questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionsword2int:
            ints.append(questionsword2int['<OUT>'])
        else:
            ints.append(questionsword2int[word])
    questions_into_int.append(ints)
    
answers_into_int = []
for answer in answers:
    ints = []
    for word in answer.split():
        if word not in answersword2int:
            ints.append(answersword2int['<OUT>'])
        else:
            ints.append(answersword2int[word])
    answers_into_int.append(ints)


sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 26):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
            
print("Created Sorted lists.")