from __future__ import print_function, division
from scipy import stats
from Debias.we import WordEmbedding
import matplotlib.pyplot as plt
import numpy as np


def Average(lst):
    return sum(lst) / len(lst)


E = WordEmbedding('./Embeddings/GoogleNews-vectors-negative300.bin')

adjective_words =[]
with open('./Data/adj.txt') as fp:
   line = fp.readline()
   while line:
     line = line.strip()
     adjective_words.append(line)
     line = fp.readline()

adjective_in_words = []
for word in adjective_words:
  if word in E.words:
    adjective_in_words.append(word)

v_gender = E.diff('she', 'he')

sp = sorted([(E.v(w).dot(v_gender), w) for w in adjective_in_words])
sp_word = []
sp_value= []
n = 0

while n<len(sp):
    sp_word.append(sp[n][1])
    sp_value.append(sp[n][0])
    n=n+1

human_relatedness = []
word2vec_relatedness = []

extreme = open("./Data/human_extreme.txt")
for line in extreme.readlines():
    word = line.split(";")[0]
    h_r = line.split(";")[1]
    human_relatedness.append(abs(float(h_r)-10))
    word2vec_relatedness.append(sp_value[sp_word.index(word)])


plt.scatter(word2vec_relatedness,color='blue')
m,b = np.polyfit(human_relatedness,word2vec_relatedness, 1)
plt.plot(human_relatedness,m*int(human_relatedness) + b)
plt.show()


print(stats.pearsonr(human_relatedness, word2vec_relatedness))





