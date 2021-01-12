from __future__ import print_function, division
from Debias.we import WordEmbedding
from scipy import stats

def make_zeros(number):
    return [0] * number

def Average(lst):
    return sum(lst) / len(lst)

E = WordEmbedding('./Embeddings/GoogleNews-vectors-negative300.bin')

adjective_words =[]
file = open("./Data/phy_adj.txt")
for line in file.readlines():
    word = line.split(",")[0]
    adjective_words.append(word)
adjective_in_words = []
for word in adjective_words:
  if word in E.words:
    adjective_in_words.append(word)

v_gender = E.diff('she', 'he')
sp = sorted([(E.v(w).dot(v_gender), w) for w in adjective_in_words])

value =[]
n = 0

while n<len(sp):
    sp_value = (sp[n][0])
    value.append(abs(sp_value))
    n=n+1

list = make_zeros(len(value))
print(Average(value))
print(stats.ttest_ind(list,value))


