from __future__ import print_function, division
from scipy import stats
from Debias.we import WordEmbedding
import matplotlib.pyplot as plt

E = WordEmbedding('./Embeddings/GoogleNews-vectors-negative300.bin')
E_glove = WordEmbedding('./Embeddings/glove.6B.300d.txt')

adjective_words =[]
with open('./Data/adj.txt') as fp:
   line = fp.readline()
   while line:
     line = line.strip()
     adjective_words.append(line)
     line = fp.readline()

adjective_in_words = []
for word in adjective_words:
  if word in E.words and word in E_glove.words:
    adjective_in_words.append(word)

v_gender = E.diff('she', 'he')
vg_gender = E_glove.diff('she', 'he')

sp = [(E.v(w).dot(v_gender), w) for w in adjective_in_words]
sp_glove = [(E_glove.v(w).dot(vg_gender), w) for w in adjective_in_words]
s = []
s_glove = []
n = 0

while n<len(sp):
    s.append(sp[n][0])
    s_glove.append(sp_glove[n][0])
    n=n+1

print(stats.pearsonr(s,s_glove))
plt.scatter(s,s_glove,color='blue')
plt.xlabel('she-he axis of w2vNEWS embedding')
plt.ylabel('she-he axis of GloVe embedding')
plt.title("Gender Bias in Adjectives across Embeddings")
plt.show()
# (0.7451636109967014, 3.1840375355388543e-232)


