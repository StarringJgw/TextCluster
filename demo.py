import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
wordLeammer=WordNetLemmatizer()
porterStemmer=PorterStemmer()
stopDict=set(stopwords.words('english'))

# target=open('science.sql','r')
target=open('source.txt','r')
remap={
    ord('\''):' ',
    ord('\n'): None,
    ord('\r'): None,
    ord('\\'):' ',
    # ord(','):' '
}
text=[]
for x in target:
    x=x.replace('\\n','')
    text.append("".join(x.translate(remap).split(',')[5:-2]))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer=TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
transformer=TfidfTransformer()
tfidf=transformer.fit_transform(vectorizer.fit_transform(text))
# temp=vectorizer.fit_transform(text)
word=vectorizer.get_feature_names()
weight=tfidf.toarray()

#---------Cluster
# from sklearn.cluster import KMeans
# clf=KMeans(n_clusters=10)
# s=clf.fit(weight)
# print(clf.cluster_centers_)

def vectorDistance(v1,v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
final=[]
for i in range(len(weight)):
    final.append([])
    for j in range(len(weight)):
        final[i].append((j,vectorDistance(weight[i],weight[j])))
    final[i].sort(key=lambda x:x[1],reverse=True)


for i in range(len(final)):
    print("Document",i)
    for j in range(5):
        print(final[i][j+1])