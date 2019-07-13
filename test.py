import numpy as np
import nltk
# nltk.download()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
wordLeammer=WordNetLemmatizer()
porterStemmer=PorterStemmer()
stopDict=set(stopwords.words('english'))

target=open('science.sql','r')
# target=open('source.txt','r')
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
    # inTarget.append(x.translate(remap))
    text.append("".join(x.translate(remap).split(',')[5:-2]))

# text=['Authorities in Colorado restored an American flag to its place Friday evening after protesters demonstrating outside a U.S. Immigration and Customs Enforcement (ICE) facility pulled down the star-spangled banner and flew the flag of Mexico in its place. The protesters also removed a “Blue Lives Matter” flag, honoring law enforcement, spray-painted it with the words “Abolish ICE,” then raised the flag upside-down, on a pole next to the Mexican flag, according to local media.'
#     ,'Hundreds of protesters had gathered in Aurora, Colo., outside the federal facility that holds illegal immigrants, to protest ICE raids scheduled to begin Sunday in Denver and other major U.S. cities, FOX 31 Denver reported.'
#     ,'Aurora police Chief Nick Metz said the majority of protesters remained peaceful and some even thanked officers for their evening efforts.'
#     ,'The protest, part of a network of #LightsForLiberty events, also dubbed the “March to Close Concentration Camps,” called for detention centers at the U.S.-Mexico border to be closed and for all immigrants being held in those locations to be granted entry to the U.S., according to the event’s Facebook page.'
#     ,'Beginning Sunday, ICE agents will reportedly work to round up thousands of illegal immigrants across the U.S.'
#     ,'President Trump delayed the operation by two weeks to allow Dems to propose a bipartisan solution to the humanitarian crisis at the border.'
#     ,'Speaking to Fox News during his visit to the border Friday, Vice President Mike Pence said the upcoming ICE raids will not be done at random and will be focused on “removing those deported by courts.”'
#     ,'Besides Denver, the raids were expected to take place in Atlanta, Baltimore, Chicago, Houston, Los Angeles, Miami, New York and San Francisco. Raids scheduled for New Orleans may be delayed due to Tropical Storm Barry, KCNC reported. Other #LightsForLiberty protests took place across the U.S. Friday, including in San Ysidro, Calif.; Portland, Ore.; and New York City.'
#       ]
# for i in range(len(text)):
#     text[i]=[wordLeammer.lemmatize(porterStemmer.stem(x)) for x in text[i].split() if x not in stopDict]
# after=[i for i in text if i not in stopDict]
# after=[wordLeammer.lemmatize(porterStemmer.stem(x)) for x in after]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer=TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
transformer=TfidfTransformer()
tfidf=transformer.fit_transform(vectorizer.fit_transform(text))
# temp=vectorizer.fit_transform(text)
word=vectorizer.get_feature_names()
weight=tfidf.toarray()
# for i in range(len(text)):
#     print("Document",i,"------")
#     tempList=[]
#     for j in range(len(word)):
#         tempList.append((word[j],tfidf[i,j]))
#     tempList.sort(key=lambda x:x[1],reverse=True)
#     print(tempList[:10])



# print(vectorizer.vocabulary_)
# print(vectorizer.idf_)
# vector=vectorizer.transform(text)

#---------Cluster
from sklearn.cluster import KMeans
clf=KMeans(n_clusters=10)
s=clf.fit(weight)
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
# print(final)