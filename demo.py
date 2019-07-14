import numpy as np
import nltk
import pickle
# nltk.download()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


wordLeammer=WordNetLemmatizer()
porterStemmer=PorterStemmer()
stopDict=set(stopwords.words('english'))

target=open('/kaggle/input/science.sql','r')
# target=open('source.txt','r')
remap={
    ord('\''):' ',
    ord('\n'): None,
    ord('\r'): None,
    ord('\\'):' ',
    # ord(','):' '
}
textPast=[]
for x in target:
    x=x.replace('\\n','')
    # inTarget.append(x.translate(remap))
    textPast.append("".join(x.translate(remap).split(',')[5:-2]))
text=list(set(textPast))
text.sort(key=textPast.index)
# text=text[:100]
# text=text[:1000]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer=TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
transformer=TfidfTransformer()
tfidf=transformer.fit_transform(vectorizer.fit_transform(text))
word=vectorizer.get_feature_names()
weight=tfidf.toarray()

def vectorDistance(v1,v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
final=[]



for i in range(len(weight)):
    if(i%50==0):
        print("Now",i)
    final.append([])
    for j in range(len(weight)):
        # final[i].append((j,sess.run(tf.reduce_sum(tf.multiply(weight[i],weight[j])))))
        final[i].append((j,vectorDistance(weight[i],weight[j])))
    final[i].sort(key=lambda x:x[1],reverse=True)

import csv
csvFile=open('saveCsv.csv','w')
writer=csv.writer(csvFile)
for row in final:
    writer.writerow(row)

csvFile=open('saveText.csv','w')
writer=csv.writer(csvFile)
for row in text:
    writer.writerow(row)

csvFile=open('saveReport.txt','w')
for i in range(len(final)):
    print("Document",i,file=csvFile)
    for j in range(5):
        print(final[i][j+1],file=csvFile)
