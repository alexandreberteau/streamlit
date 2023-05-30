#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Text manipulation libraries
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords') <-- we run this command to download the stopwords in the project
# nltk.download('punkt') <-- essential for tokenization

stop_words = stopwords.words('french')

stop_words.extend(['NaN','<','html','>','body','p','&','#','%','.','a','à'])
print(stop_words)


# In[2]:


def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """Function that cleans the input text by going to:
    - remove links
    - remove special characters
    - remove numbers
    - remove stopwords
    - convert to lowercase
    - remove excessive white spaces
    Arguments:
        text (str): text to clean
        remove_stopwords (bool): whether to remove stopwords
    Returns:
        str: cleaned text
    """
    # remove links
    #text = re.sub("http\S+", "", text)
    # remove numbers and special characters
    #text = re.sub("[^A-Za-z]+", " ", text)
    text = text.replace("\\xe9", "é")
    # remove stopwords
    
    if remove_stopwords:
        # 1. create tokens
        tokens = nltk.word_tokenize(text)
        # 2. check if it's a stopword
        tokens = [w.lower().strip() for w in tokens if not w.lower() in stop_words]
        # return a list of cleaned tokens
        return tokens


# In[ ]:





# In[3]:


import pandas as pd


df = pd.read_csv('check.csv',delimiter = ';' ,skip_blank_lines = True, keep_default_na=False)


# In[4]:


fasttext = pd.read_csv('result_fasttext.csv',delimiter = ',' ,skip_blank_lines = True, keep_default_na=False)


# In[5]:


fasttext.head()


# In[6]:


df = pd.concat([df,fasttext.fasttext],axis=1)


# In[7]:


df.head()


# In[8]:


df["cleaned"] = df.Objet.apply(
    lambda x: preprocess_text(x, remove_stopwords=True)
  )


# In[9]:


df.head()


# In[10]:


print(df["Objet"].iloc[2])


# In[11]:


texts = df.cleaned.tolist()
print(texts[2])


# In[12]:


from gensim.models import Word2Vec
model = Word2Vec(sentences=texts, min_count=1,workers=3, window =3, sg = 1)



# In[13]:


#from gensim.test.utils import datapath
#wv_from_text = KeyedVectors.load_word2vec_format(datapath('C:\Users\Alex\Documents\PSB\BI\projet\bin\cc.fr.300.bin'), binary=True, unicode_errors="ignore")  # C text format


# In[14]:


model.wv["transport"]


# In[15]:


model.wv.most_similar(positive=['finance'], topn=50)



# In[16]:


print(len(model.wv))


# In[17]:


model.wv.similarity('transport', 'santé')


# In[ ]:





# In[18]:


def classification(mots,v_model =""):

    per_cat = {'santé':[],'sécurité':[],'finance':[], 'économie':[], 'environnement':[], 'logement':[], 'éducation':[], 'transport':[], 'droits':[], 'culture':[]}

    for field in per_cat.keys() :
        temp = []
        for word in mots :
            if v_model == "" :
                if model.wv.similarity(word, field) > 0.40 : temp.append(model.wv.similarity(word, field))
            elif v_model=="2":
                if model2.wv.similarity(word, field) > 0.40 : temp.append(model2.wv.similarity(word, field))
        if len(temp) == 0 :
            per_cat[field].append('None')
        else :
            per_cat[field].append(np.mean(temp))
    print(per_cat)
   

    max_field = "Nan"
    max_value = 0
    for field,value in per_cat.items() :
        
        if value[0] == 'None':continue
        elif value[0] > max_value :
            max_field = field
            max_value = value
        else : pass

    if max_field == 'droits' : return('droits_sociaux')
    elif max_field == 'finance' : return('finance_publique')
    return(max_field)

def classification_2(mots):

    per_cat = {'santé':[],'sécurité':[],'finance':[], 'économie':[], 'environnement':[], 'logement':[], 'éducation':[], 'transport':[], 'droits':[], 'culture':[]}

    for field in per_cat.keys() :
        temp = []
        for word in mots :
            if model2.wv.similarity(word, field) > 0.40 : temp.append(model2.wv.similarity(word, field))
        if len(temp) == 0 :
            per_cat[field].append('None')
        else :
            per_cat[field].append(np.mean(temp))
    print(per_cat)
   

    max_field = "Nan"
    max_value = 0
    for field,value in per_cat.items() :
        
        if value[0] == 'None':continue
        elif value[0] > max_value :
            max_field = field
            max_value = value
        else : pass
    
    if max_field == 'droits' : return('droits_sociaux')
    elif max_field == 'finance' : return('finance_publique')


    else : return(max_field)


# In[19]:


print(classification(texts[2]))


# In[20]:


df["secteur"] = df.cleaned.apply(
    lambda x: classification(x)
  )


# In[21]:


df.head()


# In[22]:


df.iloc[178]


# In[23]:


import os
entries = os.listdir('weights/')


# In[24]:


from bs4 import BeautifulSoup

with open('87fd58793cfc31d58352af333ff983ce.txt') as mytext :
    mytext = BeautifulSoup(mytext, "lxml").text
    soup = BeautifulSoup(mytext)
    rectif = BeautifulSoup(soup.decode('utf-8'))
    encoded = rectif.encode('cp1252') 
    goodtext = encoded.decode('utf-8')
    weights = pd.DataFrame([goodtext])    


# In[25]:


#with open('weights/copy.txt') as newtext :
#    mytext = BeautifulSoup(newtext, "lxml").text
#    soup = BeautifulSoup(mytext)
#    rectif = BeautifulSoup(soup.decode('utf-8'))
#    encoded = rectif.encode('cp1252') 
#    goodtext = encoded.decode('utf-8')
#    df3 = pd.Series(goodtext)

#alls = pd.concat([df2 , df3], ignore_index=True , axis=0)


# In[26]:


print(weights)


# In[27]:


weights["cleaned"] = weights[0].apply(
    lambda x: preprocess_text(str(x), remove_stopwords=True)
  )


# In[28]:


parlement = []
for ele in weights['cleaned'] :
    parlement.append(ele)


# In[29]:


model2 = Word2Vec(sentences=parlement, min_count=1,workers=3, window =4, sg = 1)


# In[30]:


model2.wv.most_similar(positive=['transport'], topn=10)


# In[31]:


print(df['cleaned'][1])


# In[32]:


model2.wv.most_similar(positive=['transport'], topn=10)


# In[33]:


model2.wv.similarity('train', 'avion')


# In[34]:


model2.build_vocab(texts, update=True)
#model2.train(texts, total_examples=model2.corpus_count, epochs=model2.epochs)
#model.save('newmodel')


# In[35]:


model2.wv.most_similar(positive=['transport'], topn=10)


# In[36]:


df.head()


# In[37]:


result = lambda x : classification_2(df.cleaned[x])


# In[38]:


df["secteur2"] = df.cleaned.apply(
    lambda x: classification_2(x)
  )


# In[39]:


df.head()


# In[40]:


santé = ['santé','hospitalier','soins','patients','sanitaire']
sécurité = ['protection','routière','juridique','police']
finance_publique = ['imposition','communes','mairies','départements','impôt','taxes','collectivités','territoriales']
économie = ['économie','entreprises',"bureaux",'innovation','pouvoir','agriculture']
environnement = ['environnement','écologie','écologique','durable','transition','énergétique']
logement = ['logement','patrimoine','immobilier','quartiers','urbain','foncière']
éducation = ['éducation','formation']
transport = ['transport','mobilité','avion','autoroutes']
droits_sociaux = ['sociaux','femmes', 'droits','inclusion']
culture = ['culture','associations','sportives','sport','mécène','mécénat','musée']

catégories = santé + sécurité + finance_publique + économie + environnement + logement + éducation + transport + droits_sociaux + culture
alls = {'santé':santé,'sécurité':sécurité,'finance_publique':finance_publique, 'économie':économie, 'environnement':environnement, 'logement':logement, 'éducation':éducation, 'transport':transport, 'droits_sociaux':droits_sociaux, 'culture':culture}


# In[41]:


print(alls)


# In[42]:


model2.build_vocab(catégories, update=True)
model2.train(catégories, total_examples=model2.corpus_count, epochs=model2.epochs)


# In[43]:


#définition de la fonction qui permettra de classifier chaque amendement.
def classification_all(sentences,topics):

    per_topics = dict.fromkeys(topics.keys(), [])
   
 
  
  #transforme une liste de catégories en dictionnaire permettant de stocker la valeur  
    
    for topic,categories in topics.items() :
        per_cat = dict.fromkeys(categories, []) 
        for field in per_cat.keys() :
            temp = []
            for sentence in sentences :
                if model2.wv.similarity(sentence, field) > 0.55 : temp.append(model2.wv.similarity(sentence, field))
            if len(temp) == 0 :
                continue
   
            else :
                per_cat[field] = np.mean(temp)
      
        
        temp=0
        counter=0
        for key,value in per_cat.items() :
            if value == [] :
                continue
            if value > 0 :
                temp += value
                counter += 1
                
        if counter  == 0 :
            per_topics[topic] = 'None'
        else :
            per_topics[topic] = (temp/counter)
    print(per_topics)
            

            
    max_field = "Nan"
    max_value = 0    
    for field,value in per_topics.items() :
        if value == 'None':continue
        elif value > max_value :
            max_field = field
            max_value = value

        else : pass
    return(max_field)
    
    
    #for topic,listing_topic in topics.items():
     #   if max_field in listing_topic : return(topic)
    
    


# In[44]:


#définition de la fonction qui permettra de classifier chaque amendement.
def best_cat(sentences,catégories):

   
 
  
  #transforme une liste de catégories en dictionnaire permettant de stocker la valeur  
    
    
    per_cat = dict.fromkeys(catégories, []) 
    for field in per_cat.keys() :
        temp = []
        for sentence in sentences :
            if model2.wv.similarity(sentence, field) > 0.55 : temp.append(model2.wv.similarity(sentence, field))
        if len(temp) == 0 :
            continue
   
        else :
            per_cat[field] = np.mean(temp)
      
    print(per_cat)
            
         
    max_field = "Nan"
    max_value = 0    
    for field,value in per_cat.items() :
        if value == 'None':continue
        elif value == [] : continue
        elif value > max_value :
            max_field = field
            max_value = value

        else : pass

    
    
    for topic,listing_topic in alls.items():
        if max_field in listing_topic : return(topic)
    
    


# In[45]:


#définition de la fonction qui permettra de classifier chaque amendement.
def per_cat5(sentences,topics):

    per_topics = dict.fromkeys(topics.keys(), [])
    per_cat = dict.fromkeys(catégories, [])
 
  
  #transforme une liste de catégories en dictionnaire permettant de stocker la valeur  
    
    for topic,categories in topics.items() :
        for field in per_cat.keys() :
            temp = []
            for sentence in sentences :
                if model2.wv.similarity(sentence, field) > 0.55 : temp.append(model2.wv.similarity(sentence, field))
            if len(temp) == 0 :
                continue
   
            else :
                per_cat[field] = np.mean(temp)
      
        
        temp=0
        counter=0
        for key,value in per_cat.items() :
            if value == [] :
                continue
            if value > 0 :
                temp += value
                counter += 1
                
        if counter  == 0 :
            per_topics[topic] = 'None'
        else :
            per_topics[topic] = (temp/counter)
    print(per_topics)
            

            
    max_field = "Nan"
    max_value = 0
    equal = set()
    for field,value in per_cat.items() :
        if value == 'None':continue
        elif value == [] : continue
        elif value == max_value :
            equal.add(field)
            equal.add(field)

        elif value > max_value :
            max_field = field
            max_value = value

        else : pass
    
    cut_off = 0
    n1 = ''
    if len(equal) > 1:
        for topic,categories in topics.items() :
            for ele in equal :
                if ele in categories :
                    if per_topics[topic] > cut_off :
                        cut_off = per_topics[topic]
                        n1 = topic
        return(n1)
                    
            
            
    
    else :    
    
        for topic,listing_topic in topics.items():
            if max_field in listing_topic : return(topic)
    


# In[47]:


per_cat5(df.cleaned[20],alls)


# In[48]:


df["secteur3"] = df.cleaned.apply(
    lambda x: classification_all(x,alls)
  )


# In[49]:


df["secteur4"] = df.cleaned.apply(
    lambda x: best_cat(x,catégories)
  )


# In[50]:


df["secteur5"] = df.cleaned.apply(
    lambda x: per_cat5(x,alls)
  )


# In[51]:


df.head()


# In[52]:


décompte =  dict.fromkeys(alls.keys(), 0) 
for secteur in décompte.keys():
    for ele in df.Catégorie[:100] :
        if ele =="" : pass
        elif ele == secteur :
            décompte[ele] += 1
        else :
            pass
print(décompte)


# In[53]:


def get_accuracy(x,y,num):
    
    counter = 0
    n_iter = 0
    for i in range(num) :
        if x[i] == y[i] : counter += 1
        n_iter += 1
    
    return counter/n_iter


# In[55]:


print('secteur 1 : modèle entrainé directement sur le dataset =>',get_accuracy(df.secteur,df.Catégorie,100))
print('secteur 2 : modèle entrainé sur un plus large jeu de données issues du Parlement =>',get_accuracy(df.secteur2,df.Catégorie,100))
print('secteur 3 : modèle entrainé sur un plus large jeu de données avec ajout d\'un champ lexical =>',get_accuracy(df.secteur3,df.Catégorie,100))
print('secteur 4 : modèle classifiant le secteur sur la base unique du meilleur score obtenu sur l\'une des thématiques=>',get_accuracy(df.secteur4,df.Catégorie,100))
print('secteur 5 : modèle précédant amélioré pour gérer les égalités =>',get_accuracy(df.secteur5,df.Catégorie,100))
print('secteur 6 : modèle fasttext préentrainté avec les poids de facebook =>',get_accuracy(df.fasttext,df.Catégorie,100))


# In[58]:


df.iloc[80]


# In[67]:


classification_all(df.cleaned[20],catégories)


# In[72]:


df.to_csv('classified.csv', encoding='utf-8')

