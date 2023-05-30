from function.counter import counter
import numpy as np
import pandas as pd
import re
import os
import string
import nltk
import seaborn as sns
from nltk.corpus import stopwords
# nltk.download('stopwords') <-- we run this command to download the stopwords in the project
# nltk.download('punkt') <-- essential for tokenization
from gensim.models import Word2Vec
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

def get_month(dt):
    return dt.month

def get_day(dt):
    return dt.day

def count_rows(rows):
    return len(rows)

def count_row(rows):
    return len(rows)


st.title('Amendements déposés au Sénat en 2019')

st.write('Bonjour,')
st.write('Une analyse des amendements et discussions portées au Sénat permet d\'analyser certaines tendances politiques et sociétales')
st.write('Ce tableau de bord a pour objectif d\'examiner toutes les discussions portées au Sénat en 2019')
st.write('Les données sont issues d\'un dataset libre d\'accès sur le site : https://www.data.gouv.fr/fr/datasets/amendements-deposes-au-senat/')

df1 = pd.read_csv('classified.csv',encoding='utf-8',sep=';',dtype={'Numéro':str,'Subdivision':str,'Alinéa':str,'Auteur':str,'Date de dépôt':str,'Dispositif':str,'Objet':str,'Sort':str,'secteur5':str})
df1['Date de dépôt'] = df1['Date de dépôt'].map(pd.to_datetime)
df1['mois'] = df1['Date de dépôt'].map(get_day)

df = st.dataframe(df1[['Numéro','Subdivision','Alinéa','Auteur','Date de dépôt','Dispositif','Objet','Sort']])


st.markdown('la colonne Objet contient un paragraphe expliquant l\'objet de l\'amendement')
st.markdown('c\'est sur la base de cette chaine de charactère que j\'ai cherché à classifier les amendements par grandes thématiques de politique publique ')
st.markdown('Comme vous pouvez le voir ci-dessous, il est compliqué même pour un humain d\'étiquetter ces textes dans une catégorie bien spécifique')


if 'actual_num' not in st.session_state:
    st.session_state['actual_num'] = 0

next = st.button('Watch next')

if next:
    st.session_state['actual_num'] += 1
  
st.caption(df1['Objet'].iloc[st.session_state['actual_num']])

st.write('Grâce aux modèles de machin learning basés sur les algorithmes fasttext et word2vec, on peut estimer la répartition suivante')

per_cat = counter(df1.secteur5)
df_cat = pd.DataFrame(per_cat, index=['nombre'])
st.table(df_cat)

st.write('La classification de ces amendements permet d\'effectuer de nombreuses analyses. ')

# Affichage du dataframe df_cat


# Affichage des premières lignes du dataframe df1


# Grouper les données par mois et appliquer la fonction count_rows
by_date = df1.groupby('mois').apply(count_rows)
print(by_date)




# Tracer le graphe
plt.title('Rythme des débats au Sénat en 2019')
plt.xlabel('Mois')
plt.ylabel("Nombre d'amendements déposés")
plt.xticks([ x  for x in range(7,38,3)])
plt.plot(by_date)


date_by_cat = pd.DataFrame(
    0,
    index=range(1,31),
    columns=df_cat.columns)


for i in range(len(df1)) :
    if type(df1['secteur5'][i]) != str : continue
    else:
        i_mois , i_secteur = get_day(df1['Date de dépôt'][i]) , df1['secteur5'][i]
        date_by_cat.loc[i_mois, i_secteur] += 1


chart_by_cat = st.line_chart(date_by_cat)




new = st.pyplot(plt)

color = cm.rainbow(np.linspace(0, 1, len(per_cat.keys())))
plt.figure()

plt.xticks(df1.groupby('secteur5').apply(count_rows),rotation = 'vertical')
plt.title("répartition de la classification des amendements")
plt.xlabel('')
plt.ylabel('nombre d\'amendements par thématique')
plt.legend({'lol':'red'},loc='upper right')
plt.bar(x=per_cat.keys(),data=['secteur5'],
        height=df1.groupby('secteur5').apply(count_rows),
        tick_label=per_cat.keys(),
        
        color = color)

new2 = st.pyplot(plt)

listing_sort =df1.Sort.unique()


chart_data = pd.DataFrame(
    0,
    index=listing_sort,
    columns=df_cat.columns)



for i in range(len(df1)) :
    if type(df1['secteur5'][i]) != str : continue
    else:
        i_sort , i_secteur = df1['Sort'][i] , df1['secteur5'][i]
        chart_data.loc[i_sort, i_secteur] += 1


st.bar_chart(chart_data)






