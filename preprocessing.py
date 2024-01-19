#data_preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Data.csv")
X1=dataset.iloc[:, :-1].values #var independante iloc (fonction de pd) vas recuperer les indices dont on aura besoin (: pour toute et :-1 pour toute sauf la dernière) 
y=dataset.iloc[:, -1].values # -1 pour la dernière


dataset.describe() # mini description statistique d'une BD

#gerer les valeurs manquantes
from sklearn.impute import SimpleImputer

#verifions s'il ya des valeurs manquantes
missing_values = dataset.isna().sum()

impute = SimpleImputer(strategy="mean") # ecire plutot median s'il ya pas trop de valeur manquante
impute = impute.fit(X1[:,[1,2]]) 
X1[:,[1,2]]=impute.transform(X1[:,[1,2]]) 

#gerer les variables categoriques
""""l'orsque les var sont nominale(pas d''existance d'une relation d'ordre) il 
faut effectuer un encodoge par demi variable ou le OneHotEncoder qui 
consiste a creer pour chaque modalite des colones supplementaires"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelEncoder_X=LabelEncoder()
#X[:,0]=labelEncoder_X.fit_transform(X[:,0])

# Extraire la colonne de la variable catégorielle
var_cat= X1[:, 0]
# Redimensionner l'array de la variable catégorielle pour avoir la forme (n_samples, 1)
var_cat_reshaped = var_cat.reshape(-1, 1)
# Créer une instance de l'encodeur OneHotEncoder
encoder = OneHotEncoder()
# Appliquer l'encodage OneHotEncoder à la variable catégorielle
encoded_data = encoder.fit_transform(var_cat_reshaped).toarray()


result = np.concatenate((encoded_data,X1), axis=1)
X= np.delete(result, 3, axis=1)# est ma nouvelle matrice prette a être utilisé

#pour la variable dépendante
labelEncoder_y=LabelEncoder()
y=labelEncoder_y.fit_transform(y)
# division du dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)



#feature scaling
#pour eviter que les grandes valeurs n'écrase pas les plus petites durant l'entrainement du model
"""deux methode de scaling existe: la STANDARDISATION(cen) et la NORMALISATION 
 la STANDARDISATION: loi normal centré reduite (-moyenne divisé par l'ecart-type)
 la NORMALISATION: -min(x)divisé par max(x)-min(x)
"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test )
"""Ici nous n'avons pas applique le feature scaling sur le var independante
 parceque ses valeurs sont déjà comprises entre [-2;2] valeurs produite par le 
 feature scaling des donnees d'entrainnement
 DONC MAINTENANT NOS DONNEES SONT SCALABLE
"""


























































