
# L'objectif de ce code est de regrouper l'ensemble des cellules présentes dans ipynb pour éviter des appels succesifs à toutes les fonctions.

from math import floor, comb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from scipy.stats import expon
import matplotlib.pyplot as plt






def to_int(L):
    '''Renvoie la même liste mais avec des éléments de type 'int'.
    Renvoie une erreur si la conversion est impossible'''
    return [int(i) for i in L]


sal_etat = pd.read_csv('grilles_salariales/scrapping_salariale.csv', sep=',')


def list_geom(p, n):
    '''Renvoie une liste de taille n+1 avec les n premières valeurs de la fonction de masse (P(X=i)) 
    de la loi géométrique de paramètre p et donne la probabilité restante (1-sum(n primères proba)) en position n+1
    
    Forme du résultat : [P(X=1), P(X=2), ...., P(X=n), 1 - (P(X=1) + P(X=2) + ... + P(X=n))]'''

    q = 1 - p
    L = []
    for i in range(n):
        L.append(p * (q**i))
    # L.append(0)
    L = np.array(L)
    #L[n] = 1-np.sum(L)
    L[0] += 1 - np.sum(L)
    return L


def tirage_al(liste):
    '''list : liste/array dont les éléments somment à 1,
     représente une distribution de probabilité où la probabilité d'obtenir la classe de l'index i est l'élément L[i]
     
     La fonction renvoie un indexe, tiré au hasard en siuvant la distribution décrite par list'''

    s = np.cumsum(
        np.array(liste))  #réalise la somme cumulée des coefficients de list
    nbr = np.random.random()
    l = 0
    while s[l] <= nbr:
        l += 1
    return l


def list_expo(lam, n):
    '''Renvoie une liste de taille n+1 avec les n premières valeurs de la fonction de masse (P(X=i)) 
    d'une version discrétisée de la loi exponentielle de paramètre lambda (normalisée)
    
    Forme du résultat : [P(X=1), P(X=2), ...., P(X=n)]'''

    x = np.array(range(n))
    y1 = lam * np.exp(-lam * x)
    y1 = y1 / y1.sum()

    return list(y1)

def list_binom(p, n, eps=1e-30):
    '''Renvoie une liste de taille n+1 avec les n valeurs de la fonction de masse (P(X=i)) 
    de la loi binomiale de paramètre (n, p)
    
    Forme du résultat : [P(X=1), P(X=2), ...., P(X=n)]'''
    n -= 1
    q = 1 - p
    L = []
    res = 0
    for i in range(n + 1):
        a = comb(n, i) * (p**i) * (q**(n - i))
        if a > eps:
            L.append(comb(n, i) * (p**i) * (q**(n - i)))
        else:
            L.append(a)
            res += 0
    L = np.array(L)
    i = np.argmax(L)
    L[i] += res
    return L

def generer_pop(N , catA , catB , catC): 
    '''generer_pop retourne une liste de N vecteurs, chacun contenant:
    -catégorie (A,B ou C)
    -indice brut majoré
    -maladie
    -age
    -nombre d'enfants à charge
    -zone géographique (1,2 ou 3)
    -nbi'''
    
    L = [0 for i in range(N)] 
    
    #Répartition des fonctionnaires: Les poids correspondent aux proportions de fonctionnaires dans chaque catégorie au sein du Conseil d'Etat
    cat_lst = [1,2,3] # 1:A, 2:B, 3:C
    categorie_etat = random.choices(cat_lst, weights=(100*catA, 100*catB, 100*catC), k=N) 
    
    #Répartition des fonctionnaires: les poids correspondent aux proportions des âges.
    age_lst=[1,2,3]
    age = random.choices(age_lst, weights=(14.3,50.8,34.9), k=N)
    for i in range(N):
        if age[i]==1:
            age[i]=random.randrange(18,30)
        elif age[i]==2:
            age[i]=age[i]=random.randrange(30, 50)
        else:
            age[i]=age[i]=random.randrange(50, 68)

    
    ##IM
    def IM(age):
        a = 300 / 52
        b = 300 - 16 * a
        return a * age + b

    sigma = 80
    idm = IM(np.array(age)) + np.array(
        [random.gauss(0, sigma) for i in range(N)])
    idm = [int(k) for k in idm]
    
    
    #Compute le nombre de jours malade en fonction de la catégorie d'âge de chaque individu
    jour_malad = [0 for i in range(N)]
    jour_malad_lst = [0,4,11.5,22.5,30] 
    jour_malad_1 = random.choices(jour_malad_lst, weights=(70,19.2,4.5,1.2,5.4), k=N)
    jour_malad_2 = random.choices(jour_malad_lst, weights=(62,20.9,6.08,3.04,6.3), k=N)
    jour_malad_3 = random.choices(jour_malad_lst, weights=(64,15.48,7.92,2.16,10.44), k=N)
    for i in range(N):
        if age[i]==1:
            jour_malad[i]=jour_malad_1[i]
        elif age[i]==2:
            jour_malad[i]=jour_malad_2[i]
        else: 
            jour_malad[i]=jour_malad_3[i]
    
          
    #Compute l'âge de chaque individu au sein du Cosneil d'Etat
    sexe_lst = [1, 2] #1:F, 2:H
    sexe = random.choices(sexe_lst, weights=(57, 43), k=N)
            
    #Compute le nombre d'enfants pour chaque individu au sein du conseil d'Etat
    nbre_enf_lst = [0,1,2,3,4] # 0: sans enfants, 1: 1 enfant,..., 4: 4 enfants ou plus
    nbre_enf = []
    for i in range(N):
        if sexe[i]==1:
            r = random.choices(nbre_enf_lst, weights=(46.7,24.0916,20.4139,6.8224,1.9721), k=1)
            nbre_enf.append(r[0])
        else:
            r = random.choices(nbre_enf_lst, weights=(40.7,26.836,22.7119,7.5904,2.1941), k=1)
            nbre_enf.append(r[0])
    
    
    #Compute la zone géographique pour chaque individu faisant parti du conseil d'Etat
    zone_lst=[1,2,3]
    zone = random.choices(zone_lst, weights=(39.2,27.25,33.55), k=N)
    
    
    
    #Compute la nouvelle bonification indiciaire pour chaque agent au sein du Conseil d'Etat
    lam1 = 1/(50)
    lam2 = 1/(15)
    lam3 = 1/(13)

    nbi=[0 for i in range(N)]
    nbi_A_lst = range(15,121)
    nbi_B_lst = range(10,31)
    nbi_C_lst = range(10,21)
    nbi_A = random.choices(nbi_A_lst, weights=[np.exp(-lam1*(i-15))-np.exp(-lam1*(i+1-15)) for i in range(15,121)], k=N)
    nbi_B = random.choices(nbi_B_lst, weights=[np.exp(-lam2*(i-10))-np.exp(-lam2*(i+1-10)) for i in range(10,31)], k=N)
    nbi_C = random.choices(nbi_C_lst, weights=[np.exp(-lam3*(i-10))-np.exp(-lam3*(i+1-10)) for i in range(10,21)], k=N)
    for i in range(N):
        if categorie_etat[i]==1:
            nbi[i]=nbi_A[i]
        elif categorie_etat[i]==2:
            nbi[i]=nbi_B[i]
        else:
            nbi[i]=nbi_C[i]
                  
    
    for i in range(N):
        L[i] = np.zeros(7)
        L[i][0] = categorie_etat[i]
        L[i][1] = idm[i]
        L[i][2] = jour_malad[i]
        L[i][3] = age[i]
        L[i][4] = nbre_enf[i]
        L[i][5] = zone[i]
        L[i][6] = nbi[i]
    return L


#Nbr_fonct = 4071
#point = 4.686025 #Valeur du point

def evolve(df_fonct, jour_moyen_abs):
        
        '''Fonction prenant en argument :
        - un DataFrame avec comme colonnes :
            - anc_echelon : l'ancienneté d'un agent dans son echelon
            - fonction : la dénomination exacte de la fonction d'un agent
            - echelon : son echelon actuel
            - grade : son grade actuel
            - IM : son indice majoré actuel
            
        - le nombre d'années d'évolution
        - un taux de promotion représentant la porportion maximale d'agent pouvant augmenter son grade sur une année.

        La fonction modifie le DataFrame df_fonct avec les nouvelles données.

        Pour controler les lignes ayant eu un changement (promotion ou évolution de salaire), on peut utiliser la liste
        'changement', qui pour chaque indice correspond au nombre de changement dans la situation salairale de l'agent (0, 1 ou 2 changements) 
        '''
        taux_promo_corps = 0.05
        taux_promo_grade = 0.1
        nbr_annees = 1
        Nbr_fonct = df_fonct['fonction'].count() #Nombre de fonctionnaire dans la population

        changement = [0 for i in range(Nbr_fonct)] #Variable de contrôle

        for k in range(nbr_annees):
            for i in range(Nbr_fonct):
                #Variables pour faciliter l'accès à certains paramètres
                agent = df_fonct.iloc[i,:]
                anciennete = agent['anc_echelon']
                fonction = agent['fonction']
                categorie = agent['Cat']
                echelon = agent['echelon']
                grade = agent['grade']
                IM_i = agent['IM']
                durée= agent['durée']

                #Chargement du dataframe correspondant au bon versant de la fonction publique
    
                df = sal_etat.copy()
                df_fonct.iloc[i,0] += 1 #Mise à jour de l'âge
                

                grille = df.loc[df['fonction'] == fonction][['grade', 'Echelon', 'indice majoré calculé', 'durée']] #grille salariale de l'agent pour l'année n-1
                print(grille)

                if durée>=10:
                    durée=0
                
                duree_grille = list(grille.loc[(grille['grade'] == grade) & (grille['Echelon'] == echelon), 'durée'])

                if len(duree_grille) > 1 and duree_grille[1] <= durée:
                    if list(grille.loc[(grille['grade'] == grade) & (grille['Echelon'] == echelon),'durée'])[1] <= durée :
                        print(df_fonct.iloc[i,11]) #Evolution naturelle
                        df_fonct.iloc[i,11] = 0 #remise à 0 de l'ancienneté
                    
                        df_fonct.iloc[i,3] += 1 #Evolution de l'échelon
                    
                        #Revalorisation de l'IM
                    
                        df_fonct.iloc[i,7] = list(grille.loc[(grille['grade'] == grade ) & (grille['Echelon'] == df_fonct.iloc[i,3]),'indice majoré calculé']).pop()
                    
                        changement[i] += 1
                
                    else : 
                        df_fonct.iloc[i,11] += 1 #Augmentation de l'ancienneté si pas d'évolution d'échelon
                    
                if np.random.random() < taux_promo_grade or df_fonct.iloc[i,4] >= 7: #Promotion de grade de l'agent : passage au grade supérieur
                    if grade !=1 : #Promotion de grade
                        df_fonct.iloc[i,2] -= 1 #Mise a jour du grade
                        df_fonct.iloc[i,11] = 0 #remise à 0 de l'ancienneté
                        # Differentiel entre l'IM actuel et les IM du nouveau grade (évolution ne peut se faire que dans le sens d'augmentation de l'IM)
                        grille_diff = [i for i in grille.loc[grille['grade'] == df_fonct.iloc[i,2]]['indice majoré calculé']-df_fonct.iloc[i,6] if i>=0]

                        if len(grille_diff) == 0: #IM du nouveau grade toujours plus faible que l'IM actuel (normalement ce cas n'arrive pas)
                            print('Erreur 1')
                            break
                            IM = grille.loc[grille['grade'] == df_fonct.iloc[i,3], 'indice majoré calculé'].max()
                            echelon = grille.loc[grille['grade'] == df_fonct.iloc[i,3], 'Echelon'].max()
                        else : 
                            IM = np.min(grille_diff) + df_fonct.iloc[i,6] #nouvel IM
                            echelon = list(grille.loc[(grille['indice majoré calculé'] == IM) & (grille['grade'] == df_fonct.iloc[i,2]),'Echelon']).pop() #nouvel échelon
                            
                        df_fonct.iloc[i,6] = IM #Mise a jour de l'IM
                        df_fonct.iloc[i,3] = echelon #Mise a jour de l'échelon
                        
                        changement[i] += 1
        
                if np.random.random() < taux_promo_corps or df_fonct.iloc[i,4] >= 10 : #Promotion de corps
                    promo = False #Par défaut la promotion n'est pas possible
                    if categorie == 'C' : 
                        df_fonct.iloc[i,1] = 'B'
                        promo = True
                    if categorie == 'B' :
                        df_fonct.iloc[i,1] = 'A'
                        promo = True
                    
                    if promo == True :
                        df_fonct.iloc[i,11] = 0 #Remise à 0 de l'ancienneté

                        choix = df.loc[(df['cat'] == df_fonct.iloc[i,1]) & (df['indice majoré calculé'] >= float(df_fonct.iloc[i,6]))] # Promotion uniquement vers un métier plus rémunérateur.
                        
                        liste = choix['fonction'].unique()

                        nouv_met = np.random.choice(liste)
                        df_fonct.iloc[i,5] = nouv_met

                        nouv_grille = df.loc[df['fonction']==nouv_met][['fonction', 'cat','indice majoré calculé', 'Echelon', 'grade']] #grille du nouveau métier
                        
                        IM_min = nouv_grille['indice majoré calculé'].min()

                        if float(df_fonct.iloc[i,6]) <= IM_min: #Si l'IM minimal est plus grand que l'IM actuel
                            nouv_IM = IM_min
                            nouv_grade = list(nouv_grille.loc[nouv_grille['indice majoré calculé'] == IM_min]['grade']).pop()
                            nouv_echelon = list(nouv_grille.loc[nouv_grille['indice majoré calculé'] == IM_min]['Echelon']).pop()       

                        else:
                            nouv_grille_diff = [i for i in nouv_grille['indice majoré calculé']-df_fonct.iloc[i,6] if i>=0]
                            if len(nouv_grille_diff) == 0: # L'indice majoré du nouveau métier n'est jamais meilleur (peu importe l'échelon)
                                # --> normalement ce cas n'arrive jamais
                                print('Erreur 2')
                                nouv_IM = nouv_grille['indice majoré calculé'].max()
                                nouv_echelon = nouv_grille['Echelon'].max()
                                nouv_grade = nouv_grille['grade'].min()
                            else : #On prend l'indice majoré le plus proche par excès
                                nouv_IM = np.min(nouv_grille_diff) + df_fonct.iloc[i,6]
                                nouv_echelon = list(nouv_grille.loc[nouv_grille['indice majoré calculé'] == nouv_IM]['Echelon']).pop()
                                nouv_grade = list(nouv_grille.loc[nouv_grille['indice majoré calculé'] == nouv_IM]['grade']).pop()

                        #Actualisation des informations
                        df_fonct.iloc[i,6] = nouv_IM
                        df_fonct.iloc[i,2] = nouv_grade
                        df_fonct.iloc[i,3] = nouv_echelon


                ## Définition de la prime : 
                age_retraite = 65
                age_depart_car = 23
                z_i = (age_retraite - df_fonct['age'].iloc[i] + age_depart_car)/age_retraite 
                y_i = df_fonct['arret_maladie'].iloc[i] / jour_moyen_abs
                x_i = 5*(z_i + y_i) / 2

                Cat_i = df_fonct.iloc[i,1]

                if Cat_i == 'A':
                    IFSE = 3194 * 12
                    CIA_max = 5750
                if Cat_i == 'B': 
                    IFSE = 1253 * 12 
                    CIA_max = 1804
                if Cat_i == 'C': 
                    IFSE = 978 * 12 
                    CIA_max = 1174
                
                new_prime = round(IFSE + CIA_max * np.exp(-x_i),2)
                
                df_fonct.iloc[i,7] = new_prime

def calcul_sal(df_fonct, p):
    '''Fonction qui prend en argument un dataframe avec une colonne 'IM' et une colonne 'prime'
    qui renvoit une colonne donnant le salaire associé (IM*p + prime)'''
    return df_fonct['IM'] * p * 12 + df_fonct['prime']

import numpy as np

def Fonction_evolution(N , point , catA , catB , catC):
    ##Population de fonctionnaires

    import numpy as np

    pop = generer_pop(N, catA, catB,catC)

    #Transformation fonctionnaire --> Data frame

    fonct = np.array(pop)


    dict_fonct = {}
    dict_fonct['Cat'] = fonct[:,0]
    dict_fonct['IM'] = fonct[:,1]
    dict_fonct['arret_maladie'] = fonct[:,2]
    dict_fonct['age'] = fonct[:,3]
    dict_fonct['enfants'] = fonct[:,4]
    dict_fonct['zone_geo'] = fonct[:,5]
    dict_fonct['NBI'] = fonct[:,6]

    df_fonct = pd.DataFrame(dict_fonct)
    df_bis = df_fonct.copy()
    df_fonct

    import numpy as np

    metier = []
    durée = []
    cat = []
    IM_list = []
    grade_list = []
    echelon_list = []
    anciennete_grade = []

    for i in range(N):
        Cat_i = df_fonct['Cat'].iloc[i]
        # Redéfinition de la catégorie
        if Cat_i == 1:
            Cat_i = 'A'
        elif Cat_i == 2:
            Cat_i = 'B'
        elif Cat_i == 3:
            Cat_i = 'C'

        df = sal_etat
        IM_i = df_fonct['IM'].iloc[i]
        age = df_fonct['age'].iloc[i]

        #Métiers possibles
        choix = df.loc[df['cat'] == Cat_i]
        print(choix)        
        liste = choix['fonction']
        met = np.random.choice(liste)
        metier.append(met)

        




        # Selection du grade, de l'échelon et de l'indice majoré
        IM_min = (choix.loc[choix['fonction'] == met]['indice majoré calculé']).min()
        if IM_i <= IM_min:
            IM = IM_min
            grade = list(choix.loc[choix['fonction'] == met].loc[choix['indice majoré calculé'] == IM_min]['grade']).pop()
            echelon = list(choix.loc[choix['fonction'] == met].loc[choix['indice majoré calculé'] == IM_min]['Echelon']).pop()
        else:
            if df_fonct['age'].iloc[i] <= 27:
                grade = df.loc[df['fonction'] == met, 'grade'].max()
            else:
                nbr_grade = choix.loc[choix['fonction'] == met]['grade'].max()
                p = (nbr_grade - 1 - floor((age - 21) / 16)) / (nbr_grade + 1)
                grades_possible = list_binom(p, nbr_grade)
                grade = tirage_al(grades_possible) + 1

            grille = choix.loc[choix['fonction'] == met].loc[choix['grade'] == grade][['indice majoré calculé', 'Echelon']]
            grille_diff = [i for i in grille['indice majoré calculé'] - IM_i if i >= 0]
            if len(grille_diff) == 0:
                IM = grille['indice majoré calculé'].max()
                echelon = grille['Echelon'].max()
            else:
                IM = np.min(grille_diff) + IM_i
                echelon = list(grille.loc[grille['indice majoré calculé'] == IM]['Echelon']).pop()

        anc = np.random.randint(0, 5)
        anciennete_grade.append(anc)
        grade_list.append(grade)
        grade = to_int(grade_list)
        IM_list.append(IM)
        IM_list = to_int(IM_list)
        echelon_list.append(echelon)
        echelon_list = to_int(echelon_list)
        cat.append(Cat_i)


    for fonction in metier:
        durée.append(df.loc[df['fonction'] == fonction, 'durée'].iloc[0])

    df_fonct['fonction'] = metier
    df_fonct['Cat'] = cat
    df_fonct['IM'] = IM_list
    df_fonct['grade'] = grade_list
    df_fonct['echelon'] = echelon_list
    df_fonct['anc_echelon'] = anciennete_grade
    df_fonct['durée'] = durée

    prime_lst = []
    jour_moyen_abs = df_fonct['arret_maladie'].mean()
    age_retraite = 65
    age_depart_car = 23 #arbitraire : age de démarrage de la carrière en moyenne

    z = (age_retraite - df_fonct['age'] + age_depart_car)/age_retraite 
    y = df_fonct['arret_maladie'] / jour_moyen_abs

    x = np.array(5*(z + y) /2)

    for i in range(N):
        Cat_i = df_fonct['Cat'].iloc[i]
        if Cat_i == 'A': #Catégorie A
            IFSE = 3194 * 12
            CIA_max = 5750
        if Cat_i == 'B': #Catégorie B
            IFSE = 1253 * 12 
            CIA_max = 1804
        if Cat_i == 'C': #Catégorie C
            IFSE = 978 * 12 
            CIA_max = 1174
        prime = round(IFSE + CIA_max *(np.exp(-x[i])),2)
        prime_lst.append(prime)

    df_fonct['prime']=prime_lst

    df_fonct = df_fonct[['age', 'Cat', 'grade', 'echelon', 'anc_echelon', 'fonction', 'IM', 'prime', 'arret_maladie', 'enfants', 'zone_geo','durée']]
    df_init = df_fonct.copy()
    df_fonct

    #Différence d'indice majoré par rapport à la génération intiale :

    IM_base = df_bis['IM']
    IM_affecté = df_fonct['IM']

    A = np.zeros((IM_base.count(), 2), dtype = int)

    A[:,0] = range(IM_base.count())
    A[:,1] = IM_base - IM_affecté
    cat = 'A'
    IM_A = df_fonct.loc[df_fonct['Cat'] == cat]['IM']
    primes_A  = df_fonct.loc[df_fonct['Cat']==cat]['prime']
    nbr_A = IM_A.size
    sal_moy = (IM_A.sum()*point*12 + primes_A.sum())/nbr_A
    print(sal_moy)

    IM_sortie = df_fonct.loc[(df_fonct['Cat'] == cat) & (df_fonct['age'] >= 60)]['IM']
    primes_sortie = df_fonct.loc[(df_fonct['Cat'] == cat) & (df_fonct['age'] >= 60)]['prime']
    nbr_sortie = IM_sortie.size
    sal_moy = (IM_sortie.sum()*point*12 + primes_sortie.sum())/nbr_sortie

    IM_entrée = df_fonct.loc[(df_fonct['Cat'] == cat) & (df_fonct['age'] <= 26)]['IM']
    primes_entrée = df_fonct.loc[(df_fonct['Cat'] == cat) & (df_fonct['age'] <= 26)]['prime']
    nbr_entrée = IM_entrée.size
    sal_moy = (IM_entrée.sum()*point*12 + primes_entrée.sum())/nbr_entrée

    evolve(df_fonct,jour_moyen_abs)
    calcul_sal(df_fonct, point)


    annees = 10
    IM_matrix = np.zeros((N, annees))
    prime_matrix = np.zeros((N, annees))
    sal_matrix = np.zeros((N, annees))
    IM_matrix[:,0] = df_fonct['IM']
    prime_matrix[:,0] = df_fonct['prime']
    sal_matrix[:,0] = calcul_sal(df_fonct, point)
    for k in tqdm(range(1,annees)):
        evolve(df_fonct,jour_moyen_abs)
        IM_matrix[:,k] = df_fonct['IM']
        prime_matrix[:,k] = df_fonct['prime']
        sal_matrix[:,k] = calcul_sal(df_fonct, point)

    print(sum(sal_matrix[:,5]))

print(Fonction_evolution(4071,4.6, 59.9,10.5,25.6))