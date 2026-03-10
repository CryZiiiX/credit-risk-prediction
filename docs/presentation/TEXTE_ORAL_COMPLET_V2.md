# TEXTE ORAL COMPLET - Présentation Projet Credit Risk
## Script détaillé pour la présentation orale (10-15 minutes)

---

## SLIDE 1 - Titre et Présentation

Bonjour, je vais vous présenter mon projet de prédiction du risque de crédit bancaire. Il s'agit d'un projet académique réalisé dans le cadre du cours de Techniques d'Apprentissage Artificiel en M1 Informatique.

L'objectif principal de ce projet est d'implémenter from scratch, c'est-à-dire sans utiliser de bibliothèques de machine learning, deux algorithmes de classification en langage C. Cette approche permet de comprendre en profondeur le fonctionnement interne des algorithmes tout en démontrant l'efficacité d'une implémentation optimisée.

---

## SLIDE 2 - Problématique et Objectifs

Le risque de crédit représente un enjeu majeur pour les institutions financières. Chaque année, des milliards de dollars sont perdus à cause de défauts de paiement. L'objectif est donc de prédire, avant d'octroyer un prêt, si l'emprunteur sera un bon payeur ou s'il présentera un défaut de paiement. C'est un problème de classification binaire.

Mon projet vise à implémenter deux modèles de machine learning en C pur : une régression logistique et un arbre de décision CART. Pourquoi le C ? Pour allier performance computationnelle et compréhension profonde de l'algorithme. L'objectif était d'atteindre plus de 75% d'accuracy, objectif que nous avons largement dépassé avec 93.23%.

Le projet comprend un pipeline complet : chargement des données, prétraitement, entraînement des modèles, évaluation avec des métriques complètes, et comparaison avec scikit-learn pour valider les résultats.

---

## SLIDE 3 - Présentation du Dataset

Le dataset utilisé provient de Kaggle et contient 32 581 emprunteurs avec 12 colonnes de données. La variable cible est `loan_status`, qui indique si l'emprunteur a présenté un défaut de paiement (1) ou non (0).

Un point important à noter est le déséquilibre des classes : 78% des emprunteurs sont de bons payeurs contre seulement 22% de défauts. Ce déséquilibre modéré va impacter notre évaluation : on ne peut pas se fier uniquement à l'accuracy. Il faudra utiliser des métriques adaptées comme le F1-Score et l'AUC-ROC.

Le dataset contient un mélange de 8 variables numériques, comme l'âge, le revenu, le montant du prêt, le taux d'intérêt, et 4 variables catégorielles comme le type de propriété, l'intention du prêt, la note de crédit, et l'historique de défaut.

---

## SLIDE 4 - Analyse Exploratoire des Données

L'analyse exploratoire révèle plusieurs caractéristiques intéressantes. Pour les variables numériques, on observe par exemple que l'âge moyen est de 27.7 ans avec un écart-type de 6.3 ans, ce qui indique une population relativement jeune. Les revenus et montants de prêt présentent des distributions asymétriques, avec quelques valeurs très élevées.

Pour les variables catégorielles, on note que 45% des emprunteurs sont locataires, 35% ont une hypothèque, et seulement 18% sont propriétaires. La note de crédit est bien distribuée, avec environ 25% de chaque grade A, B et C. Un point important : la note de crédit est ordinale, c'est-à-dire qu'il y a un ordre sémantique : A est meilleur que B, qui est meilleur que C, etc. Cet ordre sera préservé lors de l'encodage.

Les visualisations générées montrent les distributions, les boxplots par classe cible, et permettent d'identifier des patterns intéressants dans les données.

---

## SLIDE 5 - Variables Importantes et Corrélations

L'analyse des corrélations révèle les variables les plus prédictives du défaut de paiement. En tête, on trouve le taux d'intérêt avec une corrélation de +0.42. Cela fait sens métier : un taux d'intérêt élevé reflète souvent un risque perçu plus important par la banque.

En deuxième position, la note de crédit avec une corrélation de +0.39. On observe que les notes A présentent environ 10% de défaut, tandis que les notes G peuvent atteindre 55% de défaut. C'est un excellent prédicteur.

L'historique de défaut arrive en troisième position avec +0.35. Si quelqu'un a déjà eu un défaut, c'est un signal fort qu'il pourrait en avoir un autre.

Le ratio prêt sur revenu et le revenu lui-même complètent le top 5. Ces variables mesurent la capacité de remboursement de l'emprunteur.

La matrice de corrélation montre des corrélations modérées entre les features, ce qui est bon : il n'y a pas de multicolinéarité forte qui pourrait poser problème.

---

## SLIDE 6 - Qualité des Données

Le dataset présente 4 011 valeurs manquantes, soit environ 12.3% des données. Ces valeurs manquantes se trouvent principalement dans la variable "années d'emploi". La stratégie choisie est l'imputation par la moyenne de la colonne, calculée sur l'ensemble d'entraînement uniquement. Cette approche est justifiée car les valeurs manquantes semblent aléatoires, sans pattern systématique.

Concernant les outliers, on observe par exemple un âge maximum de 144 ans, qui est probablement une erreur de saisie. Cependant, nous avons choisi de conserver les outliers car ils peuvent représenter des profils à risque réels, comme des revenus très élevés qui pourraient être informatifs pour le modèle.

La cohérence des données a été vérifiée : types de données corrects, encodages catégoriels validés, pas de doublons détectés.

---

## SLIDE 7 - Pipeline de Prétraitement

Le prétraitement comprend quatre étapes principales. D'abord, l'encodage catégoriel : chaque variable catégorielle est convertie en nombres. Par exemple, pour la propriété, RENT devient 0, OWN devient 1, MORTGAGE devient 2, et OTHER devient 3. Pour la note de crédit, on préserve l'ordre : A=1, B=2, jusqu'à G=7.

Ensuite, la gestion des valeurs manquantes par imputation de la moyenne, calculée uniquement sur le train set pour éviter le data leakage.

La troisième étape est la normalisation avec StandardScaler. On calcule la moyenne et l'écart-type sur le train, puis on normalise les deux ensembles avec ces mêmes paramètres. C'est crucial car les variables numériques n'ont pas la même échelle : l'âge est en années, le revenu en milliers de dollars. Sans normalisation, les algorithmes donneraient plus d'importance aux variables avec de grandes valeurs.

Enfin, le split train/test avec un ratio 80/20, soit 26 065 échantillons pour l'entraînement et 6 516 pour le test, avec mélange aléatoire et préservation de la distribution des classes.

---

## SLIDE 8 - Régression Logistique - Principe et Équation

La régression logistique est un modèle de classification binaire qui modélise la probabilité d'appartenance à une classe. Le principe est de combiner linéairement les features pour obtenir un score, puis de transformer ce score en probabilité grâce à la fonction sigmoïde.

L'équation de la droite de régression est z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ, où les wᵢ sont les poids à apprendre. Ce score z est ensuite transformé en probabilité par la fonction sigmoïde : P(y=1|x) = 1 / (1 + e^(-z)).

La fonction sigmoïde a des propriétés intéressantes : elle est bornée entre 0 et 1, ce qui garantit une probabilité valide, et elle est monotone croissante, ce qui préserve l'ordre des scores.

L'interprétation des coefficients est directe : si le coefficient de loan_int_rate est positif et élevé, cela signifie qu'un taux d'intérêt élevé augmente la probabilité de défaut. La valeur absolue du coefficient indique l'importance relative de la feature.

---

## SLIDE 9 - Régression Logistique - Paramètres et Entraînement

L'entraînement utilise la fonction de coût Cross-Entropy, qui mesure l'écart entre les probabilités prédites et les labels réels. Cette fonction est minimisée par descente de gradient.

L'algorithme fonctionne ainsi : à chaque itération, on calcule le gradient de la fonction de coût par rapport à chaque poids, puis on met à jour les poids dans la direction opposée au gradient, avec un pas défini par le learning rate.

Les hyperparamètres choisis sont un learning rate de 0.01, qui représente un bon compromis entre vitesse de convergence et stabilité. Trop petit, la convergence serait lente. Trop grand, on risquerait l'instabilité ou la divergence.

Le nombre d'itérations est fixé à 1000, ce qui s'avère suffisant pour la convergence. On observe que le coût passe d'environ 0.693 au début, ce qui correspond à une prédiction aléatoire, à environ 0.418 après 1000 itérations. La courbe de coût montre une convergence régulière.

---

## SLIDE 10 - Arbre de Décision CART - Principe et Critères

L'arbre de décision construit un modèle de prédiction en créant une structure arborescente qui partitionne récursivement l'espace des features. Chaque nœud pose une question sur une feature, par exemple "le taux d'intérêt est-il supérieur à 12% ?", et divise les données en deux groupes.

Pour décider quel split est le meilleur, on utilise des critères d'impureté. Le critère de Gini mesure l'impureté d'un nœud : G = 1 - Σ(pᵢ)², où pᵢ est la proportion de chaque classe. Cette valeur est entre 0, pour un nœud pur avec une seule classe, et 0.5, le maximum d'impureté pour deux classes.

L'entropie est une alternative : H = -Σ(pᵢ·log₂(pᵢ)). Elle mesure le désordre et est entre 0 et 1. L'entropie est plus sensible aux petites variations que Gini, mais Gini est plus rapide à calculer.

Le gain d'information mesure l'amélioration apportée par un split : c'est la différence entre l'impureté du nœud parent et la moyenne pondérée des impuretés des nœuds enfants. On choisit le split qui maximise ce gain, en testant exhaustivement tous les seuils possibles pour chaque feature.

---

## SLIDE 11 - Arbre de Décision - Paramètres et Construction

La construction de l'arbre utilise des techniques de pre-pruning pour éviter l'overfitting. Les hyperparamètres sont : une profondeur maximale de 7 niveaux, un minimum de 20 échantillons pour diviser un nœud, et un minimum de 10 échantillons dans une feuille.

L'algorithme fonctionne récursivement : pour chaque feature, on teste tous les seuils possibles, on calcule le gain d'information, et on choisit le meilleur split. On divise ensuite récursivement jusqu'à ce que les critères d'arrêt soient atteints.

Le résultat de l'entraînement est un arbre de 7 niveaux de profondeur avec 77 nœuds au total. C'est un arbre relativement simple mais performant, entraîné en 4.652 secondes.

Le pre-pruning est crucial : sans ces contraintes, l'arbre pourrait devenir très profond et mémoriser les données d'entraînement, ce qui donnerait une mauvaise généralisation. Avec ces paramètres, on obtient un bon équilibre entre performance et généralisation.

---

## SLIDE 12 - Métriques de Performance

Les résultats montrent que l'arbre de décision surpasse la régression logistique sur toutes les métriques. L'accuracy passe de 80.11% à 93.23%, soit une amélioration de près de 13 points. La précision atteint 96.48%, ce qui signifie que sur 100 prédictions de défaut, 96 sont correctes. C'est excellent pour une application bancaire où les faux positifs sont coûteux.

Le F1-Score, qui combine précision et rappel, passe de 50.31% à 79.40%, soit une amélioration de 29 points. L'AUC-ROC, qui mesure la capacité de discrimination du modèle, atteint 91.59% pour l'arbre contre 80.28% pour la régression logistique.

Un point important : le gap entre train et test est faible (0.24% pour l'arbre), ce qui indique qu'il n'y a pas d'overfitting significatif. L'objectif initial de dépasser 75% d'accuracy est largement atteint.

---

## SLIDE 13 - Matrices de Confusion et Courbes ROC

Les matrices de confusion permettent d'analyser en détail les erreurs du modèle. Pour la régression logistique, on a 692 faux positifs, c'est-à-dire 692 cas où on prédit un défaut à tort. Pour une banque, cela signifie refuser un prêt à quelqu'un qui aurait pu le rembourser.

Pour l'arbre de décision, on n'a que 31 faux positifs, ce qui est excellent. En revanche, on a 410 faux négatifs, c'est-à-dire des défauts qu'on n'a pas détectés. C'est un compromis : l'arbre privilégie la précision, ce qui est souvent préférable en finance.

Les courbes ROC montrent visuellement la différence de performance. L'arbre a une courbe plus proche du coin supérieur gauche, ce qui indique une meilleure capacité de discrimination. L'aire sous la courbe, l'AUC-ROC, est de 0.916 pour l'arbre contre 0.803 pour la régression logistique.

---

## SLIDE 14 - Comparaison C vs scikit-learn

Pour valider la correction de notre implémentation, nous avons comparé les résultats avec scikit-learn, la bibliothèque de référence en Python. Pour l'arbre de décision, les différences sont inférieures à 1% sur toutes les métriques : accuracy (0.0154%), precision (0.32%), recall (0.16%), F1-Score (0.0014%), et AUC-ROC (0.23%). Cette validation confirme que notre implémentation from scratch produit des résultats corrects.

Pour la régression logistique, les écarts sont plus variables : accuracy (5.74%), precision (24.42%), recall (9.60%), F1-Score (3.41%), et AUC-ROC (4.60%). Ces différences s'expliquent par plusieurs facteurs : scikit-learn utilise L-BFGS comme optimiseur, tandis que nous utilisons la descente de gradient. Il peut aussi y avoir des différences de précision numérique, d'initialisation du split train/test, et l'absence de régularisation L2 dans notre implémentation.

En termes de performance computationnelle, notre implémentation C est très rapide : 0.433 secondes pour entraîner la régression logistique et 4.652 secondes pour l'arbre de décision, ce qui démontre l'efficacité d'une implémentation optimisée.

---

## SLIDE 15 - Formalisme Théorique - Équations Principales

Les fondements mathématiques reposent sur plusieurs équations clés. Pour la régression logistique, la fonction sigmoïde σ(z) = 1 / (1 + e^(-z)) transforme un score linéaire en probabilité. La fonction de coût Cross-Entropy L = -[y·log(σ(z)) + (1-y)·log(1-σ(z))] mesure l'écart entre prédictions et réalité. Le gradient ∂L/∂wᵢ = (σ(z) - y)·xᵢ permet de mettre à jour les poids.

Pour l'arbre de décision, l'impureté de Gini G = 1 - Σ(pᵢ)² mesure le mélange des classes dans un nœud. L'entropie H = -Σ(pᵢ·log₂(pᵢ)) est une alternative qui mesure le désordre. Le gain d'information Gain = I(parent) - [p_left·I(left) + p_right·I(right)] quantifie l'amélioration apportée par un split.

Pour la normalisation, on calcule la moyenne μ et l'écart-type σ, puis on normalise avec z = (x - μ) / σ pour centrer et réduire les données.

Ces équations sont implémentées directement dans le code C, ce qui permet de comprendre exactement ce que fait chaque algorithme.

---

## SLIDE 16 - Algorithmes et Paramètres

Le choix des hyperparamètres a un impact significatif sur la performance. Pour la régression logistique, le learning rate de 0.01 représente un bon compromis : plus petit, la convergence serait trop lente ; plus grand, on risquerait l'instabilité. Le nombre d'itérations de 1000 est suffisant pour la convergence, comme le montre la courbe de coût.

Pour l'arbre de décision, la profondeur maximale de 7 contrôle la complexité : plus profond, on risque l'overfitting ; moins profond, on perd en performance. Le min_samples_split de 20 évite de diviser sur de trop petits échantillons, ce qui garantit la robustesse. Le min_samples_leaf de 10 assure que chaque feuille contient suffisamment de données pour faire une prédiction fiable.

Le choix entre Gini et Entropie a un impact mineur sur la performance, mais Gini est plus rapide à calculer, d'où notre choix.

Chaque modèle a ses avantages et limites : la régression logistique est interprétable et rapide, mais suppose une relation linéaire. L'arbre capture les non-linéarités, mais devient moins interprétable pour de grands arbres.

---

## SLIDE 17 - Conclusion, Limitations et Perspectives

Pour conclure, ce projet a permis d'atteindre et de dépasser les objectifs initiaux. L'arbre de décision obtient 93.23% d'accuracy et 91.59% d'AUC-ROC, largement supérieurs à l'objectif de 75%. L'implémentation a été validée par comparaison avec scikit-learn, avec des différences inférieures à 1% pour l'arbre de décision.

Cependant, le projet présente certaines limitations. Le déséquilibre des classes 78/22 peut impacter certaines métriques. Nous n'avons pas utilisé de validation croisée K-fold, ce qui pourrait donner une meilleure estimation de la variance. Les hyperparamètres ont été fixés manuellement, sans optimisation automatique. Enfin, nous avons utilisé des modèles simples, sans techniques d'ensemble comme Random Forest ou Gradient Boosting.

Les perspectives d'amélioration sont nombreuses : grid search pour optimiser les hyperparamètres, K-Fold Cross-Validation pour une validation plus robuste, Random Forest pour améliorer la généralisation, Gradient Boosting pour des modèles plus puissants, régularisation L2 pour la régression logistique, et feature engineering pour créer de nouvelles variables dérivées.

Sur le plan pédagogique, ce projet a permis une compréhension profonde des algorithmes de machine learning, la maîtrise de l'implémentation from scratch, la construction d'un pipeline complet de bout en bout, et la validation scientifique des résultats.

Merci pour votre attention. Je suis disponible pour répondre à vos questions.

---

**FIN DU TEXTE ORAL**

**Durée totale :** 10-15 minutes  
**Style :** Naturel, pédagogique, orienté compréhension  
**Conseils :** Adapter le rythme selon l'audience, faire des pauses, montrer les graphiques quand mentionnés

