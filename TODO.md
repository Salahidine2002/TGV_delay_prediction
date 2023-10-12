# Todo list

## Data

- [ ] écrire une fonction de séparation de dataset train/test > **Salah**

## Data visualisation

- [ ] boucle for pour visualiser tous les écarts types sur les trajets > **Salah**
- [ ] visualisation de la différence entre retard arrivée et retard départ > **Salah**

## Analysis

- [x] analyser la corrélation entre les différentes variables > **Agathe**
  - [x] par rapport au retard à l'arrivée
  - [x] par rapport aux différences causes de retard
- [ ] afficher la matrice de chaleur > **Agathe**

## Preprocessing

- [x] repérer si des trajets sont identiques deux fois par mois > **Laure**
  - [x] s'il y a en plusieurs choix : on fait la moyenne ou on les supprime (à tester ce qui est mieux) : il n'y en avait pas
- [x] enlever les trajets qui ont la même gare d'arrivée et de départ > **Laure**
- [ ] normaliser les données > **Ibrahim**
- [x] transformer national/international en 0/1 > **Ibrahim**
- [ ] encoder le nom des gares (tester plusieurs méthodes et comparer les meilleures) > **Ibrahim**
  - [x] one-hot encoding
  - [x] ordinal encoder method
  - [ ] extraire les coordonnées des gares

## Model

- [ ] c'est un pb de régression