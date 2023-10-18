# Todo list

## Data

- [ ] écrire une fonction de séparation de dataset train/test > **Salah**
- [ ] créer comme nouvelle feature le retard sur le mois précédent

## Data visualisation

- [x] boucle for pour visualiser tous les écarts types sur les trajets > **Salah**
- [x] visualisation de la différence entre retard arrivée et retard départ > **Salah**
- [ ] sauvegarder les images dans le dossier `figures` > **Salah**

## Analysis

- [x] analyser la corrélation entre les différentes variables > **Agathe**
  - [x] par rapport au retard à l'arrivée
  - [x] par rapport aux différences causes de retard
- [x] afficher la matrice de chaleur > **Agathe**
- [x] sauvegarder ces images > **Agathe**

## Preprocessing

- [x] réécrire la date d'une autre façon pour extraire juste le mois
- [x] repérer si des trajets sont identiques deux fois par mois > **Laure**
  - [x] s'il y a en plusieurs choix : on fait la moyenne ou on les supprime (à tester ce qui est mieux) : il n'y en avait pas
- [x] enlever les trajets qui ont la même gare d'arrivée et de départ > **Laure**
- [x] normaliser les données > **Ibrahim**
- [x] transformer national/international en 0/1 > **Ibrahim**
- [x] encoder le nom des gares (tester plusieurs méthodes et comparer les meilleures) > **Ibrahim**
  - [x] one-hot encoding
  - [x] ordinal encoder method
  - [x] extraire les coordonnées des gares
- [x] pipeline de preprocessing dans le main > **Ibrahim**
- [ ] enlever les valeurs aberrantes > **Salah**

## Model

- [ ] méthodes classiques de régression
  - [ ] polynomiale, moindres carrées > **Agathe**
  - [ ] Lasso, Ridge > **Ibrahim**
  - [ ] processus gaussiens > **Salah**
- [ ] random forest et ses variations > **Laure**

## Metrics

- [x] R2, R2 modifié > **Salah**
- [x] MSE, RMSE > **Agathe**

## Code cleaning 

- [ ] traiter tous les TODO
