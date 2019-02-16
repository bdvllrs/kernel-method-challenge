# Kernel_method_challenge

Kaggle challenge on DNA sequence classification for Kernel Methods for Machine Learning Course.

## Config
Le fichier `config/default.yaml` est chargé, puis tout les autres fichiers dans le dossier `config`
par ordre alphabétique, viennet écraser (ou ajouter) des nouvelles valeurs.

`defautl.yaml` un fichier exemple et est le seul fichier versionné. Commencer par créer un autre
fichier pour les valeurs personnels.

### kernels
Config des différents noyaux.
`kernel` peut être parmi :

#### onehot
Simple onehot où une lettre représenté un vecteur de dim 4 onehot.

Pas d'args.

#### spectrum
Spectre de la sequence, inspiré de slide 55 de [http://members.cbio.mines-paristech.fr/~jvert/talks/060727mlss/mlss.pdf](http://members.cbio.mines-paristech.fr/~jvert/talks/060727mlss/mlss.pdf)
Args :
- length : taille de la fenêtre glissante


### classifiers
Config des classfieurs (à ajouter au fur et à mesure)
Parmi :

#### svm

Args :
- lbd : lambda de la régularisation.