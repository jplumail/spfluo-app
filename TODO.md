## interface Scipion

- Protocole Import adapté à la fluo :
  - importer plusieurs images TIF
  - paramètres fluo
    - taille voxel
    - ...

- Prot import PSF séparé. Classe PSF?

- Création d'un type Volume fluo

- visualisé avec ImageJ

(- Preprocess images : redimensionnement + range (problème contraste))

Mineur
- [x] Protocol picking train : mettre un radius par défaut
- [x] Protocol picking train : Ajouter paramètres avancés

- [x] Pouvoir supprimer des particules après inférence: fait il suffit de selectionner les particules dans la fenêtre Analyse results et cliquer en bas à droite sur Add particles. Cela ajoutera un nouveau Set de particules.

(- Installation trop longue ! Pouvoir se passer de Xmipp et de EMAN serait bien. Récupérer le code pour l'inclure dans notre repo ?)

- Bugs en tout genre :
  - nom de fichier avec espaces crash à l'import

- Format TIF avec bioformat (OME)

## Estimation PSF
- widefield: appeler code Java Epidemic
- confocal: petite modif de widefield
- autre: code Ferréol

## Méthode Picking
- Prévoir les fonctionnalités de l'interface de picking
- Evaluer les différentes solutions possibles
  - ~~Voir si c'est possible de faire du picking sur ImageJ~~
  - TKinter avec code python natif : compliqué
  - **plugin napari / fork napari**

## Méthode Reconstruction
- marginalisation pose
- imposer la contrainte de symétrie (Thibaut)
  - estimation manuelle grossière
  - raffinement: axe de symétrie dans la loss (duplication) et Adam   
  - idée : estimer axe de symétrie à partir reconstruction ab initio
    $\arg\min_p \sum_{s=1}^9 \|R_s(p) x -x\|^2$, p=axe de symétrie
    ou comme un second terme dans la reconstuction ab initio

## Plugin imageJ
- [x] Symétrisation Paul : centrioleJ + multicanaux + stack
- Symétrisation Denis : raffinement
- reconstruction ab initio à partir d'un petit nombre de particules

## Evaluation
- données de benchmark
- stocker données sur seafile
