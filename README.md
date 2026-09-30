# Modélisation de la probabilité de sinistre en assurance automobile

Projet de Fin d'Année (Data Science) réalisé à l'ESSAI, Université de Carthage.

Ce projet estime la probabilité qu'une police d'assurance automobile soit sinistrée, à partir des caractéristiques du contrat, de l'assuré, de sa zone géographique et de son véhicule. Il suit une démarche actuarielle : tests statistiques, construction de classes de risque, puis régression logistique interprétable. Les résultats sont comparés à deux modèles ensemblistes, Random Forest et XGBoost.

## Problématique

> Dans quelle mesure les caractéristiques d'un contrat, de l'assuré et de son véhicule permettent-elles de prédire la survenance d'un sinistre, et quels facteurs expliquent les écarts de risque entre assurés ?

## Données

| Caractéristique | Valeur |
| :--- | :--- |
| Source | Fournies par l'enseignant encadrant (PFA) |
| Observations | 58 592 polices |
| Variables | 41 (contrat, assuré, région, véhicule, équipements de sécurité) |
| Cible | `claim_status` : 1 si la police est sinistrée, 0 sinon |
| Taux de sinistre | 6,40 % (classes fortement déséquilibrées) |

Les données ont été fournies par l'enseignant encadrant dans le cadre du PFA. Elles sont incluses dans le dépôt, dans `data/Insurance-claims-data-1copie.csv`. Le jeu de données est anonymisé : il ne contient aucun nom ni aucune adresse, seulement un identifiant de police fictif et des codes régionaux.

## Méthodologie

1. **Contrôle qualité et analyse de la cible** : analyse du déséquilibre des classes.
2. **Tests d'association** : Chi² et V de Cramér pour les variables catégorielles ; Mann-Whitney U et Kruskal-Wallis pour les variables continues, avec correction de Benjamini-Hochberg.
3. **Sélection de variables** : croisement des corrélations, des tests et de la pertinence métier, qui aboutit à 11 variables.
4. **Construction des classes de risque** : discrétisation et regroupement des modalités.
5. **Choix des références et encodage** : la modalité la plus risquée de chaque variable sert de référence, puis encodage *one-hot* sans cette référence (25 indicatrices).
6. **Modélisation** : régression logistique pénalisée (L1/L2), XGBoost et Random Forest, optimisés par GridSearchCV avec une validation croisée stratifiée à 5 plis (critère : AUC-ROC).

## Résultats

| Modèle | AUC-ROC | Brier Score |
| :--- | :---: | :---: |
| Régression logistique (L1, C = 0,1) | 0,612 | 0,243 |
| **XGBoost** | **0,625** | 0,241 |
| Random Forest | 0,617 | **0,240** |

- **Facteur dominant** : l'ancienneté du contrat. Le taux de sinistre passe de 3,6 % à 8,5 % selon les classes, et l'*odds ratio* des contrats les plus récents vaut 0,43.
- **Autres facteurs** : les véhicules de plus de 2 ans, la zone R5 et les assurés de moins de 39 ans présentent un risque plus faible.
- **Comparaison des modèles** : XGBoost est le plus discriminant, mais la régression logistique en est très proche (–0,013 d'AUC) et reste préférable quand le tarif doit être justifié.

**Limites** : les probabilités ne sont pas calibrées, à cause de la pondération des classes. La base ne contient ni exposition ni variables comportementales (kilométrage, historique de sinistres, bonus-malus). Le détail figure dans les sections *Discussion* et *Conclusion* du notebook.

## Structure du dépôt

```
.
├── pfa_sinistre_rapport.ipynb   # Notebook complet (rapport + code)
├── data/
│   └── Insurance-claims-data-1copie.csv   # Données brutes
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation et exécution

```bash
git clone https://github.com/Asma2201/<nom-du-repo>.git
cd <nom-du-repo>
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook pfa_sinistre_rapport.ipynb
```

> Le notebook lit les données via `PATH = "data/Insurance-claims-data-1copie.csv"`.

> Les recherches sur grille de XGBoost et de Random Forest peuvent prendre plus de 15 minutes selon la machine.

## Perspectives

- Calibrer les probabilités (méthode de Platt ou régression isotonique).
- Optimiser le seuil de décision selon une fonction de coût métier.
- Estimer un modèle de fréquence de Poisson avec exposition, puis un modèle de sévérité pour obtenir la prime pure.
- Interpréter les modèles ensemblistes avec les valeurs SHAP.

## Auteur

**Asma Allaigui**, élève ingénieure en Data Science à l'ESSAI
GitHub : [Asma2201](https://github.com/Asma2201)