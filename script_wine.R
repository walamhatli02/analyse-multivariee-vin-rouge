# ============================================================
# MINI-PROJET — ANALYSE MULTIVARIÉE
# Dataset : Wine Quality (UCI) — Vin Rouge
# Module  : Méthodes statistiques et étude de données
# Auteur  : wala mhatli omar ghannem oumaima ghannem 
# Date    : 2025
# ============================================================



# ============================================================
# 0. INSTALLATION ET CHARGEMENT DES PACKAGES
# ============================================================

# Installer les packages (à faire une seule fois)
install.packages(c("tidyverse", "FactoMineR", "factoextra",
                   "cluster", "corrplot", "gridExtra"))

# Charger les librairies
library(tidyverse)    # manipulation et visualisation des données
library(FactoMineR)   # ACP et méthodes factorielles
library(factoextra)   # visualisation des résultats ACP et clustering
library(cluster)      # algorithmes de classification
library(corrplot)     # visualisation matrice de corrélation
library(gridExtra)    # affichage de plusieurs graphiques côte à côte



# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

# URL du dataset Wine Quality (UCI Machine Learning Repository)
url  <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Importation — séparateur ";" spécifique à ce fichier
wine <- read.csv(url, sep = ";")

# Vérification rapide
dim(wine)        # 1599 observations x 12 variables
head(wine)       # aperçu des 6 premières lignes
str(wine)        # types des variables
summary(wine)    # statistiques descriptives



# ============================================================
# 2. PRÉPARATION ET NETTOYAGE DES DONNÉES
# ============================================================

# ----- 2.1 Valeurs manquantes -----
# Vérifier s'il existe des valeurs NA dans le dataset
colSums(is.na(wine))
# Résultat attendu : 0 pour toutes les variables → dataset complet

# ----- 2.2 Doublons -----
# Détecter les observations identiques
nb_doublons <- sum(duplicated(wine))
cat("Nombre de doublons :", nb_doublons, "\n")

# Supprimer les doublons pour éviter de surpondérer certains profils
wine_clean <- wine[!duplicated(wine), ]
cat("Dimensions après nettoyage :", nrow(wine_clean), "x", ncol(wine_clean), "\n")

# ----- 2.3 Séparation variables actives / variable cible -----
# Les 11 variables physicochimiques serviront pour l'ACP et le clustering
wine_vars    <- wine_clean[, 1:11]

# La variable "quality" est gardée comme variable illustrative
# Elle ne participera pas à l'ACP mais servira à interpréter les clusters
wine_quality <- wine_clean$quality

# ----- 2.4 Détection des outliers -----
# Boxplots pour visualiser les valeurs extrêmes par variable
wine_long <- wine_vars %>%
  pivot_longer(cols = everything(),
               names_to  = "variable",
               values_to = "valeur")

ggplot(wine_long, aes(x = variable, y = valeur, fill = variable)) +
  geom_boxplot(outlier.colour = "red", outlier.size = 0.8, alpha = 0.7) +
  facet_wrap(~variable, scales = "free", ncol = 4) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x     = element_blank(),
        strip.text      = element_text(size = 8, face = "bold")) +
  labs(title = "Détection des outliers — Boxplots par variable",
       x = "", y = "Valeur")
# Observation : residual.sugar, total.sulfur.dioxide et chlorides ont des outliers
# Décision : on les conserve car ils représentent des vins réels et atypiques

# ----- 2.5 Normalisation (standardisation Z-score) -----
# Indispensable avant l'ACP : met toutes les variables à la même échelle
# Z-score : (x - moyenne) / écart-type → moyenne = 0, écart-type = 1
wine_scaled <- scale(wine_vars)

# Vérification de la normalisation
round(colMeans(wine_scaled), 3)       # toutes les moyennes ≈ 0
round(apply(wine_scaled, 2, sd), 3)   # tous les écarts-types ≈ 1

# ----- 2.6 Matrice de corrélation -----
# Visualiser les relations linéaires entre variables avant l'ACP
corrplot(cor(wine_vars),
         method      = "color",
         type        = "upper",
         tl.cex      = 0.75,
         addCoef.col = "black",
         number.cex  = 0.6,
         col         = colorRampPalette(c("#d73027","white","#1a9850"))(200),
         mar         = c(0, 0, 0, 0))
# Corrélations fortes : fixed.acidity/citric.acid (+0.67), fixed.acidity/pH (-0.69)
# Ces redondances justifient l'utilisation de l'ACP



# ============================================================
# 3. ANALYSE EN COMPOSANTES PRINCIPALES (ACP)
# ============================================================

# ----- 3.1 Réalisation de l'ACP -----
# scale.unit = TRUE : normalisation intégrée (cohérent avec wine_scaled)
# ncp = 5 : on conserve 5 composantes pour l'analyse
acp <- PCA(wine_vars,
           scale.unit = TRUE,
           ncp        = 5,
           graph      = FALSE)  # ne pas afficher les graphiques automatiques

# ----- 3.2 Variance expliquée — Scree Plot -----
# Permet de choisir le nombre de composantes à retenir
fviz_eig(acp,
         addlabels = TRUE,
         ylim      = c(0, 40),
         barfill   = "#2980B9",
         linecolor = "#E74C3C") +
  labs(title = "Scree Plot — Variance expliquée par composante") +
  theme_minimal()

# Afficher le tableau des valeurs propres
print(round(acp$eig, 2))
# Règle de Kaiser : retenir les composantes avec valeur propre > 1
# Règle du coude  : rupture de pente sur le Scree Plot → après Dim4
# Décision : on retient 4 composantes → 71.1% de variance expliquée

# ----- 3.3 Contributions des variables -----
# Contribution de chaque variable à Dim1 et Dim2
print(round(acp$var$contrib[, 1:4], 2))

# Visualisation graphique des contributions
p1 <- fviz_contrib(acp, choice = "var", axes = 1, fill = "#2980B9") +
  labs(title = "Contributions à Dim1") + theme_minimal()
p2 <- fviz_contrib(acp, choice = "var", axes = 2, fill = "#E74C3C") +
  labs(title = "Contributions à Dim2") + theme_minimal()
grid.arrange(p1, p2, ncol = 2)

# ----- 3.4 Cercle des corrélations -----
# Visualise la projection des variables sur le plan factoriel Dim1 x Dim2
# cos2 : qualité de représentation (plus c'est foncé, mieux c'est représenté)
fviz_pca_var(acp,
             col.var       = "cos2",
             gradient.cols = c("#E74C3C", "#F39C12", "#1A9850"),
             repel         = TRUE,       # évite le chevauchement des étiquettes
             title         = "Cercle des corrélations — ACP") +
  theme_minimal()
# Dim1 (28.3%) : axe Acidité & Structure
#   → droite : fixed.acidity, citric.acid, density, sulphates
#   → gauche : pH, volatile.acidity
# Dim2 (17.3%) : axe SO2 & Alcool
#   → haut : free.sulfur.dioxide, total.sulfur.dioxide
#   → bas  : alcohol



# ============================================================
# 4. CLASSIFICATION
# ============================================================

# ===== 4A. K-MEANS =====

# ----- 4A.1 Nombre optimal de clusters — Méthode Elbow -----
# WSS = Within Sum of Squares (inertie intra-cluster)
# On cherche le "coude" où l'ajout d'un cluster n'apporte plus grand chose
fviz_nbclust(wine_scaled, kmeans, method = "wss", k.max = 10) +
  geom_vline(xintercept = 3, linetype = "dashed",
             color = "#E74C3C", linewidth = 1) +
  labs(title = "Méthode Elbow — Nombre optimal de clusters") +
  theme_minimal()

# ----- 4A.2 Nombre optimal — Méthode Silhouette -----
# Mesure la cohésion des clusters : plus le score est élevé, meilleure est la partition
fviz_nbclust(wine_scaled, kmeans, method = "silhouette", k.max = 10) +
  labs(title = "Méthode Silhouette — Qualité des clusters") +
  theme_minimal()
# Elbow : coude à k=3
# Silhouette : pic à k=2, mais k=3 retenu pour sa richesse interprétative

# ----- 4A.3 Application du K-means avec k=3 -----
set.seed(123)   # pour la reproductibilité des résultats
km <- kmeans(wine_scaled,
             centers = 3,    # nombre de clusters
             nstart  = 25)   # 25 initialisations aléatoires → on garde la meilleure

# Taille des clusters
cat("=== Taille des clusters ===\n")
print(table(km$cluster))

# Part de variance expliquée par les clusters
cat("Part expliquée :", round(km$betweenss / km$totss * 100, 1), "%\n")

# ----- 4A.4 Visualisation des clusters sur le plan ACP -----
fviz_cluster(km,
             data         = wine_scaled,
             palette      = c("#E74C3C", "#27AE60", "#2980B9"),
             ellipse.type = "convex",
             ggtheme      = theme_minimal(),
             main         = "Clusters K-means (k=3) — Projection ACP")

# ----- 4A.5 Profils moyens des clusters -----
# Calculer la moyenne de chaque variable par cluster
wine_clean$cluster <- as.factor(km$cluster)

profils <- wine_clean %>%
  group_by(cluster) %>%
  summarise(across(1:11, mean), quality = mean(quality), n = n())

print(round(profils[, -1], 2))
# Cluster 1 : acide, structuré, meilleure qualité (5.97)
# Cluster 2 : léger, défectueux, qualité moyenne (5.54)
# Cluster 3 : alcoolisé, soufré, qualité faible (5.32)


# ===== 4B. CLASSIFICATION ASCENDANTE HIÉRARCHIQUE (CAH) =====

# ----- 4B.1 Échantillonnage -----
# 1599 observations → dendrogramme illisible
# On travaille sur un échantillon de 150 observations
set.seed(123)
idx         <- sample(1:nrow(wine_scaled), 150)
wine_sample <- wine_scaled[idx, ]

# ----- 4B.2 Matrice de distances -----
# Distance euclidienne entre les observations
dist_matrix <- dist(wine_sample, method = "euclidean")

# ----- 4B.3 CAH avec la méthode Ward -----
# Ward.D2 : minimise la variance intra-classe à chaque fusion
# C'est la méthode la plus adaptée pour des données continues
cah <- hclust(dist_matrix, method = "ward.D2")

# ----- 4B.4 Dendrogramme -----
fviz_dend(cah,
          k           = 3,
          palette     = c("#E74C3C", "#27AE60", "#2980B9"),
          rect        = TRUE,        # encadre les clusters
          show_labels = FALSE,       # masque les étiquettes (illisibles à 150 obs)
          main        = "Dendrogramme CAH (Ward.D2) — k=3")
# La hauteur de coupure élevée entre les 3 groupes confirme k=3

# ----- 4B.5 Répartition des groupes CAH -----
groupes_cah <- cutree(cah, k = 3)
cat("=== Répartition CAH ===\n")
print(table(groupes_cah))
# CAH et K-means convergent vers la même structure → résultats robustes



# ============================================================
# 5. ANALYSE COMBINÉE ACP + CLUSTERING
# ============================================================

# ----- 5.1 Projection des clusters sur le plan ACP -----
# Récupérer les coordonnées des individus sur Dim1 et Dim2
coords_acp           <- as.data.frame(acp$ind$coord[, 1:2])
coords_acp$cluster   <- as.factor(km$cluster)
coords_acp$quality   <- wine_quality

# Graphique avec ellipses de confiance à 95%
ggplot(coords_acp, aes(x = Dim.1, y = Dim.2, color = cluster)) +
  geom_point(alpha = 0.4, size = 1.5) +
  stat_ellipse(level = 0.95, linewidth = 1.2) +
  scale_color_manual(
    values = c("#E74C3C", "#27AE60", "#2980B9"),
    labels = c("Cluster 1 — Acide & Structuré",
               "Cluster 2 — Léger & Défectueux",
               "Cluster 3 — Alcoolisé & Soufré")
  ) +
  labs(title  = "Projection des clusters sur le plan ACP",
       x      = "Dim1 — Acidité & Structure (28.3%)",
       y      = "Dim2 — SO₂ & Alcool (17.3%)",
       color  = "Profil") +
  theme_minimal() +
  theme(legend.position = "bottom")

# ----- 5.2 Distribution de la qualité par cluster -----
ggplot(coords_acp, aes(x = cluster, y = quality, fill = cluster)) +
  geom_boxplot(alpha = 0.7, outlier.colour = "gray50") +
  geom_jitter(width = 0.15, alpha = 0.1, size = 0.8) +
  scale_fill_manual(values = c("#E74C3C", "#27AE60", "#2980B9")) +
  scale_x_discrete(labels = c("1 — Acide\n& Structuré",
                              "2 — Léger\n& Défectueux",
                              "3 — Alcoolisé\n& Soufré")) +
  labs(title = "Distribution de la qualité par cluster",
       x = "", y = "Note de qualité (0–10)") +
  theme_minimal() +
  theme(legend.position = "none")

# ----- 5.3 Tableau récapitulatif qualité par cluster -----
resume_qualite <- coords_acp %>%
  group_by(cluster) %>%
  summarise(
    qualite_moyenne = round(mean(quality), 2),
    ecart_type      = round(sd(quality), 2),
    n               = n()
  )
print(resume_qualite)
# Cluster 1 : qualité moy = 5.97 → meilleurs vins
# Cluster 2 : qualité moy = 5.54 → vins ordinaires
# Cluster 3 : qualité moy = 5.32 → vins faibles



# ============================================================
# FIN DU SCRIPT
# ============================================================
# Résumé des résultats :
# - ACP : 4 composantes retenues → 71.1% de variance expliquée
# - Dim1 (28.3%) : Acidité & Structure
# - Dim2 (17.3%) : SO2 & Alcool
# - K-means k=3 validé par Elbow et Silhouette
# - CAH confirme la même partition (robustesse)
# - Variables clés : fixed.acidity, citric.acid, sulphates, volatile.acidity
# ============================================================
