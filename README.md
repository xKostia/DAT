# DAT
DAT projet

# Sujet: Explication des reseaux de neurones
En premier temps, nous avons un modèle de réseau de neurones pouvant nous aider à comprendre le fonctionnement de modèles. Concrètement, le projet à pour but récupérer des modèles de deep learning complexe, et d’en extraire les caractéristiques les plus impoortantes afin de comprendre comment le modèle apprends et quelles sont ses caractéristiques les plus importantes. 

# Technologies utilisées :
Amazon EC2 : Amazon EC2 fournit une interface utilisateur basée sur le Web, la console Amazon EC2. Si on est inscrit à un compte AWS, on peut accéder à la console Amazon EC2 en se connectant à la AWS Management Console et en sélectionnant EC2 depuis la page d'accueil de la console. Ils proposent également un système de « pay as you go »  avec les On-Demand instances : Payez les instances que vous utilisez à la seconde, avec un minimum de 60 secondes, sans engagement à long terme ou paiement initial.

S3 : Amazon S3 est un service de stockage en nuage qui offre un stockage évolutif et à haute durabilité pour les données. Il est conçu pour stocker et récupérer n'importe quelle quantité de données de n'importe où sur le Web et est couramment utilisé par les entreprises de toutes tailles. S3 est payant, sans frais initiaux ni engagements à long terme, et propose une gamme de classes de stockage pour différents scénarios de stockage de données.

Tensorflow : TensorFlow est un outil open source d'apprentissage automatique développé par Google. Le code source a été ouvert le 9 novembre 2015 par Google et publié sous licence Apache. Tensorflow est la deuxième génération du système de Google Brain. Tensorflow peut être lancé sur plusieurs CPU et GPU
Lambda : Amazon Lambda est un service informatique sans serveur qui permet d'exécuter du code en réponse à des événements et gère automatiquement les ressources de calcul sous-jacentes. Il est facturé à l'utilisation, sans frais initiaux ni engagements à long terme. C'est un FaaS : Function as a Service, modèle de cloud computing dans lequel un fournisseur permet aux utilisateurs d'exécuter leur propre code dans le cloud sans se soucier de l'infrastructure ou de la maintenance. C’est rentable et pratique.

 Xplique est un toolkit Python dédié à l'explicabilité, actuellement basé sur Tensorflow. L'objectif de cette bibliothèque est de rassembler l'état de l'art de l'IA explicable pour vous aider à comprendre vos modèles complexes de réseaux de neurones.
La bibliothèque est composée de plusieurs modules, le module Attribution Methods implémente diverses méthodes (e.g. Saliency, Grad-CAM, Integrated-Gradients...), avec des explications, des exemples et des liens vers des articles officiels. Le module Feature Visualization permet de voir comment les réseaux de neurones construisent leur compréhension des images en trouvant des entrées qui maximisent les neurones, les canaux, les couches ou les compositions de ces éléments. Le module Concepts permet d'extraire des concepts humains d'un modèle et de tester leur utilité par rapport à une classe. Enfin, le module Métriques couvre les métriques actuelles utilisées dans l'explicabilité. Utilisé conjointement avec le module Méthodes d'attribution, il permet de tester les différentes méthodes ou d'évaluer les explications d'un modèle.

SHAP (SHapley Additive exPlanations): est une approche théorique du jeu pour expliquer la sortie de tout modèle d'apprentissage automatique. Il relie l'allocation optimale des crédits aux explications locales en utilisant les valeurs classiques de Shapley de la théorie des jeux et leurs extensions associées (voir les articles pour les détails et les citations).

# Exemples modèles:

Deep learning example with GradientExplainer (TensorFlow/Keras/PyTorch models)
![image](https://user-images.githubusercontent.com/102509671/210154575-d167d24d-59eb-48ac-80b8-1cddbb79c1ad.png)


[League of Legends Win Prediction with XGBoost](https://slundberg.github.io/shap/notebooks/League%20of%20Legends%20Win%20Prediction%20with%20XGBoost.html) - À l'aide d'un ensemble de données Kaggle de 180 000 matchs classés de League of Legends, nous formons et expliquons un modèle d'arbre de renforcement de gradient avec XGBoost pour prédire si un joueur gagnera son match.

DeepExplainer
Une implémentation de Deep SHAP, un algorithme plus rapide (mais seulement approximatif) pour calculer les valeurs SHAP pour les modèles d'apprentissage en profondeur qui est basé sur les connexions entre SHAP et l'algorithme DeepLIFT.

[Classification des chiffres MNIST avec Keras](https://slundberg.github.io/shap/notebooks/deep_explainer/Front%20Page%20DeepExplainer%20MNIST%20Example.html) - À l'aide de l'ensemble de données de reconnaissance d'écriture manuscrite MNIST, ce bloc-notes entraîne un réseau de neurones avec Keras, puis explique les prédictions à l'aide de shap.

[Keras LSTM pour la classification des sentiments IMDB](https://slundberg.github.io/shap/notebooks/deep_explainer/Keras%20LSTM%20for%20IMDB%20Sentiment%20Classification.html) - Ce bloc-notes entraîne un LSTM avec Keras sur l'ensemble de données d'analyse des sentiments textuels IMDB, puis explique les prédictions à l'aide de shap.

KernelExplainer
Une implémentation de Kernel SHAP, une méthode indépendante du modèle pour estimer les valeurs SHAP pour n'importe quel modèle. Parce qu'il ne fait aucune hypothèse sur le type de modèle, KernelExplainer est plus lent que les autres algorithmes spécifiques au type de modèle.


[Modèle ImageNet VGG16 avec Keras](https://slundberg.github.io/shap/notebooks/ImageNet%20VGG16%20Model%20with%20Keras.html) - Expliquez les prédictions du réseau de neurones convolutionnel VGG16 classique pour une image. Cela fonctionne en appliquant la méthode Kernel SHAP indépendante du modèle à une image segmentée en super-pixels.

# Modèle 

Modèle réalisé à l'aide de cloudcraft.co

![image](https://user-images.githubusercontent.com/102509671/210156717-b051a575-e699-44d0-9541-56d63f8c426c.png)



Explication de chaque composant :

Modèle de réseau neuronal : Il s'agit du modèle d'apprentissage automatique réel qui a été formé pour effectuer une tâche spécifique, comme la classification d'images ou la traduction de langues. Le modèle est construit à l'aide de TensorFlow.

Amazon SageMaker Training Job : Il s'agit du service AWS qui est utilisé pour former le modèle de réseau neuronal. Il prend les données d'entraînement et entraîne le modèle à l'aide d'une variété d'algorithmes et d'hyperparamètres.

Amazon S3 Bucket : Il s'agit d'un service de stockage sur AWS où le modèle formé est enregistré une fois le travail de formation terminé. Le modèle est enregistré sous la forme d'un artefact de modèle.

Dépôt Amazon Elastic Container Registry (ECR) : Il s'agit d'un service qui vous permet de stocker des images Docker, y compris celles qui contiennent votre modèle TensorFlow.

Amazon Elastic Container Service (ECS) Fargate Task : Il s'agit d'un service qui vous permet d'exécuter des conteneurs Docker dans le cloud, y compris ceux qui contiennent votre modèle TensorFlow. Vous pouvez utiliser la tâche ECS Fargate pour déployer votre modèle en tant que microservice.

Amazon API Gateway Endpoint : Il s'agit d'un service qui agit comme un proxy inverse pour la tâche ECS Fargate, ce qui vous permet d'accéder en toute sécurité à la tâche depuis des clients externes.

Application client : Il peut s'agir de n'importe quelle application qui souhaite utiliser le modèle de réseau neuronal formé et déployé, comme une application Web ou une application mobile.

Cette architecture vous permet de former et de déployer un modèle TensorFlow sur AWS en tant que microservice, et d'y accéder via un point de terminaison API qui peut être appelé par des applications clientes. Vous pouvez ensuite utiliser des techniques telles que l'importance des caractéristiques ou la propagation de la pertinence par couches pour interpréter les prédictions du modèle et comprendre comment le modèle est...
