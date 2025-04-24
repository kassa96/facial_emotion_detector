# 🎭 Emotion Recognition & Facial Tracking App

Ce projet propose une solution complète de reconnaissance des émotions faciales à partir d’images ou de flux vidéo, combinée à un système de tracking en temps réel. 

Il repose sur des techniques de vision par ordinateur, des réseaux de neurones convolutifs (CNN), et une intégration Python orientée application.

---

## 🚀 Objectifs du Projet

- Détecter et reconnaître les émotions à partir d’expressions faciales.
- Optimiser un modèle CNN pour la classification des émotions.
- Suivre les visages en temps réel à l’aide d’une application Python.

---

## 📂 Ressources

🔗 **Challenge Kaggle** [données d'entraînement](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/overview)  
📓 **Notebook d'entraînement du modèle** : [CNN architecture](https://www.kaggle.com/code/kassadiallo/cnn-emotion-classifier) *(Activez les GPU pour de meilleures performances)*  
📺 **Vidéos pédagogiques sur les CNN ** : [cours CNN fiddle](https://www.youtube.com/watch?v=JfBf5eYptSs)  
📘 **Cours sur la vision par ordinateur** : [base computer vision](https://www.analyticsvidhya.com/blog/2022/04/face-detection-using-the-caffe-model/)  
📓 **Modele de detection de visage** : [face detection](https://github.com/vinuvish/Face-detection-with-OpenCV-and-deep-learning/tree/master/models) 

---

## 🧠 Technologies et Outils Utilisés

- Python 3
- TensorFlow / Keras
- OpenCV
- prototxt et caffemodel

---

## 🔧 Installation

```bash
git clone https://github.com/kassa96/facial_emotion_detector.git
cd facial_emotion_detector

python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt

```
---

## 📁 Structure du projet
```bash

emotion-recognition/   
  ├── data/ # Dossier pour stocker les datasets (train.csv et test_with_emotions.csv) que tu dois télécharger à partir de kaggle pour excuter le notebook en locale   
  ├── face_detection_models/ # Le modéle de reconnaissance des visages doit etre téléchargé à travers le notebook kaggle.
  ├── cnn-emotion-classifier.ipynb # Notebook d’entraînement du modèle de classification des émotions 
  ├── final_emotion_model.keras # Le modele de classification d'émotion que tu dois téléchargé à partir du notebook kaggle https://www.kaggle.com/code/kassadiallo/cnn-emotion-classifier
  ├── live_stream_app.py # Script pour lancer le tracking des visages et reconnaitre les émotions 
  ├── requirements.txt # Dépendances Python   
  └── README.md  
```

---

## 🚀 Lancement de l'application

Une fois toutes les dépendances installées, vous pouvez exécuter l’application de reconnaissance d’émotion et de suivi facial en temps réel.

### ▶️ Exécution du script de tracking :

```bash
python3 live_stream_app.py
```
Pour arreter le programme cliqué sur l'application puis taper le caractere q.

---

##  📊 Résultats
Le modèle a été entraîné sur le dataset fourni par Kaggle, avec les classes d’émotions suivantes :

😃 Joie

😢 Tristesse

😠 Colère

😮 Surprise

😐 Neutre

😨 Peur

😖 Dégout

---

## 🎯 Performance du modèle :
Précision sur le jeu de validation : 65%

Méthodes d’optimisation utilisées : Data augmentation, normalisation, early stopping, dropout

L'application affiche en temps réel les visages détectés avec leur émotion dominante prédite, encadrée et annotée sur le flux vidéo.

---

## 🤝 Contributions bienvenues
Ce projet est entièrement open-source et a été pensé pour évoluer avec l’aide de la communauté. Toute contribution est la bienvenue !

Vous pouvez aider à :

## 🔧 Améliorer la précision du modèle

📦 Intégrer d'autres datasets plus riches

🧠 Ajouter d’autres classes d’émotions ou personnaliser la classification

📖 Gerer le probléme d'équilibrage des classes.

💻 Créer une interface web avec Streamlit, Flask ou autre

🐛 Corriger des bugs ou proposer des refactorings du code

Pour toute contribution, merci de créer une pull request ou d’ouvrir une issue sur GitHub.

---

## 🙏 Remerciements

Je tiens à remercier chaleureusement toutes les personnes qui rendent le savoir accessible :

Les créateurs de contenus pédagogiques sur YouTube, Medium, GitHub…

Les développeurs qui publient leurs projets en open-source

Les enseignants et chercheurs qui rendent leurs cours disponibles en ligne

Grâce à vous, j’ai pu apprendre, expérimenter et mener ce projet à terme.
Votre travail est une source d’inspiration précieuse. Merci infiniment à Guillaume Saint-Cirgue de Machine Learnia. 💙

