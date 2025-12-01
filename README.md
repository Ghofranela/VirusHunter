# VirusHunter 🛡️

Système intelligent de détection de malwares combinant Deep Learning (PyTorch) et IA conversationnelle (Llama3:8b).

## 🚀 Lancement rapide

### Prérequis
- Docker et Docker Compose installés
- Accès au serveur Ollama distant (déjà configuré)

### Démarrage

```bash
# 1. Cloner le projet
git clone https://github.com/Ghofranela/VirusHunter.git
cd VirusHunter

# 2. Lancer l'application
docker-compose up -d

# 3. Accéder à l'interface
# → http://localhost:8501
```

### Arrêt

```bash
docker-compose down
```

## 📁 Structure

```
VirusHunter/
├── app/
│   └── streamlit_complete.py    # Interface web
├── src/                          # Code source
│   ├── model.py                  # Architectures DNN/CNN/LSTM
│   ├── training.py               # Entraînement
│   ├── preprocessing.py          # Preprocessing
│   ├── explainability.py         # SHAP, LIME, IG
│   └── ...
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/                         # Données EMBER (local uniquement)
└── requirements.txt
```

## ✨ Fonctionnalités

- **Détection de malwares** : Classification binaire avec score de risque
- **Chatbot IA** : Analyse conversationnelle via Llama3:8b
- **Explicabilité** : SHAP, LIME, Integrated Gradients
- **Interface web** : Upload, analyse, historique, rapports
- **Formats supportés** : `.npy`, `.exe`, `.dll`, `.pdf`, `.docx`, `.zip`, `.py`, `.js`

## 🌿 Workflow Git

```
main (production)
 └── dev (développement)
      └── feature/chatbot-intelligent-analysis (en cours)
      └── feature/nouvelle-feature
```

### Créer une feature

```bash
git checkout dev
git checkout -b feature/ma-feature
# développer...
git add .
git commit -m "feat: description"
git push -u origin feature/ma-feature
# Pull Request vers dev
```

## 🛠️ Technologies

- **PyTorch 2.0+** : Deep Learning
- **Streamlit** : Interface web
- **Ollama + Llama3:8b** : LLM distant
- **SHAP/LIME** : Explicabilité
- **EMBER** : Dataset (2,381 features)

## 📊 Dataset

Les données d'entraînement EMBER se trouvent sur le PC du collègue dans le dossier `data/`.

## 🔧 Configuration

L'URL Ollama est configurée dans [docker-compose.yml](docker/docker-compose.yml) :
- **Serveur** : `http://51.254.200.139:11434`
- **Modèle** : `llama3:8b`

## 📄 Licence

Projet éducatif - Recherche en cybersécurité uniquement.
