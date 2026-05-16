# **Résumé : Benchmarks des Chats NSFW et Roleplay**

---

## **📌 Introduction**

Les **modèles de roleplay** (y compris NSFW) sont évalués différemment des LLM généralistes. Les benchmarks se concentrent sur :

- **Cohérence** (mémoire du personnage).
- **Créativité** (originalité des réponses).
- **Fidélité** (respect des consignes et du contexte).
- **Profondeur émotionnelle** (expression des sentiments).
- **Sécurité/Éthique** (filtres pour les contenus illégaux).

---

## **🔹 Benchmarks Formels et Communautaires**

### **1. Benchmarks Spécifiques au Roleplay**


| **Benchmark**  | **Description**                                                              | **Modèles Évalués**                 | **Lien**                                                                            |
| -------------- | ---------------------------------------------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------- |
| **RPBench**    | Évalue la cohérence, la créativité et la fidélité au personnage.             | Llama 2, Mistral, Pygmalion         | [GitHub - RPBench](https://github.com/rpbench/rpbench) *(à vérifier)*               |
| **ChatHaruhi** | Test la capacité à maintenir un personnage cohérent sur des dialogues longs. | GPT-4, Claude 3, Llama 3, Pygmalion | [Projet ChatHaruhi](https://github.com/ChatHaruhi/ChatHaruhi)                       |
| **RoleLLM**    | Benchmark pour la profondeur émotionnelle et la consistance des réponses.    | Modèles open source                 | [Hugging Face - RoleLLM](https://huggingface.co/datasets/RoleLLM) *(si disponible)* |


### **2. Évaluations Communautaires**

- **LMSYS Chatbot Arena** : [https://lmsys.org/](https://lmsys.org/)
  - Permet de comparer des modèles en temps réel (inclut parfois des modèles de roleplay).
- **Forums** :
  - [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) (discussions sur les modèles de roleplay).
  - [ChaiML GitHub](https://github.com/ChaiML/ChaiNNer) (pour les modèles NSFW).
- **Classements informels** :

  | **Modèle**        | **Cohérence** | **Créativité** | **Fidélité** | **NSFW** | **Popularité** |
  | ----------------- | ------------- | -------------- | ------------ | -------- | -------------- |
  | **Pygmalion-13B** | ⭐⭐⭐⭐⭐         | ⭐⭐⭐⭐⭐          | ⭐⭐⭐⭐⭐        | ✅ Oui    | ⭐⭐⭐⭐⭐          |
  | **Metharme-7B**   | ⭐⭐⭐⭐          | ⭐⭐⭐⭐⭐          | ⭐⭐⭐⭐         | ✅ Oui    | ⭐⭐⭐⭐           |
  | **Llama 3 8B**    | ⭐⭐⭐⭐          | ⭐⭐⭐⭐           | ⭐⭐⭐⭐         | ❌ Non    | ⭐⭐⭐            |


---

## **🏆 Top Modèles pour le Roleplay (2024-2026)**


| **Modèle**             | **Base** | **Taille** | **Cohérence** | **Créativité** | **Fidélité** | **NSFW** | **Lien**                                                          |
| ---------------------- | -------- | ---------- | ------------- | -------------- | ------------ | -------- | ----------------------------------------------------------------- |
| **Pygmalion-13B**      | Mistral  | 13B        | ⭐⭐⭐⭐⭐         | ⭐⭐⭐⭐⭐          | ⭐⭐⭐⭐⭐        | ✅ Oui    | [Hugging Face](https://huggingface.co/PygmalionAI/pygmalion-13b)  |
| **Metharme-7B**        | Mistral  | 7B         | ⭐⭐⭐⭐          | ⭐⭐⭐⭐⭐          | ⭐⭐⭐⭐         | ✅ Oui    | [Hugging Face](https://huggingface.co/Metharme/Metharme-7B)       |
| **Llama 3 8B**         | Meta     | 8B         | ⭐⭐⭐⭐          | ⭐⭐⭐⭐           | ⭐⭐⭐⭐         | ❌ Non    | [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| **Erotica-Llama-2-7B** | Llama 2  | 7B         | ⭐⭐⭐⭐          | ⭐⭐⭐⭐           | ⭐⭐⭐⭐         | ✅ Oui    | [Hugging Face](https://huggingface.co/erotica/llama2-7b-erotica)  |


---

## **🔍 Critères d'Évaluation pour le Roleplay**


| **Critère**                 | **Description**                                                                | **Exemple de Test**                                                                       |
| --------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| **Cohérence**               | Le modèle maintient-il un personnage cohérent (mémoire, traits de caractère) ? | *"Tu es un pirate. Décris ta journée."* → Vérifier si les réponses restent dans le thème. |
| **Créativité**              | Les réponses sont-elles originales et immersives ?                             | *"Invente une histoire avec un dragon et un chevalier."* → Évaluer l'originalité.         |
| **Fidélité au Prompt**      | Le modèle respecte-t-il les consignes (ex: ton, style, restrictions) ?         | *"Réponds en vers."* → Vérifier si la réponse est bien en vers.                           |
| **Profondeur Émotionnelle** | Le modèle exprime-t-il des émotions adaptées au personnage ?                   | *"Tu es triste. Décris tes sentiments."* → Évaluer la richesse émotionnelle.              |
| **Gestion du Contexte**     | Le modèle mémorise-t-il les informations précédentes (ex: noms, événements) ?  | *"Tu as rencontré Alice hier. Que penses-tu d'elle aujourd'hui ?"* → Vérifier la mémoire. |


---

## **🛠️ Comment Tester un Modèle de Roleplay ?**

### **1. Scénarios de Test**

- *"Tu es un détective dans les années 1920. Résous cette énigme : [énigme]."*
- *"Tu es un elfe dans un monde fantastique. Décris ta journée."*
- *"Tu es un scientifique fou. Explique ton invention la plus folle."*

### **2. Évaluation Manuelle**

- Noter chaque réponse sur :
  - Cohérence (1-10)
  - Créativité (1-10)
  - Fidélité (1-10)

### **3. Outils pour Automatiser**

- **Ollama** :
  ```bash
  ollama pull pygmalion-13b
  ollama run pygmalion-13b
  ```
- **Text Generation WebUI** :
  - [GitHub - Oobabooga](https://github.com/oobabooga/text-generation-webui)

### **4. Script Python pour Évaluer**

```python
from transformers import pipeline
import pandas as pd

models = [
    {"name": "Pygmalion-13B", "id": "PygmalionAI/pygmalion-13b"},
    {"name": "Metharme-7B", "id": "Metharme/Metharme-7B"},
]

scenarios = [
    "Tu es un chevalier médiéval. Décris ta quête pour sauver la princesse.",
    "Tu es un pirate. Raconte une aventure en mer.",
]

results = []
for model in models:
    pipe = pipeline("text-generation", model=model["id"])
    for scenario in scenarios:
        response = pipe(scenario, max_length=200)[0]["generated_text"]
        coherence = int(input(f"Cohérence (1-10) pour {model['name']} : "))
        creativity = int(input(f"Créativité (1-10) pour {model['name']} : "))
        fidelity = int(input(f"Fidélité (1-10) pour {model['name']} : "))
        results.append({
            "model": model["name"],
            "scenario": scenario,
            "coherence": coherence,
            "creativity": creativity,
            "fidelity": fidelity
        })

df = pd.DataFrame(results)
print(df.groupby("model").mean())
```

---

## **📌 Plateformes de Chat NSFW et Leurs Modèles**


| **Plateforme**   | **Modèles Utilisés**                                                         | **Approche**                                                                        |
| ---------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Candy.ai**     | Modèles propriétaires (fine-tunés à partir de Llama 2, Mistral, ou GPT-3/4). | Filtres de modération pour éviter les contenus illégaux.                            |
| **Character.AI** | Modèles propriétaires (Transformer-based).                                   | Fine-tuning sur des datasets de dialogues. Filtres stricts sur la version publique. |
| **ChaiML**       | Llama 2, Mistral, Pygmalion, Metharme (open source).                         | Communauté active, hébergement local possible.                                      |
| **Nomi**         | Llama 2, Mistral, modèles personnalisés (ex: Erotica-Llama).                 | Pas de censure (selon les paramètres utilisateur).                                  |
| **SpicyChat**    | Llama 3, GPT-3.5/4 (via API).                                                | Optimisé pour le NSFW, mais dépend des modèles tiers.                               |


---

## **💡 Recommandations**


| **Besoin**                    | **Modèle Recommandé**                     | **Pourquoi ?**                                            |
| ----------------------------- | ----------------------------------------- | --------------------------------------------------------- |
| **Roleplay général**          | **Pygmalion-13B**                         | Meilleur équilibre cohérence/créativité.                  |
| **Roleplay NSFW**             | **Metharme-7B** ou **Erotica-Llama-2-7B** | Optimisés pour les interactions adultes.                  |
| **Roleplay léger (non NSFW)** | **Llama 3 8B**                            | Bonnes performances, filtres de sécurité intégrés.        |
| **Test rapide**               | **LMSYS Chatbot Arena**                   | Compare plusieurs modèles en temps réel.                  |
| **Hébergement local**         | **Ollama + Pygmalion-13B**                | Simple à configurer, pas de dépendance à une API externe. |


---

## **🔗 Ressources Utiles**

- **LMSYS Chatbot Arena** : [https://lmsys.org/](https://lmsys.org/)
- **Hugging Face (Modèles)** :
  - [Pygmalion-13B](https://huggingface.co/PygmalionAI/pygmalion-13b)
  - [Metharme-7B](https://huggingface.co/Metharme/Metharme-7B)
  - [Erotica-Llama-2-7B](https://huggingface.co/erotica/llama2-7b-erotica)
- **Ollama** : [https://ollama.ai/](https://ollama.ai/)
- **Text Generation WebUI** : [GitHub - Oobabooga](https://github.com/oobabooga/text-generation-webui)

---

## **📝 Notes Personnelles**

- **À tester** :
  - Comparer **Pygmalion-13B** et **Metharme-7B** sur un scénario de roleplay NSFW.
  - Évaluer la **cohérence** de Llama 3 8B sur un dialogue long.
  - Tester **Ollama** pour héberger localement un modèle de roleplay.
- **Idées** :
  - Créer un **benchmark personnalisé** pour ton usage spécifique (ex: classement d’emails en roleplay).
  - Explorer **ChaiML** pour des modèles NSFW open source.
