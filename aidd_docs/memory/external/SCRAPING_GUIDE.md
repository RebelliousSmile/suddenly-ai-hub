# Guide de Scraping - La Cour d'Obéron

> **Note** : la cible La Cour d'Obéron a été mise de côté par décision éthique en 2026-05-15 (cf. `issues-analysis.md` #47). Ce document reste comme méthodologie de référence pour un scraping légitime d'archives publiques avec authentification — réutilisable pour d'autres sources sous accord explicite. La finalité a changé : alimenter le mining vers les tables de Muses (cf. `corpus-public.md`), pas le fine-tune d'un modèle LoRA.

## Objectif
Méthodologie pour scraper les archives publiques d'un forum RP sous authentification, anonymiser les contenus, et produire un JSONL exploitable par le pipeline de mining.

## Prérequis

### 1. Créer un compte
- Va sur: https://couroberon.com/Salons/ucp.php?mode=register
- Remplis le formulaire
- Valide ton email (si nécessaire)

### 2. Installer les dépendances
```bash
cd /home/user/suddenly-muses
source venv/bin/activate
pip install requests python-dotenv
```

### 3. Configurer les cookies

**Méthode A: Via le script (recommandé)**
```bash
# Exécute le script, il te guidera
python scrape_couroberon.py
```

**Méthode B: Manuellement**
1. Dans ton navigateur, va sur https://couroberon.com/Salons
2. Connecte-toi
3. Ouvre les DevTools (F12) → Network
4. Actualise la page
5. Clique sur la requête `viewforum.php`
6. Va dans l'onglet `Cookies`
7. Copie les cookies:
   - `PHPSESSID`
   - `user_id`
   - `user_hash`
   - `csrf_token` (si présent)
8. Sauvegarde dans `cookies.json`

### 4. Lancer le scraping
```bash
cd /home/user/suddenly-muses
source venv/bin/activate
python scrape_couroberon.py
```

## Configuration

Le script utilise ces paramètres par défaut:
- **Max forums:** 3
- **Max topics par forum:** 50
- **Délai entre requêtes:** 3 secondes

Tu peux modifier ces valeurs dans la fonction `main()`:
```python
scraper.scrape_all(
    max_forums=5,        # Nombre de forums à scraper
    max_topics_per_forum=100  # Topics par forum
)
```

## Format des données scrapées

Les données brutes sont sauvegardées au format **JSONL** (avant anonymisation et avant le mining vers les tables Muses) :
```json
{
  "topic_id": "12345",
  "title": "Titre du sujet",
  "url": "https://couroberon.com/Salons/viewtopic.php?t=12345",
  "posts": [
    {
      "author": "Author_1",
      "original_author": "NomOriginal",
      "content": "Contenu du message...",
      "date": "2026-05-13T14:30:00"
    },
    ...
  ],
  "scraped_at": "2026-05-13T15:00:00"
}
```

Ce format est **intermédiaire**. La conversion en rows de tables (format défini par `data-format.md`) se fait dans une étape ultérieure de mining.

## Bonnes pratiques

1. **Respect des délais:** 3 secondes entre chaque requête
2. **Limitation:** Commence petit (3 forums, 50 topics)
3. **Anonymisation au scraping** : les noms d'utilisateurs sont remplacés par `Author_X` à la collecte. L'anonymisation finale (incluant les noms propres dans le contenu) est traitée par `pipelines/anonymization/` au mining.
4. **Respect des CGU:** Utilise uniquement après accord explicite des administrateurs du site cible, jamais pour des forums fermés.
5. **Vérification:** Vérifie les données avant de les passer au mining.

## Dépannage

### Erreur 503
- **Cause:** Protection Cloudflare
- **Solution:** Attends 30 secondes, réessaie

### Erreur de connexion
- **Cause:** Cookies expirés ou incomplets
- **Solution:**
  1. Déconnecte-toi du site
  2. Supprime `cookies.json`
  3. Reconnecte-toi avec ton navigateur
  4. Récupère les nouveaux cookies
  5. Relance le script

### Aucun forum trouvé
- **Cause:** Comptes sans droits de lecture
- **Solution:** Vérifie que ton compte a accès aux archives publiques

## Étapes suivantes

Après le scraping:

1. **Vérifier les données**
   ```bash
   head -n 5 data/couroberon_*.jsonl
   ```

2. **Anonymiser** : pipeline `pipelines/anonymization/` (noms propres → placeholders typés).

3. **Miner vers les tables Muses** : extraction des entités, templates, beats, fragments, et tagging axial. Cf. `corpus-public.md` § Du corpus brut aux rows de tables. Le futur `mining-pipeline.md` formalisera les étapes opérationnelles.

## Avertissements

- Ne scrappe pas trop rapidement (risque de ban)
- Respecte les conditions d'utilisation du site
- Le contenu collecté alimente le pool de rows du service Muses partagé — il doit donc être obtenu sous une licence ou un accord compatible avec un usage par une couche tierce. Pas de scraping de forums RP privés (décision éthique, cf. `issues-analysis.md`).
- Anonymise toujours les données personnelles avant insertion dans les tables.

---

**Dernière mise à jour:** 2026-05-17 (rewrite post-pivot)
