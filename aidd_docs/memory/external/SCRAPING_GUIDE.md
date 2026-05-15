# Guide de Scraping - La Cour d'Obéron

## 🎯 Objectif
Scrapper les archives publiques de La Cour d'Obéron pour entraîner des LoRAs RP.

## 📋 Prérequis

### 1. Créer un compte
- Va sur: https://couroberon.com/Salons/ucp.php?mode=register
- Remplis le formulaire
- Valide ton email (si nécessaire)

### 2. Installer les dépendances
```bash
cd /home/user/suddenly-ai-hub
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
cd /home/user/suddenly-ai-hub
source venv/bin/activate
python scrape_couroberon.py
```

## ⚙️ Configuration

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

## 📊 Format des données

Les données sont sauvegardées au format **JSONL**:
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

## 🛡️ Bonnes pratiques

1. **Respect des délais:** 3 secondes entre chaque requête
2. **Limitation:** Commence petit (3 forums, 50 topics)
3. **Anonymisation:** Les noms d'utilisateurs sont remplacés par `Author_X`
4. **Respect des CGU:** Utilise uniquement pour l'entraînement personnel
5. **Vérification:** Vérifie les données avant de les utiliser

## 🔧 Dépannage

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

## 📈 Étapes suivantes

Après le scraping:

1. **Vérifier les données**
   ```bash
   head -n 5 data/couroberon_*.jsonl
   ```

2. **Nettoyer et structurer**
   - Voir `clean_dataset.py` (à créer)

3. **Convertir en format Axolotl**
   - Voir `convert_to_axolotl.py` (à créer)

4. **Fine-tuning**
   - Utiliser Fireworks.ai ou Together.ai

## ⚠️ Avertissements

- Ne scrappe pas trop rapidement (risque de ban)
- Respecte les conditions d'utilisation du site
- Utilise uniquement pour l'entraînement personnel (non commercial)
- Anonymise toujours les données personnelles

## 📞 Support

Pour les problèmes:
- Vérifie le fichier `logs/scraping.log`
- Consulte ce README
- Contacte l'équipe (si open source)

---

**Dernière mise à jour:** 2026-05-13
