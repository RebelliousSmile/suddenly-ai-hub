# JDROLL Scraper

Scraper pour extraire les campagnes de jdroll.org

## Pourquoi Playwright ?

JDROLL.org utilise AngularJS pour generer le contenu dynamiquement:
- HTML statique vide
- Pas d'API REST accessible
- Playwright execute le JavaScript AngularJS
- Extraction des donnees du DOM genere

## Installation

```bash
pip install -r requirements_scraper.txt
playwright install chromium
```

## Utilisation

```bash
# Usage de base
python scripts/scraper_jdroll.py

# Personnaliser la sortie
python scripts/scraper_jdroll.py --output data/jdroll_campaigns.json

# Scroller plus de pages
python scripts/scraper_jdroll.py --pages 20
```

## Format de sortie

### JSON (output.json)
```json
{
  "total_campaigns": 150,
  "scraped_at": "2026-05-13T22:30:00",
  "source": "jdroll.org",
  "campaigns": [
    {
      "id": 0,
      "title": "Nom de la campagne",
      "description": "Description...",
      "universe": "Fantasy",
      "system": "d20",
      "author": "Auteur",
      "link": "http://www.jdroll.org/campagne/123",
      "scraped_at": "2026-05-13T22:30:00"
    }
  ]
}
```

### JSONL (output.jsonl)
```json
{"id": 0, "title": "...", ...}
{"id": 1, "title": "...", ...}
{"id": 2, "title": "...", ...}
```

## Configuration

- User-Agent: Chrome 124
- Headless: Oui
- Timeout: 60s par page
- Delais: 2s entre les pages

## Probleme

Le site JDROLL.org utilise AngularJS pour charger le contenu dynamiquement.
Le scraping traditionnel (curl, requests) ne fonctionne pas car le HTML
statique ne contient que les scripts, pas les donnees.

Playwright resout ce probleme en executant le JavaScript dans un navigateur
headless.

---

**Fait partie de Suddenly AI Hub - Phase 2: Scraping**
