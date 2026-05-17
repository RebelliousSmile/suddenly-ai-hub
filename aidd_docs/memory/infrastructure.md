---
name: infrastructure
description: Spec opérationnelle de l'infrastructure et de l'API du service Muses
---

# Infrastructure et API opérationnelle

> Projet pré-MVP. Cette spec couvre l'API actuellement implémentée (M0-M3) et fixe les contraintes pour le déploiement public M4. La haute disponibilité au-delà du SPOF assumé (cf. `DECISIONS.md` D15) n'est pas dans le périmètre tant que la charge réelle ne le justifie pas.

## Topologie cible

```
[ Instance Suddenly A ]──┐
[ Instance Suddenly B ]──┼── HTTPS signé ActivityPub ──►  [ Service Muses, single VM CPU ]
[ Instance Suddenly C ]──┘                                   │
                                                             ├── tables/      JSONL versionnés git
                                                             ├── feedback/    SQLite local (trust, profil, learner, instance_rep, event log)
                                                             └── snapshots/   copies horodatées des SQLite (T40)
```

## Endpoints HTTP

| Méthode | Path | Auth | Description | Codes |
|---|---|---|---|---|
| GET | `/v1/health` | aucune | Liveness check. Renvoie `tables_count`, `encoder_dim`, `feedback_enabled` | 200 |
| POST | `/v1/suggest/dialogue` | Signature HTTP | Feature dialogue (#77). Body : `SuggestRequest`. Réponse : `SuggestResponse` avec traçabilité (rows servies + scores + axes relâchés) | 200, 400 (mauvaise feature), 401 (signature manquante/invalide), 422 (tags hors canon) |
| POST | `/v1/feedback/signal` | Signature HTTP | Capture d'un signal UI (5 types). Body : `FeedbackSignalRequest`. Dispatche vers trust + style + learner | 200, 401, 503 (feedback désactivé) |
| GET | `/v1/admin/coverage` | `X-Admin-Token` (optionnel) | Carte de couverture par cellule (counts_by_level, distinct_contributors, last_contribution) | 200, 403 |

Schémas détaillés : `muses/api/schemas.py` et `muses/api/server.py`.

## Authentification ActivityPub (cible M4)

Le MVP M2/M3 utilise `require_signature_stub` qui parse le header `Signature` (RFC draft-cavage) mais ne vérifie pas la signature cryptographique. À implémenter en M4 :

1. **Parsing** : `keyId`, `algorithm`, `headers`, `signature` (déjà fait par `parse_http_signature`).
2. **Résolution de l'acteur** : `keyId` = URI ActivityPub d'un objet `publicKey`. Fetch l'acteur (`GET <actor>` avec `Accept: application/activity+json`) → récupérer `publicKey.publicKeyPem`.
3. **Vérification** :
   - Reconstruire le `signingString` canonical depuis les headers listés (`request-target`, `host`, `date`, `digest`, etc.).
   - Vérifier `signature` avec la clé publique RSA-SHA256.
4. **Validation du `Digest`** si présent : `SHA-256=base64(sha256(body))`.
5. **Anti-replay** : rejeter `date` trop ancienne (>5 min).

Le résultat enrichit `ParsedSignature` avec `actor_uri` validé, utilisé pour peupler `user_id` / `instance_id` dans les signals et les rows.

Implémentation suggérée : librairie `httpsig` ou implémentation maison ~150 lignes.

## Mode dégradé côté client (M4/T38)

`MusesClient.suggest` lève `MusesUnavailable` sur :

- timeout HTTP
- erreurs réseau (DNS, refusal)
- codes 5xx du service

Convention côté instance Suddenly : attraper `MusesUnavailable` → griser le bouton IA + message « Assistant indisponible, réessayez plus tard » (cf. `external/use-cases.md` §4.3). Aucune unité d'usage n'est débitée sur cet échec.

Les 4xx restent propagés via `httpx.HTTPStatusError` — ce sont des erreurs de la requête (signature invalide, tags hors canon), pas du service.

## Stockage opérationnel

| Type | Backend | Schéma source | Backup |
|---|---|---|---|
| Tables (rows) | JSONL versionnés git | `external/data-format.md` | git remote |
| Index FTS5 des rows | SQLite local | `muses/tables/sqlite_index.py` | reconstruit depuis JSONL |
| Embeddings rows | `.npy` adjacent | `muses/tables/embeddings.py` | reconstruit depuis JSONL |
| Trust contributeur | SQLite local | `muses/feedback/trust.py` | snapshots T40 |
| Réputation instance | SQLite local | `muses/feedback/instance_reputation.py` | snapshots T40 |
| Profil de style | SQLite local | `muses/feedback/style_profile.py` | snapshots T40 |
| Online learner | SQLite local | `muses/feedback/online_learning.py` | snapshots T40 |
| Event log signaux | JSONL append-only | `muses/feedback/events.py` | rotation périodique + archivage |

Les SQLite sont reconstructibles depuis l'event log (rejouable). Les snapshots permettent un rollback rapide sans rejouer l'historique complet.

## Snapshots et rollback (M4/T40)

`muses.feedback.snapshots` :

- `snapshot_directory(source_dir, snapshot_dir)` — copie horodatée d'un dossier (typiquement `feedback/`).
- `list_snapshots(snapshot_dir)` — liste antichronologique.
- `restore_snapshot(snapshot_path, target_dir)` — restauration (écrase la cible).

Politique :
- Snapshot automatique toutes les N heures (à figer en production).
- Conservation roulante des K derniers snapshots.
- Snapshot pré-déploiement systématique.

Le rollback **ne supprime pas** les rows JSONL ajoutées après le snapshot — seuls les poids ML / trust / profils sont rétablis.

## Déploiement v0 (M4/T37 — hors session)

Cible : VM unique CPU-only, distribution Linux standard.

Étapes attendues :

1. Provisioner une VM (ex: Hetzner CX22 — 2 vCPU, 8 GB RAM, 50 GB SSD pour ~6€/mois).
2. Cloner le repo, `pip install -e .[api,embeddings,pipelines]`.
3. Configurer un service systemd qui lance `uvicorn muses.api.server:app` (avec `app = create_app(...)` configuré par un module wrapper qui lit les chemins depuis `.env`).
4. Reverse proxy nginx avec TLS Let's Encrypt sur `muses.<domaine>`.
5. Persistance : `tables/`, `feedback/`, `snapshots/` sur un disque dédié.
6. Cron : snapshots horaires + scan anti-sleeper quotidien.
7. Monitoring : Prometheus node_exporter + healthcheck `/v1/health` via Blackbox.

À documenter en script `scripts/deploy/install_v0.sh` quand T37 est exécutée.

## Capacity planning indicatif

| Mesure | M2 | M3 | M4 nominal |
|---|---|---|---|
| Tables (KB total) | ~50 | ~50 | 1-100 MB |
| Embeddings (MB) | <1 | <1 | 10-500 |
| Event log (lignes/jour) | 0 | 0-100 | 1k-100k |
| Trust DB (rows) | 0 | 10-100 | 1k-1M |
| Latence p95 suggest | <50ms (StubEncoder) | <50ms | <300ms (sentence-transformer CPU) |
| RAM idle | ~50MB | ~80MB | 500MB-1GB |

## Hors périmètre

- **Haute disponibilité** (réplication, failover) — D15 acte le SPOF pour le MVP.
- **Multi-region** — non pertinent au volume cible initial.
- **Authentification admin via ActivityPub** — pour MVP, `X-Admin-Token` statique suffit.
- **Rate limiting public** — à ajouter avant ouverture publique (`slowapi` ou middleware custom).
