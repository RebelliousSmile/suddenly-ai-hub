# 🔧 Installation des Skills

## Prérequis

- Hermes Agent installé
- Accès internet pour cloner hermes-vault

## Installation Rapide

### Option 1 : Via l'interface (Recommandé)

1. Ouvrir l'interface Hermes Agent
2. Cliquer sur **"Add skills"**
3. Choisir **"Install from GitHub/Git URL"**
4. Entrer : `RebelliousSmile/hermes-vault`
5. Cocher **"Enable after install"**
6. Cliquer sur **"INSTALL FROM GIT"**

### Option 2 : Via CLI

```bash
# Installer toutes les skills
hermes skills install RebelliousSmile/hermes-vault

# Vérifier l'installation
hermes skills list
```

### Option 3 : Manuel

```bash
# Cloner le dépôt
git clone https://github.com/RebelliousSmile/hermes-vault.git ~/.hermes/plugins/hermes-vault

# Redémarrer Hermes Agent
```

## Vérification

```bash
# Lister les skills installées
hermes skills list

# Tester une skill
hermes -s aidd-workflow "Créer un plan pour phase 3"
```

## Maintenance

### Mettre à jour

```bash
# Pull des dernières modifications
cd ~/.hermes/plugins/hermes-vault
git pull origin main

# Reconnaître les changements
hermes skills reload
```

### Désinstaller

```bash
# Via l'interface
# Cliquer sur les skills et supprimer

# Via CLI (si disponible)
hermes skills uninstall RebelliousSmile/hermes-vault
```

## FAQ

### Pourquoi utiliser hermes-vault ?

- ✅ Centralisation de toutes les compétences
- ✅ Versionning des modifications
- ✅ Réutilisation entre projets
- ✅ Partage avec la communauté

### Quelles compétences sont incluses ?

Voir le README de hermes-vault pour la liste complète :
https://github.com/RebelliousSmile/hermes-vault#-organisation

### Comment ajouter une nouvelle compétence ?

1. Voir le template dans `hermes-vault/README.md`
2. Créer un fichier `SKILL.md` dans la catégorie appropriée
3. Documenter la compétence
4. Pusher sur hermes-vault

## Références

- [hermes-vault sur GitHub](https://github.com/RebelliousSmile/hermes-vault)
- [Documentation Hermes Agent](https://hermes-agent.nousresearch.com/)
- [Guide AIDD](./aidd_docs/AIDD_GUIDE.md)
