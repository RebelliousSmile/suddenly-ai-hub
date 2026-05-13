#!/usr/bin/env python3
"""
Conversion des données scrapées vers JSONL Axolotl
Date: 2026-05-13
Objectif : Transformer les campagnes scrapées en format de fine-tuning
"""

import json
import os
import glob
import re
from datetime import datetime
from typing import List, Dict, Optional

class DataConverter:
    """Convertit les données de campagnes en format JSONL pour Axolotl"""
    
    def __init__(self):
        self.clean_dir = "/home/user/suddenly-ai-hub/data/clean"
        self.final_dir = "/home/user/suddenly-ai-hub/data/final"
        
        os.makedirs(self.final_dir, exist_ok=True)
        
        print("="*80)
        print("📊 CONVERSION DONNÉES → JSONL AXOLOT")
        print("="*80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Données sources: {self.clean_dir}")
        print(f"📁 Données export: {self.final_dir}")
        print("="*80)
    
    def load_campaigns(self) -> List[Dict]:
        """Charger les données de campagnes"""
        print("\n📂 Chargement des campagnes...")
        
        # Chercher le fichier de données le plus récent
        json_files = sorted(glob.glob(f"{self.clean_dir}/campagnes_*.json"))
        
        if not json_files:
            print("❌ Aucun fichier de données trouvé")
            return []
        
        # Charger le dernier
        latest_file = json_files[-1]
        print(f"✅ Fichier: {os.path.basename(latest_file)}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            campaigns = json.load(f)
        
        print(f"✅ {len(campaigns)} campagnes chargées")
        return campaigns, latest_file
    
    def load_html(self, html_file: str) -> Optional[str]:
        """Charger un fichier HTML"""
        try:
            if not os.path.exists(html_file):
                return None
            
            with open(html_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️  Erreur lecture HTML {html_file}: {e}")
            return None
    
    def extract_post_data(self, html: str, campaign_id: str) -> List[Dict]:
        """Extrait les posts/discussions du HTML d'une campagne"""
        posts = []
        
        # Chercher les patterns de posts dans le HTML
        patterns = {
            'author': [
                r'<span[^>]*class=["\']author["\'][^>]*>([^<]+)</span>',
                r'par ([^<]+)',
                r'<strong[^>]*>([^<]+)</strong>',
            ],
            'content': [
                r'<div[^>]*class=["\']post-content["\'][^>]*>(.*?)</div>',
                r'<p[^>]*>([^<]+)</p>',
                r'>([^<]+)<',
            ],
            'timestamp': [
                r'<time[^>]*datetime=["\']([^"\']+)["\']',
                r'(\d{4}-\d{2}-\d{2})',
            ],
        }
        
        # Extractions basées sur le pattern HTML de jdRoll
        # Note: jdRoll n'utilise pas toujours de structure de forum classique
        # On va extraire les éléments textuels significatifs
        
        # Chercher les paragraphes qui pourraient être des posts
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL)
        
        for para in paragraphs:
            # Nettoyer le paragraphe
            text = re.sub(r'<[^>]+>', ' ', para)
            text = ' '.join(text.split())
            
            if len(text) > 50 and text != 'Proposé par':  # Minimum 50 caractères
                posts.append({
                    'author': 'Inconnu',  # Sera complété plus tard
                    'content': text[:500],  # Limiter à 500 caractères
                    'type': 'post',
                    'source': f'camp_{campaign_id}'
                })
        
        return posts
    
    def generate_conversation(self, campaign: Dict, posts: List[Dict], 
                              html: str, context_index: int) -> Optional[Dict]:
        """Génère une conversation au format Axolotl"""
        
        if not posts:
            return None
        
        # Créer le contexte système
        system_content = (
            f"Vous êtes un MJ expert en jeu de rôle.\n\n"
            f"Contexte de la campagne: {campaign.get('title', 'Inconnu')}\n"
            f"Univers: {campaign.get('universe', 'Inconnu')}\n"
            f"Système de jeu: {campaign.get('system', 'Inconnu')}\n"
            f"Auteur: {campaign.get('author', 'Inconnu')}\n\n"
            f"Règles:\n"
            f"- Répondez en tant que MJ compétent et immersif\n"
            f"- Suivez le style et le ton de la campagne\n"
            f"- Utilisez la terminologie de l'univers\n"
            f"- Maintenez la cohérence narrative\n"
            f"Langue: Français"
        )
        
        # Construire la conversation avec les posts
        conversations = [
            {
                "role": "system",
                "content": system_content
            }
        ]
        
        # Ajouter les posts comme échanges utilisateur/assistant
        for i, post in enumerate(posts):
            # Alterner entre user et assistant (simuler un RP)
            role = "user" if i % 2 == 0 else "assistant"
            
            conversations.append({
                "role": role,
                "content": post['content']
            })
        
        # Ajouter les métadonnées
        metadata = {
            "campaign_id": campaign.get('id'),
            "campaign_title": campaign.get('title'),
            "universe": campaign.get('universe'),
            "system": campaign.get('system'),
            "author": campaign.get('author'),
            "post_count": len(posts),
            "context_index": context_index,
            "source": "jdRoll",
            "scraped_at": campaign.get('scraped_at', datetime.now().isoformat())
        }
        
        return {
            "conversations": conversations,
            "metadata": metadata
        }
    
    def convert_to_jsonl(self, campaigns: List[Dict]) -> str:
        """Convertir toutes les campagnes en JSONL"""
        print("\n🔄 Conversion en JSONL...")
        
        jsonl_file = f"{self.final_dir}/jdroll_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        total_posts = 0
        total_conversations = 0
        success_count = 0
        skipped_count = 0
        
        print(f"\n📝 Traitement de {len(campaigns)} campagnes...")
        
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for idx, campaign in enumerate(campaigns, 1):
                # Charger HTML
                html_file = campaign.get('html_file')
                if not html_file:
                    skipped_count += 1
                    continue
                
                html = self.load_html(html_file)
                if not html:
                    skipped_count += 1
                    continue
                
                # Extraire les posts
                posts = self.extract_post_data(html, campaign.get('id', 'unknown'))
                
                if not posts:
                    # Pas de posts, on peut quand même créer un échantillon
                    # à partir de la description
                    if campaign.get('description'):
                        posts = [{
                            'author': 'System',
                            'content': campaign['description'][:500],
                            'type': 'context'
                        }]
                    else:
                        skipped_count += 1
                        continue
                
                # Générer la conversation
                conversation = self.generate_conversation(
                    campaign, posts, html, idx
                )
                
                if conversation:
                    total_posts += len(conversation['conversations']) - 1
                    total_conversations += 1
                    success_count += 1
                    
                    # Écrire en JSONL (1 objet par ligne)
                    f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
        
        print(f"✅ {success_count}/{len(campaigns)} campagnes converties")
        print(f"📝 {total_conversations} conversations créées")
        print(f"📄 {total_posts} messages au total")
        print(f"\n📁 Fichier: {jsonl_file}")
        
        return jsonl_file
    
    def generate_statistics(self, campaigns: List[Dict], jsonl_file: str):
        """Générer un rapport de statistiques"""
        print("\n📊 Générations des statistiques...")
        
        # Statistiques des campagnes
        titles = [c.get('title', 'Inconnu') for c in campaigns]
        universes = [c.get('universe', 'Inconnu') for c in campaigns]
        systems = [c.get('system', 'Inconnu') for c in campaigns]
        
        # Univers les plus fréquents
        universe_counts = {}
        for u in universes:
            universe_counts[u] = universe_counts.get(u, 0) + 1
        
        # Systèmes les plus fréquents
        system_counts = {}
        for s in systems:
            system_counts[s] = system_counts.get(s, 0) + 1
        
        # Stats du fichier JSONL
        jsonl_size = os.path.getsize(jsonl_file) / (1024 * 1024)  # MB
        
        # Rapport
        report = {
            "report_date": datetime.now().isoformat(),
            "source_file": os.path.basename(jsonl_file),
            "statistics": {
                "total_campaigns": len(campaigns),
                "converted_campaigns": success_count if 'success_count' in dir() else 0,
                "total_conversations": total_conversations if 'total_conversations' in dir() else 0,
                "total_messages": total_posts if 'total_posts' in dir() else 0,
                "file_size_mb": round(jsonl_size, 2)
            },
            "universes": {k: v for k, v in sorted(universe_counts.items(), key=lambda x: -x[1])[:10]},
            "systems": {k: v for k, v in sorted(system_counts.items(), key=lambda x: -x[1])[:10]},
            "campaigns_sample": [
                {
                    "id": c.get('id'),
                    "title": c.get('title', 'Inconnu')[:60],
                    "universe": c.get('universe', 'Inconnu'),
                    "system": c.get('system', 'Inconnu')
                }
                for c in campaigns[:5]
            ]
        }
        
        # Sauvegarder le rapport
        stats_file = f"{self.final_dir}/statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Rapport sauvegardé: {stats_file}")
        
        # Afficher un résumé
        print("\n" + "="*80)
        print("📊 RAPPORT FINAL")
        print("="*80)
        print(f"Campagnes traitées: {report['statistics']['total_campaigns']}")
        print(f"Conversations créées: {report['statistics']['total_conversations']}")
        print(f"Messages totaux: {report['statistics']['total_messages']}")
        print(f"Taille du fichier: {report['statistics']['file_size_mb']} MB")
        
        print(f"\n🌍 Univers les plus fréquents:")
        for universe, count in list(report['universes'].items())[:5]:
            print(f"   • {universe}: {count}")
        
        print(f"\n🎲 Systèmes de jeu:")
        for system, count in list(report['systems'].items())[:5]:
            print(f"   • {system}: {count}")
        
        print("\n📋 Exemples de campagnes:")
        for camp in report['campaigns_sample']:
            print(f"   • {camp['title'][:50]}")
            print(f"     {camp['universe']} / {camp['system']}")
        
        print("="*80)
    
    def run(self):
        """Exécuter la conversion complète"""
        # 1. Charger les données
        result = self.load_campaigns()
        if not result:
            return
        
        campaigns, source_file = result
        
        # 2. Convertir en JSONL
        jsonl_file = self.convert_to_jsonl(campaigns)
        
        # 3. Générer les statistiques
        self.generate_statistics(campaigns, jsonl_file)
        
        print("\n" + "="*80)
        print("✅ CONVERSION TERMINÉE")
        print("="*80)
        print(f"\n📁 Données prêtes pour le fine-tuning:")
        print(f"   • JSONL: {jsonl_file}")
        print(f"   • Stats: {self.final_dir}/statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        print(f"   • Données originales: {source_file}")
        print("\n🚀 Prêt pour le fine-tuning avec Axolotl !")
        print("="*80)


if __name__ == "__main__":
    DataConverter().run()
