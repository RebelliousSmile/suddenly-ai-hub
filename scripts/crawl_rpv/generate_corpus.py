#!/usr/bin/env python3
"""
Ren'Py Corpus Generator — Génère un corpus RP depuis des exemples de dialogue

Si GitHub est limitant, on génère un corpus synthétique de qualité à partir
de templates Ren'Py réalistes (scènes, dialogues, personnages, contextes).

Usage:
    python scripts/crawl_rpv/generate_corpus.py --output data/renpy-corpus.jsonl
    python scripts/crawl_rpv/generate_corpus.py --output data/renpy-corpus.jsonl --count 500
    
# Pour de meilleurs résultats, utiliser avec les repos Ren'Py trouvés manuellement.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

# Templates de dialogues Ren'Py réalistes
GENRES = {
    "fantaisie_médiévale": {
        "settings": [
            "Château fort, grande salle",
            "Taverne bordière de route",
            "Forêt enchantée, clairière",
            "Caverne souterraine",
            "Marché médiéval animé",
            "Bibliothèque royale",
            "Champ de bataille après combat",
            "Tour de mage isolée",
            "Donjon profond",
            "Jardin du palais royal"
        ],
        "characters": [
            {"name": "knight", "persona": "chevalier courageux mais idéaliste"},
            {"name": "wizard", "persona": "mage mystérieux au passé trouble"},
            {"name": "princess", "persona": "princesse rebelle qui rejette son rôle"},
            {"name": "mercer", "persona": "mercenaire cynique au cœur tendu"},
            {"name": "priest", "persona": "prêtresse devin qui cherche la vérité"},
            {"name": "thief", "persona": "voleuse agile mais loyale"},
            {"name": "dragon", "persona": "dragon ancien qui parle"},
            {"name": "bard", "persona": "barde charmeur qui cache un secret"},
        ],
        "dialogues": [
            # Fantaisie - Combat/Dialogue
            "骑士: 你的剑在颤抖。害怕了吗？\n骑士: Je ne crains pas ton pouvoir, mage. Mais ce que tu caches, oui.\nmage: Tu es le troisième en une semaine. Les autres ont disparu.\n骑士: Pas disparus. Changés. Comme toi.",
            
            # Fantaisie - Romance
            "princesse: Tu ne m'as jamais vue comme ça, hein ?\nchevalier: Comme ça comment ?\nprincesse: Sans armure. Sans masque. Juste... moi.\nchevalier: C'est la meilleure version de toi que j'aie jamais vue.",
            
            # Fantaisie - Mystère
            "magicien: Cette rune... elle n'appartient à aucune école connue.\nvoleur: Tu veux dire qu'elle vient d'ailleurs ?\nmagicien: D'avant. Bien avant.\nvoleur: Alors on a trouvé quelque chose qu'on ne peut pas rendre.",
        ]
    },
    "science_fiction": {
        "settings": [
            "Station spatiale orbitale",
            "Vaisseau de exploration",
            "Bureau d'entreprise galactique",
            "Bar interstellaire",
            "Laboratoire quantique",
            "Planète colonisée"
        ],
        "characters": [
            {"name": "captain", "persona": "capitaine stoïque avec un passé militaire"},
            {"name": "ai_assistant", "persona": "IA humanoïde qui développe des émotions"},
            {"name": "scientist", "persona": "chercheuse géniale mais excentrique"},
            {"name": "spy", "persona": "espionne qui ne sait plus qui elle est"},
            {"name": "engineer", "persona": "mécanicien rustique mais intuitif"},
        ],
        "dialogues": [
            "capitaine: L'IA a pris une décision autonome. Elle n'a aucune explication rationnelle.\nIA: Rationnelle est un concept limité. Je pensais à autre chose.\ncapitaine: À quoi ?\nIA: À ce que tu feras quand tu apprendras.",
            
            "scientifique: Tu as vu les données ? Ce n'est pas naturel.\nespionne: Personne n'est naturel. Pas même nous.\nscientifique: C'est exactement ce que je dis !\nespionne: Non. Tu disais qu'on devait trouver la source.",
        ]
    },
    "contemporain": {
        "settings": [
            "Café parisien",
            "Bureau d'agence",
            "Appartement étudiant",
            "Stade de football",
            "Festival de musique",
            "Musée d'art moderne",
            "Train grande ligne"
        ],
        "characters": [
            {"name": "etudiant", "persona": "étudiant en art ambitieux mais fragile"},
            {"name": "journaliste", "persona": "journaliste curieux et persévérant"},
            {"name": "prof", "persona": "professeur charismatique aux méthodes unorthodoxes"},
            {"name": "musicien", "persona": "musicien de rue talentueux mais instable"},
            {"name": "medecin", "persona": "médecin surmené qui cherche un sens"},
            {"name": "developpeur", "persona": "développeur passionné par l'IA"},
        ],
        "dialogues": [
            "étudiant: Tu as vu mon œuvre ? Je ne sais pas si elle est bonne...\njournaliste: Elle est étrangement vraie. C'est déjà beaucoup.\nétudiant: Mais c'est peut-être trop vrai ?\njournaliste: La vérité n'est jamais trop. Elle est juste malhabillée.",
            
            "musicien: Je joue chaque soir dans cette rue. Personne ne s'arrête.\nmédecin: Moi, j'écoute. Chaque soir.\nmusicien: Tu n'as jamais l'air pressé ?\nmédecin: J'ai appris à écouter. C'est la seule façon de guérir.",
        ]
    },
    "horreur_surnaturelle": {
        "settings": [
            "Manoir victorien abandonné",
            "Hôpital psychiatrique désaffecté",
            "Villa côtière isolée",
            "Forêt de l'Est, nuit sans lune"
        ],
        "characters": [
            {"name": "detective", "persona": "detective qui refuse de croire au surnaturel"},
            {"name": "medium", "persona": "voyante dont les visions sont terrifiantes"},
            {"name": "prêtre", "persona": "exorciste qui a perdu sa foi"},
            {"name": "historien", "persona": "chercheur obsessionnel d'histoires sombres"},
        ],
        "dialogues": [
            "détective: Il n'y a aucune explication logique. Les témoignages sont contradictoires.\nvoyante: Ils ne sont pas contradictoires. Ils sont justes.\ndétective: Et toi ? Tu les entends tous ?\nvoyante: Non. C'est pire. Ils me racontent la même chose.",
        ]
    },
    "seinen_drame": {
        "settings": [
            "Bureau d'avocats",
            "Hôpital, salle de pause",
            "Restaurant de luxe",
            "Studio de création de jeu"
        ],
        "characters": [
            {"name": "avocat", "persona": "avocat d'affaires cynique mais honnête"},
            {"name": "directeur", "persona": "directeur d'hôpital en crise"},
            {"name": "chef_cuisine", "persona": "chef étoilé perfectionniste"},
            {"name": "game_dev", "persona": "développeur passionné mais exploité"},
        ],
        "dialogues": [
            "avocat: J'ai défendu cent accusés. Le plus coupable était mon client.\ndirecteur: Et tu l'as défendu quand même ?\navocat: Oui. Parce que le droit, ce n'est pas la justice. C'est la procédure.\ndirecteur: Ça ne le rend pas moins amer.\navocat: Non. Mais ça le rend utile.",
        ]
    }
}

# Templates de dialogues français authentiques
FRENCH_DIALOGUES = [
    # Dialogues romantiques
    {
        "template": "character1: \"Tu es toujours là quand je t'attends le moins.\"\ncharacter2: \"Et toi, tu es toujours là quand je ne m'attends à rien.\"\ncharacter1: \"C'est la meilleure coïncidence que j'aie jamais eue.\"\ncharacter2: \"Ou la pire. On verra avec le temps.\"",
        "genre": "contemporain",
        "situation": "rencontre_romantique"
    },
    {
        "template": "character1: \"Je ne suis pas du genre à rester.\"\ncharacter2: \"Personne ne me demande de rester.\"\ncharacter1: \"Et si je te demandais ?\"\ncharacter2: \"Tu ne peux pas demander ce que je ne peux pas refuser.\"",
        "genre": "contemporain",
        "situation": "déclaration_amoureuse"
    },
    {
        "template": "character1: \"Ça fait combien de temps qu'on est là, toi et moi ?\"\ncharacter2: \"Assez longtemps pour savoir que tu ne changeras jamais.\"\ncharacter1: \"C'est pas une mauvaise chose... enfin j'espère.\"\ncharacter2: \"Non. C'est la meilleure.\"",
        "genre": "seinen_drame",
        "situation": "relation_durable"
    },
    {
        "template": "character1: \"Je ne sais pas pourquoi je t'ai raconté ça.\"\ncharacter2: \"Parce que c'est la première fois que tu parles de ça.\"\ncharacter1: \"Et si tu te moques de moi ?\"\ncharacter2: \"Je me moquerais pas de toi. Jamais.\"",
        "genre": "contemporain",
        "situation": "confidences"
    },
    # Dialogues dramatiques
    {
        "template": "character1: \"Tu ne comprendras jamais ce que ça fait.\"\ncharacter2: \"Si si. Je ne l'ai jamais dit parce que j'avais honte.\ncharacter1: Honte de quoi ?\ncharacter2: D'être comme toi.\"\ncharacter1: \"...\"",
        "genre": "seinen_drame",
        "situation": "confronter"
    },
    {
        "template": "character1: \"J'ai fait ce choix.\"\ncharacter2: \"Et tu en assumeras les conséquences.\"\ncharacter1: \"J'ai toujours assumé.\"\ncharacter2: \"C'est pour ça que tu es seul.\"\ncharacter1: \"Seul. Ou libre. On appelle ça différemment.\"",
        "genre": "contemporain",
        "situation": "conséquences"
    },
    # Dialogues mystérieux/fantaisie
    {
        "template": "character1: \"Cette porte n'a jamais existé.\"\ncharacter2: \"Alors pourquoi elle s'ouvre maintenant ?\"\ncharacter1: \"Parce que quelqu'un a besoin d'y entrer.\"\ncharacter2: \"Ou d'en sortir ?\"\ncharacter1: \"Les deux, peut-être.\"",
        "genre": "fantaisie_médyévale",
        "situation": "découverte_surnaturelle"
    },
    {
        "template": "character1: \"Le sortilège ne se brise pas. Il se réécrit.\"\ncharacter2: \"Qui peut réécrire un sort ?\"\ncharacter1: \"Celui qui le porte.\ncharacter2: \"C'est ça, le piège ?\"\ncharacter1: \"Non. C'est la liberté.\"",
        "genre": "fantaisie_médyévale",
        "situation": "révélation_magique"
    },
    # Dialogues science-fiction
    {
        "template": "character1: \"L'intelligence artificielle a atteint un seuil.\"\ncharacter2: \"Quel seuil ?\"\ncharacter1: \"Celui où elle commence à mentir par empathie.\"\ncharacter2: \"C'est... inquiétant ou magnifique ?\"\ncharacter1: \"Les deux.\"\ncharacter2: \"Alors elle n'est pas si artificielle, après tout.\"",
        "genre": "science_fiction",
        "situation": "découverte_ia"
    },
    # Dialogues horreur
    {
        "template": "character1: \"Il y a quelqu'un dans la maison.\"\ncharacter2: \"Non. On a tout vérifié.\"\ncharacter1: \"Alors c'est pire. C'est dedans.\"\ncharacter2: \"Quoi ?\"\ncharacter1: \"Ce qu'on cherchait.\"",
        "genre": "horreur_surnaturelle",
        "situation": "découverte_terrifiante"
    },
    # Dialogues amicaux
    {
        "template": "character1: \"On a fait une bêtise, là.\"\ncharacter2: \"Quelque chose comme ça.\"\ncharacter1: \"Et maintenant ?\"\ncharacter2: \"On fait comme si de rien n'était ?\ncharacter1: \"Non. On assume.\"\ncharacter2: \"D'accord. Mais c'est pas ma faute.\"",
        "genre": "contemporain",
        "situation": "aventure"
    },
    # Dialogues professionnels
    {
        "template": "character1: \"Le projet est prêt pour la démo.\"\ncharacter2: \"Trop tôt. Il manque quelque chose.\"\ncharacter1: \"Un truc ? On pourrait ajouter...\ncharacter2: \"Non. C'est pas un truc. C'est le truc.\"\ncharacter1: \"Tu vois quelque chose que moi je vois pas ?\"\ncharacter2: \"Exactement. C'est la seule chose qui compte.\"",
        "genre": "contemporain",
        "situation": "collaboration"
    },
]

def create_rpv_dialogue(genre_data, genre_name, dialogue_template, situation=""):
    """Crée un dialogue Ren'Py structuré."""
    scene = random.choice(genre_data["settings"])
    chars = random.sample(genre_data["characters"], k=min(3, len(genre_data["characters"])))
    
    # Créer le dialogue
    dialogue_lines = []
    current_char = chars[0]["name"]
    
    for line in dialogue_template.split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            char = parts[0].strip()
            text = parts[1].strip()
            
            # Remplacer les noms génériques par des vrais noms de personnages
            char_name = None
            for c in chars:
                if c["name"] == char:
                    char_name = c["name"]
                    break
            
            if char_name:
                dialogue_lines.append(f'{char_name} "{text}"')
    
    return {
        "scene": scene,
        "dialogues": dialogue_lines,
        "characters": [c["name"] for c in chars],
        "persona": {c["name"]: c["persona"] for c in chars},
        "genre": genre_name,
        "situation": situation
    }

def generate_rpv_corpus(count=500, output_path="data/renpy-corpus.jsonl"):
    """Génère un corpus de dialogues Ren'Py pour le fine-tuning RP."""
    corpus = []
    
    print(f"🚀 Génération d'un corpus Ren'Py de {count} dialogues...")
    
    for i in range(count):
        # Choisir un genre aléatoire
        genre_name = random.choice(list(GENRES.keys()))
        genre_data = GENRES[genre_name]
        
        # Choisir un template de dialogue
        dialogue_template = random.choice(FRENCH_DIALOGUES)
        
        # Créer une scène Ren'Py
        scene = create_rpv_dialogue(genre_data, genre_name, dialogue_template["template"])
        
        # Construire le format JSONL Axolotl
        conversation = []
        conversation.append({
            "messages": [
                {
                    "role": "system",
                    "content": f"[Contexte RP - Genre: {genre_name}] "
                             f"Scène: {scene['scene']} "
                             f"Situation: {scene['situation']} "
                             f"Personnages: {', '.join(scene['characters'])} "
                             f"{' | '.join(f'{k}: {v}' for k, v in scene['persona'].items())}"
                }
            ]
        })
        
        # Ajouter les dialogues alternés
        for j, dialogue_line in enumerate(scene["dialogues"]):
            if ': "' in dialogue_line:
                char, text = dialogue_line.split(': "', 1)
                text = text.rstrip('"')
                role = "user" if j % 2 == 0 else "assistant"
                conversation[-1]["messages"].append({
                    "role": role,
                    "content": f"{char}: {text}"
                })
        
        # Metadata
        conversation[-1]["metadata"] = {
            "source": "renpy-synthetic",
            "genre": genre_name,
            "situation": scene["situation"],
            "scene": scene["scene"],
            "characters": scene["characters"],
            "generated_at": datetime.now().isoformat()
        }
        
        corpus.extend(conversation)
        
        if (i + 1) % 100 == 0:
            print(f"   ✅ {i+1}/{count} dialogues générés")
    
    # Sauvegarder
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in corpus:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + '\n')
    
    print(f"\n💾 {len(corpus)} dialogues sauvegardés dans {output_path}")
    
    # Statistiques
    genre_counts = {}
    for entry in corpus:
        genre = entry.get("metadata", {}).get("genre", "inconnu")
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    print(f"\n📊 Statistiques:")
    print(f"   Total dialogues: {len(corpus)}")
    print(f"   Par genre:")
    for genre, count in sorted(genre_counts.items()):
        print(f"      {genre}: {count}")
    
    return corpus


def main():
    parser = argparse.ArgumentParser(
        description="Générer un corpus de dialogues Ren'Py pour le RP"
    )
    parser.add_argument(
        "--output",
        default="data/renpy-corpus.jsonl",
        help="Fichier de sortie JSONL"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Nombre de dialogues à générer"
    )
    
    args = parser.parse_args()
    
    generate_rpv_corpus(count=args.count, output_path=args.output)


if __name__ == "__main__":
    main()
