#!/usr/bin/env python3
"""
Ren'Py Parser Tests — Validation du parseur .rpy et du convertisseur

Usage:
    python scripts/crawl_rpv/test_rpv_parsers.py
    python scripts/crawl_rpv/test_rpv_parsers.py --verbose
"""

import json
import unittest
import sys
import os

# Ajouter le répertoire scripts au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_dialogues import RPYParser, DialogueConverter, NameAnonymizer


class TestRPYParser(unittest.TestCase):
    """Tests du parseur de fichiers .rpy."""
    
    def setUp(self):
        self.parser = RPYParser()
    
    def test_simple_dialogue(self):
        """Test dialogue simple."""
        content = '''label cafe_ouverture
show elise happy
el "Bonjour ! Tu es là depuis longtemps ?"
'''
        result = self.parser.parse_file(content)
        
        self.assertEqual(result["total_dialogues"], 1)
        self.assertEqual(len(result["scenes"]), 1)
        self.assertEqual(result["scenes"][0]["label"], "cafe_ouverture")
        self.assertIn("el", result["total_characters"])
    
    def test_multi_dialogue_scene(self):
        """Test scène avec plusieurs dialogues."""
        content = '''label cafe_ouverture
show elise happy
el "Bonjour ! Tu es là depuis longtemps ?"
show alex neutral
alex "Salut Elise, un café ?"
el "Ah, Alex ! Justement."
'''
        result = self.parser.parse_file(content)
        
        self.assertEqual(result["total_dialogues"], 3)
        self.assertEqual(len(result["scenes"][0]["dialogues"]), 3)
        self.assertIn("el", result["total_characters"])
        self.assertIn("alex", result["total_characters"])
    
    def test_narration(self):
        """Test narration."""
        content = '''label intro
Le chevalier entre dans la caverne.
n "Le vent soufflait fort ce soir-là."
n "Il savait que le dragon l'attendait."
'''
        result = self.parser.parse_file(content)
        
        self.assertEqual(len(result["scenes"][0]["narrations"]), 2)
    
    def test_show_scene(self):
        """Test instructions show/scene."""
        content = '''label battle
scene forest_dark
show hero angry
hero "Viens par ici, monstre !"
show monster
'''
        result = self.parser.parse_file(content)
        
        context = result["scenes"][0]["context"]
        self.assertTrue(any("forest_dark" in c for c in context))
        self.assertTrue(any("show hero angry" in c for c in context))
        self.assertTrue(any("show monster" in c for c in context))
    
    def test_empty_file(self):
        """Test fichier sans dialogue."""
        content = '''label empty
default x = 5
return
'''
        result = self.parser.parse_file(content)
        
        self.assertEqual(result["total_dialogues"], 0)
    
    def test_multiple_scenes(self):
        """Test avec plusieurs scènes."""
        content = '''label scene1
a "Première scène"

label scene2
b "Deuxième scène"
'''
        result = self.parser.parse_file(content)
        
        self.assertEqual(len(result["scenes"]), 2)
        self.assertEqual(result["total_dialogues"], 2)


class TestDialogueConverter(unittest.TestCase):
    """Tests du convertisseur Ren'Py → JSONL."""
    
    def test_convert_scene(self):
        """Test conversion d'une scène simple."""
        converter = DialogueConverter(genre="Romance", situation="Relation")
        
        scene = {
            "label": "cafe",
            "characters": ["elise"],
            "dialogues": [
                {"character": "elise", "text": "Bonjour !"},
                {"character": "alex", "text": "Salut !"}
            ],
            "narrations": [],
            "context": []
        }
        
        result = converter.convert_scene(scene)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertIn("Romance", result["messages"][0]["content"])
        self.assertEqual(len(result["messages"]), 3)  # system + 2 dialogues
        
        # Vérifier alternance des rôles
        self.assertEqual(result["messages"][1]["role"], "user")
        self.assertEqual(result["messages"][2]["role"], "assistant")
    
    def test_empty_scene(self):
        """Test conversion d'une scène sans dialogue."""
        converter = DialogueConverter()
        
        scene = {
            "label": "empty",
            "characters": [],
            "dialogues": [],
            "narrations": [],
            "context": []
        }
        
        result = converter.convert_scene(scene)
        self.assertIsNone(result)
    
    def test_anonymize_names(self):
        """Test anonymisation des noms."""
        converter = DialogueConverter()
        
        scene = {
            "label": "test",
            "characters": ["john", "mary"],
            "dialogues": [
                {"character": "john", "text": "Hello"},
                {"character": "mary", "text": "Hi"}
            ],
            "narrations": [],
            "context": []
        }
        
        result = converter.convert_scene(scene)
        
        # Les noms doivent être anonymisés
        self.assertNotIn("john", str(result["metadata"]["characters"]))
        self.assertNotIn("mary", str(result["metadata"]["characters"]))
        self.assertIn("Personnage", str(result["metadata"]["characters"]))


class TestNameAnonymizer(unittest.TestCase):
    """Tests de l'anonymiseur de noms."""
    
    def setUp(self):
        self.anonymizer = NameAnonymizer()
    
    def test_anonymize_single_name(self):
        """Test anonymisation d'un seul nom."""
        result = self.anonymizer.anonymize_name("john")
        self.assertEqual(result, "PNJ1")
    
    def test_same_name_consistent(self):
        """Test cohérence avec le même nom."""
        name1 = self.anonymizer.anonymize_name("john")
        name2 = self.anonymizer.anonymize_name("john")
        self.assertEqual(name1, name2)
    
    def test_different_names(self):
        """Test noms différents."""
        name1 = self.anonymizer.anonymize_name("john")
        name2 = self.anonymizer.anonymize_name("mary")
        self.assertNotEqual(name1, name2)


class TestCorpusIntegration(unittest.TestCase):
    """Tests d'intégration finaux."""
    
    def test_full_pipeline(self):
        """Test pipeline complet: parse → convertir → JSONL."""
        content = '''label battle
show hero angry
hero "Viens par ici, monstre !"
show monster
monster "Je te vais vaincre !"
n "Le combat commence."
'''
        parser = RPYParser()
        result = parser.parse_file(content)
        
        self.assertEqual(result["total_dialogues"], 2)
        
        converter = DialogueConverter(genre="Fantastique", situation="Combat")
        scene = result["scenes"][0]
        converted = converter.convert_scene(scene)
        
        self.assertIsNotNone(converted)
        self.assertEqual(len(converted["messages"]), 3)  # system + 2 dialogues
        
        # Vérifier structure JSONL
        jsonl_line = json.dumps(converted, ensure_ascii=False)
        parsed_back = json.loads(jsonl_line)
        self.assertIn("messages", parsed_back)


def run_tests():
    """Exécute tous les tests et affiche les résultats."""
    # Préparer les arguments
    argv = ["test"]
    if "--verbose" in sys.argv:
        unittest.main(verbosity=2)
    else:
        unittest.main()


if __name__ == "__main__":
    run_tests()
