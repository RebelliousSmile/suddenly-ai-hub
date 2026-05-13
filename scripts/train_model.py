#!/usr/bin/env python3
"""
Fine-tuning JDR - Script de training Axolotl
Date: 2026-05-13
Objectif : Entraîner un modèle Mistral/Mixtral sur les données JDROLL

Principes AIDD :
- TDD : Tests avant implémentation
- DRY : Code modulaire
- YAGNI : Minimal viable
"""

import os
import sys
import yaml
import json
import time
from datetime import datetime
from pathlib import Path

# =========================================
# CONFIGURATION
# =========================================

CONFIG_DIR = "/home/user/suddenly-ai-hub/configs"
DATA_DIR = "/home/user/suddenly-ai-hub/data"
LOGS_DIR = "/home/user/suddenly-ai-hub/logs"
OUTPUT_DIR = "/home/user/suddenly-ai-hub/output"

# Modèles disponibles
MODELS = {
    "mistral_7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "vram": "16GB",
        "lora_rank": 64,
    },
    "mixtral_8x7b": {
        "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "vram": "48GB",
        "lora_rank": 32,
    },
}

# =========================================
# CLASSE DE TRAINING
# =========================================

class JDRFineTuning:
    """Fine-tuning d'un modèle JDR avec Axolotl"""
    
    def __init__(self, model="mistral_7b", dataset="data/final"):
        self.model_key = model
        self.model_config = MODELS[model]
        self.dataset_path = dataset
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("="*80)
        print("🎯 FINE-TUNING JDROLL - MODEL")
        print("="*80)
        print(f"Modèle: {self.model_config['name']}")
        print(f"VRAM requise: {self.model_config['vram']}")
        print(f"LoRA rank: {self.model_config['lora_rank']}")
        print(f"Run ID: {self.run_id}")
        print("="*80)
    
    def check_prerequisites(self):
        """Vérifier les prérequis"""
        print("\n🔍 Vérification des prérequis...")
        
        checks = {}
        
        # Dataset
        dataset_files = list(Path(self.dataset_path).glob("*.jsonl"))
        checks["dataset_exists"] = len(dataset_files) > 0
        checks["dataset_count"] = len(dataset_files)
        
        if not checks["dataset_exists"]:
            print("❌ Dataset JSONL non trouvé")
            return False
        
        print(f"✅ Dataset trouvé: {checks['dataset_count']} fichier(s)")
        
        # Output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print("✅ Output directory créé")
        
        # Config directory
        os.makedirs(CONFIG_DIR, exist_ok=True)
        print("✅ Config directory prêt")
        
        return True
    
    def create_axolotl_config(self):
        """Créer le fichier de configuration Axolotl"""
        print("\n📝 Création de la configuration Axolotl...")
        
        config = {
            "base_model": self.model_config["name"],
            "base_model_config": self.model_config["name"],
            
            # Dataset
            "dataset": {
                "type": "sharegpt",
                "datasets": [
                    {
                        "path": self.dataset_path,
                        "type": "sharegpt",
                    }
                ],
            },
            
            # Training parameters
            "lr": 2e-4,
            "lr_scheduler": "cosine",
            "warmup_ratio": 0.03,
            "epochs": 3,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            
            # LoRA parameters
            "lora": {
                "r": self.model_config["lora_rank"],
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
            
            # Quantization
            "load_in_4bit": True,
            "load_in_8bit": False,
            "nf4": True,
            
            # Gradient checkpointing
            "gradient_checkpointing": True,
            ".flash_attention": True,
            
            # Logging
            "log_level": "info",
            "logging_steps": 10,
            
            # Outputs
            "output_dir": f"{OUTPUT_DIR}/fine_tuned_{self.run_id}",
            "evaluation_strategy": "steps",
            "eval_steps": 100,
            "save_strategy": "steps",
            "save_steps": 100,
            "save_total_limit": 2,
            
            # Optimization
            "bf16": True,
            "fp16": False,
            "tf32": True,
            "weight_decay": 0.01,
            
            # Warmup
            "warmup_steps": 10,
            
            # Seed
            "seed": 42,
        }
        
        config_file = f"{CONFIG_DIR}/axolotl_config_{self.run_id}.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Configuration sauvegardée: {config_file}")
        return config_file
    
    def run_training(self, config_file):
        """Lancer le training"""
        print(f"\n🚀 Lancement du training...")
        print("="*80)
        
        try:
            # Lancer Axolotl
            import subprocess
            
            result = subprocess.run(
                ["axolotl", "--config", config_file],
                cwd="/home/user/suddenly-ai-hub",
                capture_output=True,
                text=True,
                timeout=36000  # 10 heures max
            )
            
            if result.returncode == 0:
                print("✅ Training terminé avec succès")
            else:
                print(f"❌ Training terminé avec erreur (code {result.returncode})")
                print(result.stderr)
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("⏰ Timeout - Le training tourne toujours")
            return True
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False
    
    def monitor_training(self, config_file):
        """Monitorer le training en temps réel"""
        print(f"\n📊 Monitoring du training...")
        
        logs_file = f"{LOGS_DIR}/training_{self.run_id}.log"
        
        with open(logs_file, 'w', encoding='utf-8') as log:
            log.write(f"Training started: {datetime.now().isoformat()}\n")
            log.write(f"Config: {config_file}\n")
            log.write(f"Model: {self.model_config['name']}\n")
            log.write("="*80 + "\n")
        
        print(f"✅ Logs sauvegardés: {logs_file}")
        print("\nLe training est lancé en arrière-plan...")
        print("Vous pouvez surveiller les logs avec:")
        print(f"  tail -f {logs_file}")
    
    def get_summary(self):
        """Générer un résumé"""
        print("\n" + "="*80)
        print("📋 RÉSUMÉ DU FINE-TUNING")
        print("="*80)
        
        summary = {
            "run_id": self.run_id,
            "model": self.model_config["name"],
            "vram": self.model_config["vram"],
            "lora_rank": self.model_config["lora_rank"],
            "config_file": f"{CONFIG_DIR}/axolotl_config_{self.run_id}.yaml",
            "output_dir": f"{OUTPUT_DIR}/fine_tuned_{self.run_id}",
            "status": "pending",
            "started_at": datetime.now().isoformat(),
        }
        
        summary_file = f"{LOGS_DIR}/training_summary_{self.run_id}.json"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Run ID: {self.run_id}")
        print(f"Modèle: {self.model_config['name']}")
        print(f"VRAM requise: {self.model_config['vram']}")
        print(f"LoRA rank: {self.model_config['lora_rank']}")
        print(f"Config: {summary_file}")
        print("="*80)
        
        return summary


# =========================================
# POINT D'ENTRÉE
# =========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuning JDR - Axolotl")
    parser.add_argument("--model", choices=["mistral_7b", "mixtral_8x7b"], default="mistral_7b")
    parser.add_argument("--dataset", default="data/final")
    parser.add_argument("--dry-run", action="store_true", help="Vérifier sans lancer")
    
    args = parser.parse_args()
    
    # Créer l'instance
    tuner = JDRFineTuning(model=args.model, dataset=args.dataset)
    
    # Vérifier les prérequis
    if not tuner.check_prerequisites():
        print("❌ Prérequis non satisfaits - Arrêt")
        sys.exit(1)
    
    # Créer la configuration
    config_file = tuner.create_axolotl_config()
    
    # Afficher le résumé
    summary = tuner.get_summary()
    
    # Lancer le training ou dry-run
    if args.dry_run:
        print("\n🔍 MODE DRY-RUN - Pas de training effectué")
    else:
        tuner.monitor_training(config_file)
        # Uncomment pour lancer le training :
        # tuner.run_training(config_file)
    
    print("\n✅ Setup terminé !")
    print("\nPour lancer le training:")
    print(f"  axolotl --config {config_file}")
    print("\nPour monitorer:")
    print(f"  tail -f {LOGS_DIR}/training_{summary['run_id']}.log")
