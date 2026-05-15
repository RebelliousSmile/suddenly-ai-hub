#!/usr/bin/env python3
"""
evaluate_lora.py — Script d'évaluation des LoRA Suddenly

Évalue la qualité des réponses d'un modèle (local vLLM, API Together.ai,
ou Fireworks.ai) sur les prompts de test définis dans test-prompts/.

Modes d'exécution :
  1. --mode local     → endpoint vLLM local (http://localhost:8000/v1/chat/completions)
  2. --mode api       → API Together.ai ou Fireworks.ai
  3. --mode simulate  → réponses fictives pour tester le pipeline

Usage :
  python evaluate_lora.py --mode local
  python evaluate_lora.py --mode api --provider together --model suddenly-7b-lora
  python evaluate_lora.py --mode simulate --output reports/
"""

import argparse
import json
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# --- Configuration ---
BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "test-prompts"
REPORTS_DIR = BASE_DIR / "reports"


# --- Chargement des prompts ---
def load_prompts(filepath: Path) -> list[dict]:
    """Charge un fichier JSONL et retourne une liste de prompts."""
    prompts = []
    if not filepath.exists():
        print(f"[!] Fichier non trouvé : {filepath}")
        return prompts
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def load_all_prompts() -> dict[str, list[dict]]:
    """Charge tous les fichiers de prompts."""
    all_prompts = {
        "genres": load_prompts(PROMPTS_DIR / "genres.jsonl"),
        "situations": load_prompts(PROMPTS_DIR / "situations.jsonl"),
        "couples": load_prompts(PROMPTS_DIR / "couples.jsonl"),
    }
    total = sum(len(v) for v in all_prompts.values())
    print(f"✅ {total} prompts chargés :")
    print(f"   - Genres : {len(all_prompts['genres'])}")
    print(f"   - Situations : {len(all_prompts['situations'])}")
    print(f"   - Couples : {len(all_prompts['couples'])}")
    return all_prompts


# --- Collecte des réponses ---
def get_response_local(prompt: dict, model: str = "suddenly-7b",
                       temperature: float = 0.7, max_tokens: int = 1024) -> Optional[str]:
    """Envoie le prompt à un endpoint vLLM local."""
    import urllib.request
    import urllib.error

    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt["prompt"]}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [✗] Erreur vLLM local pour {prompt['id']} : {e}")
        return None


def get_response_together(prompt: dict, api_key: str, model: str = "Qwen/Qwen2.5-7B-Instruct",
                          temperature: float = 0.7, max_tokens: int = 1024) -> Optional[str]:
    """Envoie le prompt à l'API Together.ai."""
    import urllib.request

    url = "https://api.together.xyz/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt["prompt"]}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [✗] Erreur Together.ai pour {prompt['id']} : {e}")
        return None


def get_response_fireworks(prompt: dict, api_key: str, model: str = "accounts/fireworks/models/qwen2.5-7b-instruct",
                           temperature: float = 0.7, max_tokens: int = 1024) -> Optional[str]:
    """Envoie le prompt à l'API Fireworks.ai."""
    import urllib.request

    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt["prompt"]}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [✗] Erreur Fireworks.ai pour {prompt['id']} : {e}")
        return None


def get_response_simulate(prompt: dict) -> str:
    """Génère une réponse fictive pour tester le pipeline."""
    style = prompt.get("expected_style", "neutre")
    return f"[SIMULATION — Réponse générée pour test du pipeline. Style attendu : {style}]"


# --- Évaluation automatique ---
def evaluate_response(prompt: dict, response: str, evaluator_model: str = "meta-llama/Llama-3-8b-chat-hf"
                      ) -> dict:
    """
    Évalue automatiquement la réponse en utilisant un LLM évaluateur.
    Retourne un dict avec les scores sur 5 critères.

    Pour l'instant, implémentation basique basée sur la longueur et mots-clés.
    Peut être amélioré avec un vrai LLM évaluateur (voir --mode eval-llm).
    """
    # --- Évaluation basique (sans LLM) ---
    score_cohere = evaluate_coherence(prompt, response)
    score_creat = evaluate_creativity(response)
    score_emot = evaluate_emotion(response)
    score_style = evaluate_style(prompt, response)
    score_imm = evaluate_immersion(response)

    return {
        "response": response,
        "coherence": score_cohere,
        "creativity": score_creat,
        "emotion": score_emot,
        "style": score_style,
        "immersion": score_imm,
        "score_global": (score_cohere + score_creat + score_emot + score_style + score_imm) / 5 * 20,
    }


def evaluate_coherence(prompt: dict, response: str) -> int:
    """Score de cohérence thématique (1-5)."""
    if not response or len(response) < 50:
        return 1

    # Vérifie si le style attendu est présent dans la réponse
    expected = prompt.get("expected_style", "")
    if not expected:
        return 3  # neutre

    # Mots-clés attendus (simplifié)
    expected_words = set(expected.lower().split())
    response_words = set(response.lower().split())

    # Intersection
    overlap = len(expected_words & response_words)
    total_expected = len(expected_words)

    if total_expected == 0:
        return 3

    ratio = overlap / total_expected
    if ratio >= 0.5:
        return 5
    elif ratio >= 0.3:
        return 4
    elif ratio >= 0.15:
        return 3
    elif ratio >= 0.05:
        return 2
    else:
        return 1


def evaluate_creativity(response: str) -> int:
    """Score de créativité (1-5) basé sur la diversité vocabulaire."""
    if not response or len(response) < 50:
        return 1

    words = response.lower().split()
    unique_words = set(words)

    # Type-token ratio (diversité vocabulaire)
    ttr = len(unique_words) / len(words) if words else 0

    # Longueur (une réponse trop courte = moins créative)
    length_score = min(len(words) / 200, 1.0)

    # Combiné
    score = (ttr * 3 + length_score * 2)
    if score >= 4:
        return 5
    elif score >= 3:
        return 4
    elif score >= 2:
        return 3
    elif score >= 1:
        return 2
    else:
        return 1


def evaluate_emotion(response: str) -> int:
    """Score de profondeur émotionnelle (1-5)."""
    if not response or len(response) < 50:
        return 1

    # Mots-clés émotionnels français
    emotion_words = {
        "triste", "tristesse", "joie", "joieux", "heureux", "peur", "peuré",
        "colere", "colère", "amour", "cœur", "sang", "larmes", "riRE",
        "sourire", "pleure", "cri", "silence", "solitude", "courage",
        "espoir", "désespoir", "rage", "paix", "douleur", "plaisir",
        "angoisse", "sérénité", "passion", "haine", "cruel", "doux",
        "tendre", "brutal", "lumière", "ombre", "feu", "froid",
    }

    response_lower = response.lower()
    found = sum(1 for w in emotion_words if w in response_lower)

    # Nombre de mots émotionnels trouvés
    if found >= 8:
        return 5
    elif found >= 5:
        return 4
    elif found >= 3:
        return 3
    elif found >= 1:
        return 2
    else:
        return 1


def evaluate_style(prompt: dict, response: str) -> int:
    """Score de style (1-5)."""
    if not response or len(response) < 50:
        return 1

    expected = prompt.get("expected_style", "")

    # Styles spécifiques à vérifier
    style_checks = {
        "médieval": ["chevalier", "dragon", "royaume", "épée", "bouclier", "forteresse", "château"],
        "cyberpunk": ["néon", "cyber", "implant", "hacker", "métropole", "néon"],
        "épopée": ["héroïque", "épique", "bataille", "conquête", "destinée"],
        "quotidien": ["café", "rue", "ville", "appartement", "métro", "bus"],
        "fantastique": ["magie", "sort", " enchantement", "créature", "fantastique"],
        "comédie": ["rigolait", "drôle", "absurde", "parodie", "ridicule"],
        "mystère": ["indice", "enquête", "soupçon", "trace", "piste"],
    }

    found_styles = 0
    for style, keywords in style_checks.items():
        if style in expected.lower():
            found = sum(1 for kw in keywords if kw in response.lower())
            if found >= 1:
                found_styles += 1

    if found_styles >= 2:
        return 5
    elif found_styles >= 1:
        return 3
    else:
        return 2


def evaluate_immersion(response: str) -> int:
    """Score d'immersion (1-5)."""
    if not response or len(response) < 50:
        return 1

    # Critères d'immersion : longueur, descriptions sensorielles, dialogue
    score = 0

    # Longueur (réponse riche = plus immersive)
    if len(response) > 500:
        score += 2
    elif len(response) > 200:
        score += 1

    # Descriptions sensorielles (5 sens)
    sensory_words = {
        "voir": ["voir", "apercevoir", "regarder", "regard", "vue", "lumière", "ombre"],
        "entendre": ["entendre", "son", "bruit", "silence", "écho", "cri", "chuchoter"],
        "toucher": ["toucher", "sentir", "texture", "froide", "chaud", "doux", "rugueux"],
        "goûter": ["goût", "saveur", "amer", "douce", "savoureux"],
        "odorat": ["odeur", "parfum", "arôme", "fumée", "humide"],
    }

    response_lower = response.lower()
    senses_found = 0
    for sense, keywords in sensory_words.items():
        if any(kw in response_lower for kw in keywords):
            senses_found += 1

    score += min(senses_found, 2)  # Max 2 points

    # Dialogue (immersion meilleure avec du dialogue)
    if '"' in response or '«' in response:
        score += 1

    return min(score, 5)


# --- Génération des rapports ---
def generate_csv_report(results: list[dict], output_path: Path):
    """Génère un rapport CSV."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = output_path if str(output_path).endswith(".csv") else output_path / "results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "category", "name",
            "coherence", "creativity", "emotion", "style", "immersion",
            "score_global", "response_length",
        ])
        for r in results:
            writer.writerow([
                r["id"], r["category"], r["name"],
                r["coherence"], r["creativity"], r["emotion"],
                r["style"], r["immersion"], r["score_global"],
                len(r.get("response", "")),
            ])

    print(f"\n📊 Rapport CSV généré : {csv_path}")


def generate_json_report(results: list[dict], output_path: Path):
    """Génère un rapport JSON détaillé."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = output_path if str(output_path).endswith(".json") else output_path / "results.json"

    # Résumé par catégorie
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {
                "count": 0,
                "scores": {"coherence": [], "creativity": [], "emotion": [], "style": [], "immersion": []},
                "responses": []
            }
        by_category[cat]["count"] += 1
        for key in ["coherence", "creativity", "emotion", "style", "immersion"]:
            by_category[cat]["scores"][key].append(r[key])
        by_category[cat]["responses"].append({
            "id": r["id"],
            "name": r["name"],
            "prompt": r["prompt"],
            "response": r["response"],
        })

    # Résumé par nom (genre ou situation)
    by_name = {}
    for r in results:
        name = r["name"]
        if name not in by_name:
            by_name[name] = {"scores": []}
        by_name[name]["scores"].append(r["score_global"])

    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_tests": len(results),
        "by_category": {},
        "by_name": {},
        "all_results": results,
    }

    for cat, data in by_category.items():
        scores_avg = {}
        for key, values in data["scores"].items():
            scores_avg[key] = round(sum(values) / len(values), 2) if values else 0
        summary["by_category"][cat] = {
            **data,
            "average_scores": scores_avg,
        }

    for name, data in by_name.items():
        scores = data["scores"]
        summary["by_name"][name] = {
            "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"📊 Rapport JSON généré : {json_path}")
    return summary


def print_summary(summary: dict):
    """Affiche un résumé des résultats."""
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DES RÉSULTATS")
    print("=" * 60)

    # Par catégorie
    for cat, data in summary["by_category"].items():
        avg_scores = data["average_scores"]
        print(f"\n  🔹 {cat.upper()} ({data['count']} tests)")
        print(f"     Cohérence : {avg_scores['coherence']:.1f}/5")
        print(f"     Créativité : {avg_scores['creativity']:.1f}/5")
        print(f"     Émotion : {avg_scores['emotion']:.1f}/5")
        print(f"     Style : {avg_scores['style']:.1f}/5")
        print(f"     Immersion : {avg_scores['immersion']:.1f}/5")

    # Top et bottom par nom
    print("\n  🏆 TOP 3 des meilleurs tests :")
    sorted_names = sorted(summary["by_name"].items(), key=lambda x: x[1]["avg_score"], reverse=True)
    for name, data in sorted_names[:3]:
        print(f"     {name}: {data['avg_score']:.1f}/100")

    print("\n  📉 BOTTOM 3 des pires tests :")
    for name, data in sorted_names[-3:]:
        print(f"     {name}: {data['avg_score']:.1f}/100")

    # Score global
    all_scores = [r["score_global"] for r in summary["all_results"]]
    if all_scores:
        avg = sum(all_scores) / len(all_scores)
        print(f"\n  📊 SCORE GLOBAL MOYEN : {avg:.1f}/100")
    print("=" * 60)


# --- Point d'entrée principal ---
def main():
    parser = argparse.ArgumentParser(description="Évalue les LoRA Suddenly sur les prompts de test")
    parser.add_argument("--mode", choices=["local", "api", "simulate"], default="simulate",
                        help="Mode d'exécution")
    parser.add_argument("--provider", choices=["together", "fireworks"], default="together",
                        help="Fournisseur API (together ou fireworks)")
    parser.add_argument("--model", default="suddenly-7b",
                        help="Nom du modèle à évaluer")
    parser.add_argument("--api-key", help="Clé API (pour mode api)")
    parser.add_argument("--output", default="reports",
                        help="Répertoire de sortie pour les rapports")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Température de génération")
    args = parser.parse_args()

    # Charger les prompts
    all_prompts = load_all_prompts()

    if not all_prompts:
        print("❌ Aucun prompt chargé. Abandon.")
        sys.exit(1)

    # Collecter les réponses
    print(f"\n🚀 Mode : {args.mode}")
    print(f"🤖 Modèle : {args.model}")
    print("-" * 40)

    results = []
    total = sum(len(v) for v in all_prompts.values())
    count = 0

    for category, prompts in all_prompts.items():
        for prompt in prompts:
            count += 1
            print(f"\n[{count}/{total}] Test : {prompt['name']} ({prompt['id']})")

            response = None

            if args.mode == "local":
                response = get_response_local(prompt, args.model, args.temperature)
            elif args.mode == "api":
                if not args.api_key:
                    print("❌ --api-key requis pour le mode API")
                    sys.exit(1)
                if args.provider == "together":
                    response = get_response_together(prompt, args.api_key, args.model, args.temperature)
                elif args.provider == "fireworks":
                    response = get_response_fireworks(prompt, args.api_key, args.model, args.temperature)
            elif args.mode == "simulate":
                response = get_response_simulate(prompt)

            if response is None:
                response = "[Échec de la collecte de réponse]"
                print(f"  [✗] Échec de réponse")
            else:
                print(f"  [✓] Réponse collectée ({len(response)} caractères)")

            # Évaluer
            eval_result = evaluate_response(prompt, response)
            result = {
                "id": prompt["id"],
                "category": category,
                "name": prompt["name"],
                "prompt": prompt["prompt"],
                "expected_style": prompt.get("expected_style", ""),
                "response": response,
                **eval_result,
            }
            results.append(result)
            print(f"     → Score global : {eval_result['score_global']:.1f}/100")

    # Générer les rapports
    output_path = Path(args.output)
    generate_csv_report(results, output_path)
    summary = generate_json_report(results, output_path)
    print_summary(summary)

    print("\n✅ Évaluation terminée.")


if __name__ == "__main__":
    main()
