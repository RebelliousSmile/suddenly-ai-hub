# Model Choice: Qwen2.5-7B-Instruct

**Décision** : Qwen/Qwen2.5-7B-Instruct  
**Date** : 2026-05-15  
**Contexte** : Choix du modèle de base pour `suddenly-7b` (RP en français, inference Fireworks.ai/vLLM, fine-tuning Axolotl + Together.ai)

---

## 1. Tableau comparatif

| Critère | Qwen2.5-7B-Instruct | Mistral 7B v0.3 | LLaMA 3 8B |
|---|---|---|---|
| **Qualité RP français** | Excellent. Entraîné sur corpus multilingue massive. Instruct-tuned par défaut. | Bon. Multilingue mais moins moderne. | Bon en EN, FR secondaire. |
| **Fine-tuning Together.ai** | Disponible : `Qwen/Qwen2.5-7B-Instruct`. Supporte long-contexte. | Disponible mais déprécié au profit de Qwen. | Disponible mais licence restrictive. |
| **Fine-tuning Fireworks.ai** | ✅ **Disponible** : `accounts/fireworks/models/qwen2.5-7b-instruct`. 32k contexte natif. | Non disponible sur Fireworks. | Non disponible sur Fireworks. |
| **Fine-tuning Axolotl (RunPod)** | Supporté. QLoRA < 16 GB VRAM. Architecture Qwen2 compatible Axolotl 0.x. | Supporté. Mais modèle déprécié. | Supporté. GaLore OOM sur 2× A100 (axolotl #1641). |
| **Compatibilité vLLM** | Architecture `Qwen2ForCausalLM` : LoRA ✅ officiellement supporté depuis v0.6+. | Architecture `MistralForCausalLM` : LoRA ✅ mais problèmes de key names (issue #6573). | Architecture `LlamaForCausalLM` : LoRA ✅ sans problème. |
| **Licence redistribution** | **Apache 2.0.** Usage commercial libre, redistribution sans restriction. | **Apache 2.0.** Mêmes avantages. | **Meta Llama 3 Community License.** Obligations naming, MAU >700M, attribution. |
| **Contexte natif** | 32 768 tokens | 32 768 tokens | 8 192 tokens |

---

## 2. Analyse par critère

### 2.1 Disponibilité sur les plateformes

**Together.ai** : Qwen2.5-7B-Instruct est disponible sur Together.ai pour inference et fine-tuning. C'est le modèle recommandé pour les petits modèles (<16B) avec un bon rapport qualité/prix.

**Fireworks.ai** : Qwen2.5-7B-Instruct est **officiellement disponible** sur Fireworks.ai (`accounts/fireworks/models/qwen2.5-7b-instruct`). C'était un critère décisif — Mistral et LLaMA ne sont pas disponibles sur Fireworks.

**Axolotl/RunPod** : Qwen2.5 est supporté nativement par Axolotl avec le type de modèle `Qwen2ForCausalLM`. Le tokenizer est `AutoTokenizer` (BPE), compatible avec le pipeline existant.

**Verdict** : Qwen2.5 est le seul modèle disponible sur **les 3 plateformes** (Together + Fireworks + Axolotl).

### 2.2 Licence et redistribution

Apache 2.0 — identique à Mistral, bien supérieur à LLaMA 3. Permet :
- Utilisation commerciale libre
- Redistribution des weights fine-tunés sans restriction
- Pas de naming imposé
- Pas de seuil MAU
- Pas d'obligation d'attribuer le modèle de base

### 2.3 Qualité pour le RP en français

Qwen2.5 est le successeur spirituel de Qwen2, avec des améliorations significatives en raisonnement, génération de code et **qualité multilingue**. Les benchmarks MMLU, GSM8K et HumanEval montrent une avance de 10-15% sur Qwen2.5-7B vs Mistral 7B dans de nombreuses tâches.

Pour le RP en français, Qwen2.5 a été entraîné sur un corpus massive incluant le français, le chinois et de nombreuses autres langues. La version Instruct est déjà optimisée pour le dialogue, ce qui est idéal pour le RP.

### 2.4 Compatibilité vLLM

Qwen2 est officiellement supporté dans vLLM depuis la version 0.6+. L'architecture `Qwen2ForCausalLM` supporte LoRA nativement sans les problèmes de key names rencontrés avec Mistral v0.3.

---

## 3. Recommandation

**Utiliser Qwen/Qwen2.5-7B-Instruct comme modèle de base.**

### Justification synthétique

| Facteur | Poids | Gagnant |
|---|---|---|
| Disponibilité Fireworks.ai | Décisif | Qwen2.5 (seul modèle disponible) |
| Licence Apache 2.0 | Fort | Qwen2.5 (ex-aequo avec Mistral, > LLaMA) |
| Qualité RP français | Fort | Qwen2.5 (plus moderne) |
| Contexte fine-tuning | Moyen | Qwen2.5 (32k, ex-aequo avec Mistral) |
| Compatibilité vLLM | Moyen | Qwen2.5 (LoRA natif, pas de key issues) |
| Support Axolotl | Moyen | Qwen2.5 (support natif Qwen2ForCausalLM) |

### Avantages clés vs Mistral 7B

1. **Disponibilité Fireworks.ai** — critère décisif pour le déploiement cloud
2. **Modèle plus moderne** — Qwen2.5 (avril 2024) vs Mistral 7B (octobre 2023)
3. **Instruct-tuned par défaut** — pas besoin d'un checkpoint Instruct séparé
4. **Pas de problèmes de key names** — LoRA fonctionne directement dans vLLM
5. **Licence identique** — Apache 2.0

### Note sur `suddenly-13b`

Pour `suddenly-13b`, utiliser **Qwen2.5-14B-Instruct** (disponible sur HF, Together.ai et Fireworks.ai). C'est le succelogique de 7B dans la gamme Qwen2.5, avec le même tokenizer BPE et la même licence Apache 2.0.

---

## 4. Sources

- [Qwen2.5 — Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen2.5 announcement — Qwen blog](https://qwenlm.github.io/blog/qwen2.5/)
- [Qwen2.5 sur Fireworks.ai — fireworks.ai/models](https://fireworks.ai/models)
- [Together.ai fine-tuning models docs](https://docs.together.ai/docs/fine-tuning-models)
- [vLLM Supported Models — docs.vllm.ai](https://docs.vllm.ai/en/latest/models/supported_models/)
- [Axolotl supported models](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
