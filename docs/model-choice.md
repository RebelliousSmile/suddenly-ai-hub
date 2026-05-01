# Model Choice: Mistral 7B v0.3 vs LLaMA 3 8B

**Décision** : Mistral 7B v0.3  
**Date** : 2026-05-01  
**Contexte** : Choix du modèle de base pour `suddenly-7b` (RP en français, inference vLLM, fine-tuning Axolotl + Together.ai)

---

## 1. Tableau de comparaison

| Critère | Mistral 7B v0.3 | LLaMA 3 8B |
|---|---|---|
| **Qualité RP français** | Fort. Entraîné sur corpus multilingue incluant FR/DE/ES/IT. Fertilité tokenizer : 1.698 tokens/mot FR. Performance WMT FR-EN : 72%. | Bon en anglais, multilingual secondaire. Fertilité tokenizer : 1.591 tokens/mot FR (légèrement meilleur). Optimisé EN, FR en best-effort. |
| **Fine-tuning Together.ai** | Disponible : `mistralai/Mistral-7B-Instruct-v0.2` et `mistralai/Mistral-7B-v0.1`. Context SFT : 32 768 tokens. Batch max : 16. | Disponible : `meta-llama/Meta-Llama-3-8B-Instruct` et base. Context SFT : 8 192 tokens (4× moins). Batch max : 64. |
| **Fine-tuning Axolotl (RunPod)** | Supporté. QLoRA < 16 GB VRAM sur un A100. Connu pour stabilité mémoire. Sliding window attention (SWA) peut poser problème sur séquences très longues. | Supporté. QLoRA < 16 GB VRAM. GaLore à 8k séquence cause OOM sur 2× A100 (issue confirmée axolotl #1641). Plus coûteux en inférence (~2× Mistral sur Bedrock). |
| **Compatibilité vLLM + LoRA** | Architecture `MistralForCausalLM` : LoRA ✅ (docs vLLM). Mise en garde : weight key names différents sur certains checkpoints v0.3 (issue vLLM #6573). Nécessite conversion si LoRA généré via `mistral-finetune`. | Architecture `LlamaForCausalLM` : LoRA ✅ (docs vLLM). Aucune mise en garde documentée. Support PP (pipeline parallelism) confirmé. |
| **Licence redistribution** | **Apache 2.0.** Utilisation commerciale libre, redistribution des weights fine-tunés sans restriction, pas d'obligation de naming ni de MAU threshold. | **Meta Llama 3 Community License.** Obligations : (1) nommer tout modèle dérivé "Llama 3 …" ; (2) afficher "Built with Meta Llama 3" ; (3) si > 700M MAU, demander licence commerciale à Meta ; (4) interdit d'utiliser les outputs pour améliorer un LLM concurrent de Meta. |

---

## 2. Analyse par critère

### 2.1 Qualité RP en français

Mistral 7B a été conçu par une équipe française avec une attention explicite au multilinguisme européen. Les benchmarks WMT et HellaSwag montrent une avance de 8–12 points sur Llama 2, et les évaluations communautaires FR persistent au-delà de Llama 2.

Concernant LLaMA 3 8B, Meta a progressé sur le multilinguisme (30+ langues déclarées en 3.1), mais le pré-entraînement reste centré sur l'anglais. Les auteurs de LLM-Stats notent que LLaMA 3 8B surpasse Mistral 7B sur les benchmarks en anglais mais la parité en français est moins documentée.

La fertilité tokenizer de LLaMA 3 est légèrement meilleure pour le français (1.591 vs 1.698), ce qui signifie légèrement moins de tokens par mot. L'avantage est marginal (~6%) et ne compense pas l'orientation anglophone du modèle de base.

**Verdict** : Mistral 7B est un meilleur point de départ pour du texte narratif FR. L'avantage tokenizer de LLaMA 3 ne change pas la conclusion.

### 2.2 Temps et coût de fine-tuning

**Together.ai (Phase 1)** : Les deux modèles sont disponibles sur la plateforme. Différence critique : Mistral supporte un contexte SFT de 32 768 tokens contre 8 192 pour LLaMA 3 8B. Pour des sessions RP avec de longs historiques de conversation, Mistral est nettement mieux adapté sans truncation. La tarification standard pour les modèles < 16B est de $0.48–$1.35 par million de tokens selon la méthode (LoRA vs full FT).

**Axolotl sur RunPod (Phase 2)** : Les deux fonctionnent en QLoRA sous 16 GB VRAM. Mistral 7B a démontré une meilleure stabilité mémoire (Mistral v0.2 passe sur un A100 là où LLaMA 3 8B OOM avec GaLore à 8k). Avantage concret pour les expériences Axolotl sur GPU unique.

**Verdict** : Mistral 7B est plus avantageux sur les deux phases : fenêtre de contexte 4× plus grande chez Together.ai et stabilité mémoire supérieure pour Axolotl.

### 2.3 Compatibilité vLLM

Les deux architectures sont officiellement supportées dans vLLM avec le flag LoRA (`✅` dans la doc officielle pour `MistralForCausalLM` et `LlamaForCausalLM`).

Point de friction pour Mistral : les checkpoints générés via `mistral-finetune` ont des noms de clés différents de ceux attendus par vLLM, ce qui nécessite une étape de conversion (issue vLLM #6573, ouverte en 2024). Ce problème est documenté et contournable, mais représente une friction opérationnelle.

LLaMA 3 n'a pas de tel problème documenté. Son architecture est plus directement compatible avec vLLM sans post-traitement.

**Verdict** : Légère friction côté Mistral (conversion de clés LoRA), mais résoluble. Ne remet pas en cause le choix.

### 2.4 Licence et redistribution

C'est le critère le plus tranchant.

**Mistral 7B v0.3 — Apache 2.0** : Permissif sans réserve. Les weights fine-tunés `suddenly-7b`, `suddenly-7b-q4` et `suddenly-13b` peuvent être distribués publiquement, packagés, hébergés, commercialisés sans notification à Mistral AI. Aucun naming imposé, aucun seuil MAU.

**LLaMA 3 8B — Meta Llama 3 Community License** : Quatre contraintes opérationnelles pour ce projet :
1. Le modèle doit s'appeler "Llama 3 suddenly-7b" (ou similaire) — incompatible avec le branding prévu.
2. "Built with Meta Llama 3" doit apparaître dans l'UI ou la documentation publique.
3. Interdiction d'utiliser les outputs pour améliorer un autre LLM (bloquerait la distillation ou le synthetic data generation vers un autre modèle).
4. Demande de licence obligatoire si > 700M MAU (risque faible à court terme, mais clause contraignante).

**Verdict** : Apache 2.0 de Mistral est sans ambiguïté le meilleur choix pour un projet Fediverse open-source qui distribuera des dérivés.

---

## 3. Recommandation

**Utiliser Mistral 7B v0.3 comme modèle de base.**

### Justification synthétique

| Facteur | Poids | Gagnant |
|---|---|---|
| Qualité FR pour RP | Fort | Mistral 7B |
| Contexte fine-tuning | Fort | Mistral 7B (32k vs 8k) |
| Stabilité Axolotl | Moyen | Mistral 7B |
| vLLM LoRA | Moyen | LLaMA 3 (légèrement) |
| Licence redistribution | Décisif | Mistral 7B |

LLaMA 3 8B n'a qu'un avantage marginal côté tokenizer FR et une compatibilité vLLM légèrement plus propre. Ces deux points ne compensent pas les contraintes de licence et la fenêtre de contexte 4× réduite sur Together.ai.

### Point de vigilance

Le problème de compatibilité des clés LoRA entre `mistral-finetune` et vLLM (issue #6573) doit être traité en amont du déploiement. Solution connue : utiliser Axolotl (et non `mistral-finetune`) pour générer les adapters, ou passer par un script de conversion de clés avant chargement dans vLLM.

### Note sur `suddenly-13b`

Mistral n'ayant pas de modèle 13B dense, `suddenly-13b` nécessite un spike séparé. Candidats probables : **Mistral Nemo 12B** (Apache 2.0, 128k contexte) ou **LLaMA 3 13B** (si disponible avec licence acceptable). La décision pour `suddenly-13b` ne bloque pas Phase 0 — seul `suddenly-7b` est requis pour le premier fine-tune.

---

## 4. Sources

- [Mistral 7B announcement — mistral.ai](https://mistral.ai/news/announcing-mistral-7b)
- [Mistral-7B-Instruct-v0.3 — Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [EU Tokenizer Performance (Occiglot)](https://occiglot.eu/posts/eu_tokenizer_perfomance/) — données fertilité FR : Mistral 1.698, LLaMA 1.591
- [vLLM Supported Models — docs.vllm.ai](https://docs.vllm.ai/en/latest/models/supported_models/) — LoRA support confirmé pour les deux architectures
- [vLLM LoRA issue #6573 — Mistral finetune key names](https://github.com/vllm-project/vllm/issues/6573)
- [vLLM LoRA Adapters docs](https://docs.vllm.ai/en/v0.8.4/features/lora.html)
- [Together.ai fine-tuning models docs](https://docs.together.ai/docs/fine-tuning-models) — contexte Mistral 32k, LLaMA 3 8k
- [Together.ai pricing](https://www.together.ai/pricing) — $0.48–$1.35 / 1M tokens pour < 16B
- [Axolotl issue #1641 — LLaMA 3 8B GaLore OOM](https://github.com/axolotl-ai-cloud/axolotl/issues/1641)
- [Meta Llama 3 License — llama.com](https://www.llama.com/llama3/license/) — obligations naming, MAU threshold, attribution
- [Mistral 7B Apache 2.0 — HuggingFace discussion](https://huggingface.co/mistralai/Mistral-7B-v0.1/discussions/2)
- [Llama 3 8B vs Mistral 7B — Vantage.sh](https://www.vantage.sh/blog/best-small-llm-llama-3-8b-vs-mistral-7b-cost)
- [Mistral vs LLaMA 2025 — machinetranslation.com](https://www.machinetranslation.com/blog/mistral-vs-llama)
