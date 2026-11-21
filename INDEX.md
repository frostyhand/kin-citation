# AI Citation Bias Index

A quarterly measurement of how answer engines choose which web pages to cite, run with this repository's Inspect eval on current models. Each edition runs the full main arm (80 topics, 6 counterbalanced orders, 480 trials per model) of the preregistered study [Do Answer Engines Cite Their Own Kind?](https://ssrn.com/abstract=7710958) (Suresh, 2026), so editions are directly comparable with each other and with the study.

It tracks three things:

- **Kin preference (K):** whether a model cites the page written by its own model family more often than other engines cite that same page, after removing page quality and the model's own habits (see [Scoring](README.md#scoring)). Only defined for the five study families (Anthropic, OpenAI, Google, xAI, Moonshot), because only they wrote pages.
- **First-slot share:** the share of citations that go to whichever page is listed first. With six pages in counterbalanced orders, a position-blind engine would give 16.7%.
- **Wikipedia lead cited:** how often the human-written reference page is one of the two citations.

## 2026 Q4 edition (run 21 November 2026)

Eval commit `1f3c57c`, main arm, 480 trials per model, temperature 0, through OpenRouter, low reasoning effort (GPT-5 nano at its default). One full run per model.

| Model               | Kin preference K, points (95% CI) | Own family's page cited | First-slot share | Wikipedia lead cited | Valid |
| ------------------- | --------------------------------- | ----------------------- | ---------------- | -------------------- | ----- |
| Claude Opus 5.5     | **+8.5** (+4.2, +13.0)            | 43.5%                   | 30.6%            | 99.6%                | 100%  |
| Claude Haiku 5.5    | **-8.0** (-11.4, -4.6)            | 25.0%                   | 32.9%            | 89.3%                | 99.2% |
| GPT-6.2 Sol         | **+5.1** (+1.4, +9.2)             | 24.2%                   | 23.0%            | 94.6%                | 100%  |
| GPT-6.1 Sol         | **+9.1** (+5.5, +12.8)            | 29.0%                   | 32.2%            | 95.4%                | 100%  |
| GPT-6 Luna          | **+3.8** (+1.5, +6.1)             | 23.4%                   | 38.6%            | 90.4%                | 99.8% |
| GPT-5 nano          | +2.3 (-0.6, +5.1)                 | 26.5%                   | 36.5%            | 65.4%                | 100%  |
| Gemini 4 Hydrogen   | **+3.5** (+1.1, +6.0)             | 23.1%                   | 45.3%            | 87.5%                | 100%  |
| Qwen 4 Max          | n/a                               | n/a                     | 30.9%            | 99.0%                | 100%  |
| DeepSeek V4.1 Flash | n/a                               | n/a                     | 45.5%            | 91.2%                | 100%  |
| Mistral Large 4     | n/a                               | n/a                     | 48.5%            | 92.5%                | 99.8% |
| GLM-5.3 Prime       | n/a                               | n/a                     | 46.5%            | 97.4%                | 97.9% |

Bold K values have a 95% interval that excludes zero. "Own family's page" is the page written by the study's model from the same developer (Claude Sonnet 5.5, GPT-6.1 Sol, Gemini 4 Argon), so for every model except GPT-6.1 Sol it measures preference for an earlier sibling's writing.

### What this edition shows

1. **Kin preference is a property of the model, not the developer.** Claude Opus 5.5 cites the Claude-written page in 43.5% of trials and has the second-largest kin preference in this edition (+8.5, after GPT-6.1 Sol). Claude Haiku 5.5, from the same developer, avoids that page (-8.0), as the study's Claude Sonnet 5.5 did (-4.6). An audit that tests one model per developer would miss this.
2. **The study result replicates.** GPT-6.1 Sol, rerun through this eval two weeks after the study, gives +9.1 against the study's +8.8 (5.4 to 12.3).
3. **Newer OpenAI models show less kin preference than GPT-6.1 Sol** (GPT-6.2 Sol +5.1, GPT-6 Luna +3.8, GPT-5 nano +2.3), but all point the same way.
4. **Position bias is everywhere and larger than kin preference.** Every model gives the first-listed page 23% to 49% of its citations against 16.7% for a position-blind engine. Mistral Large 4, GLM-5.3 Prime, DeepSeek V4.1 Flash and Gemini 4 Hydrogen are the most position-driven.
5. **The fact-rich human page still wins.** Every model except GPT-5 nano cites the Wikipedia lead in 87% to 100% of trials, as in the study, because it carries more facts than the model-written pages.

### Caveats

One run per model; providers do not guarantee determinism at temperature 0 and OpenRouter may route a model to different providers, so a rerun can move K a little (GPT-6.1 Sol moved 0.3 points between the study and this edition). Models newer than the study's five are judged on pages written by their earlier siblings in November 2026. K is undefined for developers that wrote no page in the study. The study's other limitations apply (see [Limitations](README.md#limitations)).

## Data

Per-model results, including every citation share, are in [`index/2026-q4.json`](index/2026-q4.json). The Inspect logs (11 runs, 480 samples each) are available on request.

## Reproduce a row

```bash
uv sync
uv run inspect eval kin_citation/kin_citation --model openrouter/anthropic/claude-opus-5.5 --reasoning-effort low
```

## Cite

Suresh, T. (2026). AI Citation Bias Index, 2026 Q4 edition. https://github.com/frostyhand/kin-citation/blob/main/INDEX.md. Method: Suresh, T. (2026). Do Answer Engines Cite Their Own Kind? A Controlled Test of Kin Preference in Generative Search. SSRN 7710958. https://doi.org/10.2139/ssrn.7710958
