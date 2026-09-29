# Curated Evidence Retrieval

The evidence layer provides cited background passages for all 30 saved model classes. It remains separate from prediction, calibrated probabilities, uncertainty, abstention, and SHAP output. A class outside the curated set returns `supported: false` with no passages; a covered class with insufficient retrieval signal returns `status: no_sufficient_evidence` and no passages.

## Sources and Reuse

The corpus contains short, verbatim excerpts from official U.S. federal health agencies selected for patient-facing disease definitions and symptoms. Each JSON document records its title, organization, URL, section (and page number where available), access date, publication/update/review date when available, attribution, and rights note. Ingestion copies this provenance onto each stable chunk. HTML sections are not paginated; the source page and named section are retained instead. No images, logos, or third-party linked materials are copied, and excerpts are not paraphrased during ingestion.

CDC says most CDC/ATSDR website information is public domain, but warns that some contractor, grantee, or third-party materials are restricted. Its reuse terms require agency attribution and a clear statement that reuse does not imply government endorsement. The corpus uses CDC-authored text, attributes CDC, links to the original pages, and the UI/API disclaim endorsement. See [CDC Use of Agency Materials](https://www.cdc.gov/other/agencymaterials.html).

NIDDK says most site information is copyright-free and asks that unchanged reproduced content acknowledge NIDDK; it also lists exceptions for some co-sponsored documents and graphics. The included excerpts are NIDDK page text with attribution, without graphics or logos. NIH says its public-site content is generally public domain unless noted, with exceptions for licensed material. MedlinePlus explicitly identifies its Genetics summaries as public-domain federal content, and NIAMS states its website text is public domain. See [NIDDK Copyright](https://www.niddk.nih.gov/copyright), [NIH copyright FAQ](https://www.nih.gov/about-nih/frequently-asked-questions), [MedlinePlus content use](https://medlineplus.gov/about/using/usingcontent/), and [NIAMS disclaimer](https://www.niams.nih.gov/disclaimer). Source notices should be checked again before reuse outside this project or in a commercial distribution.

| Model class | Curated source |
| --- | --- |
| Asthma | [CDC, About Asthma](https://www.cdc.gov/asthma/about/index.html) |
| COPD | [CDC, About COPD](https://www.cdc.gov/copd/about/index.html) |
| Common_Cold | [CDC, About Common Cold](https://www.cdc.gov/common-cold/about/) |
| Flu | [CDC, Signs and Symptoms of Flu](https://www.cdc.gov/flu/signs-symptoms/) |
| COVID19 | [CDC, Symptoms of COVID-19](https://www.cdc.gov/covid/signs-symptoms/) |
| Tuberculosis | [CDC, Signs and Symptoms of Tuberculosis](https://www.cdc.gov/tb/signs-symptoms/index.html) |
| Pneumonia | [CDC, About Pneumonia](https://www.cdc.gov/pneumonia/about/index.html) |
| Stroke | [CDC, Signs and Symptoms of Stroke](https://www.cdc.gov/stroke/signs-symptoms/index.html) |
| Type2_Diabetes | [NIDDK, Type 2 Diabetes](https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes/type-2-diabetes) |
| Chronic_Kidney_Disease | [NIDDK, What Is Chronic Kidney Disease in Adults?](https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/what-is-chronic-kidney-disease) |
| Hypertension | [NIDDK, High Blood Pressure & Kidney Disease](https://www.niddk.nih.gov/health-information/kidney-disease/high-blood-pressure) |
| Migraine | [NINDS, Migraine Information Page](https://www.ninds.nih.gov/Disorders/All-Disorders/Migraine-Information-Page) |
| Amyotrophic_Lateral_Sclerosis | [NINDS, Amyotrophic Lateral Sclerosis (ALS)](https://www.ninds.nih.gov/sites/default/files/2025-05/NINDS_ALS_Booklet_Digital-508c.pdf) |
| Coronary_Artery_Disease | [NHLBI, Coronary Heart Disease Symptoms](https://www.nhlbi.nih.gov/health/coronary-heart-disease/symptoms) |
| Cystic_Fibrosis | [NHLBI, Cystic Fibrosis Symptoms](https://www.nhlbi.nih.gov/health/cystic-fibrosis/symptoms) |
| Depression | [NIMH, Depression](https://www.nimh.nih.gov/health/publications/depression) |
| Ehlers_Danlos_Syndrome | [MedlinePlus Genetics, Ehlers-Danlos syndrome](https://medlineplus.gov/genetics/condition/ehlers-danlos-syndrome/) |
| Epilepsy | [NINDS, Epilepsy and Seizures](https://www.ninds.nih.gov/node/647) |
| Gastritis | [NIDDK, Gastritis & Gastropathy](https://www.niddk.nih.gov/health-information/digestive-diseases/gastritis-gastropathy) |
| Generalized_Anxiety_Disorder | [NIMH, Generalized Anxiety Disorder](https://www.nimh.nih.gov/health/publications/generalized-anxiety-disorder-gad) |
| Hyperthyroidism | [NIDDK, Hyperthyroidism](https://www.niddk.nih.gov/health-information/endocrine-diseases/hyperthyroidism) |
| Hypothyroidism | [NIDDK, Hypothyroidism](https://www.niddk.nih.gov/health-information/endocrine-diseases/hypothyroidism) |
| Irritable_Bowel_Syndrome | [NIDDK, Symptoms & Causes of IBS](https://www.niddk.nih.gov/health-information/digestive-diseases/irritable-bowel-syndrome/symptoms-causes) |
| Marfan_Syndrome | [NIAMS, Marfan Syndrome](https://www.niams.nih.gov/health-topics/marfan-syndrome) |
| Obesity | [CDC, Consequences of Obesity](https://www.cdc.gov/obesity/php/about/consequences.html) |
| Osteoarthritis | [NIAMS, Osteoarthritis](https://www.niams.nih.gov/health-topics/osteoarthritis) |
| Osteopetrosis | [MedlinePlus Genetics, Osteopetrosis](https://medlineplus.gov/genetics/condition/osteopetrosis/) |
| Peptic_Ulcer_Disease | [NIDDK, Symptoms & Causes of Peptic Ulcers](https://www.niddk.nih.gov/health-information/digestive-diseases/peptic-ulcers-stomach-ulcers/symptoms-causes) |
| Polycystic_Ovary_Syndrome | [NICHD, PCOS Symptoms](https://www.nichd.nih.gov/health/topics/pcos/conditioninfo/symptoms) |
| Systemic_Lupus_Erythematosus | [NIAMS, Systemic Lupus Erythematosus](https://www.niams.nih.gov/health-topics/lupus) |

The expanded corpus was reviewed for this milestone on 2026-09-30. Source pages can change; `accessed_on` and page dates are stored with each source. CDC recommends linking to resources and checking copied material for revisions. This project does not fetch or scrape pages at runtime. GARD pages with mixed or unclear third-party source provenance were not used.

## Index and Retrieval

The TF-IDF baseline remains the default and keeps its existing deterministic behavior. Scikit-learn's `TfidfVectorizer` turns each chunk into normalized sparse word unigram/bigram vectors. Cosine similarity ranks passages for a query made from the model class and selected symptom context; if the context query is below the configured threshold, it retries with the class name alone. Retrieval is restricted to the exact canonical class or its curated aliases, so another disease's passages cannot be returned.

Chunking follows sentence boundaries with a 700-character limit and one-sentence overlap. Section and full source metadata are copied onto each chunk. The default response contains the top 3 passages. `rag/config.json` controls corpus path, chunk size, overlap, retrieval thresholds, selected strategy, embedding model, index location, and fusion weights.

Semantic retrieval uses `sentence-transformers/all-MiniLM-L6-v2` at pinned Hub revision `bc57282bc374d33e0d6c4de27f12dc1c2a87f37a` (Apache-2.0; 384-dimensional sentence embeddings), with `sentence-transformers==5.7.0` and CPU inference. Install the optional runtime with `python -m pip install -r rag/requirements-semantic.txt`. The model is downloaded locally on first semantic use; no paid API or hosted inference service is used. Encoded chunks are stored in an ignored local compressed NumPy matrix with a JSON manifest containing the model revision, ordered chunk IDs, and corpus digest. A changed model revision or corpus text causes deterministic index rebuild. Because the corpus has only 30 source records, exact NumPy cosine search is simpler than operating a separate vector database service.

Hybrid retrieval combines the top-ranked TF-IDF and semantic lists using weighted Reciprocal Rank Fusion. With configurable weights `w_t` and `w_s` (default 0.5 each, normalized to sum to 1) and `rrf_k=60`, the fusion score is `w_t / (rrf_k + lexical_rank) + w_s / (rrf_k + semantic_rank)`. Candidates must pass the configured per-retriever retrieval thresholds before fusion. These scores are ranking values, not medical confidence; the React panel displays the selected method but never displays numeric scores.

The React client calls `POST /evidence` after the existing prediction. The optional request field `strategy` accepts `tfidf`, `semantic`, or `hybrid`; omission uses `default_strategy` from config, which remains `tfidf`. The response adds `sufficient_evidence` and a status of `evidence_available`, `no_sufficient_evidence`, or `unsupported_condition`. A missing/low-signal retrieval returns no passages and never invents evidence or calls an LLM. Evidence is displayed independently and does not change the model result. Retrieved passages are contextual information only, not a diagnosis or treatment recommendation; inclusion does not imply CDC, NIH, or NIDDK endorsement.

## Rebuild and Evaluation

From the repository root, run:

    python scripts/build_evidence_index.py
    python -m pip install -r rag/requirements-semantic.txt
    python scripts/build_evidence_index.py --semantic
    python -m unittest tests.test_evidence_retrieval
    python scripts/evaluate_evidence_retrieval.py

The evaluation script compares TF-IDF, semantic, and hybrid retrieval on the manual 30-class query set in `rag/evaluation_queries.json` plus one unsupported-class check. It reports Recall@1, Recall@3, MRR@3, missing-relevant-evidence count, and expected-source coverage. These are retrieval-only metrics, not clinical metrics; no classifier test samples (including the held-out 19,974 samples) are read, and ML metrics are not recalculated. The focused tests cover all 30 evidence classes, chunk/index construction, metadata preservation, each retrieval strategy, insufficient/unsupported evidence, the FastAPI route, and retrieval evaluation behavior.
