# Curated Evidence Retrieval

The evidence layer provides cited background passages for a subset of the saved model's classes. It does not alter prediction, calibrated probabilities, uncertainty, abstention, or SHAP output. A class outside the curated set returns supported: false with no passages.

## Sources and Reuse

The initial corpus contains short, verbatim excerpts from official U.S. federal health agencies selected for patient-facing disease definitions, symptoms, and symptom caveats. Each JSON document records its title, agency, URL, section, access date, page update or review date when available, attribution, and rights note. The HTML sections are not paginated; the source page ID and section are retained instead. No images, logos, or third-party linked materials are copied.

CDC says most CDC/ATSDR website information is public domain, but warns that some contractor, grantee, or third-party materials are restricted. Its reuse terms require agency attribution and a clear statement that reuse does not imply government endorsement. The corpus uses CDC-authored text, attributes CDC, links to the original pages, and the UI/API disclaim endorsement. See [CDC Use of Agency Materials](https://www.cdc.gov/other/agencymaterials.html).

NIDDK says most site information is copyright-free and asks that unchanged reproduced content acknowledge NIDDK; it also lists exceptions for some co-sponsored documents and graphics. The included excerpts are NIDDK page text with attribution, without graphics or logos. NIH says its public-site content is generally public domain unless noted, with exceptions for licensed material; the NINDS excerpt uses page text only. See [NIDDK Copyright](https://www.niddk.nih.gov/copyright) and [NIH copyright FAQ](https://www.nih.gov/about-nih/frequently-asked-questions). Source notices should be checked again before reuse outside this project or in a commercial distribution.

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

The corpus was reviewed for this milestone on 2026-09-30. Source pages can change; accessed_on and page dates are stored with each source. CDC recommends linking to resources and checking copied material for revisions. This project does not fetch or scrape pages at runtime.

## Index and Retrieval

The index is built locally from rag/documents/*.json the first time /evidence is called. Scikit-learn's existing TfidfVectorizer turns each chunk into normalized sparse word unigram/bigram vectors. Cosine similarity ranks chunks for a query made from the model class and selected symptom context. Retrieval is first restricted to the exact canonical class or its curated aliases, so passages from another disease cannot be returned. If the context query scores below the configured minimum, retrieval retries with the class name alone.

Chunking follows sentence boundaries with a 700-character limit and one-sentence overlap. Section and source metadata are copied onto every chunk. The default response contains the top 3 passages; rag/config.json controls corpus path, chunk size, overlap, score threshold, and default top-k. The sparse matrix is small and rebuilt in memory; no vector database service or model download is required. This lexical embedding is deterministic but does not capture semantic similarity the way a neural embedding model can.

The React client calls POST /evidence after the existing prediction. Evidence is displayed independently and does not change the model result. The endpoint returns an empty, explicitly unsupported response for the other 18 of 30 saved classes rather than fabricating content. Retrieved passages are contextual information only, not a diagnosis or treatment recommendation; inclusion does not imply CDC, NIH, or NIDDK endorsement.

## Rebuild and Evaluation

From the repository root, run:

    python scripts/build_evidence_index.py
    python -m unittest tests.test_evidence_retrieval

The focused retrieval tests cover the 12-document count, chunk and sparse-index construction, passage/source metadata, representative symptom queries across the corpus, unknown classes, and the FastAPI evidence route. They do not read classifier evaluation samples, tune the classifier, or recalculate ML metrics.
