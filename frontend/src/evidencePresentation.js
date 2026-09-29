const methodLabels = {
  tfidf_cosine: 'TF-IDF',
  semantic_cosine: 'Semantic',
  hybrid_weighted_rrf: 'Hybrid',
}

export function evidencePresentation(evidence) {
  const passages = Array.isArray(evidence?.passages) ? evidence.passages : []
  const sufficient = evidence?.sufficient_evidence ?? passages.length > 0
  return {
    methodLabel: methodLabels[evidence?.retrieval_method] || '',
    emptyMessage: !evidence
      ? ''
      : !evidence.supported
        ? 'This class is outside the curated evidence set.'
        : !sufficient
          ? 'No sufficient source evidence was found for this context.'
          : '',
    hasPassages: Boolean(evidence?.supported && sufficient && passages.length),
  }
}
