const statusMessages = {
  abstained: 'The model abstained from a ranked prediction. No generated explanation was produced.',
  insufficient_evidence: 'Insufficient evidence for a grounded explanation.',
  llm_unavailable: 'Generated synthesis is unavailable. Retrieved evidence is shown below.',
  validation_failed: 'A grounded explanation could not be validated; no generated synthesis is shown.',
}

export function groundedExplanationPresentation(response) {
  const evidence = Array.isArray(response?.evidence) ? response.evidence : []
  const evidenceById = new Map(
    evidence
      .filter((item) => item?.evidence_id && item?.passage?.source?.title && item?.passage?.source?.url)
      .map((item) => [item.evidence_id, item]),
  )
  const claims = Array.isArray(response?.explanation?.claims) ? response.explanation.claims : []
  const validClaims = claims.length > 0 && claims.every((claim) => (
    typeof claim?.text === 'string'
    && claim.text.trim().length > 0
    && Array.isArray(claim.evidence_ids)
    && claim.evidence_ids.length > 0
    && claim.evidence_ids.every((id) => evidenceById.has(id))
  ))
  const grounded = response?.grounding_status === 'grounded'
    && response?.explanation?.grounding_status === 'grounded'
    && validClaims
  const citedIds = new Set(grounded ? claims.flatMap((claim) => claim.evidence_ids) : [])

  return {
    grounded,
    claims: grounded ? claims : [],
    evidence: grounded
      ? evidence.filter((item) => citedIds.has(item.evidence_id))
      : evidence,
    statusMessage: grounded
      ? 'Citations were validated against the retrieved passages.'
      : response?.status_message || statusMessages[response?.grounding_status] || '',
  }
}
