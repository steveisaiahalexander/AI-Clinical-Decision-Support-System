import test from 'node:test'
import assert from 'node:assert/strict'
import { resultPresentation } from '../src/resultsPresentation.js'
import { evidencePresentation } from '../src/evidencePresentation.js'
import { groundedExplanationPresentation } from '../src/groundedExplanationPresentation.js'

test('confident response separates calibrated probability from the ranked class', () => {
  const view = resultPresentation({
    decision_status: 'ranked_prediction',
    probability_status: 'calibrated',
    predicted_disease: 'Condition A',
  })
  assert.equal(view.insufficientEvidence, false)
  assert.equal(view.primaryLabel, 'Top model-ranked class')
  assert.equal(view.primaryHeading, 'Condition A')
  assert.equal(view.probabilityLabel, 'Calibrated model probability')
})

test('abstention hides the top class as a primary finding and requests more information', () => {
  const view = resultPresentation({
    decision_status: 'insufficient_evidence',
    probability_status: 'calibrated',
    predicted_disease: 'Condition A',
  })
  assert.equal(view.insufficientEvidence, true)
  assert.equal(view.primaryLabel, 'Insufficient evidence')
  assert.equal(view.primaryHeading, 'No clear leading class')
  assert.match(view.primaryMessage, /Additional information may be useful/)
  assert.equal(view.rankedHeading, 'Ranked model probabilities')
})

test('evidence status distinguishes unsupported classes and insufficient retrieval', () => {
  assert.equal(evidencePresentation({ supported: false, passages: [] }).emptyMessage, 'This class is outside the curated evidence set.')
  assert.equal(evidencePresentation({ supported: true, sufficient_evidence: false, passages: [] }).emptyMessage, 'No sufficient source evidence was found for this context.')
})

test('evidence presentation labels strategy without exposing retrieval scores', () => {
  const view = evidencePresentation({
    supported: true,
    sufficient_evidence: true,
    retrieval_method: 'hybrid_weighted_rrf',
    passages: [{ relevance_score: 0.25 }],
  })
  assert.equal(view.methodLabel, 'Hybrid')
  assert.equal(view.hasPassages, true)
  assert.equal(Object.hasOwn(view, 'score'), false)
})

test('grounded summary displays only claims whose evidence IDs resolve to attributed sources', () => {
  const response = {
    grounding_status: 'grounded',
    explanation: {
      grounding_status: 'grounded',
      summary: 'Wheezing is described in the retrieved evidence. [E1]',
      claims: [{ text: 'Wheezing is described in the retrieved evidence.', evidence_ids: ['E1'] }],
    },
    evidence: [
      { evidence_id: 'E1', passage: { source: { title: 'About Asthma', url: 'https://example.gov/asthma' } } },
      { evidence_id: 'E2', passage: { source: { title: 'Unused source', url: 'https://example.gov/unused' } } },
    ],
  }
  const view = groundedExplanationPresentation(response)
  assert.equal(view.grounded, true)
  assert.equal(view.claims.length, 1)
  assert.deepEqual(view.evidence.map((item) => item.evidence_id), ['E1'])
})

test('grounded summary suppresses unknown citations but retains evidence on fallback states', () => {
  const invalid = groundedExplanationPresentation({
    grounding_status: 'grounded',
    explanation: {
      grounding_status: 'grounded',
      claims: [{ text: 'Unsupported claim.', evidence_ids: ['E9'] }],
    },
    evidence: [{ evidence_id: 'E1', passage: { source: { title: 'About Asthma', url: 'https://example.gov/asthma' } } }],
  })
  assert.equal(invalid.grounded, false)
  assert.deepEqual(invalid.claims, [])

  const unavailable = groundedExplanationPresentation({
    grounding_status: 'llm_unavailable',
    status_message: 'Generated synthesis is unavailable.',
    evidence: [{ evidence_id: 'E1', passage: { source: { title: 'About Asthma', url: 'https://example.gov/asthma' } } }],
  })
  assert.equal(unavailable.statusMessage, 'Generated synthesis is unavailable.')
  assert.deepEqual(unavailable.evidence.map((item) => item.evidence_id), ['E1'])
})
