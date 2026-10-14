import test from 'node:test'
import assert from 'node:assert/strict'
import { resultPresentation } from '../src/resultsPresentation.js'
import { evidencePresentation } from '../src/evidencePresentation.js'
import { groundedExplanationPresentation } from '../src/groundedExplanationPresentation.js'
import { applyQuestionResponse, nextQuestionPanelPresentation, unaskedSuggestions } from '../src/questioningPresentation.js'
import { fetchNextQuestions } from '../src/services/api.js'
import viteConfig, { apiProxyPaths } from '../vite.config.js'

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

test('Yes adds a positive symptom while No and Skip remain separate questioning states', () => {
  const initial = { selected: ['fatigue'], answeredNegative: [], skipped: [] }
  assert.deepEqual(applyQuestionResponse(initial, 'cough', 'yes'), {
    selected: ['fatigue', 'cough'], answeredNegative: [], skipped: [],
  })
  assert.deepEqual(applyQuestionResponse(initial, 'cough', 'no'), {
    selected: ['fatigue'], answeredNegative: ['cough'], skipped: [],
  })
  assert.deepEqual(applyQuestionResponse(initial, 'cough', 'skip'), {
    selected: ['fatigue'], answeredNegative: [], skipped: ['cough'],
  })
  assert.deepEqual(initial, { selected: ['fatigue'], answeredNegative: [], skipped: [] })
})

test('responded symptoms are not suggested again', () => {
  const suggestions = ['cough', 'nausea', 'headache'].map((feature_id) => ({ feature_id }))
  assert.deepEqual(
    unaskedSuggestions(suggestions, ['cough'], ['nausea'], ['headache']),
    [],
  )
})

test('question panel presents ranked candidates for the response controls', () => {
  const response = {
    status: 'suggestions_available',
    status_message: 'Suggestions available.',
    suggestions: [
      { feature_id: 'symptom_orthopnea', label: 'Orthopnea', information_gain_bits: 0.09 },
      { feature_id: 'symptom_edema', label: 'Peripheral Edema', information_gain_bits: 0.08 },
    ],
  }
  const view = nextQuestionPanelPresentation({
    response,
    error: '',
    loading: false,
    selected: ['symptom_fatigue'],
    answeredNegative: [],
    skipped: [],
  })
  assert.equal(view.state, 'ready')
  assert.deepEqual(view.suggestions.map((item) => item.label), ['Orthopnea', 'Peripheral Edema'])
})

test('question panel displays the request error state rather than an empty result', () => {
  const view = nextQuestionPanelPresentation({
    response: null,
    error: 'Please review the selected symptoms and try again.',
    loading: false,
    selected: [],
    answeredNegative: [],
    skipped: [],
  })
  assert.deepEqual(view, {
    state: 'error',
    message: 'Please review the selected symptoms and try again.',
  })
})

test('Vite forwards the next-question API route to FastAPI', () => {
  assert.ok(apiProxyPaths.includes('/next-question'))
  assert.ok(Object.hasOwn(viteConfig.server.proxy, '/next-question'))
})

test('browser API sends the request contract and returns ranked candidates', async () => {
  const originalFetch = globalThis.fetch
  const payload = { status: 'suggestions_available', suggestions: [{ feature_id: 'symptom_orthopnea' }] }
  let captured
  globalThis.fetch = async (url, options) => {
    captured = { url, options }
    return { ok: true, json: async () => payload }
  }
  try {
    const response = await fetchNextQuestions(
      ['symptom_fatigue'],
      ['symptom_cough'],
      ['symptom_nausea'],
      3,
    )
    assert.equal(captured.url, '/next-question')
    assert.deepEqual(JSON.parse(captured.options.body), {
      symptoms: ['symptom_fatigue'],
      answered_negative_symptoms: ['symptom_cough'],
      skipped_symptoms: ['symptom_nausea'],
      top_k: 3,
    })
    assert.deepEqual(response, payload)
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('browser API converts an unproxied HTML 404 into the displayed request error', async () => {
  const originalFetch = globalThis.fetch
  globalThis.fetch = async () => ({
    ok: false,
    json: async () => { throw new SyntaxError('Unexpected token <') },
  })
  try {
    await assert.rejects(
      fetchNextQuestions(['symptom_fatigue']),
      /Please review the selected symptoms and try again\./,
    )
  } finally {
    globalThis.fetch = originalFetch
  }
})
