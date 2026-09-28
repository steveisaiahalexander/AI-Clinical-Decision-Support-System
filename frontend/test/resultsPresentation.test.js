import test from 'node:test'
import assert from 'node:assert/strict'
import { resultPresentation } from '../src/resultsPresentation.js'

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
