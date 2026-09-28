export function resultPresentation(result) {
  const insufficientEvidence = result?.decision_status === 'insufficient_evidence'
  const probabilityLabel = result?.probability_status === 'calibrated'
    ? 'Calibrated model probability'
    : 'Raw model probability'

  return {
    insufficientEvidence,
    probabilityLabel,
    primaryLabel: insufficientEvidence ? 'Insufficient evidence' : 'Top model-ranked class',
    primaryHeading: insufficientEvidence ? 'No clear leading class' : result?.predicted_disease || '',
    primaryMessage: insufficientEvidence
      ? 'Model probability, class separation, or distribution uncertainty did not meet the evidence thresholds. Additional information may be useful.'
      : '',
    rankedHeading: insufficientEvidence ? 'Ranked model probabilities' : 'Other ranked predictions',
  }
}
