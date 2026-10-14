export function applyQuestionResponse(state, featureId, response) {
  const next = {
    selected: [...state.selected],
    answeredNegative: [...state.answeredNegative],
    skipped: [...state.skipped],
  }
  next.answeredNegative = next.answeredNegative.filter((item) => item !== featureId)
  next.skipped = next.skipped.filter((item) => item !== featureId)

  if (response === 'yes' && !next.selected.includes(featureId)) next.selected.push(featureId)
  if (response === 'no' && !next.selected.includes(featureId)) next.answeredNegative.push(featureId)
  if (response === 'skip' && !next.selected.includes(featureId)) next.skipped.push(featureId)
  return next
}

export function unaskedSuggestions(suggestions, selected, answeredNegative, skipped) {
  const excluded = new Set([...selected, ...answeredNegative, ...skipped])
  return (suggestions || []).filter((item) => !excluded.has(item.feature_id))
}
