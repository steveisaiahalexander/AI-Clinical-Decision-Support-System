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

export function nextQuestionPanelPresentation({
  response,
  error,
  loading,
  selected,
  answeredNegative,
  skipped,
}) {
  if (loading) return { state: 'loading' }
  if (error) return { state: 'error', message: error.message || String(error) }
  if (!response) return { state: 'idle' }

  const suggestions = unaskedSuggestions(
    response.suggestions,
    selected,
    answeredNegative,
    skipped,
  )
  if (suggestions.length > 0) return { state: 'ready', suggestions }
  return { state: 'empty', message: response.status_message }
}
