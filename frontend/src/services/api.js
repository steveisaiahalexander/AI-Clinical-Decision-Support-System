const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${apiBaseUrl}${path}`, {
      ...options,
      headers: {
        Accept: 'application/json',
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        ...options.headers,
      },
    })
  } catch {
    throw new Error('The assessment service could not be reached. Check that the API is running and try again.')
  }

  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = typeof payload?.detail === 'string'
      ? payload.detail
      : typeof payload?.detail?.message === 'string'
        ? payload.detail.message
        : 'Please review the selected symptoms and try again.'
    throw new Error(detail)
  }
  return payload
}

export function fetchSymptoms() {
  return request('/symptoms')
}

export function predictFromSymptoms(symptoms, topK = 5) {
  return request('/predict', {
    method: 'POST',
    body: JSON.stringify({ symptoms, top_k: topK }),
  })
}

export function explainSymptoms(symptoms) {
  return request('/explain', {
    method: 'POST',
    body: JSON.stringify({ symptoms }),
  })
}
