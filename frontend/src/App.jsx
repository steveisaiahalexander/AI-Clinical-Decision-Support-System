import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, AlertCircle, ChevronRight, LoaderCircle, Plus, Search, ShieldAlert, Sparkles, X } from 'lucide-react'
import { explainSymptoms, fetchSymptoms, predictFromSymptoms } from './services/api.js'
import { resultPresentation } from './resultsPresentation.js'

function labelFor(value, keepSymptomSuffix = false) {
  let label = value.replace(/^(?:rare_)?symptom_/, '').replace(/_rare$/, '')
  if (!keepSymptomSuffix) label = label.replace(/_symptom$/, '')
  return label
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .replace(/\bGi\b/g, 'GI')
}

function formatAttribution(value) {
  if (!Number.isFinite(value)) return '—'
  const percentagePoints = value * 100
  if (Math.abs(percentagePoints) < 0.005) return '0.00 pp'
  return `${percentagePoints > 0 ? '+' : ''}${percentagePoints.toFixed(2)} pp`
}

function App() {
  const [symptoms, setSymptoms] = useState([])
  const [selected, setSelected] = useState([])
  const [query, setQuery] = useState('')
  const [loadingSymptoms, setLoadingSymptoms] = useState(true)
  const [symptomError, setSymptomError] = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisError, setAnalysisError] = useState('')
  const [result, setResult] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [explanationLoading, setExplanationLoading] = useState(false)
  const [explanationError, setExplanationError] = useState('')
  const resultsRef = useRef(null)
  const resultsTitleRef = useRef(null)

  async function loadSymptoms() {
    setLoadingSymptoms(true)
    setSymptomError('')
    try {
      const data = await fetchSymptoms()
      if (!Array.isArray(data?.symptoms)) throw new Error('The symptom list returned an unexpected response.')
      setSymptoms(data.symptoms)
    } catch (error) {
      setSymptomError(error.message)
    } finally {
      setLoadingSymptoms(false)
    }
  }

  useEffect(() => { loadSymptoms() }, [])

  useEffect(() => {
    if (!result) return
    resultsTitleRef.current?.focus({ preventScroll: true })
    resultsRef.current?.scrollIntoView({
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
      block: 'start',
    })
  }, [result])

  const symptomLabels = useMemo(() => {
    const labels = symptoms.map((symptom) => labelFor(symptom))
    const frequencies = new Map()
    labels.forEach((label) => frequencies.set(label, (frequencies.get(label) || 0) + 1))

    return new Map(symptoms.map((symptom, index) => [
      symptom,
      frequencies.get(labels[index]) > 1 && /_symptom$/.test(symptom)
        ? labelFor(symptom, true)
        : labels[index],
    ]))
  }, [symptoms])

  const filteredSymptoms = useMemo(() => {
    const tokens = query.trim().toLowerCase().split(/\s+/).filter(Boolean)
    return symptoms.filter((symptom) => {
      const label = symptomLabels.get(symptom).toLowerCase()
      return !selected.includes(symptom) && tokens.every((token) => label.includes(token))
    })
  }, [query, selected, symptomLabels, symptoms])

  const rankedPredictions = Array.isArray(result?.top_predictions) ? result.top_predictions : []
  const primaryPrediction = rankedPredictions.find((prediction) => Number(prediction.rank) === 1) || rankedPredictions[0]
  const primaryDisease = result?.predicted_disease || primaryPrediction?.disease
  const primaryPercentage = Number(result?.predicted_percentage ?? primaryPrediction?.percentage)
  const alternatives = rankedPredictions.filter((prediction) => prediction.disease !== primaryDisease)
  const resultView = result ? resultPresentation(result) : null
  const visibleRankings = resultView?.insufficientEvidence ? rankedPredictions : alternatives

  function addSymptom(symptom) {
    setSelected((current) => [...current, symptom])
    setQuery('')
    setResult(null)
    setExplanation(null)
    setExplanationError('')
    setAnalysisError('')
  }

  function removeSymptom(symptom) {
    setSelected((current) => current.filter((item) => item !== symptom))
    setResult(null)
    setExplanation(null)
    setExplanationError('')
    setAnalysisError('')
  }

  function clearSelection() {
    setSelected([])
    setResult(null)
    setExplanation(null)
    setExplanationError('')
    setAnalysisError('')
  }

  async function analyze() {
    if (selected.length === 0 || analyzing) return
    setAnalyzing(true)
    setAnalysisError('')
    setResult(null)
    setExplanation(null)
    setExplanationError('')
    try {
      const prediction = await predictFromSymptoms(selected)
      setResult(prediction)
      await loadExplanation(selected)
    } catch (error) {
      setAnalysisError(error.message)
    } finally {
      setAnalyzing(false)
    }
  }

  async function loadExplanation(symptomsToExplain) {
    setExplanationLoading(true)
    setExplanationError('')
    try {
      const data = await explainSymptoms(symptomsToExplain)
      setExplanation(data)
    } catch (error) {
      setExplanationError(error.message)
    } finally {
      setExplanationLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Clinical Assessment home">
          <span className="brand-mark"><Activity size={19} strokeWidth={2.2} /></span>
          <span>Clinical<span className="brand-light">Assessment</span></span>
        </a>
        <div className="topbar-right"><span className="status-dot" /> Decision support demo</div>
      </header>

      <main id="top" className="main-content">
        <section className="intro" aria-labelledby="page-title">
          <div className="eyebrow"><span>ASSESSMENT</span><span className="eyebrow-line" /></div>
          <h1 id="page-title">Start with what<br className="desktop-break" /> you’re experiencing.</h1>
          <p className="intro-copy">Select the symptoms that apply. The model will return a ranked set of possible classes for review.</p>
        </section>

        <section className="assessment" aria-label="Symptom assessment">
          <div className="section-heading">
            <div><span className="step-number">01</span><h2>Choose symptoms</h2></div>
            <span className="available-count" role="status" aria-live="polite">{loadingSymptoms ? 'Loading list' : query.trim() ? `${filteredSymptoms.length} matches` : `${symptoms.length} available`}</span>
          </div>

          <div className="workspace">
            <div className="picker-column">
              <label className="search-label" htmlFor="symptom-search">Search symptoms</label>
              <div className="search-wrap">
                <Search size={18} aria-hidden="true" />
                <input id="symptom-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Try “headache” or “fatigue”" autoComplete="off" aria-controls="symptom-options" />
                {query && <button className="icon-button search-clear" type="button" onClick={() => setQuery('')} aria-label="Clear search"><X size={16} /></button>}
              </div>

              <div id="symptom-options" className="symptom-list" aria-label="Available symptoms" aria-busy={loadingSymptoms}>
                {loadingSymptoms && <div className="list-state"><LoaderCircle className="spin" size={20} /><span>Loading available symptoms…</span></div>}
                {!loadingSymptoms && symptomError && <div className="list-state error-state"><AlertCircle size={20} /><span>{symptomError}</span><button className="text-button" type="button" onClick={loadSymptoms}>Try again</button></div>}
                {!loadingSymptoms && !symptomError && filteredSymptoms.length === 0 && <div className="list-state"><span>{query ? 'No symptoms match your search.' : symptoms.length === 0 ? 'No symptoms are available from the service.' : 'All available symptoms are selected.'}</span></div>}
                {!loadingSymptoms && !symptomError && filteredSymptoms.map((symptom) => (
                  <button className="symptom-option" type="button" key={symptom} onClick={() => addSymptom(symptom)} disabled={analyzing}>
                    <span>{symptomLabels.get(symptom)}</span><Plus size={17} aria-hidden="true" />
                  </button>
                ))}
              </div>
              {!loadingSymptoms && !symptomError && <p className="list-footnote">Choose all symptoms currently present.</p>}
            </div>

            <aside className="selection-column" aria-labelledby="selected-title">
              <div className="selection-heading">
                <div><h3 id="selected-title">Selected</h3><span className="selection-count">{selected.length}</span></div>
                {selected.length > 0 && <button type="button" className="text-button" onClick={clearSelection} disabled={analyzing}>Clear all</button>}
              </div>

              <div className={`selected-area ${selected.length === 0 ? 'is-empty' : ''}`} aria-label="Selected symptoms" aria-live="polite">
                {selected.length === 0 ? <div className="empty-selection"><span className="empty-icon"><Plus size={19} /></span><p>Your selected symptoms<br />will appear here.</p></div> : (
                  <ul className="chip-list">
                    {selected.map((symptom) => <li key={symptom} className="symptom-chip"><span>{symptomLabels.get(symptom)}</span><button type="button" onClick={() => removeSymptom(symptom)} aria-label={`Remove ${symptomLabels.get(symptom)}`} disabled={analyzing}><X size={14} /></button></li>)}
                  </ul>
                )}
              </div>

              <button className="analyze-button" type="button" onClick={analyze} disabled={selected.length === 0 || analyzing || loadingSymptoms || Boolean(symptomError)}>
                {analyzing ? <><LoaderCircle className="spin" size={17} /> Analyzing</> : <>Analyze symptoms <ChevronRight size={18} /></>}
              </button>
              <p className="action-note">{selected.length === 0 ? 'Select at least one symptom to continue' : 'Results are generated by the saved model'}</p>
            </aside>
          </div>
        </section>

        {analyzing && <div className="feedback loading-feedback" role="status"><LoaderCircle className="spin" size={19} /><span>{result ? 'Explaining the model output…' : 'Analyzing the selected symptoms…'}</span></div>}
        {analysisError && <div className="feedback error-feedback" role="alert"><AlertCircle size={19} /><span>{analysisError}</span></div>}

        {result && <section ref={resultsRef} className="results" aria-labelledby="results-title">
          <div className="results-top"><div className="eyebrow"><span>ENSEMBLE PREDICTION</span><span className="eyebrow-line" /></div><span className="result-badge"><Sparkles size={14} /> {explanationLoading ? 'Assessment ready' : 'Assessment complete'}</span></div>
          <h2 id="results-title" ref={resultsTitleRef} tabIndex={-1}>Assessment results</h2>
          <p className="results-intro">A ranked model assessment based on the symptoms selected above.</p>
          {!primaryDisease || !Number.isFinite(primaryPercentage) ? (
            <div className="results-empty" role="status"><AlertCircle size={19} /><p>No ranked predictions were returned. Review the selected symptoms and try again.</p></div>
          ) : (
            <>
              <div className={'primary-result ' + (resultView.insufficientEvidence ? 'is-insufficient' : '')}>
                <div className="primary-copy">
                  <span className="primary-label">{resultView.primaryLabel}</span>
                  <h3>{resultView.primaryHeading ? labelFor(resultView.primaryHeading) : ''}</h3>
                  {resultView.primaryMessage && <p className="insufficient-message">{resultView.primaryMessage}</p>}
                </div>
                <div className="primary-probability"><span>{resultView.probabilityLabel}</span><strong>{primaryPercentage.toFixed(2)}%</strong></div>
              </div>
              {result.uncertainty && <section className={'uncertainty-summary ' + (resultView.insufficientEvidence ? 'is-insufficient' : '')} aria-label="Model uncertainty and separation">
                <div className="uncertainty-heading">
                  <span className="eyebrow">UNCERTAINTY / SEPARATION</span>
                  <strong>{resultView.insufficientEvidence ? 'Insufficient evidence' : 'Model separation'}</strong>
                </div>
                <div className="uncertainty-values">
                  <div><span>Top probability</span><strong>{(Number(result.uncertainty.top_probability) * 100).toFixed(2)}%</strong></div>
                  <div><span>Top-two margin</span><strong>{(Number(result.uncertainty.top_two_margin) * 100).toFixed(2)} pp</strong></div>
                  <div><span>Normalized entropy</span><strong>{(Number(result.uncertainty.normalized_entropy) * 100).toFixed(1)}%</strong></div>
                </div>
                <p className="uncertainty-note">These signals describe the model’s probability distribution, not clinical certainty. Abstention thresholds are development settings and are not clinically validated.</p>
              </section>}
              <div className="alternatives-heading"><h3>{resultView.rankedHeading}</h3><span>{visibleRankings.length}</span></div>
              {visibleRankings.length === 0 ? <p className="no-alternatives">No alternative predictions were included.</p> : (
                <div className="prediction-list">
                  {visibleRankings.map((prediction) => {
                    const percentage = Number(prediction.percentage)
                    const safePercentage = Number.isFinite(percentage) ? percentage : 0
                    return <div className="prediction-row" key={`${prediction.rank}-${prediction.disease}`}>
                      <span className="prediction-rank">{String(prediction.rank).padStart(2, '0')}</span>
                      <div className="prediction-detail"><span className="prediction-name">{labelFor(prediction.disease)}</span><span className="probability-track" aria-hidden="true"><span style={{ width: `${Math.max(0, Math.min(safePercentage, 100))}%` }} /></span></div>
                      <span className="prediction-score">{Number.isFinite(percentage) ? `${percentage.toFixed(2)}%` : '—'}</span>
                    </div>
                  })}
                </div>
              )}
              <p className="probability-note">Probabilities are model outputs for this selected feature pattern. They are not a diagnosis or a measure of clinical certainty.</p>
              <section className="explanation" aria-labelledby="explanation-title">
                <div className="explanation-heading">
                  <div><span className="eyebrow">MODEL EXPLANATION</span><h3 id="explanation-title">Why was this considered?</h3></div>
                </div>
                <p className="explanation-note">{explanation?.explanation_note || 'Calculating how the selected symptoms contribute to the ensemble probability for this class.'}</p>
                {explanationLoading && <div className="explanation-status" role="status"><LoaderCircle className="spin" size={17} /><span>Calculating feature attributions…</span></div>}
                {explanationError && <div className="explanation-error" role="alert"><AlertCircle size={17} /><span>{explanationError}</span><button className="text-button" type="button" onClick={() => loadExplanation(selected)} disabled={explanationLoading}>Try again</button></div>}
                {explanation?.features?.length > 0 && <>
                  <div className="explanation-components">
                    <div className="explanation-component">
                      <div className="component-heading"><h4>Ensemble probability attribution</h4><span>{explanation.is_exact ? 'Exact SHAP' : 'Approximate SHAP'}</span></div>
                      <ul className="attribution-list">
                        {explanation.features.map((item) => {
                          const value = Number(item.shap_value)
                          const negligible = !Number.isFinite(value) || Math.abs(value) < 1e-12
                          const direction = negligible ? 'does not shift' : value > 0 ? 'increases' : 'decreases'
                          return <li key={item.feature}>
                            <span className="attribution-name">{symptomLabels.get(item.feature) || labelFor(item.feature)}</span>
                            <span className={`attribution-value ${!negligible && value > 0 ? 'positive' : !negligible && value < 0 ? 'negative' : ''}`} aria-label={`${direction} the ensemble probability`}>
                              {formatAttribution(value)}
                            </span>
                          </li>
                        })}
                      </ul>
                    </div>
                  </div>
                  <p className="explanation-disclaimer">Feature attributions describe model behavior relative to a reference input. They do not establish causation or provide a clinical explanation.</p>
                </>}
              </section>
            </>
          )}
        </section>}

        <section className="disclaimer" aria-label="Clinical decision support disclaimer">
          <span className="disclaimer-icon"><ShieldAlert size={18} /></span>
          <p><strong>For decision support only.</strong> This educational demonstration is not a medical diagnosis and does not replace advice from a qualified clinician. If symptoms are severe or rapidly worsening, seek urgent medical care.</p>
        </section>
        <footer className="page-footer"><span>AI Clinical Decision Support System</span><span>Educational use only</span></footer>
      </main>
    </div>
  )
}

export default App
