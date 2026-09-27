import { useEffect, useMemo, useState } from 'react'
import { Activity, AlertCircle, Check, ChevronRight, LoaderCircle, Plus, Search, ShieldAlert, Sparkles, X } from 'lucide-react'
import { fetchSymptoms, predictFromSymptoms } from './services/api.js'

function labelFor(value) {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
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

  const filteredSymptoms = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    return symptoms.filter((symptom) => !selected.includes(symptom) && labelFor(symptom).toLowerCase().includes(normalized))
  }, [query, selected, symptoms])

  function addSymptom(symptom) {
    setSelected((current) => [...current, symptom])
    setQuery('')
    setResult(null)
    setAnalysisError('')
  }

  function removeSymptom(symptom) {
    setSelected((current) => current.filter((item) => item !== symptom))
    setResult(null)
  }

  function clearSelection() {
    setSelected([])
    setResult(null)
    setAnalysisError('')
  }

  async function analyze() {
    if (selected.length === 0 || analyzing) return
    setAnalyzing(true)
    setAnalysisError('')
    setResult(null)
    try {
      setResult(await predictFromSymptoms(selected))
    } catch (error) {
      setAnalysisError(error.message)
    } finally {
      setAnalyzing(false)
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
            <span className="available-count">{loadingSymptoms ? 'Loading list' : `${symptoms.length} available`}</span>
          </div>

          <div className="workspace">
            <div className="picker-column">
              <label className="search-label" htmlFor="symptom-search">Search symptoms</label>
              <div className="search-wrap">
                <Search size={18} aria-hidden="true" />
                <input id="symptom-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Try “headache” or “fatigue”" autoComplete="off" />
                {query && <button className="icon-button search-clear" type="button" onClick={() => setQuery('')} aria-label="Clear search"><X size={16} /></button>}
              </div>

              <div className="symptom-list" aria-label="Available symptoms" aria-busy={loadingSymptoms}>
                {loadingSymptoms && <div className="list-state"><LoaderCircle className="spin" size={20} /><span>Loading available symptoms…</span></div>}
                {!loadingSymptoms && symptomError && <div className="list-state error-state"><AlertCircle size={20} /><span>{symptomError}</span><button className="text-button" type="button" onClick={loadSymptoms}>Try again</button></div>}
                {!loadingSymptoms && !symptomError && filteredSymptoms.length === 0 && <div className="list-state"><span>{query ? 'No symptoms match your search.' : symptoms.length === 0 ? 'No symptoms are available from the service.' : 'All available symptoms are selected.'}</span></div>}
                {!loadingSymptoms && !symptomError && filteredSymptoms.map((symptom) => (
                  <button className="symptom-option" type="button" key={symptom} onClick={() => addSymptom(symptom)}>
                    <span>{labelFor(symptom)}</span><Plus size={17} aria-hidden="true" />
                  </button>
                ))}
              </div>
              {!loadingSymptoms && !symptomError && <p className="list-footnote">Choose all symptoms currently present.</p>}
            </div>

            <aside className="selection-column" aria-labelledby="selected-title">
              <div className="selection-heading">
                <div><h3 id="selected-title">Selected</h3><span className="selection-count">{selected.length}</span></div>
                {selected.length > 0 && <button type="button" className="text-button" onClick={clearSelection}>Clear all</button>}
              </div>

              <div className={`selected-area ${selected.length === 0 ? 'is-empty' : ''}`} aria-live="polite">
                {selected.length === 0 ? <div className="empty-selection"><span className="empty-icon"><Plus size={19} /></span><p>Your selected symptoms<br />will appear here.</p></div> : (
                  <ul className="chip-list">
                    {selected.map((symptom) => <li key={symptom} className="symptom-chip"><span>{labelFor(symptom)}</span><button type="button" onClick={() => removeSymptom(symptom)} aria-label={`Remove ${labelFor(symptom)}`}><X size={14} /></button></li>)}
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

        {analysisError && <div className="feedback error-feedback" role="alert"><AlertCircle size={19} /><span>{analysisError}</span></div>}

        {result && <section className="results" aria-labelledby="results-title" aria-live="polite">
          <div className="results-top"><div className="eyebrow"><span>MODEL OUTPUT</span><span className="eyebrow-line" /></div><span className="result-badge"><Sparkles size={14} /> Assessment complete</span></div>
          <h2 id="results-title">Ranked possibilities</h2>
          <p className="results-intro">These are model classifications for the selected symptom pattern.</p>
          <div className="prediction-list">
            {result.top_predictions?.map((prediction) => <div className={`prediction-row ${prediction.rank === 1 ? 'prediction-primary' : ''}`} key={prediction.rank}>
              <span className="prediction-rank">{String(prediction.rank).padStart(2, '0')}</span>
              <span className="prediction-name">{labelFor(prediction.disease)}</span>
              <span className="prediction-score">{Number(prediction.percentage).toFixed(2)}%</span>
              {prediction.rank === 1 && <Check className="prediction-check" size={17} aria-label="Highest ranked" />}
            </div>)}
          </div>
          <p className="probability-note">Percentages are model outputs for this feature vector, not a measure of clinical certainty.</p>
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
