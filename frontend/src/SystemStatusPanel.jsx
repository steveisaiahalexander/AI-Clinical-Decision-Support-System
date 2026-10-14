import { useState } from 'react'
import { AlertCircle, CheckCircle2, RefreshCw } from 'lucide-react'
import { fetchSystemStatus } from './services/api.js'
import './systemStatus.css'

function readableStatus(status) {
  return status === 'ready' ? 'Ready' : status === 'degraded' ? 'Degraded' : 'Unavailable'
}

export default function SystemStatusPanel() {
  const [opened, setOpened] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [status, setStatus] = useState(null)

  async function refresh() {
    setLoading(true)
    setError('')
    try {
      setStatus(await fetchSystemStatus())
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }

  function toggle(event) {
    const nextOpened = event.currentTarget.open
    setOpened(nextOpened)
    if (nextOpened && !status && !loading) refresh()
  }

  return <details className="system-status" onToggle={toggle}>
    <summary>Development service status</summary>
    {opened && <div className="system-status-content">
      <div className="system-status-heading">
        <div><span className="eyebrow">DEVELOPMENT</span><h3>Service status</h3></div>
        <button className="icon-button" type="button" onClick={refresh} disabled={loading} aria-label="Refresh service status" title="Refresh service status"><RefreshCw size={16} className={loading ? 'status-spin' : ''} /></button>
      </div>
      {loading && <p className="status-message" role="status">Checking local services…</p>}
      {error && <p className="status-message status-error" role="alert"><AlertCircle size={16} />{error}</p>}
      {status && <>
        <p className={`overall-status status-${status.overall_status}`}><span>{status.overall_status === 'ready' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}</span>Overall: {readableStatus(status.overall_status)}</p>
        <ul className="component-status-list">{Object.entries(status.components).map(([name, component]) => <li key={name}>
          <span>{name.replaceAll('_', ' ')}</span><strong className={`status-${component.status}`}>{readableStatus(component.status)}</strong>
          {component.details?.configured_model && <small>{component.details.configured_model}</small>}
          {component.details?.documents != null && <small>{component.details.documents} documents · {component.details.chunks} passages</small>}
        </li>)}</ul>
        <div className="telemetry-summary">
          <span>Requests <strong>{status.observability.requests_total}</strong></span>
          <span>Errors <strong>{status.observability.errors_total}</strong></span>
          <span>Mean latency <strong>{status.observability.request_latency.mean_ms == null ? '—' : `${status.observability.request_latency.mean_ms} ms`}</strong></span>
          <span>Retrievals <strong>{status.observability.retrieval.requests}</strong></span>
          <span>LLM generated <strong>{status.observability.llm_generation.outcomes.generated}</strong></span>
          <span>Abstained <strong>{status.observability.predictions.abstained}</strong></span>
        </div>
        {status.evaluation_report_available && <a className="evaluation-report-link" href={status.evaluation_report_url}>Open unified evaluation report</a>}
        <p className="status-disclaimer">{status.disclaimer} Counters reset when the API process restarts.</p>
      </>}
    </div>}
  </details>
}
