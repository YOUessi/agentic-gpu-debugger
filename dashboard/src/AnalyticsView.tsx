import { useCallback, useEffect, useMemo, useState } from 'react'
import { getAnalyticsOverview, getBatchDetail, getEvaluationDetail } from './api'
import { formatTime, humanize, shortId, statusTone } from './format'
import type {
  AnalyticsOverview,
  BatchCard,
  BatchDetail,
  EvaluationCard,
  EvaluationDetail,
  EvaluationModeSummary,
} from './types'

function StatusPill({ value }: { value: string | null | undefined }) {
  return <span className={'status-pill ' + statusTone(value)}>{humanize(value)}</span>
}

function percent(value: number | null): string {
  return value === null ? '—' : (value * 100).toFixed(1) + '%'
}

function seconds(value: number | null): string {
  return value === null ? '—' : (value / 1000).toFixed(1) + 's'
}

function compact(value: number | null): string {
  if (value === null) return '—'
  if (Math.abs(value) >= 1000000) return (value / 1000000).toFixed(1) + 'M'
  if (Math.abs(value) >= 1000) return (value / 1000).toFixed(1) + 'K'
  return value.toFixed(value % 1 === 0 ? 0 : 1)
}

function money(value: number | null): string {
  return value === null ? '—' : '$' + value.toFixed(4)
}

function Metric({
  label,
  value,
  hint,
}: {
  label: string
  value: string | number
  hint: string
}) {
  return (
    <article className="analytics-metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </article>
  )
}

function ModeCard({ metric }: { metric: EvaluationModeSummary }) {
  return (
    <article className="mode-card">
      <div className="mode-head">
        <span>Mode {metric.mode}</span>
        <strong>{percent(metric.verified_rate)}</strong>
      </div>
      <div className="mode-stats">
        <div><span>Units</span><strong>{metric.record_count}</strong></div>
        <div><span>Diagnosed</span><strong>{metric.diagnosed}</strong></div>
        <div><span>Latency</span><strong>{seconds(metric.latency_mean_ms)}</strong></div>
        <div><span>LLM calls</span><strong>{compact(metric.llm_calls_mean)}</strong></div>
      </div>
    </article>
  )
}

function FailureDistribution({ families }: { families: Record<string, number> }) {
  const entries = Object.entries(families).sort((a, b) => b[1] - a[1])
  const max = Math.max(...entries.map(([, value]) => value), 1)
  return (
    <div className="analytics-family-list">
      {entries.length === 0 && <div className="empty-mini">No diagnosed failure families.</div>}
      {entries.slice(0, 10).map(([family, count]) => (
        <div className="analytics-family-row" key={family}>
          <div><span>{humanize(family)}</span><strong>{count}</strong></div>
          <div className="analytics-family-track">
            <i style={{ width: Math.max(5, count / max * 100) + '%' }} />
          </div>
        </div>
      ))}
    </div>
  )
}

function EvaluationDrawer({
  run,
  onClose,
}: {
  run: EvaluationCard
  onClose: () => void
}) {
  const [detail, setDetail] = useState<EvaluationDetail | null>(null)
  const [page, setPage] = useState(1)
  const [mode, setMode] = useState('')
  const [verdict, setVerdict] = useState('')
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const pageSize = 40

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setDetail(
        await getEvaluationDetail(run.run_id, {
          page,
          pageSize,
          mode: mode || undefined,
          verdict: verdict || undefined,
          query: query || undefined,
        }),
      )
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Evaluation analytics unavailable')
    } finally {
      setLoading(false)
    }
  }, [mode, page, query, run.run_id, verdict])

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void load()
  }, [load])

  const pages = Math.max(1, Math.ceil((detail?.total ?? 0) / pageSize))

  return (
    <div className="drawer-backdrop" onMouseDown={onClose}>
      <aside className="analytics-drawer" onMouseDown={(event) => event.stopPropagation()}>
        <header className="drawer-header">
          <div>
            <span className="eyebrow">Evaluation analytics</span>
            <h2>{shortId(run.run_id, 14)}</h2>
            <p>{humanize(run.split)} · {run.executed_units} executed units</p>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close analytics">×</button>
        </header>
        <div className="analytics-drawer-scroll">
          <section className="analytics-summary-strip">
            <Metric label="Verified" value={percent(run.verified_rate)} hint={String(run.verified_fixed) + ' fixed'} />
            <Metric label="Diagnosed" value={run.diagnosed} hint="grounded diagnoses" />
            <Metric label="Latency" value={seconds(run.latency_mean_ms)} hint="mean / unit" />
            <Metric label="LLM calls" value={compact(run.llm_calls_mean)} hint="mean / unit" />
            <Metric label="Tokens" value={compact(run.tokens_mean)} hint="mean / unit" />
            <Metric label="Known cost" value={money(run.known_cost_usd)} hint="public records" />
          </section>

          {error && <div className="error-banner">{error}</div>}

          <section className="analytics-section">
            <div className="analytics-section-head">
              <div><span className="eyebrow">Ablation view</span><h3>Mode comparison</h3></div>
            </div>
            <div className="mode-grid">
              {(detail?.mode_metrics ?? []).map((metric) => <ModeCard key={metric.mode} metric={metric} />)}
            </div>
          </section>

          <section className="analytics-section two-column">
            <div>
              <div className="analytics-section-head">
                <div><span className="eyebrow">Diagnosis mix</span><h3>Failure families</h3></div>
              </div>
              <FailureDistribution families={detail?.failure_families ?? {}} />
            </div>
            <div className="evaluation-meta">
              <div><span>Corpus cutoff</span><strong>{run.corpus_cutoff ?? '—'}</strong></div>
              <div><span>Expected units</span><strong>{run.expected_units ?? '—'}</strong></div>
              <div><span>Executed units</span><strong>{run.executed_units}</strong></div>
              <div><span>Repeats</span><strong>{run.repeats ?? '—'}</strong></div>
              <div className="wide"><span>Modes</span><strong>{run.modes.join(' · ') || '—'}</strong></div>
            </div>
          </section>

          <section className="analytics-section">
            <div className="analytics-section-head records-head">
              <div><span className="eyebrow">Data grid</span><h3>Evaluation records</h3></div>
              <span className="section-count">{detail?.total ?? 0}</span>
            </div>
            <div className="analytics-filters">
              <input
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value)
                  setPage(1)
                }}
                placeholder="Case, template, or failure family"
              />
              <select
                value={mode}
                onChange={(event) => {
                  setMode(event.target.value)
                  setPage(1)
                }}
              >
                <option value="">All modes</option>
                {run.modes.map((item) => <option key={item} value={item}>Mode {item}</option>)}
              </select>
              <select
                value={verdict}
                onChange={(event) => {
                  setVerdict(event.target.value)
                  setPage(1)
                }}
              >
                <option value="">All verdicts</option>
                <option value="VERIFIED_FIXED">Verified fixed</option>
                <option value="NOT_FIXED">Not fixed</option>
                <option value="INCONCLUSIVE">Inconclusive</option>
                <option value="REGRESSION_DETECTED">Regression</option>
              </select>
              <button className="ghost-button" onClick={() => void load()}>Refresh</button>
            </div>
            <div className="analytics-table-wrap">
              <table className="analytics-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Case</th>
                    <th>Mode</th>
                    <th>Diagnosis</th>
                    <th>Failure</th>
                    <th>Verdict</th>
                    <th>Latency</th>
                    <th>LLM</th>
                    <th>San.</th>
                    <th>Tokens</th>
                    <th>Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {loading && (
                    <tr><td colSpan={11} className="table-message">Loading evaluation records…</td></tr>
                  )}
                  {!loading && (detail?.records.length ?? 0) === 0 && (
                    <tr><td colSpan={11} className="table-message">No records match the filters.</td></tr>
                  )}
                  {!loading && detail?.records.map((record) => (
                    <tr key={record.record_id}>
                      <td>{record.ordinal}</td>
                      <td>
                        <strong>{record.case_id}</strong>
                        <small>{record.template_id}</small>
                      </td>
                      <td><span className="mode-badge">{record.mode}</span></td>
                      <td><StatusPill value={record.diagnosis_outcome} /></td>
                      <td>{humanize(record.failure_family)}</td>
                      <td><StatusPill value={record.verdict} /></td>
                      <td>{seconds(record.latency_ms)}</td>
                      <td>{record.physical_calls ?? '—'}</td>
                      <td>{record.sanitizer_calls ?? '—'}</td>
                      <td>{compact(record.total_tokens)}</td>
                      <td>{money(record.cost_usd)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pagination">
              <span>Page {page} of {pages}</span>
              <div>
                <button disabled={page <= 1} onClick={() => setPage((value) => value - 1)}>
                  Previous
                </button>
                <button disabled={page >= pages} onClick={() => setPage((value) => value + 1)}>
                  Next
                </button>
              </div>
            </div>
          </section>
        </div>
      </aside>
    </div>
  )
}

function BatchDrawer({ run, onClose }: { run: BatchCard; onClose: () => void }) {
  const [detail, setDetail] = useState<BatchDetail | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    void getBatchDetail(run.run_id)
      .then((value) => {
        if (!cancelled) setDetail(value)
      })
      .catch((caught) => {
        if (!cancelled) setError(caught instanceof Error ? caught.message : 'Batch unavailable')
      })
    return () => {
      cancelled = true
    }
  }, [run.run_id])

  return (
    <div className="drawer-backdrop" onMouseDown={onClose}>
      <aside className="analytics-drawer batch-drawer" onMouseDown={(event) => event.stopPropagation()}>
        <header className="drawer-header">
          <div>
            <span className="eyebrow">Seed validation batch</span>
            <h2>{shortId(run.run_id, 14)}</h2>
            <p>{run.case_count} cases · {formatTime(run.finished_at)}</p>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close batch">×</button>
        </header>
        <div className="analytics-drawer-scroll">
          <section className="analytics-summary-strip batch">
            <Metric label="Cases" value={run.case_count} hint="public seeds" />
            <Metric label="Registered" value={run.registered} hint="ledger registered" />
            <Metric label="Validated" value={run.validated} hint="validation passed" />
            <Metric label="Failed" value={run.failed} hint="blocked cases" />
            <Metric label="Running" value={run.running} hint="in progress" />
          </section>

          {error && <div className="error-banner">{error}</div>}

          <section className="analytics-section">
            <div className="analytics-section-head">
              <div><span className="eyebrow">Clean / mutant</span><h3>Seed case results</h3></div>
              <StatusPill value={run.status} />
            </div>
            <div className="analytics-table-wrap">
              <table className="analytics-table batch-table">
                <thead>
                  <tr>
                    <th>Case</th>
                    <th>Tool</th>
                    <th>Status</th>
                    <th>Clean runtime</th>
                    <th>Clean sanitizer</th>
                    <th>Mutant runtime</th>
                    <th>Mutant sanitizer</th>
                    <th>Detection</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {!detail && !error && (
                    <tr><td colSpan={9} className="table-message">Loading batch…</td></tr>
                  )}
                  {detail?.cases.map((item) => (
                    <tr key={item.case_id}>
                      <td>
                        <strong>{item.case_id}</strong>
                        <small>{item.repetitions}× sanitizer</small>
                      </td>
                      <td>{humanize(item.target_tool)}</td>
                      <td><StatusPill value={item.status} /></td>
                      <td>{humanize(item.clean_runtime_status)}</td>
                      <td>{item.clean_sanitizer_outcomes.join(' · ') || '—'}</td>
                      <td>{humanize(item.mutant_runtime_status)}</td>
                      <td>{item.mutant_sanitizer_outcomes.join(' · ') || '—'}</td>
                      <td>
                        {item.target_detections.length
                          ? String(item.target_detections.filter(Boolean).length) +
                            '/' +
                            String(item.target_detections.length)
                          : '—'}
                      </td>
                      <td>{humanize(item.reason_code)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {detail?.stopped_reason && (
              <div className="error-banner">{humanize(detail.stopped_reason)}</div>
            )}
          </section>
        </div>
      </aside>
    </div>
  )
}

export default function AnalyticsView() {
  const [overview, setOverview] = useState<AnalyticsOverview | null>(null)
  const [selectedEvaluation, setSelectedEvaluation] = useState<EvaluationCard | null>(null)
  const [selectedBatch, setSelectedBatch] = useState<BatchCard | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setOverview(await getAnalyticsOverview())
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Analytics unavailable')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void load()
  }, [load])

  const totals = useMemo(() => {
    const evaluations = overview?.evaluations ?? []
    const units = evaluations.reduce((sum, item) => sum + item.executed_units, 0)
    const fixed = evaluations.reduce((sum, item) => sum + item.verified_fixed, 0)
    const cost = evaluations.reduce((sum, item) => sum + (item.known_cost_usd ?? 0), 0)
    const weightedLatency = evaluations.reduce(
      (sum, item) => sum + (item.latency_mean_ms ?? 0) * item.executed_units,
      0,
    )
    return {
      units,
      fixed,
      rate: units ? fixed / units : null,
      cost: evaluations.some((item) => item.known_cost_usd !== null) ? cost : null,
      latency: units ? weightedLatency / units : null,
    }
  }, [overview])

  return (
    <main className="analytics-page">
      <section className="analytics-hero">
        <div>
          <span className="eyebrow">Test intelligence</span>
          <h1>Batch & Evaluation Analytics</h1>
          <p>
            Public operational summaries for CUDA seed validation and A–E evaluation runs.
            Evaluator/private artifacts are not exposed here.
          </p>
        </div>
        <div className="analytics-source">
          <span>Public analytics store</span>
          <code>{overview?.store_root ?? 'Loading…'}</code>
          <button className="ghost-button" onClick={() => void load()}>Refresh</button>
        </div>
      </section>

      {error && <div className="global-error">{error}</div>}
      {(overview?.projection_errors?.length ?? 0) > 0 && (
        <div className="analytics-warning">
          <strong>Projection warning</strong>
          <span>
            {overview?.projection_errors.length} public analytics run(s) could not be projected.
            The console did not silently count them as successful or empty runs.
          </span>
        </div>
      )}

      <section className="analytics-overview-grid">
        <Metric label="Evaluation runs" value={overview?.evaluation_count ?? 0} hint="public manifests" />
        <Metric label="Executed units" value={totals.units} hint="across visible evaluations" />
        <Metric label="Verified rate" value={percent(totals.rate)} hint={String(totals.fixed) + ' fixed'} />
        <Metric label="Mean latency" value={seconds(totals.latency)} hint="weighted / unit" />
        <Metric label="Known cost" value={money(totals.cost)} hint="public records only" />
        <Metric label="Seed batches" value={overview?.batch_count ?? 0} hint="public validation runs" />
      </section>

      <section className="analytics-layout">
        <section className="panel analytics-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">A–E runs</span><h2>Evaluation history</h2></div>
            <span className="panel-count">{overview?.evaluations.length ?? 0}</span>
          </div>
          <div className="analytics-table-wrap">
            <table className="analytics-table evaluation-list">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Split</th>
                  <th>Units</th>
                  <th>Modes</th>
                  <th>Verified</th>
                  <th>Diagnosed</th>
                  <th>Latency</th>
                  <th>LLM calls</th>
                  <th>Tokens</th>
                  <th>Cost</th>
                </tr>
              </thead>
              <tbody>
                {loading && (
                  <tr><td colSpan={10} className="table-message">Loading evaluation manifests…</td></tr>
                )}
                {!loading && (overview?.evaluations.length ?? 0) === 0 && (
                  <tr><td colSpan={10} className="table-message">No public evaluation runs in this store.</td></tr>
                )}
                {!loading && overview?.evaluations.map((item) => (
                  <tr
                    key={item.run_id}
                    className="analytics-click-row"
                    onClick={() => setSelectedEvaluation(item)}
                  >
                    <td><strong>{shortId(item.run_id, 10)}</strong><small>{item.status}</small></td>
                    <td>{humanize(item.split)}</td>
                    <td>{item.executed_units}/{item.expected_units ?? '—'}</td>
                    <td>{item.modes.join(' · ')}</td>
                    <td><strong className="rate-value">{percent(item.verified_rate)}</strong></td>
                    <td>{item.diagnosed}</td>
                    <td>{seconds(item.latency_mean_ms)}</td>
                    <td>{compact(item.llm_calls_mean)}</td>
                    <td>{compact(item.tokens_mean)}</td>
                    <td>{money(item.known_cost_usd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel analytics-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">GPU seed validation</span><h2>Batch history</h2></div>
            <span className="panel-count">{overview?.batches.length ?? 0}</span>
          </div>
          <div className="batch-card-grid">
            {!loading && (overview?.batches.length ?? 0) === 0 && (
              <div className="table-message">No seed batches in this store.</div>
            )}
            {overview?.batches.map((item) => (
              <button
                className="batch-summary-card"
                key={item.run_id}
                onClick={() => setSelectedBatch(item)}
              >
                <div className="batch-summary-head">
                  <div>
                    <strong>{shortId(item.run_id, 11)}</strong>
                    <small>{formatTime(item.finished_at)}</small>
                  </div>
                  <StatusPill value={item.status} />
                </div>
                <div className="batch-summary-stats">
                  <div><span>Cases</span><strong>{item.case_count}</strong></div>
                  <div><span>Registered</span><strong>{item.registered}</strong></div>
                  <div><span>Failed</span><strong>{item.failed}</strong></div>
                  <div><span>Tools</span><strong>{Object.keys(item.target_tools).length}</strong></div>
                </div>
                <div className="tool-tags">
                  {Object.entries(item.target_tools).map(([tool, count]) => (
                    <span key={tool}>{humanize(tool)} · {count}</span>
                  ))}
                </div>
              </button>
            ))}
          </div>
        </section>
      </section>

      {selectedEvaluation && (
        <EvaluationDrawer
          run={selectedEvaluation}
          onClose={() => setSelectedEvaluation(null)}
        />
      )}
      {selectedBatch && (
        <BatchDrawer run={selectedBatch} onClose={() => setSelectedBatch(null)} />
      )}
    </main>
  )
}
