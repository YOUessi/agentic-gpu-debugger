import { useCallback, useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'
import {
  getArtifact,
  getCases,
  getRepairJob,
  getRun,
  getRuns,
  getStats,
  startRepairJob,
  verifyRun,
} from './api'
import { formatBytes, formatTime, humanize, shortId, statusTone } from './format'
import type {
  ArtifactSummary,
  CaseSummary,
  CitationTarget,
  RepairJob,
  RepairRequest,
  RunDetail,
  RunStats,
  RunSummary,
} from './types'

type JsonObject = Record<string, unknown>

function asObject(value: unknown): JsonObject {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as JsonObject) : {}
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function StatusPill({ value }: { value: string | null | undefined }) {
  return <span className={'status-pill ' + statusTone(value)}>{humanize(value)}</span>
}

function MetricCard({
  label,
  value,
  hint,
  accent,
}: {
  label: string
  value: number
  hint: string
  accent?: boolean
}) {
  return (
    <article className={'metric-card' + (accent ? ' accent' : '')}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-hint">{hint}</div>
    </article>
  )
}

function FailureFamilies({ stats }: { stats: RunStats | null }) {
  const entries = Object.entries(stats?.failure_families ?? {}).sort((a, b) => b[1] - a[1])
  const max = Math.max(...entries.map(([, count]) => count), 1)
  return (
    <section className="panel side-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">AI diagnosis</span>
          <h2>Failure families</h2>
        </div>
        <span className="panel-count">{entries.reduce((sum, [, value]) => sum + value, 0)}</span>
      </div>
      <div className="family-list">
        {entries.length === 0 && <div className="empty-mini">No classified failures yet.</div>}
        {entries.map(([family, count]) => (
          <div className="family-row" key={family}>
            <div className="family-copy">
              <span>{humanize(family)}</span>
              <strong>{count}</strong>
            </div>
            <div className="family-track">
              <span style={{ width: Math.max(8, (count / max) * 100) + '%' }} />
            </div>
          </div>
        ))}
      </div>
      <div className="side-note">
        <span className="signal-dot" />
        Classification is derived from persisted diagnosis artifacts, not recomputed in the UI.
      </div>
    </section>
  )
}

function RunTable({
  runs,
  loading,
  onSelect,
}: {
  runs: RunSummary[]
  loading: boolean
  onSelect: (run: RunSummary) => void
}) {
  return (
    <div className="table-wrap">
      <table className="run-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>Status</th>
            <th>Diagnosis</th>
            <th>Failure</th>
            <th>Repair</th>
            <th>Verification</th>
            <th>Updated</th>
          </tr>
        </thead>
        <tbody>
          {loading && (
            <tr>
              <td colSpan={7} className="table-message">Loading run inventory…</td>
            </tr>
          )}
          {!loading && runs.length === 0 && (
            <tr>
              <td colSpan={7} className="table-message">No runs match the current filters.</td>
            </tr>
          )}
          {!loading &&
            runs.map((run) => (
              <tr key={run.id} onClick={() => onSelect(run)} className="run-row">
                <td>
                  <div className="run-id">{shortId(run.id, 10)}</div>
                  <div className="run-kind">{humanize(run.kind)}</div>
                </td>
                <td><StatusPill value={run.status} /></td>
                <td>
                  <div className="diagnosis-cell">
                    <StatusPill value={run.diagnosis_outcome} />
                    {run.confidence && <span className="confidence">{run.confidence}</span>}
                  </div>
                </td>
                <td>{run.failure_family ? <span className="failure-tag">{humanize(run.failure_family)}</span> : '—'}</td>
                <td><StatusPill value={run.repair_stop_reason} /></td>
                <td><StatusPill value={run.verification_verdict} /></td>
                <td className="time-cell">{formatTime(run.last_event_at)}</td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  )
}

function RepairRound({ round }: { round: JsonObject }) {
  const number = Number(round.round ?? 0)
  const result = asObject(round.result)
  const candidate = asObject(round.candidate)
  const checks = asObject(result.checks)
  const diff = asString(candidate.unified_diff)
  return (
    <article className="repair-round">
      <div className="round-head">
        <div className="round-number">#{number}</div>
        <div>
          <strong>Candidate {number}</strong>
          <span>{shortId(asString(candidate.patched_source_hash), 12)}</span>
        </div>
        <StatusPill value={asString(result.status)} />
      </div>
      <div className="check-grid">
        {Object.entries(checks).map(([name, outcome]) => (
          <div className="check-chip" key={name}>
            <span>{humanize(name)}</span>
            <StatusPill value={String(outcome)} />
          </div>
        ))}
      </div>
      {diff && <pre className="diff-view">{diff}</pre>}
    </article>
  )
}

function InvestigationTrace({ actions }: { actions: JsonObject[] }) {
  return (
    <div className="trace-list">
      {actions.length === 0 && <div className="empty-mini">No persisted agent actions.</div>}
      {actions.map((entry) => {
        const proposal = asObject(entry.proposal)
        const action = asObject(proposal.action)
        const decision = asObject(entry.decision)
        const actionType = asString(action.action_type)
        return (
          <div className="trace-row" key={String(entry.step)}>
            <div className="trace-index">{Number(entry.step) + 1}</div>
            <div className="trace-copy">
              <strong>{humanize(actionType)}</strong>
              <span>{asString(action.rationale) || 'Controller-recorded investigation action'}</span>
            </div>
            <StatusPill value={decision.allowed === false ? 'FAILED' : 'COMPLETED'} />
          </div>
        )
      })}
    </div>
  )
}

function ArtifactViewer({
  runId,
  artifacts,
}: {
  runId: string
  artifacts: ArtifactSummary[]
}) {
  const [selected, setSelected] = useState<ArtifactSummary | null>(null)
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(false)

  async function openArtifact(artifact: ArtifactSummary) {
    setSelected(artifact)
    setLoading(true)
    try {
      setContent(await getArtifact(runId, artifact.id))
    } catch (error) {
      setContent(error instanceof Error ? error.message : 'Unable to read artifact')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="artifact-layout">
      <div className="artifact-list">
        {artifacts.map((artifact) => (
          <button key={artifact.id} onClick={() => void openArtifact(artifact)} className={selected?.id === artifact.id ? 'active' : ''}>
            <span>{artifact.name}</span>
            <small>{formatBytes(artifact.byte_count)}</small>
          </button>
        ))}
      </div>
      <pre className="artifact-content">
        {loading ? 'Loading artifact…' : selected ? content : 'Select an artifact to inspect its persisted contents.'}
      </pre>
    </div>
  )
}

function EvidenceClaims({
  runId,
  diagnosis,
  citations,
}: {
  runId: string
  diagnosis: JsonObject
  citations: Record<string, CitationTarget>
}) {
  const [selected, setSelected] = useState<CitationTarget | null>(null)
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(false)
  const groups = [
    ['Observed facts', diagnosis.observed_facts],
    ['Tool findings', diagnosis.tool_findings],
    ['Documentation', diagnosis.documentation_evidence],
  ] as const

  async function openCitation(target: CitationTarget) {
    setSelected(target)
    setContent(target.preview || '')
    setLoading(true)
    try {
      setContent(await getArtifact(runId, target.artifact_id))
    } catch (caught) {
      setContent(
        target.preview ||
          (caught instanceof Error ? caught.message : 'Citation artifact is unavailable'),
      )
    } finally {
      setLoading(false)
    }
  }

  const hasClaims = groups.some(([, value]) => Array.isArray(value) && value.length > 0)
  if (!hasClaims) return <div className="empty-mini">No structured evidence claims persisted.</div>

  return (
    <div className="evidence-claims">
      <div className="claim-groups">
        {groups.map(([label, value]) => {
          const claims = Array.isArray(value) ? value : []
          if (claims.length === 0) return null
          return (
            <div className="claim-group" key={label}>
              <h4>{label}</h4>
              {claims.map((raw, index) => {
                const claim = asObject(raw)
                const ids = Array.isArray(claim.citation_ids)
                  ? claim.citation_ids.filter((item): item is string => typeof item === 'string')
                  : []
                return (
                  <article className="claim-card" key={label + index}>
                    <p>{asString(claim.text) || 'Evidence claim'}</p>
                    <div className="citation-row">
                      {ids.map((id) => {
                        const target = citations[id]
                        return target ? (
                          <button
                            type="button"
                            className="citation-chip"
                            key={id}
                            onClick={() => void openCitation(target)}
                          >
                            {target.kind === 'document' ? 'DOC' : 'ART'} · {shortId(id, 12)}
                          </button>
                        ) : (
                          <span className="citation-chip unresolved" key={id}>
                            UNRESOLVED · {shortId(id, 12)}
                          </span>
                        )
                      })}
                    </div>
                  </article>
                )
              })}
            </div>
          )
        })}
      </div>
      <aside className="citation-preview">
        {selected ? (
          <>
            <div className="citation-preview-head">
              <div>
                <span className="eyebrow">{selected.kind}</span>
                <strong>{selected.label}</strong>
              </div>
              <span>{shortId(selected.citation_id, 14)}</span>
            </div>
            <pre>{loading ? 'Loading persisted citation…' : content}</pre>
          </>
        ) : (
          <div className="citation-empty">Select a citation to inspect its persisted evidence.</div>
        )}
      </aside>
    </div>
  )
}

function ControllerTimeline({ events }: { events: JsonObject[] }) {
  if (events.length === 0) {
    return <div className="empty-mini">No controller state events persisted.</div>
  }
  return (
    <div className="controller-timeline">
      {events.map((event, index) => {
        const status = asString(event.status)
        const phase = asString(event.phase)
        const at = asString(event.at)
        return (
          <div className="timeline-event" key={at + index}>
            <div className={'timeline-dot ' + statusTone(status)} />
            <div className="timeline-event-copy">
              <div>
                <strong>{phase ? humanize(phase) : humanize(status)}</strong>
                <StatusPill value={status} />
              </div>
              <span>{formatTime(at || null)}</span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function SourceCompare({
  runId,
  artifacts,
  candidate,
}: {
  runId: string
  artifacts: ArtifactSummary[]
  candidate: JsonObject | null
}) {
  const sourceArtifact = artifacts.find((item) => item.name === 'sources/kernel.cu')
  const sourceArtifactId = sourceArtifact?.id
  const [source, setSource] = useState('')
  const [error, setError] = useState('')
  const diff = asString(candidate?.unified_diff)

  useEffect(() => {
    let cancelled = false
    if (!sourceArtifactId) return undefined
    void getArtifact(runId, sourceArtifactId)
      .then((value) => {
        if (!cancelled) setSource(value)
      })
      .catch((caught) => {
        if (!cancelled) setError(caught instanceof Error ? caught.message : 'Source unavailable')
      })
    return () => {
      cancelled = true
    }
  }, [runId, sourceArtifactId])

  if (!sourceArtifact && !diff) {
    return <div className="empty-mini">No source/candidate comparison is available.</div>
  }

  return (
    <div className="source-compare">
      <div className="source-pane">
        <div className="source-pane-head">
          <span>Original kernel.cu</span>
          <small>{sourceArtifact ? formatBytes(sourceArtifact.byte_count) : '—'}</small>
        </div>
        <pre>{error || source || 'Loading original source…'}</pre>
      </div>
      <div className="source-pane">
        <div className="source-pane-head">
          <span>Selected candidate diff</span>
          <small>{shortId(asString(candidate?.patched_source_hash), 12)}</small>
        </div>
        <pre className="candidate-diff">{diff || 'No registered candidate diff.'}</pre>
      </div>
    </div>
  )
}

function RunDrawer({
  detail,
  onClose,
  onChanged,
}: {
  detail: RunDetail
  onClose: () => void
  onChanged: () => Promise<void>
}) {
  const [verifying, setVerifying] = useState(false)
  const [error, setError] = useState('')
  const diagnosis = asObject(detail.diagnosis)
  const repairSummary = asObject(detail.repair_summary)
  const rounds = detail.repair_rounds as JsonObject[]
  const actions = detail.actions as JsonObject[]

  async function verify() {
    setVerifying(true)
    setError('')
    try {
      await verifyRun(detail.summary.id, true)
      await onChanged()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Verification failed')
    } finally {
      setVerifying(false)
    }
  }

  const pipeline = [
    ['Investigation', actions.length > 0 ? 'COMPLETED' : detail.summary.status],
    ['Diagnosis', detail.summary.diagnosis_outcome],
    ['Repair', asString(repairSummary.stop_reason)],
    ['Verification', detail.summary.verification_verdict],
  ]

  return (
    <div className="drawer-backdrop" onMouseDown={onClose}>
      <aside className="drawer" onMouseDown={(event) => event.stopPropagation()}>
        <header className="drawer-header">
          <div>
            <span className="eyebrow">Run detail</span>
            <h2>{shortId(detail.summary.id, 14)}</h2>
            <p>{humanize(detail.summary.failure_family)} · {detail.summary.artifact_count} artifacts</p>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close">×</button>
        </header>

        <div className="drawer-scroll">
          <section className="pipeline">
            {pipeline.map(([label, status], index) => (
              <div className="pipeline-step" key={label}>
                <div className="pipeline-marker">{index + 1}</div>
                <div>
                  <span>{label}</span>
                  <StatusPill value={status} />
                </div>
              </div>
            ))}
          </section>

          <section className="detail-section">
            <div className="section-title">
              <div><span className="eyebrow">Controller state</span><h3>Live pipeline timeline</h3></div>
              <span className="section-count">{detail.events.length} events</span>
            </div>
            <ControllerTimeline events={detail.events as JsonObject[]} />
          </section>

          <section className="detail-section">
            <div className="section-title">
              <div><span className="eyebrow">Grounded analysis</span><h3>Diagnosis</h3></div>
              <StatusPill value={detail.summary.confidence} />
            </div>
            <div className="detail-grid">
              <div className="detail-card wide">
                <label>Root cause</label>
                <p>{asString(diagnosis.root_cause) || 'No root-cause statement persisted.'}</p>
              </div>
              <div className="detail-card wide">
                <label>Recommended change</label>
                <p>{asString(diagnosis.recommended_change) || 'No recommendation persisted.'}</p>
              </div>
            </div>
            <div className="evidence-block">
              <div className="evidence-block-title">
                <span>Evidence & citations</span>
                <small>{Object.keys(detail.citations).length} resolved</small>
              </div>
              <EvidenceClaims
                runId={detail.summary.id}
                diagnosis={diagnosis}
                citations={detail.citations}
              />
            </div>
          </section>

          <section className="detail-section">
            <div className="section-title">
              <div><span className="eyebrow">Agent trajectory</span><h3>Investigation trace</h3></div>
              <span className="section-count">{actions.length} steps</span>
            </div>
            <InvestigationTrace actions={actions} />
          </section>

          <section className="detail-section">
            <div className="section-title">
              <div><span className="eyebrow">Bounded revision</span><h3>Repair candidates</h3></div>
              <StatusPill value={asString(repairSummary.stop_reason)} />
            </div>
            <div className="round-stack">
              {rounds.length === 0 && <div className="empty-mini">No iterative repair rounds persisted.</div>}
              {rounds.map((round) => <RepairRound key={String(round.round)} round={round} />)}
            </div>
          </section>

          <section className="detail-section">
            <div className="section-title">
              <div><span className="eyebrow">Source review</span><h3>Original vs candidate</h3></div>
              <span className="section-count">kernel.cu</span>
            </div>
            <SourceCompare
              runId={detail.summary.id}
              artifacts={detail.artifacts}
              candidate={detail.candidate as JsonObject | null}
            />
          </section>

          <section className="detail-section">
            <div className="section-title">
              <div><span className="eyebrow">Independent gate</span><h3>Verification</h3></div>
              <button className="secondary-button" onClick={() => void verify()} disabled={verifying}>
                {verifying ? 'Verifying…' : 'Strict verify'}
              </button>
            </div>
            {error && <div className="error-banner">{error}</div>}
            <div className="verification-list">
              {detail.verifications.length === 0 && <div className="empty-mini">Independent verification has not produced a public result.</div>}
              {detail.verifications.map((item) => (
                <div className="verification-card" key={asString(item.run_id)}>
                  <div>
                    <StatusPill value={asString(item.verdict)} />
                    <strong>{humanize(asString(item.reason_code))}</strong>
                  </div>
                  <span>public checks: {String(item.public_passed_count ?? 0)}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="detail-section">
            <div className="section-title">
              <div><span className="eyebrow">RunStore</span><h3>Artifacts</h3></div>
              <span className="section-count">{detail.artifacts.length}</span>
            </div>
            <ArtifactViewer runId={detail.summary.id} artifacts={detail.artifacts} />
          </section>
        </div>
      </aside>
    </div>
  )
}

function RepairModal({
  onClose,
  onSubmitted,
}: {
  onClose: () => void
  onSubmitted: (job: RepairJob) => void
}) {
  const [form, setForm] = useState<RepairRequest>({
    case_id: 'case_0021',
    mode: 'E',
    max_candidates: 3,
    max_llm_calls: 40,
    allow_paid_calls: false,
  })
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [cases, setCases] = useState<CaseSummary[]>([])
  const [casesLoading, setCasesLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    void getCases()
      .then((items) => {
        if (cancelled) return
        setCases(items)
        setForm((current) => {
          const ready = items.some(
            (item) => item.case_id === current.case_id && item.repair_ready,
          )
          if (ready) return current
          const fallback = items.find((item) => item.repair_ready)
          return fallback ? { ...current, case_id: fallback.case_id } : current
        })
      })
      .catch((caught) => {
        if (!cancelled) setError(caught instanceof Error ? caught.message : 'Case catalog unavailable')
      })
      .finally(() => {
        if (!cancelled) setCasesLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const selectedCase = cases.find((item) => item.case_id === form.case_id)

  async function submit(event: FormEvent) {
    event.preventDefault()
    setSubmitting(true)
    setError('')
    try {
      const job = await startRepairJob(form)
      onSubmitted(job)
      onClose()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Repair could not start')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <form className="repair-modal" onSubmit={(event) => void submit(event)} onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <div><span className="eyebrow">New workflow</span><h2>Start CUDA repair</h2></div>
          <button type="button" className="icon-button" onClick={onClose}>×</button>
        </div>
        <label>Public case
          <select
            value={form.case_id}
            disabled={casesLoading || cases.length === 0}
            onChange={(event) => setForm({ ...form, case_id: event.target.value })}
          >
            {casesLoading && <option>Loading cases…</option>}
            {!casesLoading && cases.length === 0 && <option>No repair-ready cases</option>}
            {cases.map((item) => (
              <option key={item.case_id} value={item.case_id} disabled={!item.repair_ready}>
                {item.case_id} · {humanize(item.algorithm)} · {humanize(item.target_tool)}
              </option>
            ))}
          </select>
        </label>
        {selectedCase && (
          <div className="case-preview">
            <div className="case-preview-head">
              <strong>{humanize(selectedCase.template_id || selectedCase.algorithm)}</strong>
              <StatusPill value={selectedCase.target_tool} />
            </div>
            <p>{selectedCase.requirement}</p>
            <small>
              Mutation: {humanize(selectedCase.mutation_id)} · Expected: {selectedCase.expected_finding || '—'}
            </small>
          </div>
        )}
        <div className="form-split">
          <label>Investigation mode
            <select value={form.mode} onChange={(event) => setForm({ ...form, mode: event.target.value as 'D' | 'E' })}>
              <option value="E">E · Agent planner</option>
              <option value="D">D · Rule router</option>
            </select>
          </label>
          <label>Candidate limit
            <input type="number" min={1} max={20} value={form.max_candidates} onChange={(event) => setForm({ ...form, max_candidates: Number(event.target.value) })} />
          </label>
        </div>
        <label>LLM call limit
          <input type="number" min={1} max={40} value={form.max_llm_calls} onChange={(event) => setForm({ ...form, max_llm_calls: Number(event.target.value) })} />
        </label>
        <label className="check-label">
          <input type="checkbox" checked={form.allow_paid_calls} onChange={(event) => setForm({ ...form, allow_paid_calls: event.target.checked })} />
          <span><strong>Allow paid model calls</strong><small>Explicit opt-in; the controller still enforces the call boundary.</small></span>
        </label>
        {error && <div className="error-banner">{error}</div>}
        <div className="modal-actions">
          <button type="button" className="ghost-button" onClick={onClose}>Cancel</button>
          <button className="primary-button" disabled={submitting}>{submitting ? 'Starting…' : 'Start repair'}</button>
        </div>
      </form>
    </div>
  )
}

function RepairJobBanner({
  job,
  onOpenRun,
  onDismiss,
}: {
  job: RepairJob
  onOpenRun: (runId: string) => void
  onDismiss: () => void
}) {
  const terminal = job.status === 'COMPLETED' || job.status === 'FAILED'
  return (
    <section className={'job-banner ' + statusTone(job.status)}>
      <div className="job-banner-main">
        <div className="job-pulse" />
        <div>
          <span className="eyebrow">Background repair job</span>
          <strong>
            {job.case_id} · Mode {job.mode}
          </strong>
          <small>
            {job.run_id
              ? 'Run ' + shortId(job.run_id, 12)
              : job.status === 'QUEUED'
                ? 'Queued before controller execution'
                : 'Starting controller run…'}
          </small>
        </div>
      </div>
      <div className="job-banner-actions">
        <StatusPill value={job.status} />
        {job.verification_verdict && <StatusPill value={job.verification_verdict} />}
        {job.error_code && <span className="job-error">{humanize(job.error_code)}</span>}
        {job.run_id && (
          <button className="secondary-button" onClick={() => onOpenRun(job.run_id!)}>
            Open run
          </button>
        )}
        {terminal && (
          <button className="icon-button small" onClick={onDismiss} aria-label="Dismiss job">
            ×
          </button>
        )}
      </div>
    </section>
  )
}

export default function App() {
  const [stats, setStats] = useState<RunStats | null>(null)
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<RunDetail | null>(null)
  const [repairJob, setRepairJob] = useState<RepairJob | null>(null)
  const [jobRunOpened, setJobRunOpened] = useState<string | null>(null)
  const [showRepair, setShowRepair] = useState(false)
  const [error, setError] = useState('')
  const repairJobId = repairJob?.id
  const repairJobStatus = repairJob?.status
  const pageSize = 20

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const [nextStats, nextRuns] = await Promise.all([
        getStats(),
        getRuns({ page, pageSize, query: query || undefined, status: status || undefined, kind: 'diagnosis' }),
      ])
      setStats(nextStats)
      setRuns(nextRuns.items)
      setTotal(nextRuns.total)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Failed to load dashboard')
    } finally {
      setLoading(false)
    }
  }, [page, query, status])

  useEffect(() => {
    // The effect synchronizes the UI with the external RunStore API.
    // oxlint-disable-next-line react/set-state-in-effect
    void load()
  }, [load])

  useEffect(() => {
    const selectedId = selected?.summary.id
    const needsPolling = (stats?.active ?? 0) > 0 || selected?.summary.status === 'RUNNING'
    if (!needsPolling) return undefined
    const timer = window.setInterval(() => {
      void load()
      if (selectedId) {
        void getRun(selectedId).then(setSelected).catch(() => undefined)
      }
    }, 2500)
    return () => window.clearInterval(timer)
  }, [load, selected?.summary.id, selected?.summary.status, stats?.active])

  useEffect(() => {
    if (!repairJobId || repairJobStatus === 'COMPLETED' || repairJobStatus === 'FAILED') {
      return undefined
    }
    const jobId = repairJobId
    let cancelled = false

    async function pollJob() {
      try {
        const latest = await getRepairJob(jobId)
        if (cancelled) return
        setRepairJob(latest)
        if (latest.run_id && latest.run_id !== jobRunOpened) {
          setJobRunOpened(latest.run_id)
          void getRun(latest.run_id).then(setSelected).catch(() => undefined)
        }
        if (latest.status === 'COMPLETED' || latest.status === 'FAILED') {
          void load()
        }
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : 'Repair job polling failed')
        }
      }
    }

    void pollJob()
    const timer = window.setInterval(() => void pollJob(), 1000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [jobRunOpened, load, repairJobId, repairJobStatus])

  async function openRun(runId: string) {
    setError('')
    try {
      setSelected(await getRun(runId))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unable to load run')
    }
  }

  async function refreshSelected() {
    await load()
    if (selected) setSelected(await getRun(selected.summary.id))
  }

  const pages = Math.max(1, Math.ceil(total / pageSize))
  const completion = useMemo(() => {
    if (!stats?.total_diagnoses) return 0
    return Math.round((stats.verified_fixed / stats.total_diagnoses) * 100)
  }, [stats])

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">GPU</div>
          <div>
            <strong>Agentic GPU Debugger</strong>
            <span>CUDA Test & Repair Console</span>
          </div>
        </div>
        <div className="top-actions">
          <div className="system-health">
            <span className="signal-dot" />
            {(stats?.active ?? 0) > 0 ? `${stats?.active} active · live polling` : 'Controller online'}
          </div>
          <button className="primary-button" onClick={() => setShowRepair(true)}>+ New repair</button>
        </div>
      </header>

      <main>
        <section className="hero">
          <div>
            <span className="eyebrow">Operator workspace</span>
            <h1>Evidence-first CUDA debugging</h1>
            <p>Investigate GPU failures, inspect bounded repair revisions, and verify candidates against independent checks.</p>
          </div>
          <div className="hero-progress">
            <span>Verified repair ratio</span>
            <strong>{completion}%</strong>
            <div><i style={{ width: completion + '%' }} /></div>
          </div>
        </section>

        {repairJob && (
          <RepairJobBanner
            job={repairJob}
            onOpenRun={(runId) => void openRun(runId)}
            onDismiss={() => {
              setRepairJob(null)
              setJobRunOpened(null)
            }}
          />
        )}

        <section className="metrics">
          <MetricCard label="Diagnosis runs" value={stats?.total_diagnoses ?? 0} hint="public RunStore workflows" />
          <MetricCard label="Active" value={stats?.active ?? 0} hint="currently executing" />
          <MetricCard label="Diagnosed" value={stats?.diagnosed ?? 0} hint="grounded diagnosis persisted" />
          <MetricCard label="Verified fixed" value={stats?.verified_fixed ?? 0} hint="independent strict verdict" accent />
          <MetricCard label="Needs attention" value={stats?.needs_attention ?? 0} hint="failed, inconclusive, or blocked" />
        </section>

        {error && <div className="global-error">{error}</div>}

        <section className="workspace-grid">
          <section className="panel runs-panel">
            <div className="panel-heading">
              <div><span className="eyebrow">RunStore explorer</span><h2>CUDA repair runs</h2></div>
              <span className="panel-count">{total}</span>
            </div>
            <div className="filters">
              <div className="search-box">
                <span>⌕</span>
                <input
                  value={query}
                  onChange={(event) => {
                    setQuery(event.target.value)
                    setPage(1)
                  }}
                  placeholder="Search run ID or failure family"
                />
              </div>
              <select
                value={status}
                onChange={(event) => {
                  setStatus(event.target.value)
                  setPage(1)
                }}
              >
                <option value="">All statuses</option>
                <option value="RUNNING">Running</option>
                <option value="COMPLETED">Completed</option>
                <option value="FAILED">Failed</option>
              </select>
              <button className="ghost-button" onClick={() => void load()}>Refresh</button>
            </div>
            <RunTable runs={runs} loading={loading} onSelect={(run) => void openRun(run.id)} />
            <div className="pagination">
              <span>Page {page} of {pages}</span>
              <div>
                <button
                  disabled={page <= 1}
                  onClick={() => setPage((value) => Math.max(1, value - 1))}
                >
                  Previous
                </button>
                <button
                  disabled={page >= pages}
                  onClick={() => setPage((value) => Math.min(pages, value + 1))}
                >
                  Next
                </button>
              </div>
            </div>
          </section>
          <FailureFamilies stats={stats} />
        </section>
      </main>

      {selected && (
        <RunDrawer
          detail={selected}
          onClose={() => setSelected(null)}
          onChanged={refreshSelected}
        />
      )}
      {showRepair && (
        <RepairModal
          onClose={() => setShowRepair(false)}
          onSubmitted={(job) => {
            setRepairJob(job)
            setJobRunOpened(null)
          }}
        />
      )}
    </div>
  )
}
