import type {
  AnalyticsOverview,
  BatchDetail,
  CaseSummary,
  EvaluationComparison,
  EvaluationDetail,
  RepairJob,
  RepairRequest,
  RepairResponse,
  RunDetail,
  RunListResponse,
  RunStats,
} from './types'

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = (await response.json()) as { detail?: string }
      detail = body.detail || detail
    } catch {
      // Keep HTTP status text when the response is not JSON.
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export function getCases(): Promise<CaseSummary[]> {
  return request<CaseSummary[]>('/api/cases')
}

export function getStats(): Promise<RunStats> {
  return request<RunStats>('/api/stats')
}

export function getRuns(params: {
  page: number
  pageSize: number
  query?: string
  status?: string
  kind?: string
}): Promise<RunListResponse> {
  const search = new URLSearchParams({
    page: String(params.page),
    page_size: String(params.pageSize),
  })
  if (params.query) search.set('query', params.query)
  if (params.status) search.set('status', params.status)
  if (params.kind) search.set('kind', params.kind)
  return request<RunListResponse>('/api/runs?' + search.toString())
}

export function getRun(runId: string): Promise<RunDetail> {
  return request<RunDetail>('/api/runs/' + runId)
}

export function getArtifact(runId: string, artifactId: string): Promise<string> {
  return fetch('/api/runs/' + runId + '/artifacts/' + artifactId).then(async (response) => {
    if (!response.ok) throw new Error('ARTIFACT_NOT_FOUND')
    return response.text()
  })
}

export function startRepairJob(payload: RepairRequest): Promise<RepairJob> {
  return request<RepairJob>('/api/jobs/repair', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function getRepairJob(jobId: string): Promise<RepairJob> {
  return request<RepairJob>('/api/jobs/' + jobId)
}

export function startRepair(payload: RepairRequest): Promise<RepairResponse> {
  return request<RepairResponse>('/api/repair', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function verifyRun(runId: string, strict = true): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>('/api/runs/' + runId + '/verify', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ strict }),
  })
}

export function getAnalyticsOverview(): Promise<AnalyticsOverview> {
  return request<AnalyticsOverview>('/api/analytics/overview')
}

export function getEvaluationDetail(
  runId: string,
  params: {
    page: number
    pageSize: number
    mode?: string
    verdict?: string
    query?: string
  },
): Promise<EvaluationDetail> {
  const search = new URLSearchParams({
    page: String(params.page),
    page_size: String(params.pageSize),
  })
  if (params.mode) search.set('mode', params.mode)
  if (params.verdict) search.set('verdict', params.verdict)
  if (params.query) search.set('query', params.query)
  return request<EvaluationDetail>(
    '/api/analytics/evaluations/' + runId + '?' + search.toString(),
  )
}

export function getBatchDetail(runId: string): Promise<BatchDetail> {
  return request<BatchDetail>('/api/analytics/batches/' + runId)
}

export function getAnalyticsRun(runId: string): Promise<RunDetail> {
  return request<RunDetail>('/api/analytics/runs/' + runId)
}

export function getAnalyticsArtifact(runId: string, artifactId: string): Promise<string> {
  return fetch('/api/analytics/runs/' + runId + '/artifacts/' + artifactId).then(
    async (response) => {
      if (!response.ok) throw new Error('ANALYTICS_ARTIFACT_NOT_FOUND')
      return response.text()
    },
  )
}


export function compareEvaluations(
  baselineRunId: string,
  candidateRunId: string,
): Promise<EvaluationComparison> {
  const search = new URLSearchParams({
    baseline: baselineRunId,
    candidate: candidateRunId,
  })
  return request<EvaluationComparison>(
    '/api/analytics/evaluations/compare?' + search.toString(),
  )
}

export function evaluationExportUrl(
  runId: string,
  format: 'csv' | 'json',
): string {
  return '/api/analytics/evaluations/' + runId + '/export.' + format
}
