export type CaseSummary = {
  case_id: string
  algorithm: string
  requirement: string
  template_id: string | null
  mutation_id: string | null
  target_tool: string | null
  expected_finding: string | null
  repair_ready: boolean
}

export type CitationTarget = {
  citation_id: string
  artifact_id: string
  artifact_name: string
  kind: 'artifact' | 'document'
  label: string
  preview: string | null
  source_url: string | null
}

export type RunSummary = {
  id: string
  kind: string
  parent_run_id: string | null
  status: string
  phase: string | null
  last_event_at: string | null
  artifact_count: number
  diagnosis_outcome: string | null
  failure_family: string | null
  confidence: string | null
  repair_stop_reason: string | null
  verification_verdict: string | null
}

export type RunStats = {
  total_diagnoses: number
  active: number
  diagnosed: number
  verified_fixed: number
  needs_attention: number
  failure_families: Record<string, number>
}

export type RunListResponse = {
  items: RunSummary[]
  total: number
  page: number
  page_size: number
}

export type ArtifactSummary = {
  id: string
  name: string
  byte_count: number
  sha256: string
}

export type RunDetail = {
  summary: RunSummary
  events: Array<Record<string, unknown>>
  diagnosis: Record<string, unknown> | null
  repair_summary: Record<string, unknown> | null
  repair_rounds: Array<Record<string, unknown>>
  candidate: Record<string, unknown> | null
  verifications: Array<Record<string, unknown>>
  actions: Array<Record<string, unknown>>
  citations: Record<string, CitationTarget>
  artifacts: ArtifactSummary[]
}

export type RepairJob = {
  id: string
  status: 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED'
  case_id: string
  mode: 'D' | 'E'
  created_at: string
  updated_at: string
  run_id: string | null
  verification_verdict: string | null
  error_code: string | null
}

export type RepairRequest = {
  case_id: string
  mode: 'D' | 'E'
  max_candidates: number
  max_llm_calls: number
  allow_paid_calls: boolean
}

export type RepairResponse = {
  run_id: string
  status: string
  verification_verdict: string | null
}


export type BatchCard = {
  run_id: string
  status: string
  started_at: string | null
  finished_at: string | null
  case_count: number
  registered: number
  validated: number
  failed: number
  running: number
  not_run: number
  target_tools: Record<string, number>
}

export type EvaluationModeSummary = {
  mode: string
  record_count: number
  diagnosed: number
  verified_fixed: number
  verified_rate: number | null
  latency_mean_ms: number | null
  llm_calls_mean: number | null
  tokens_mean: number | null
  known_cost_usd: number | null
}

export type EvaluationCard = {
  run_id: string
  status: string
  last_event_at: string | null
  split: string | null
  corpus_cutoff: number | null
  expected_units: number | null
  executed_units: number
  modes: string[]
  repeats: number | null
  verified_fixed: number
  verified_rate: number | null
  diagnosed: number
  latency_mean_ms: number | null
  llm_calls_mean: number | null
  tokens_mean: number | null
  known_cost_usd: number | null
}

export type AnalyticsOverview = {
  store_root: string
  batch_count: number
  evaluation_count: number
  projection_errors: string[]
  batches: BatchCard[]
  evaluations: EvaluationCard[]
}

export type EvaluationRecordRow = {
  ordinal: number
  record_id: string
  case_id: string
  template_id: string
  mode: string
  repeat: number
  status: string
  diagnosis_run_id: string | null
  candidate_run_id: string | null
  verification_run_id: string | null
  diagnosis_outcome: string | null
  failure_family: string | null
  verdict: string | null
  oracle_passed: boolean | null
  latency_ms: number | null
  physical_calls: number | null
  sanitizer_calls: number | null
  total_tokens: number | null
  cost_usd: number | null
  failure_reason: string | null
}

export type EvaluationDetail = {
  summary: EvaluationCard
  mode_metrics: EvaluationModeSummary[]
  failure_families: Record<string, number>
  records: EvaluationRecordRow[]
  total: number
  page: number
  page_size: number
}

export type BatchCaseRow = {
  case_id: string
  target_tool: string
  repetitions: number
  status: string
  clean_run_id: string | null
  clean_runtime_status: string | null
  clean_oracle_passed: boolean | null
  clean_sanitizer_outcomes: string[]
  mutant_run_id: string | null
  mutant_runtime_status: string | null
  mutant_oracle_passed: boolean | null
  mutant_sanitizer_outcomes: string[]
  target_detections: boolean[]
  reason_code: string | null
}

export type BatchDetail = {
  summary: BatchCard
  register_requested: boolean
  stopped_reason: string | null
  cases: BatchCaseRow[]
}


export type EvaluationDelta = {
  verified_rate_delta: number | null
  latency_mean_ms_delta: number | null
  llm_calls_mean_delta: number | null
  tokens_mean_delta: number | null
  known_cost_usd_delta: number | null
}

export type EvaluationModeComparison = {
  mode: string
  baseline: EvaluationModeSummary | null
  candidate: EvaluationModeSummary | null
  delta: EvaluationDelta
}

export type EvaluationRegressionRow = {
  case_id: string
  template_id: string
  mode: string
  repeat: number
  baseline_verdict: string | null
  candidate_verdict: string | null
  baseline_diagnosis_run_id: string | null
  candidate_diagnosis_run_id: string | null
}

export type EvaluationComparison = {
  comparable: boolean
  reasons: string[]
  baseline: EvaluationCard
  candidate: EvaluationCard
  overall_delta: EvaluationDelta
  mode_comparisons: EvaluationModeComparison[]
  matched_units: number
  regressions: number
  improvements: number
  unchanged: number
  regression_rows: EvaluationRegressionRow[]
}
