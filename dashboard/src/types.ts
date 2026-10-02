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
  diagnosis: Record<string, unknown> | null
  repair_summary: Record<string, unknown> | null
  repair_rounds: Array<Record<string, unknown>>
  candidate: Record<string, unknown> | null
  verifications: Array<Record<string, unknown>>
  actions: Array<Record<string, unknown>>
  citations: Record<string, CitationTarget>
  artifacts: ArtifactSummary[]
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
