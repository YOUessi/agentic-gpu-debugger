import { expect, test } from '@playwright/test'

const runId = '0123456789abcdef0123456789abcdef'
const analyticsDiagnosisId = '99999999999999999999999999999999'

test.beforeEach(async ({ page }) => {
  await page.route('**/api/cases', async (route) => {
    await route.fulfill({
      json: [{
        case_id: 'case_0021',
        algorithm: 'stencil2d-cpu-v1',
        requirement: 'Apply the five-point stencil to the row-major grid.',
        template_id: 'stencil2d',
        mutation_id: 'omit-top-boundary',
        target_tool: 'memcheck',
        expected_finding: 'Invalid __global__ read',
        repair_ready: true,
      }],
    })
  })

  await page.route('**/api/stats', async (route) => {
    await route.fulfill({
      json: {
        total_diagnoses: 7,
        active: 1,
        diagnosed: 6,
        verified_fixed: 4,
        needs_attention: 2,
        failure_families: {
          out_of_bounds: 3,
          shared_memory_race: 2,
          barrier_misuse: 1,
        },
      },
    })
  })


  await page.route('**/api/analytics/overview', async (route) => {
    await route.fulfill({
      json: {
        store_root: '/public/analytics',
        batch_count: 1,
        evaluation_count: 1,
        projection_errors: [],
        evaluations: [{
          run_id: 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
          status: 'COMPLETED',
          last_event_at: '2026-09-23T19:00:00Z',
          split: 'development',
          corpus_cutoff: 16,
          expected_units: 240,
          executed_units: 240,
          modes: ['A', 'B', 'C', 'D', 'E'],
          repeats: 3,
          verified_fixed: 83,
          verified_rate: 0.3458333333,
          diagnosed: 105,
          latency_mean_ms: 39353.0,
          llm_calls_mean: 2.8667,
          tokens_mean: 7163.3,
          known_cost_usd: 2.6858,
        }],
        batches: [{
          run_id: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
          status: 'COMPLETED',
          started_at: '2026-09-23T18:30:58Z',
          finished_at: '2026-09-23T18:33:38Z',
          case_count: 16,
          registered: 16,
          validated: 0,
          failed: 0,
          running: 0,
          not_run: 0,
          target_tools: { memcheck: 4, racecheck: 4, initcheck: 4, synccheck: 4 },
        }],
      },
    })
  })

  await page.route('**/api/analytics/evaluations/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee?**', async (route) => {
    await route.fulfill({
      json: {
        summary: {
          run_id: 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
          status: 'COMPLETED',
          last_event_at: '2026-09-23T19:00:00Z',
          split: 'development',
          corpus_cutoff: 16,
          expected_units: 240,
          executed_units: 240,
          modes: ['A', 'B', 'C', 'D', 'E'],
          repeats: 3,
          verified_fixed: 83,
          verified_rate: 0.3458333333,
          diagnosed: 105,
          latency_mean_ms: 39353.0,
          llm_calls_mean: 2.8667,
          tokens_mean: 7163.3,
          known_cost_usd: 2.6858,
        },
        mode_metrics: [
          { mode: 'A', record_count: 48, diagnosed: 8, verified_fixed: 5, verified_rate: 0.1042, latency_mean_ms: 17267.8, llm_calls_mean: 1.0, tokens_mean: 2400, known_cost_usd: 0.2 },
          { mode: 'C', record_count: 48, diagnosed: 44, verified_fixed: 36, verified_rate: 0.75, latency_mean_ms: 63475.9, llm_calls_mean: 3.2, tokens_mean: 7200, known_cost_usd: 0.7 },
          { mode: 'D', record_count: 48, diagnosed: 43, verified_fixed: 35, verified_rate: 0.7292, latency_mean_ms: 65639.7, llm_calls_mean: 3.1, tokens_mean: 7000, known_cost_usd: 0.7 },
          { mode: 'E', record_count: 48, diagnosed: 10, verified_fixed: 4, verified_rate: 0.0833, latency_mean_ms: 37301.8, llm_calls_mean: 4.0, tokens_mean: 9100, known_cost_usd: 0.8 },
        ],
        failure_families: { unknown: 135, barrier_misuse: 31, shared_memory_race: 24 },
        records: [{
          ordinal: 0,
          record_id: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
          case_id: 'case_0006',
          template_id: 'vector-add-oob-load',
          mode: 'C',
          repeat: 1,
          status: 'COMPLETED',
          diagnosis_run_id: analyticsDiagnosisId,
          candidate_run_id: 'cccccccccccccccccccccccccccccccc',
          verification_run_id: 'dddddddddddddddddddddddddddddddd',
          diagnosis_outcome: 'DIAGNOSED',
          failure_family: 'out_of_bounds',
          verdict: 'VERIFIED_FIXED',
          oracle_passed: true,
          latency_ms: 73713.179,
          physical_calls: 3,
          sanitizer_calls: 1,
          total_tokens: 5419,
          cost_usd: 0.00886644,
          failure_reason: null,
        }],
        total: 1,
        page: 1,
        page_size: 40,
      },
    })
  })

  await page.route('**/api/analytics/batches/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', async (route) => {
    await route.fulfill({
      json: {
        summary: {
          run_id: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
          status: 'COMPLETED',
          started_at: '2026-09-23T18:30:58Z',
          finished_at: '2026-09-23T18:33:38Z',
          case_count: 16,
          registered: 16,
          validated: 0,
          failed: 0,
          running: 0,
          not_run: 0,
          target_tools: { memcheck: 4, racecheck: 4, initcheck: 4, synccheck: 4 },
        },
        register_requested: true,
        stopped_reason: null,
        cases: [{
          case_id: 'case_0001',
          target_tool: 'memcheck',
          repetitions: 1,
          status: 'REGISTERED',
          clean_run_id: '11111111111111111111111111111111',
          clean_runtime_status: 'SUCCESS',
          clean_oracle_passed: true,
          clean_sanitizer_outcomes: ['CLEAN'],
          mutant_run_id: '22222222222222222222222222222222',
          mutant_runtime_status: 'SUCCESS',
          mutant_oracle_passed: true,
          mutant_sanitizer_outcomes: ['FINDING'],
          target_detections: [true],
          reason_code: null,
        }],
      },
    })
  })


  await page.route('**/api/analytics/runs/' + analyticsDiagnosisId + '/artifacts/**', async (route) => {
    await route.fulfill({
      contentType: 'text/plain',
      body: '========= MEMCHECK finding: invalid global read in kernel.cu:7',
    })
  })

  await page.route('**/api/analytics/runs/' + analyticsDiagnosisId, async (route) => {
    await route.fulfill({
      json: {
        summary: {
          id: analyticsDiagnosisId,
          kind: 'diagnosis',
          parent_run_id: null,
          status: 'COMPLETED',
          phase: null,
          last_event_at: '2026-09-23T18:31:00Z',
          artifact_count: 4,
          diagnosis_outcome: 'DIAGNOSED',
          failure_family: 'out_of_bounds',
          confidence: 'high',
          repair_stop_reason: null,
          verification_verdict: 'VERIFIED_FIXED',
        },
        events: [
          { at: '2026-09-23T18:30:59Z', status: 'QUEUED', phase: null },
          { at: '2026-09-23T18:31:00Z', status: 'RUNNING', phase: 'DIAGNOSING' },
          { at: '2026-09-23T18:31:01Z', status: 'COMPLETED', phase: null },
        ],
        diagnosis: {
          diagnostic_outcome: 'DIAGNOSED',
          failure_family: 'out_of_bounds',
          root_cause: 'The load marker reads input[n + 16], beyond the allocated buffer.',
          recommended_change: 'Restrict the load marker to the valid allocation range.',
          confidence_label: 'high',
          observed_facts: [{
            text: 'Build succeeded and runtime completed.',
            citation_ids: ['analyticsartifact'],
          }],
          tool_findings: [{
            text: 'Invalid __global__ read',
            citation_ids: ['analyticsartifact'],
          }],
          documentation_evidence: [],
        },
        repair_summary: null,
        repair_rounds: [],
        candidate: {
          run_id: 'cccccccccccccccccccccccccccccccc',
          unified_diff: '--- a/kernel.cu\n+++ b/kernel.cu\n- input[n + 16]\n+ input[n - 1]\n',
          patched_source_hash: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        },
        verifications: [{
          run_id: 'dddddddddddddddddddddddddddddddd',
          verdict: 'VERIFIED_FIXED',
          reason_code: 'ALL_REQUIRED_CHECKS_PASSED',
          public_passed_count: 1,
        }],
        actions: [],
        citations: {
          analyticsartifact: {
            citation_id: 'analyticsartifact',
            artifact_id: 'analyticsartifact',
            artifact_name: 'sanitizer/memcheck.log',
            kind: 'artifact',
            label: 'sanitizer/memcheck.log',
            preview: null,
            source_url: null,
          },
        },
        artifacts: [{
          id: 'analyticsartifact',
          name: 'sanitizer/memcheck.log',
          byte_count: 128,
          sha256: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
        }],
      },
    })
  })

  await page.route('**/api/runs?**', async (route) => {
    await route.fulfill({
      json: {
        items: [{
          id: runId,
          kind: 'diagnosis',
          parent_run_id: null,
          status: 'COMPLETED',
          phase: null,
          last_event_at: '2026-10-03T00:00:00Z',
          artifact_count: 18,
          diagnosis_outcome: 'DIAGNOSED',
          failure_family: 'shared_memory_race',
          confidence: 'high',
          repair_stop_reason: 'PUBLIC_CHECKS_PASSED',
          verification_verdict: 'VERIFIED_FIXED',
        }],
        total: 1,
        page: 1,
        page_size: 20,
      },
    })
  })

  await page.route('**/api/runs/' + runId + '/artifacts/**', async (route) => {
    await route.fulfill({
      contentType: 'text/plain',
      body: '========= RACECHECK SUMMARY: 1 hazard displayed',
    })
  })

  await page.route('**/api/runs/' + runId, async (route) => {
    await route.fulfill({
      json: {
        summary: {
          id: runId,
          kind: 'diagnosis',
          parent_run_id: null,
          status: 'COMPLETED',
          phase: null,
          last_event_at: '2026-10-03T00:00:00Z',
          artifact_count: 18,
          diagnosis_outcome: 'DIAGNOSED',
          failure_family: 'shared_memory_race',
          confidence: 'high',
          repair_stop_reason: 'PUBLIC_CHECKS_PASSED',
          verification_verdict: 'VERIFIED_FIXED',
        },
        events: [
          { at: '2026-10-03T00:00:00Z', status: 'QUEUED', phase: null },
          { at: '2026-10-03T00:00:01Z', status: 'RUNNING', phase: 'PREPARING' },
          { at: '2026-10-03T00:00:02Z', status: 'RUNNING', phase: 'DIAGNOSING' },
          { at: '2026-10-03T00:00:03Z', status: 'COMPLETED', phase: null },
        ],
        diagnosis: {
          root_cause: 'Shared-memory writes race before synchronization.',
          recommended_change: 'Separate writers and synchronize before consuming values.',
          tool_findings: [{
            text: 'Racecheck reported a shared-memory hazard.',
            citation_ids: ['evidenceartifact'],
          }],
          documentation_evidence: [{
            text: 'NVIDIA documents this hazard class.',
            citation_ids: ['nvcuda-doc-1'],
          }],
        },
        repair_summary: { stop_reason: 'PUBLIC_CHECKS_PASSED' },
        repair_rounds: [],
        candidate: null,
        verifications: [{
          run_id: 'fedcba9876543210fedcba9876543210',
          verdict: 'VERIFIED_FIXED',
          reason_code: 'ALL_REQUIRED_CHECKS_PASSED',
          public_passed_count: 1,
        }],
        actions: [{
          step: 0,
          proposal: { action: { action_type: 'run_racecheck', rationale: 'Check the suspected race.' } },
          decision: { allowed: true, action_type: 'run_racecheck' },
        }],
        citations: {
          evidenceartifact: {
            citation_id: 'evidenceartifact',
            artifact_id: 'evidenceartifact',
            artifact_name: 'sanitizer/racecheck.log',
            kind: 'artifact',
            label: 'sanitizer/racecheck.log',
            preview: null,
            source_url: null,
          },
          'nvcuda-doc-1': {
            citation_id: 'nvcuda-doc-1',
            artifact_id: 'docartifact',
            artifact_name: 'docs/nvcuda-doc-1.json',
            kind: 'document',
            label: 'Compute Sanitizer Guide · Racecheck',
            preview: 'Racecheck reports shared-memory hazards.',
            source_url: 'https://docs.nvidia.com/compute-sanitizer/',
          },
        },
        artifacts: [],
      },
    })
  })
})

test('renders the operator dashboard and opens a run', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Evidence-first CUDA debugging' })).toBeVisible()
  await expect(page.getByRole('table').getByText('Shared Memory Race')).toBeVisible()
  await page.getByRole('table').getByText('0123456789').click()
  await expect(page.getByRole('heading', { name: 'Diagnosis' })).toBeVisible()
  await expect(page.getByText('Shared-memory writes race before synchronization.')).toBeVisible()
  await expect(page.getByText('Run Racecheck')).toBeVisible()
  await page.getByRole('button', { name: /ART · evidencearti/ }).click()
  await expect(page.getByText('========= RACECHECK SUMMARY: 1 hazard displayed')).toBeVisible()
})

test('loads the public case catalog into the repair selector', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '+ New repair' }).click()
  await expect(page.getByLabel('Public case')).toHaveValue('case_0021')
  await expect(page.getByLabel('Public case').locator('option')).toHaveCount(1)
  await expect(page.getByText('Apply the five-point stencil to the row-major grid.')).toBeVisible()
  await expect(page.getByText(/Mutation: Omit-Top-Boundary/)).toBeVisible()
})


test('starts repair asynchronously and attaches the run when the controller binds it', async ({ page }) => {
  const jobId = 'dddddddddddddddddddddddddddddddd'
  let polls = 0

  await page.route('**/api/jobs/repair', async (route) => {
    await route.fulfill({
      status: 202,
      json: {
        id: jobId,
        status: 'QUEUED',
        case_id: 'case_0021',
        mode: 'E',
        created_at: '2026-10-03T00:00:00Z',
        updated_at: '2026-10-03T00:00:00Z',
        run_id: null,
        verification_verdict: null,
        error_code: null,
      },
    })
  })

  await page.route('**/api/jobs/' + jobId, async (route) => {
    polls += 1
    await route.fulfill({
      json: {
        id: jobId,
        status: polls > 1 ? 'COMPLETED' : 'RUNNING',
        case_id: 'case_0021',
        mode: 'E',
        created_at: '2026-10-03T00:00:00Z',
        updated_at: '2026-10-03T00:00:0' + Math.min(polls, 9) + 'Z',
        run_id: runId,
        verification_verdict: polls > 1 ? 'VERIFIED_FIXED' : null,
        error_code: null,
      },
    })
  })

  await page.goto('/')
  await page.getByRole('button', { name: '+ New repair' }).click()
  await expect(page.getByLabel('Public case')).toHaveValue('case_0021')
  await page.getByRole('button', { name: 'Start repair' }).click()

  await expect(page.getByText('Background repair job')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Diagnosis' })).toBeVisible({ timeout: 5000 })
  await expect(page.getByText('Live pipeline timeline')).toBeVisible()
  await expect(page.locator('.job-banner').getByText('VERIFIED FIXED')).toBeVisible({ timeout: 5000 })
})


test('opens evaluation analytics and renders the A-E data grid', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Analytics' }).click()

  await expect(page.getByRole('heading', { name: 'Batch & Evaluation Analytics' })).toBeVisible()
  await expect(page.locator('.analytics-overview-grid').getByText('240')).toBeVisible()
  await expect(page.getByRole('table').getByText('eeeeeeeeee')).toBeVisible()

  await page.getByRole('table').getByText('eeeeeeeeee').click()
  await expect(page.getByRole('heading', { name: 'Mode comparison' })).toBeVisible()
  await expect(page.getByText('Verified rate by mode')).toBeVisible()
  await expect(page.getByText('Mean latency by mode')).toBeVisible()
  await expect(page.locator('.mode-card').filter({ hasText: 'Mode C' }).getByText('75.0%')).toBeVisible()
  await expect(page.getByText('case_0006')).toBeVisible()
  await expect(page.getByText('Vector-Add-Oob-Load')).toBeVisible()
  await expect(page.getByText('Out Of Bounds')).toBeVisible()
  await expect(page.getByRole('link', { name: 'Export CSV' })).toHaveAttribute(
    'href',
    '/api/analytics/evaluations/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee/export.csv',
  )
  await expect(page.getByRole('link', { name: 'Export JSON' })).toHaveAttribute(
    'href',
    '/api/analytics/evaluations/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee/export.json',
  )

  await page.getByText('case_0006').click()
  await expect(page.getByRole('heading', { name: /case_0006 · Mode C/ })).toBeVisible()
  await expect(page.getByText(analyticsDiagnosisId)).toBeVisible()
  await page.getByRole('button', { name: 'Open diagnosis run' }).click()
  await expect(page.getByText('Historical public diagnosis')).toBeVisible()
  await expect(page.getByText('The load marker reads input[n + 16], beyond the allocated buffer.')).toBeVisible()
  await page.locator('.analytics-run-drawer .citation-chip').first().click()
  await expect(page.getByText('========= MEMCHECK finding: invalid global read in kernel.cu:7')).toBeVisible()
  await expect(page.getByText('Read only')).toBeVisible()
})

test('opens seed batch analytics and shows clean-mutant evidence summary', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Analytics' }).click()

  await page.getByRole('button', { name: /bbbbbbbbbbb/ }).click()
  await expect(page.getByRole('heading', { name: 'Seed case results' })).toBeVisible()
  await expect(page.getByText('case_0001')).toBeVisible()
  await expect(page.getByRole('cell', { name: 'Memcheck' })).toBeVisible()
  await expect(page.getByRole('cell').filter({ hasText: 'REGISTERED' })).toBeVisible()
  await expect(page.getByText('1/1')).toBeVisible()
})


test('compares compatible evaluations and drills into a regression lineage', async ({ page }) => {
  const baselineId = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
  const candidateId = 'ffffffffffffffffffffffffffffffff'
  const card = (runId: string, lastEvent: string, verified: number) => ({
    run_id: runId,
    status: 'COMPLETED',
    last_event_at: lastEvent,
    split: 'development',
    corpus_cutoff: 16,
    expected_units: 3,
    executed_units: 3,
    modes: ['D'],
    repeats: 1,
    verified_fixed: verified,
    verified_rate: verified / 3,
    diagnosed: 3,
    latency_mean_ms: 1000,
    llm_calls_mean: 2,
    tokens_mean: 1200,
    known_cost_usd: 0.03,
  })

  await page.route('**/api/analytics/overview', async (route) => {
    await route.fulfill({
      json: {
        store_root: '/public/analytics',
        batch_count: 0,
        evaluation_count: 2,
        projection_errors: [],
        batches: [],
        evaluations: [
          card(candidateId, '2026-10-03T02:00:00Z', 2),
          card(baselineId, '2026-10-03T01:00:00Z', 2),
        ],
      },
    })
  })

  await page.route('**/api/analytics/evaluations/compare?**', async (route) => {
    await route.fulfill({
      json: {
        comparable: true,
        reasons: [],
        baseline: card(baselineId, '2026-10-03T01:00:00Z', 2),
        candidate: card(candidateId, '2026-10-03T02:00:00Z', 2),
        overall_delta: {
          verified_rate_delta: 0,
          latency_mean_ms_delta: 100,
          llm_calls_mean_delta: 0.5,
          tokens_mean_delta: 200,
          known_cost_usd_delta: 0.01,
        },
        mode_comparisons: [{
          mode: 'D',
          baseline: {
            mode: 'D',
            record_count: 3,
            diagnosed: 3,
            verified_fixed: 2,
            verified_rate: 2 / 3,
            latency_mean_ms: 1000,
            llm_calls_mean: 2,
            tokens_mean: 1200,
            known_cost_usd: 0.03,
          },
          candidate: {
            mode: 'D',
            record_count: 3,
            diagnosed: 3,
            verified_fixed: 2,
            verified_rate: 2 / 3,
            latency_mean_ms: 1100,
            llm_calls_mean: 2.5,
            tokens_mean: 1400,
            known_cost_usd: 0.04,
          },
          delta: {
            verified_rate_delta: 0,
            latency_mean_ms_delta: 100,
            llm_calls_mean_delta: 0.5,
            tokens_mean_delta: 200,
            known_cost_usd_delta: 0.01,
          },
        }],
        matched_units: 3,
        regressions: 1,
        improvements: 1,
        unchanged: 1,
        regression_rows: [{
          case_id: 'case_0006',
          template_id: 'vector-add-oob-load',
          mode: 'D',
          repeat: 0,
          baseline_verdict: 'VERIFIED_FIXED',
          candidate_verdict: 'NOT_FIXED',
          baseline_diagnosis_run_id: analyticsDiagnosisId,
          candidate_diagnosis_run_id: null,
        }],
      },
    })
  })

  await page.goto('/')
  await page.getByRole('button', { name: 'Analytics' }).click()
  const comparison = page.locator('.comparison-panel')
  await expect(page.locator('.trend-card')).toHaveCount(2)
  await expect(comparison.getByLabel('Baseline')).toHaveValue(baselineId)
  await expect(comparison.getByLabel('Candidate')).toHaveValue(candidateId)
  await comparison.getByRole('button', { name: 'Compare' }).click()

  await expect(comparison.getByText('Regression signals')).toBeVisible()
  await expect(comparison.getByText('case_0006')).toBeVisible()
  await expect(
    comparison.locator('.regression-counts .regression').getByText('1', { exact: true }),
  ).toBeVisible()
  await expect(
    comparison.locator('.regression-counts .improvement').getByText('1', { exact: true }),
  ).toBeVisible()
  await comparison.getByRole('button', { name: 'Baseline run' }).click()
  await expect(page.getByText('Historical public diagnosis')).toBeVisible()
  await expect(page.getByText('Read only')).toBeVisible()
})
