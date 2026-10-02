import { expect, test } from '@playwright/test'

const runId = '0123456789abcdef0123456789abcdef'

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
