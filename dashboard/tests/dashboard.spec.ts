import { expect, test } from '@playwright/test'

const runId = '0123456789abcdef0123456789abcdef'

test.beforeEach(async ({ page }) => {
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
})
