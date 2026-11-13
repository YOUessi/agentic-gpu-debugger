import { describe, expect, it } from 'vitest'
import { formatBytes, humanize, shortId, statusTone } from './format'

describe('dashboard formatting', () => {
  it('shortens immutable run identifiers for dense grids', () => {
    expect(shortId('1234567890abcdef', 8)).toBe('12345678')
  })

  it('humanizes controller reason codes', () => {
    expect(humanize('PUBLIC_CHECKS_PASSED')).toBe('PUBLIC CHECKS PASSED')
  })

  it('maps persisted verdicts to visual tones', () => {
    expect(statusTone('VERIFIED_FIXED')).toBe('good')
    expect(statusTone('REGRESSION_DETECTED')).toBe('bad')
    expect(statusTone('INCONCLUSIVE')).toBe('warn')
  })

  it('formats artifact sizes without hiding the byte scale', () => {
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(2048)).toBe('2.0 KB')
  })
})
