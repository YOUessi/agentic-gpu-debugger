export function shortId(value: string, width = 8): string {
  return value.length <= width ? value : value.slice(0, width)
}

export function humanize(value: string | null | undefined): string {
  if (!value) return '—'
  return value
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export function statusTone(value: string | null | undefined): 'good' | 'bad' | 'warn' | 'muted' {
  if (!value) return 'muted'
  if (['COMPLETED', 'DIAGNOSED', 'VERIFIED_FIXED', 'PASSED', 'CLEAN', 'PUBLIC_CHECKS_PASSED'].includes(value)) return 'good'
  if (['FAILED', 'NOT_FIXED', 'REGRESSION_DETECTED', 'FINDING'].includes(value)) return 'bad'
  if (['RUNNING', 'INCONCLUSIVE', 'UNAVAILABLE', 'LLM_UNAVAILABLE', 'REPEATED_CANDIDATE'].includes(value)) return 'warn'
  return 'muted'
}

export function formatTime(value: string | null): string {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

export function formatBytes(value: number): string {
  if (value < 1024) return value + ' B'
  if (value < 1024 * 1024) return (value / 1024).toFixed(1) + ' KB'
  return (value / 1024 / 1024).toFixed(1) + ' MB'
}
