import type { AppConfig } from '../types/config'

/** Display label/unit for any feature, engineered feature or group id the API may return. */
export function featureMeta(
  config: AppConfig | null,
  id: string,
): { label: string; unit: string; group?: string; type?: string; value_labels?: Record<string, string> } {
  const f = config?.features.features.find((x) => x.id === id)
  if (f) return { label: f.label, unit: f.unit, group: f.group, type: f.type, value_labels: f.value_labels }
  const e = config?.features.engineered.find((x) => x.id === id)
  if (e) return { label: e.label, unit: e.unit, group: e.group }
  const g = config?.features.groups.find((x) => x.id === id)
  if (g) return { label: g.label, unit: '' }
  if (id === 'other') return { label: 'All other features', unit: '' }
  return { label: id, unit: '' }
}

export const TARGET_LABEL: Record<string, string> = {
  CAD: 'Overall CAD',
  LAD: 'LAD stenosis',
  LCX: 'LCX stenosis',
  RCA: 'RCA stenosis',
  P_GE2: 'Multi-vessel (≥2)',
}

/** Display text for a binary value: the feature's own value labels (e.g. Female / Male) or No / Yes. */
export function binaryText(f: { value_labels?: Record<string, string> } | undefined, v: unknown): string {
  const on = Number(v) === 1
  return f?.value_labels?.[on ? '1' : '0'] ?? (on ? 'Yes' : 'No')
}
