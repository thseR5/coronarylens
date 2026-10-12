import { load as loadYaml } from 'js-yaml'
import type { AnalyzeResponse, ExplainResponse, Features, Preset, SimilarResponse, CounterfactualResponse, TargetId, ThresholdMode } from '../types/api'
import type { AnatomyConfig, AppConfig, FeaturesConfig, TargetsConfig } from '../types/config'
import featuresRaw from '../../../config/features.yaml?raw'
import targetsRaw from '../../../config/targets.yaml?raw'
import anatomyRaw from '../../../config/anatomy.yaml?raw'

async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`/api/${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`)
  return res.json() as Promise<T>
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`/api/${path}`)
  if (!res.ok) throw new Error(`${path}: ${res.status}`)
  return res.json() as Promise<T>
}

/** Config from the same YAML files the backend reads, bundled at build time (no duplication). */
export function bundledConfig(): AppConfig {
  return {
    features: loadYaml(featuresRaw) as FeaturesConfig,
    targets: loadYaml(targetsRaw) as TargetsConfig,
    anatomy: loadYaml(anatomyRaw) as AnatomyConfig,
  }
}

export const api = {
  config: () => get<AppConfig>('config').catch(() => bundledConfig()),
  presets: () => get<Preset[]>('presets'),
  metrics: () => get<Record<string, unknown>>('metrics'),
  globalExplanations: () => get<Record<string, unknown>>('global-explanations'),
  analyze: (features: Features, mode: ThresholdMode, signal?: AbortSignal) =>
    post<AnalyzeResponse>('analyze', { features, mode }, signal),
  explain: (features: Features, signal?: AbortSignal) => post<ExplainResponse>('explain', { features }, signal),
  counterfactual: (features: Features, target: TargetId) =>
    post<CounterfactualResponse>('counterfactual', { features, target }),
  similar: (features: Features, k = 5) => post<SimilarResponse>('similar', { features, k }),
}
