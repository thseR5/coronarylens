import { useEffect, useState } from 'react'
import { api } from '../api/client'

type Auc = Record<'CAD' | 'LAD' | 'LCX' | 'RCA', number>

/** Nested-CV ROC-AUC of the served model, read from metrics.json via /api/metrics (null until loaded). */
export function useHeadlineAuc(): Auc | null {
  const [auc, setAuc] = useState<Auc | null>(null)
  useEffect(() => {
    let live = true
    api
      .metrics()
      .then((m) => {
        const mm = m as { cv?: { variants?: Record<string, Record<string, { roc_auc?: { mean?: number } }>> }; test?: { served?: { variant?: string } } }
        const v = mm.cv?.variants?.[mm.test?.served?.variant ?? '']
        const get = (t: string) => v?.[t]?.roc_auc?.mean
        const vals = { CAD: get('CAD'), LAD: get('LAD'), LCX: get('LCX'), RCA: get('RCA') }
        if (live && Object.values(vals).every((x) => typeof x === 'number')) setAuc(vals as Auc)
      })
      .catch(() => {})
    return () => {
      live = false
    }
  }, [])
  return auc
}
