import { useEffect, useState } from 'react'
import { useStore } from '../store'

interface Agreement {
  segments_majority_agree: number
  disagreeing_segments: { segment: number; name: string; standard: string; partition: string; reason: string }[]
}

/** How the 3D nearest-artery territories line up with the AHA 17-segment convention (tools/territory_check.py). */
export default function TerritoryAgreement({ detailed = false }: { detailed?: boolean }) {
  const [a, setA] = useState<Agreement | null>(null)
  const mode = useStore((s) => s.view.territoryMode)
  useEffect(() => {
    fetch('/models/territory_agreement.json')
      .then((r) => (r.ok ? r.json() : null))
      .then(setA)
      .catch(() => setA(null))
  }, [])
  if (!a) return null
  return (
    <div className="text-[10px] leading-snug text-muted">
      <p>
        {mode === 'aha' ? (
          <>
            3D territories: the 17 AHA segments of the LV surface with the standard vessel assignment (identical to this
            bullseye). Optional view: nearest-artery estimate ({a.segments_majority_agree}/17 segments agree with AHA).
          </>
        ) : (
          <>
            3D territories: nearest-artery (Voronoi) partition of the LV surface, aligned with the AHA 17-segment convention (
            {a.segments_majority_agree}/17 segments agree).
          </>
        )}{' '}
        Expected supply territory (illustrative); coronary dominance varies between people.
      </p>
      {detailed && (
        <ul className="mt-1 list-disc pl-4">
          {a.disagreeing_segments.map((d) => (
            <li key={d.segment}>
              Segment {d.segment} ({d.name}): AHA {d.standard}, partition {d.partition}. {d.reason}.
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
