import { useEffect, useState, type ReactNode } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import { api } from '../api/client'

// Loose view of ml/artifacts/metrics.json (+ ablation) — only the fields this tab reads.
type Stat = { mean: number; sd: number }
type CI = { value: number; ci95: [number, number] }
type Diff = { mean: number; sd: number; se: number; folds_improved: number; n_folds: number; better_is: string }
interface Sprint {
  n_screened: number
  n_finalists: number
  finalists: string[]
  outer: string
  chosen_counts: Record<string, number>
  paired_diff_vs_previous_served: Record<string, Diff>
  previous_served: Record<string, Record<string, Stat>> & { pattern_log_loss: Stat; brier_mean: Stat }
  served_on_full_training_split: { chosen: string }
}
interface Metrics {
  sprint?: Sprint
  test_first_look?: { test: { served: { metrics: Record<string, Record<string, CI>> } } }
  data: { n: number; n_train: number; n_test: number }
  cv: {
    settings: { outer_folds: number; outer_repeats: number; inner_folds: number }
    variants: Record<string, Record<string, Record<string, Stat>> & { pattern_log_loss: Stat; coherence_violation_rate: Stat }>
    selection?: { chosen?: string; rule?: string } & Record<string, unknown>
  }
  test: {
    served: { variant: string; metrics: Record<string, Record<string, CI>> }
    comparison_served_vs_independent: Record<string, { delong: { diff: number; p_value: number } }>
  }
  coherence: Record<string, Stat>
  calibration: { method: string; test: Record<string, { slope: number; ece: number; reliability: { mean_pred: number; frac_pos: number; n: number }[] }> }
  conformal: { target_coverage: number; test: Record<string, { coverage: number; mean_set_size: number; frac_uncertain: number }> }
  dca: { test: Record<string, { thresholds: number[]; model: number[]; treat_all: number[] }> }
  subgroups: { groups: Record<string, { n: number } & Record<string, { roc_auc: number }>> }
  explanations: { deletion_auc_shap: number; deletion_auc_random: number; max_additivity_error: number }
  ablation?: { model: string; cv: string; targets: Record<string, Record<string, { roc_auc: Stat; accuracy: Stat }>> }
  benchmarks?: Record<string, { summary: { coherence_violation_rate: Stat } }>
}

const TARGETS = ['CAD', 'LAD', 'LCX', 'RCA'] as const
const VARIANT_LABEL: Record<string, string> = {
  naive_lr: 'Naive logistic regression ×4',
  joint_vector: 'Joint (served)',
  joint_none: 'Joint, uncalibrated',
  independent: 'Independent',
  chains: 'Classifier chains',
  lightgbm_temperature: 'LightGBM joint',
}
const f2 = (x: number | undefined) => (x == null ? '–' : x.toFixed(2))
const f3 = (x: number | undefined) => (x == null ? '–' : x.toFixed(3))
const pc = (x: number | undefined, d = 1) => (x == null ? '–' : `${(x * 100).toFixed(d)}%`)
const AX = { stroke: 'var(--color-muted)', fontSize: 10 }
const TIP = { contentStyle: { background: '#121b25', border: '1px solid #243445', fontSize: 11 } }

function Section({ title, children, note }: { title: string; children: ReactNode; note?: ReactNode }) {
  return (
    <section className="card">
      <h3 className="h-section">{title}</h3>
      {children}
      {note && <p className="mt-2 text-[10px] leading-snug text-muted">{note}</p>}
    </section>
  )
}

const Th = ({ children }: { children?: ReactNode }) => <th className="px-1 py-1 text-right font-medium first:text-left">{children}</th>
const Td = ({ children, strong }: { children: ReactNode; strong?: boolean }) => (
  <td className={`px-1 py-1 text-right font-mono first:text-left first:font-sans ${strong ? 'text-text' : 'text-text/80'}`}>{children}</td>
)

export default function ModelPerformance() {
  const [m, setM] = useState<Metrics | null>(null)
  const [err, setErr] = useState(false)
  const [dcaTarget, setDcaTarget] = useState<(typeof TARGETS)[number]>('CAD')
  const reg = m?.benchmarks?.regularized_lr

  useEffect(() => {
    api.metrics().then((x) => setM(x as unknown as Metrics)).catch(() => setErr(true))
  }, [])
  if (err) return <p className="text-xs text-muted">Metrics unavailable (backend offline).</p>
  if (!m) return <div className="card h-40 animate-pulse" />

  const served = m.test.served.metrics
  const variants = Object.keys(VARIANT_LABEL).filter((v) => m.cv.variants[v])
  const coherence = Object.keys(VARIANT_LABEL)
    .filter((v) => m.coherence[v])
    .map((v) => ({ name: VARIANT_LABEL[v], rate: m.coherence[v].mean * 100 }))
  const dca = m.dca.test[dcaTarget]
  const dcaRows = dca.thresholds.map((t, i) => ({ t, model: dca.model[i], all: dca.treat_all[i], none: 0 }))
  const COLORS = ['#4fd1c5', '#f6c35b', '#a78bfa', '#f472b6']

  return (
    <div className="flex flex-col gap-3 text-xs">
      <p className="text-muted">
        n = {m.data.n} patients.{' '}
        {m.sprint ? (
          <>
            Primary estimate: nested cross-validation of the whole model-selection procedure ({m.sprint.outer}) on the {m.data.n_train}
            -patient training split. Secondary: a locked {m.data.n_test}-patient test split (looked at twice: before and after the
            model search; never used to choose anything), with 2,000-resample bootstrap 95% CIs.
          </>
        ) : (
          <>
            Primary estimate: repeated nested cross-validation ({m.cv.settings.outer_repeats}×{m.cv.settings.outer_folds} outer,{' '}
            {m.cv.settings.inner_folds} inner) on the {m.data.n_train}-patient training split. Secondary: a locked {m.data.n_test}
            -patient test split evaluated once, with 2,000-resample bootstrap 95% CIs.
          </>
        )}
      </p>

      <Section
        title={
          m.sprint
            ? `Headline · nested CV of the served procedure (mean ± SD, ${m.sprint.outer})`
            : `Headline · nested CV, served model (mean ± SD over ${m.cv.settings.outer_repeats}×${m.cv.settings.outer_folds} outer folds)`
        }
        note="Every number here comes from patients the model (and every choice about it) never saw while training. Threshold metrics use the max-F1 threshold chosen inside each training fold."
      >
        <table className="w-full">
          <thead className="text-[10px] uppercase text-muted">
            <tr>
              <Th>Target</Th>
              <Th>ROC-AUC</Th>
              <Th>PR-AUC</Th>
              <Th>Brier</Th>
              <Th>Acc</Th>
              <Th>Prec</Th>
              <Th>Rec</Th>
              <Th>F1</Th>
            </tr>
          </thead>
          <tbody>
            {[...TARGETS, 'P_GE2', 'P_3VD'].map((t) => {
              const r = m.cv.variants[m.test.served.variant]?.[t]
              if (!r) return null
              return (
                <tr key={t} className="border-t border-line/60">
                  <Td strong>{t === 'P_GE2' ? '≥2 vessels' : t === 'P_3VD' ? '3 vessels' : t}</Td>
                  <Td strong>
                    {f3(r.roc_auc?.mean)}
                    <span className="block text-[9px] text-muted">±{f2(r.roc_auc?.sd)}</span>
                  </Td>
                  <Td>{f2(r.pr_auc?.mean)}</Td>
                  <Td>{f3(r.brier?.mean)}</Td>
                  <Td>{f2(r.accuracy?.mean)}</Td>
                  <Td>{f2(r.precision?.mean)}</Td>
                  <Td>{f2(r.recall?.mean)}</Td>
                  <Td>{f2(r.f1?.mean)}</Td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </Section>

      {m.sprint && (
        <Section
          title={`Model search · ${m.sprint.n_screened} procedures screened, ${m.sprint.n_finalists} finalists in nested CV`}
          note={
            <>
              Screening used the training split only. Finalists were chosen by a rule written before the results were read; inside
              every outer fold the procedure with the best inner cross-fitted log-loss was selected and then scored on unseen
              patients, so these numbers already account for the search. Served (selected on the full training split):{' '}
              <span className="font-mono">{m.sprint.served_on_full_training_split.chosen}</span>. LCX/RCA did not improve: the
              ceiling held.
            </>
          }
        >
          <table className="w-full">
            <thead className="text-[10px] uppercase text-muted">
              <tr>
                <Th>Metric</Th>
                <Th>Previous model</Th>
                <Th>Paired gain</Th>
                <Th>Folds better</Th>
              </tr>
            </thead>
            <tbody>
              {(
                [
                  ['pattern_log_loss', 'Pattern log-loss'],
                  ['brier_mean', 'Mean Brier'],
                  ['auc_CAD', 'CAD AUC'],
                  ['auc_LAD', 'LAD AUC'],
                  ['auc_LCX', 'LCX AUC'],
                  ['auc_RCA', 'RCA AUC'],
                ] as const
              ).map(([k, label]) => {
                const d = m.sprint!.paired_diff_vs_previous_served[k]
                const prev = k.startsWith('auc_')
                  ? m.sprint!.previous_served[k.slice(4)]?.roc_auc?.mean
                  : (m.sprint!.previous_served[k as 'pattern_log_loss' | 'brier_mean'] as Stat).mean
                const good = d.better_is === 'lower' ? d.mean < -d.se : d.mean > d.se
                return (
                  <tr key={k} className="border-t border-line/60">
                    <Td strong>{label}</Td>
                    <Td>{f3(prev)}</Td>
                    <Td strong>
                      <span className={good ? 'text-emerald-300' : 'text-muted'}>
                        {d.mean >= 0 ? '+' : '−'}
                        {Math.abs(d.mean).toFixed(4)}
                      </span>
                      <span className="block text-[9px] text-muted">± {d.sd.toFixed(4)} (SE {d.se.toFixed(4)})</span>
                    </Td>
                    <Td>
                      {d.folds_improved}/{d.n_folds}
                    </Td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </Section>
      )}

      <Section
        title="Benchmarks · same 5×5 nested CV for every approach (ROC-AUC mean ± SD)"
        note={
          <>
            Same outer folds for every model. Selection rule: lowest pattern log-loss, then AUC, simplest within 1 SE. On the test set the
            joint and independent models do not differ significantly in AUC (DeLong p ={' '}
            {TARGETS.map((t) => f2(m.test.comparison_served_vs_independent[t]?.delong.p_value)).join(' / ')}); the joint model's gain is
            coherence and calibration.
          </>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="text-[10px] uppercase text-muted">
              <tr>
                <Th>Model</Th>
                {TARGETS.map((t) => (
                  <Th key={t}>{t}</Th>
                ))}
                <Th>Log-loss</Th>
                <Th>Contra&shy;dictions</Th>
              </tr>
            </thead>
            <tbody>
              {variants.map((v) => {
                const r = m.cv.variants[v]
                return (
                  <tr key={v} className={`border-t border-line/60 ${v === m.test.served.variant ? 'bg-accent/5' : ''}`}>
                    <Td strong={v === m.test.served.variant}>{VARIANT_LABEL[v]}</Td>
                    {TARGETS.map((t) => (
                      <Td key={t}>
                        {f3(r[t]?.roc_auc?.mean)}
                        <span className="block text-[9px] text-muted">±{f2(r[t]?.roc_auc?.sd)}</span>
                      </Td>
                    ))}
                    <Td>{f3(r.pattern_log_loss?.mean)}</Td>
                    <Td strong>{pc(r.coherence_violation_rate?.mean)}</Td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Section>

      <Section
        title="Coherence · patients where P(CAD) < some vessel's P(stenosis)"
        note={
          <>
            Logically impossible, since CAD means at least one stenotic vessel. Measured on nested-CV out-of-fold predictions. The
            joint model is 0% by construction: P(CAD) = 1 − P(no vessel stenotic). Naive logistic regression = four separate
            scikit-learn LogisticRegression models (L2, C = 1, lbfgs, standardized inputs).
            {reg && (
              <>
                {' '}
                Footnote: four elastic-net logistic regressions (C = 0.3, l1_ratio 0.5) on the same folds still contradict themselves
                in {pc(reg.summary.coherence_violation_rate.mean)} of patients.
              </>
            )}
          </>
        }
      >
        <div className="h-36">
          <ResponsiveContainer>
            <BarChart data={coherence} layout="vertical" margin={{ left: 8, right: 16 }}>
              <XAxis type="number" unit="%" {...AX} />
              <YAxis type="category" dataKey="name" width={110} {...AX} />
              <Tooltip {...TIP} formatter={(v) => `${Number(v).toFixed(1)}%`} />
              <Bar dataKey="rate" fill="#f472b6" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section
        title={`Secondary · locked test set (n = ${m.data.n_test}), served model${m.test_first_look ? ' (second look, after the model search)' : ''}`}
        note={
          <>
            Held out before any modelling. With n = {m.data.n_test} the 95% bootstrap CIs are wide, so nested CV above is the primary
            estimate.
            {m.test_first_look && (
              <>
                {' '}
                The test set has been looked at twice: once before the model search (previous model, ROC-AUC CAD/LAD/LCX/RCA{' '}
                {TARGETS.map((t) => f3(m.test_first_look!.test.served.metrics[t]?.roc_auc?.value)).join(' / ')}) and once after it
                (this table). Nothing was chosen or tuned on it.
              </>
            )}
          </>
        }
      >
        <table className="w-full">
          <thead className="text-[10px] uppercase text-muted">
            <tr>
              <Th>Target</Th>
              <Th>ROC-AUC [95% CI]</Th>
              <Th>PR-AUC</Th>
              <Th>Brier</Th>
              <Th>Acc</Th>
              <Th>Prec</Th>
              <Th>Rec</Th>
              <Th>F1</Th>
            </tr>
          </thead>
          <tbody>
            {[...TARGETS, 'P_GE2', 'P_3VD'].map((t) => {
              const r = served[t]
              if (!r) return null
              return (
                <tr key={t} className="border-t border-line/60">
                  <Td strong>{t === 'P_GE2' ? '≥2 vessels' : t === 'P_3VD' ? '3 vessels' : t}</Td>
                  <Td strong>
                    {f3(r.roc_auc?.value)}
                    <span className="block text-[9px] text-muted">[{f2(r.roc_auc?.ci95[0])}–{f2(r.roc_auc?.ci95[1])}]</span>
                  </Td>
                  <Td>{f2(r.pr_auc?.value)}</Td>
                  <Td>{f3(r.brier?.value)}</Td>
                  <Td>{f2(r.accuracy?.value)}</Td>
                  <Td>{f2(r.precision?.value)}</Td>
                  <Td>{f2(r.recall?.value)}</Td>
                  <Td>{f2(r.f1?.value)}</Td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </Section>

      <Section
        title={`Calibration on test (${m.calibration.method} scaling)`}
        note={`Calibration slope / ECE: ${TARGETS.map((t) => `${t} ${f2(m.calibration.test[t]?.slope)} / ${f3(m.calibration.test[t]?.ece)}`).join(' · ')}. Diagonal = perfect calibration; dot size = patients in bin.`}
      >
        <div className="h-56">
          <ResponsiveContainer>
            <ScatterChart margin={{ left: -10, right: 10, top: 5 }}>
              <CartesianGrid stroke="#243445" />
              <XAxis type="number" dataKey="mean_pred" domain={[0, 1]} name="Predicted" {...AX} />
              <YAxis type="number" dataKey="frac_pos" domain={[0, 1]} name="Observed" {...AX} />
              <ZAxis type="number" dataKey="n" range={[15, 120]} />
              <Tooltip {...TIP} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Scatter
                name="ideal"
                data={[
                  { mean_pred: 0, frac_pos: 0, n: 0 },
                  { mean_pred: 1, frac_pos: 1, n: 0 },
                ]}
                line={{ stroke: '#8aa0b4', strokeDasharray: '4 4' }}
                shape={() => <g />}
                legendType="none"
              />
              {TARGETS.map((t, i) => (
                <Scatter key={t} name={t} data={m.calibration.test[t]?.reliability ?? []} fill={COLORS[i]} line={{ stroke: COLORS[i] }} isAnimationActive={false} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section
        title="Decision curve analysis (test)"
        note="Net benefit of acting on the model vs treating everyone or no one, across risk thresholds (Vickers & Elkin 2006)."
      >
        <div className="mb-1 flex gap-1">
          {TARGETS.map((t) => (
            <button key={t} type="button" className={`btn !py-0.5 ${t === dcaTarget ? '!ring-accent' : ''}`} onClick={() => setDcaTarget(t)}>
              {t}
            </button>
          ))}
        </div>
        <div className="h-48">
          <ResponsiveContainer>
            <LineChart data={dcaRows} margin={{ left: -10, right: 10, top: 5 }}>
              <CartesianGrid stroke="#243445" />
              <XAxis dataKey="t" type="number" domain={[0, 1]} {...AX} />
              <YAxis domain={[-0.1, 'auto']} allowDataOverflow {...AX} />
              <Tooltip {...TIP} formatter={(v) => Number(v).toFixed(3)} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Line dataKey="model" name="Model" stroke="#4fd1c5" dot={false} isAnimationActive={false} />
              <Line dataKey="all" name="Treat all" stroke="#f6c35b" dot={false} isAnimationActive={false} />
              <Line dataKey="none" name="Treat none" stroke="#8aa0b4" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section title={`Conformal prediction · ${pc(m.conformal.target_coverage, 0)} target coverage (test)`}>
        <table className="w-full">
          <thead className="text-[10px] uppercase text-muted">
            <tr>
              <Th>Target</Th>
              <Th>Coverage</Th>
              <Th>Mean set size</Th>
              <Th>"Uncertain" rate</Th>
            </tr>
          </thead>
          <tbody>
            {TARGETS.map((t) => (
              <tr key={t} className="border-t border-line/60">
                <Td strong>{t}</Td>
                <Td>{pc(m.conformal.test[t]?.coverage)}</Td>
                <Td>{f2(m.conformal.test[t]?.mean_set_size)}</Td>
                <Td>{pc(m.conformal.test[t]?.frac_uncertain)}</Td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      {m.ablation && (
        <Section
          title="Leakage ablation · why we don't report 97–99%"
          note={`${m.ablation.model}, ${m.ablation.cv}. Analysis only, never served. Doing SMOTE or feature selection on the full dataset before splitting lets test-fold information leak into training.`}
        >
          <table className="w-full">
            <thead className="text-[10px] uppercase text-muted">
              <tr>
                <Th>ROC-AUC</Th>
                <Th>SMOTE + selection before CV</Th>
                <Th>Correct (in-fold)</Th>
                <Th>Inflation</Th>
              </tr>
            </thead>
            <tbody>
              {TARGETS.map((t) => {
                const a = m.ablation!.targets[t]
                if (!a) return null
                const bad = a.A_smote_and_selection_before_cv?.roc_auc.mean
                const good = (a.D_correct_selection_and_smote_in_fold ?? a.D_correct_selection_in_fold)?.roc_auc.mean
                return (
                  <tr key={t} className="border-t border-line/60">
                    <Td strong>{t}</Td>
                    <Td>{f3(bad)}</Td>
                    <Td>{f3(good)}</Td>
                    <Td strong>+{f3(bad - good)}</Td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </Section>
      )}

      <Section
        title="Subgroups (nested-CV out-of-fold ROC-AUC)"
        note="Small subgroups give noisy AUCs; check n before reading a number. Performance is weakest in patients aged ≥ 65 (see Limitations)."
      >
        <table className="w-full">
          <thead className="text-[10px] uppercase text-muted">
            <tr>
              <Th>Group</Th>
              <Th>n</Th>
              {TARGETS.map((t) => (
                <Th key={t}>{t}</Th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(m.subgroups.groups).map(([g, r]) => (
              <tr key={g} className="border-t border-line/60">
                <Td strong>{g}</Td>
                <Td>{r.n}</Td>
                {TARGETS.map((t) => (
                  <Td key={t}>{f2(r[t]?.roc_auc)}</Td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section title="Explanation fidelity">
        <p className="text-text/80">
          Removing the top-SHAP features first lowers the prediction far more than removing random features (deletion-curve area{' '}
          {f3(m.explanations.deletion_auc_shap)} vs {f3(m.explanations.deletion_auc_random)}). Additivity error of the served explanation:{' '}
          {m.explanations.max_additivity_error.toExponential(1)}.
        </p>
      </Section>
    </div>
  )
}
