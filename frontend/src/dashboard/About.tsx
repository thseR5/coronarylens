import { useStore } from '../store'
import { DISCLAIMER } from '../lib/disclaimer'
import TerritoryAgreement from './TerritoryAgreement'

export default function About() {
  const anatomy = useStore((s) => s.config?.anatomy)
  return (
    <div className="flex flex-col gap-3 text-xs leading-relaxed text-text/90">
      <div className="rounded-md bg-warn/10 p-2 text-warn ring-1 ring-warn/40">{DISCLAIMER}</div>
      <section>
        <h3 className="h-section">Data</h3>
        <p>
          Extension of Z-Alizadeh Sani dataset (Alizadehsani, Roshanzamir &amp; Sani; UCI Machine Learning Repository id 411,
          CC BY 4.0): 303 patients from a single centre in Iran, with demographic, examination, ECG, laboratory and echo features.
          Angiography labels are vessel-level (stenotic yes/no for LAD, LCX, RCA), not lesion locations. The dataset has no cardiac
          enzymes, total cholesterol, diastolic pressure or left-main label.
        </p>
      </section>
      <section>
        <h3 className="h-section">How the model works</h3>
        <p>
          One calibrated model predicts the probability of each of the 8 possible LAD/LCX/RCA patterns. Every number shown is a sum
          of those 8 probabilities, so the overall CAD risk can never be lower than any single vessel's risk. LAD, LCX, RCA and Cath
          are never used as inputs. Intervals come from a bootstrap ensemble; "uncertain" means the 90% conformal prediction set
          contains both outcomes or disagrees with the threshold decision, or that the patient is out of distribution or many
          inputs were imputed. Artery estimates mostly follow overall CAD risk: among patients with CAD, the model separates LAD
          modestly and LCX/RCA barely better than chance, so they are risk levels, not lesion locations.
        </p>
      </section>
      <section>
        <h3 className="h-section">Anatomy</h3>
        <p>
          {anatomy?.source.attribution}. Coronary meshes are mapped to model outputs through FMA identifiers in{' '}
          <code>config/anatomy.yaml</code>. {anatomy?.territory_note} The left main artery is shown grey because the dataset has no
          left-main label.
        </p>
        <TerritoryAgreement detailed />
      </section>
      <section>
        <h3 className="h-section">Limitations</h3>
        <ul className="list-disc space-y-0.5 pl-4">
          <li>Small single-centre cohort (n = 303); no external validation.</li>
          <li>LCX and RCA are hard to predict from clinical data (AUC ≈ 0.7); treat them as screening hints.</li>
          <li>Territories and bullseye show typical supply, not this patient's anatomy.</li>
          <li>Not a medical device. Not for diagnosis or treatment decisions.</li>
        </ul>
      </section>
    </div>
  )
}
