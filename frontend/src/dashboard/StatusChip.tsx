import type { Status } from '../types/api'
import { STATUS_CLASS, STATUS_HELP, STATUS_LABEL } from '../lib/risk'

export default function StatusChip({ status }: { status: Status }) {
  return (
    <span title={STATUS_HELP[status]} className={`rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ${STATUS_CLASS[status]}`}>
      {STATUS_LABEL[status]}
    </span>
  )
}
