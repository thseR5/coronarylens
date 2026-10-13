import type { Status, StatusReason } from '../types/api'
import { STATUS_CLASS, STATUS_HELP, STATUS_LABEL, statusReasonText } from '../lib/risk'

/** Status chip with its reason written out inline (not only in the tooltip). */
export default function StatusChip({ status, reason }: { status: Status; reason?: StatusReason }) {
  return (
    <span className="flex flex-col items-start gap-0.5" data-testid="status">
      <span title={STATUS_HELP[status]} className={`rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ${STATUS_CLASS[status]}`}>
        {STATUS_LABEL[status]}
      </span>
      {reason && (
        <span className="text-[10px] leading-tight text-muted" data-testid="status-reason">
          {statusReasonText(status, reason)}
        </span>
      )}
    </span>
  )
}
