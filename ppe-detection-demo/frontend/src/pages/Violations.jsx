import { useState, useEffect } from 'react'
import { AlertTriangle, HardHat, Shirt, ShieldAlert, Activity, Flame, ChevronLeft, ChevronRight } from 'lucide-react'
import api from '../api'

const TYPE_FILTERS = [
  { value: '', label: 'Tất cả' },
  { value: 'no_helmet', label: 'Thiếu mũ' },
  { value: 'no_vest', label: 'Thiếu áo' },
  { value: 'no_mask', label: 'Thiếu khẩu trang' },
  { value: 'fall_detected', label: 'Phát hiện ngã' },
  { value: 'fire_detected', label: 'Cháy 🔥' },
  { value: 'smoke_detected', label: 'Khói 💨' },
]

const TYPE_ICONS = {
  no_helmet: HardHat,
  no_vest: Shirt,
  no_mask: ShieldAlert,
  fall_detected: Activity,
  fire_detected: Flame,
  smoke_detected: Flame,
}

const TYPE_LABELS = {
  no_helmet: 'Thiếu mũ bảo hiểm',
  no_vest: 'Thiếu áo bảo hộ',
  no_mask: 'Thiếu khẩu trang',
  fall_detected: 'Phát hiện ngã',
  fire_detected: 'Phát hiện cháy (Hỏa hoạn)',
  smoke_detected: 'Phát hiện khói',
}

export default function Violations() {
  const [violations, setViolations] = useState([])
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [typeFilter, setTypeFilter] = useState('')
  const [hourly, setHourly] = useState([])

  useEffect(() => {
    const params = { page, per_page: 15 }
    if (typeFilter) params.type = typeFilter
    api.get('/violations', { params }).then(r => {
      setViolations(r.data.violations)
      setTotalPages(r.data.pages)
    }).catch(() => {})
  }, [page, typeFilter])

  useEffect(() => {
    api.get('/violations/stats/hourly').then(r => setHourly(r.data)).catch(() => {})
  }, [])

  const maxHourly = Math.max(...hourly.map(h => h.count), 1)

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Lịch sử vi phạm</h1>

      {/* Hourly Chart */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
        <h2 className="font-semibold mb-3">Phân bố vi phạm theo giờ (hôm nay)</h2>
        <div className="flex items-end gap-1 h-32">
          {hourly.map(h => (
            <div key={h.hour} className="flex-1 flex flex-col items-center gap-1">
              <div
                className="w-full bg-sky-500/60 rounded-t"
                style={{ height: `${(h.count / maxHourly) * 100}%`, minHeight: h.count > 0 ? 4 : 0 }}
                title={`${h.label}: ${h.count} vi phạm`}
              />
              {h.fall_count > 0 && (
                <div
                  className="w-full bg-rose-500/80 rounded-t"
                  style={{ height: `${(h.fall_count / maxHourly) * 100}%`, minHeight: 4 }}
                  title={`${h.label}: ${h.fall_count} ngã`}
                />
              )}
              {h.fire_count > 0 && (
                <div
                  className="w-full bg-red-600 rounded-t animate-pulse"
                  style={{ height: `${(h.fire_count / maxHourly) * 100}%`, minHeight: 4 }}
                  title={`${h.label}: ${h.fire_count} cháy`}
                />
              )}
              {h.smoke_count > 0 && (
                <div
                  className="w-full bg-amber-500/80 rounded-t"
                  style={{ height: `${(h.smoke_count / maxHourly) * 100}%`, minHeight: 4 }}
                  title={`${h.label}: ${h.smoke_count} khói`}
                />
              )}
            </div>
          ))}
        </div>
        <div className="flex gap-1 mt-1">
          {hourly.map(h => (
            <div key={h.hour} className="flex-1 text-center text-[10px] text-gray-500">
              {h.hour % 3 === 0 ? `${h.hour}h` : ''}
            </div>
          ))}
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-2 flex-wrap">
        {TYPE_FILTERS.map(f => (
          <button
            key={f.value}
            onClick={() => { setTypeFilter(f.value); setPage(1) }}
            className={`px-4 py-2 rounded-lg text-sm transition-colors ${
              typeFilter === f.value
                ? 'bg-sky-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Violations List */}
      <div className="space-y-3">
        {violations.length === 0 ? (
          <p className="text-gray-500 text-center py-8">Không có vi phạm nào</p>
        ) : (
          violations.map(v => {
            const Icon = TYPE_ICONS[v.type] || AlertTriangle
            const time = new Date(v.timestamp)
            const isEmergency = v.type === 'fire_detected' || v.type === 'fall_detected'
            return (
              <div key={v.id} className={`flex items-center gap-4 p-4 bg-gray-900 rounded-xl border ${isEmergency ? 'border-red-500/50 bg-red-950/10' : 'border-gray-800'}`}>
                <div className="w-16 h-16 rounded-lg overflow-hidden bg-gray-800 flex-shrink-0">
                  <img src={v.image} alt="" className="w-full h-full object-cover" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Icon className={`w-4 h-4 ${v.type === 'fire_detected' ? 'text-red-500 animate-pulse' : 'text-red-400'}`} />
                    <span className="font-medium">{TYPE_LABELS[v.type] || v.type}</span>
                    <span className="text-sm text-gray-500">{v.confidence}%</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">{time.toLocaleString('vi-VN')}</p>
                </div>
              </div>
            )
          })
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-4">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 disabled:opacity-30"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-sm text-gray-400">Trang {page} / {totalPages}</span>
          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 disabled:opacity-30"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  )
}
