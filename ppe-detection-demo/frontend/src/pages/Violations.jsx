import { useState, useEffect } from 'react'
import api from '../api'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'

import { Button } from '@/components/ui/button'
import { ChevronLeft, ChevronRight } from 'lucide-react'

const TYPE_FILTERS = [
  { value: '', label: 'Tất cả' },
  { value: 'no_helmet', label: 'Thiếu mũ' },
  { value: 'no_vest', label: 'Thiếu áo' },
  { value: 'no_mask', label: 'Thiếu khẩu trang' },
  { value: 'fall_detected', label: 'Phát hiện ngã' },
  { value: 'fire_detected', label: 'Hỏa hoạn' },
  { value: 'smoke_detected', label: 'Khói nguy hiểm' },
]

const TYPE_CONFIG = {
  no_helmet: { label: 'Thiếu mũ bảo hiểm', dot: 'bg-amber-400', border: 'border-border' },
  no_vest: { label: 'Thiếu áo bảo hộ', dot: 'bg-orange-400', border: 'border-border' },
  no_mask: { label: 'Thiếu khẩu trang', dot: 'bg-purple-400', border: 'border-border' },
  fall_detected: { label: 'Phát hiện ngã', dot: 'bg-rose-500', border: 'border-rose-500/30' },
  fire_detected: { label: 'Phát hiện cháy', dot: 'bg-red-500 animate-pulse', border: 'border-red-500/40' },
  smoke_detected: { label: 'Phát hiện khói', dot: 'bg-amber-500', border: 'border-amber-500/30' },
}

export default function Violations() {
  const [violations, setViolations] = useState([])
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [typeFilter, setTypeFilter] = useState('')
  const [hourly, setHourly] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    const params = { page, per_page: 12 }
    if (typeFilter) params.type = typeFilter
    api.get('/violations', { params })
      .then((r) => {
        setViolations(r.data.violations || [])
        setTotalPages(r.data.pages || 1)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [page, typeFilter])

  useEffect(() => {
    api.get('/violations/stats/hourly').then((r) => setHourly(r.data || [])).catch(() => {})
  }, [])

  const maxHourly = Math.max(...hourly.map((h) => h.count || 0), 1)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-foreground">Nhật ký vi phạm & sự cố</h1>
        <p className="text-xs text-muted-foreground mt-0.5">
          Tra cứu bằng chứng hình ảnh và lịch sử phân bố vi phạm theo khung giờ
        </p>
      </div>

      {/* Hourly Activity Bar Chart */}
      <Card className="bg-card border-border shadow-none">
        <CardHeader className="p-4 pb-2 border-b border-border/50">
          <div className="flex items-center justify-between">
            <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Phân bố vi phạm theo giờ trong ngày
            </CardTitle>
            <span className="text-[11px] text-muted-foreground">
              24 giờ qua
            </span>
          </div>
        </CardHeader>
        <CardContent className="p-4">
          <div className="flex items-end gap-1.5 h-24 pt-4">
            {hourly.map((h) => {
              const hasEmergency = (h.fire_count || 0) > 0 || (h.fall_count || 0) > 0
              const barHeight = Math.max((h.count / maxHourly) * 100, h.count > 0 ? 8 : 2)
              return (
                <div key={h.hour} className="flex-1 flex flex-col items-center gap-1 group relative">
                  <div
                    className={`w-full rounded-sm transition-all ${
                      hasEmergency
                        ? 'bg-red-500/80 group-hover:bg-red-400'
                        : h.count > 0
                        ? 'bg-zinc-600 group-hover:bg-zinc-400'
                        : 'bg-zinc-800/40'
                    }`}
                    style={{ height: `${barHeight}%` }}
                  />
                  {/* Tooltip on hover */}
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute bottom-full mb-1 bg-zinc-900 border border-zinc-700 text-[10px] text-zinc-200 px-2 py-1 rounded shadow-lg pointer-events-none whitespace-nowrap z-10">
                    <span className="font-semibold">{h.hour}:00</span>: {h.count} vi phạm
                    {h.fire_count > 0 && ` (${h.fire_count} cháy)`}
                    {h.fall_count > 0 && ` (${h.fall_count} ngã)`}
                  </div>
                </div>
              )
            })}
          </div>
          <div className="flex gap-1.5 mt-2">
            {hourly.map((h) => (
              <div key={h.hour} className="flex-1 text-center text-[10px] text-muted-foreground font-mono">
                {h.hour % 3 === 0 ? `${h.hour}h` : ''}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Filter Tabs */}
      <div className="flex items-center gap-1.5 flex-wrap">
        {TYPE_FILTERS.map((f) => (
          <button
            key={f.value}
            onClick={() => {
              setTypeFilter(f.value)
              setPage(1)
            }}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
              typeFilter === f.value
                ? 'bg-accent text-accent-foreground border border-border'
                : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Violations Cards Grid */}
      {loading ? (
        <div className="text-center py-16 text-xs text-muted-foreground">
          Đang tải danh sách vi phạm...
        </div>
      ) : violations.length === 0 ? (
        <div className="text-center py-16 text-xs text-muted-foreground border border-dashed border-border rounded-xl">
          Không tìm thấy bản ghi vi phạm nào phù hợp với bộ lọc
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {violations.map((v) => {
            const cfg = TYPE_CONFIG[v.type] || { label: v.type, dot: 'bg-zinc-500', border: 'border-border' }
            const time = new Date(v.timestamp).toLocaleString('vi-VN', {
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
              day: '2-digit',
              month: '2-digit',
            })
            const isEmergency = v.type === 'fire_detected' || v.type === 'fall_detected'
            return (
              <Card
                key={v.id}
                className={`bg-card ${cfg.border} shadow-none overflow-hidden flex flex-col justify-between`}
              >
                <div className="p-3 pb-2 flex items-start gap-3">
                  {/* Snapshot thumbnail */}
                  <div className="w-16 h-16 rounded-md bg-muted flex-shrink-0 overflow-hidden border border-border/40">
                    {v.image ? (
                      <img
                        src={v.image}
                        alt="Violation snapshot"
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-[10px] text-muted-foreground">
                        No img
                      </div>
                    )}
                  </div>

                  {/* Metadata */}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot} flex-shrink-0`} />
                      <span className="text-xs font-semibold text-foreground truncate">
                        {cfg.label}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[11px] text-muted-foreground mt-1">
                      <span>Độ tin cậy:</span>
                      <span className="font-mono text-foreground font-medium">{v.confidence}%</span>
                    </div>
                    <div className="text-[10px] text-muted-foreground font-mono mt-1">
                      {time}
                    </div>
                  </div>
                </div>
              </Card>
            )
          })}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between pt-2">
          <span className="text-xs text-muted-foreground font-mono">
            Trang {page} / {totalPages}
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="h-8 px-2.5 text-xs"
            >
              <ChevronLeft className="w-3.5 h-3.5 mr-1" /> Trước
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="h-8 px-2.5 text-xs"
            >
              Sau <ChevronRight className="w-3.5 h-3.5 ml-1" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
