import { useState, useEffect } from 'react'
import api from '../api'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Download,
  X,
  Clock,
  Camera,
  CheckCircle2,
  AlertTriangle,
  Flame,
  Activity,
  Calendar,
  CalendarRange,
  Filter,
  X as XIcon,
} from 'lucide-react'

const TIME_FILTERS = [
  { value: 'all', label: 'Tất cả' },
  { value: 'today', label: 'Hôm nay' },
  { value: '24h', label: '24 giờ qua' },
  { value: '7d', label: '7 ngày qua' },
  { value: '30d', label: '30 ngày qua' },
]

const TYPE_FILTERS = [
  { value: '', label: 'Tất cả loại' },
  { value: 'no_helmet', label: 'Thiếu mũ' },
  { value: 'no_vest', label: 'Thiếu áo' },
  { value: 'no_mask', label: 'Thiếu khẩu trang' },
  { value: 'fall', label: 'Phát hiện ngã' },
  { value: 'fire_detected', label: 'Hỏa hoạn' },
  { value: 'smoke_detected', label: 'Khói nguy hiểm' },
]

const TYPE_CONFIG = {
  no_helmet: { label: 'Thiếu mũ bảo hiểm' },
  no_vest: { label: 'Thiếu áo bảo hộ' },
  no_mask: { label: 'Thiếu khẩu trang' },
  fall_detected: { label: 'Phát hiện ngã' },
  fall_immobile: { label: 'Ngã bất động (>5s)' },
  fire_detected: { label: 'Phát hiện cháy' },
  smoke_detected: { label: 'Phát hiện khói' },
}

function getPageNumbers(currentPage, total) {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i + 1)
  }
  if (currentPage <= 4) {
    return [1, 2, 3, 4, 5, '...', total]
  }
  if (currentPage >= total - 3) {
    return [1, '...', total - 4, total - 3, total - 2, total - 1, total]
  }
  return [1, '...', currentPage - 1, currentPage, currentPage + 1, '...', total]
}

export default function Violations() {
  // Load saved filters from localStorage for state persistence
  const [timeFilter, setTimeFilter] = useState(() => {
    try {
      const saved = localStorage.getItem('ppe_violations_time_filter')
      return saved || 'today'
    } catch {
      return 'today'
    }
  })

  const [typeFilter, setTypeFilter] = useState(() => {
    try {
      const saved = localStorage.getItem('ppe_violations_type_filter')
      return saved || ''
    } catch {
      return ''
    }
  })

  const [fromDate, setFromDate] = useState(() => {
    try {
      const saved = localStorage.getItem('ppe_violations_from_date')
      return saved || ''
    } catch {
      return ''
    }
  })

  const [toDate, setToDate] = useState(() => {
    try {
      const saved = localStorage.getItem('ppe_violations_to_date')
      return saved || ''
    } catch {
      return ''
    }
  })

  const [page, setPage] = useState(() => {
    try {
      const saved = localStorage.getItem('ppe_violations_page')
      return saved ? Number(saved) : 1
    } catch {
      return 1
    }
  })

  const [perPage, setPerPage] = useState(() => {
    try {
      const saved = localStorage.getItem('ppe_violations_per_page')
      return saved ? Number(saved) : 10
    } catch {
      return 10
    }
  })

  const [violations, setViolations] = useState([])
  const [totalPages, setTotalPages] = useState(1)
  const [totalCount, setTotalCount] = useState(0)
  const [loading, setLoading] = useState(false)

  // Modal Detail & Zoom State
  const [selectedViolation, setSelectedViolation] = useState(null)
  const [zoomLevel, setZoomLevel] = useState(1)

  // Persist filter states
  useEffect(() => {
    try {
      localStorage.setItem('ppe_violations_time_filter', timeFilter)
      localStorage.setItem('ppe_violations_type_filter', typeFilter)
      localStorage.setItem('ppe_violations_from_date', fromDate)
      localStorage.setItem('ppe_violations_to_date', toDate)
      localStorage.setItem('ppe_violations_page', String(page))
      localStorage.setItem('ppe_violations_per_page', String(perPage))
    } catch (e) {}
  }, [timeFilter, typeFilter, fromDate, toDate, page, perPage])

  const handleFilterChange = () => {
    setPage(1)
  }

  useEffect(() => {
    setLoading(true)
    const params = { page, per_page: perPage, time_range: timeFilter }
    if (typeFilter) params.type = typeFilter
    if (fromDate) params.from_date = fromDate
    if (toDate) params.to_date = toDate

    api.get('/violations', { params })
      .then((r) => {
        setViolations(r.data.violations || [])
        setTotalPages(r.data.pages || 1)
        setTotalCount(r.data.total || 0)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [page, perPage, typeFilter, timeFilter, fromDate, toDate])

  // Close modal on Escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setSelectedViolation(null)
        setZoomLevel(1)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const startRecord = totalCount === 0 ? 0 : (page - 1) * perPage + 1
  const endRecord = Math.min(page * perPage, totalCount)

  const openDetail = (v) => {
    setSelectedViolation(v)
    setZoomLevel(1)
  }

  const closeDetail = () => {
    setSelectedViolation(null)
    setZoomLevel(1)
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-foreground">Nhật ký vi phạm & sự cố</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Danh sách sự cố và vi phạm an toàn lao động được ghi nhận theo thời gian thực
          </p>
        </div>

        <div className="text-xs text-muted-foreground font-mono">
          Tổng số: <strong className="text-foreground">{totalCount}</strong> bản ghi
        </div>
      </div>

      {/* Filter Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3 bg-card border border-border rounded-xl">
        {/* Time Filters (Quick) */}
        <div className="flex items-center gap-1 overflow-x-auto pb-1 sm:pb-0">
          <span className="text-[11px] font-medium text-muted-foreground mr-1.5 flex items-center gap-1">
            <Clock className="w-3 h-3" />
            Thời gian:
          </span>
          {TIME_FILTERS.map((tf) => (
            <button
              key={tf.value}
              onClick={() => {
                setTimeFilter(tf.value)
                setFromDate('')
                setToDate('')
                setPage(1)
              }}
              className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors cursor-pointer whitespace-nowrap ${
                timeFilter === tf.value
                  ? 'bg-accent text-accent-foreground font-semibold shadow-xs'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
              }`}
            >
              {tf.label}
            </button>
          ))}
        </div>

        {/* Date Range Picker */}
        <div className="flex items-center gap-1.5 overflow-x-auto">
          <span className="text-[11px] font-medium text-muted-foreground mr-1 flex items-center gap-1">
            <Calendar className="w-3 h-3" />
            Từ ngày:
          </span>
          <Input
            type="date"
            value={fromDate}
            onChange={(e) => { setFromDate(e.target.value); handleFilterChange(); }}
            className="bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground focus:outline-none cursor-pointer w-[150px]"
          />
          <span className="text-[11px] font-medium text-muted-foreground mx-1 flex items-center gap-1">
            <Calendar className="w-3 h-3" />
            Đến ngày:
          </span>
          <Input
            type="date"
            value={toDate}
            onChange={(e) => { setToDate(e.target.value); handleFilterChange(); }}
            className="bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground focus:outline-none cursor-pointer w-[150px]"
          />
          {(fromDate || toDate) && (
            <button
              onClick={() => { setFromDate(''); setToDate(''); handleFilterChange(); }}
              className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors cursor-pointer"
              title="Xóa khoảng ngày"
            >
              <XIcon className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Type Filter Dropdown */}
        <div className="flex items-center gap-1.5 overflow-x-auto">
          <span className="text-[11px] font-medium text-muted-foreground mr-1">
            Loại:
          </span>
          <select
            value={typeFilter}
            onChange={(e) => {
              setTypeFilter(e.target.value)
              setPage(1)
            }}
            className="bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground focus:outline-none cursor-pointer"
          >
            {TYPE_FILTERS.map((tf) => (
              <option key={tf.value} value={tf.value}>
                {tf.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Horizontal Bars List ("Mỗi lỗi là thanh ngang") */}
      {loading ? (
        <div className="text-center py-16 text-xs text-muted-foreground">
          Đang tải danh sách vi phạm...
        </div>
      ) : violations.length === 0 ? (
        <div className="text-center py-16 text-xs text-muted-foreground border border-dashed border-border rounded-xl p-8">
          <AlertTriangle className="w-8 h-8 mx-auto mb-2 opacity-30 text-amber-400" />
          <p className="font-medium text-foreground">Không tìm thấy bản ghi vi phạm nào</p>
          <p className="text-[11px] text-muted-foreground/70 mt-0.5">
            Thử thay đổi mốc thời gian hoặc loại sự cố trong bộ lọc phía trên
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {violations.map((v) => {
            const cfg = TYPE_CONFIG[v.type] || { label: v.type }
            const time = new Date(v.timestamp).toLocaleString('vi-VN', {
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
              day: '2-digit',
              month: '2-digit',
              year: 'numeric',
            })

            return (
              <div
                key={v.id}
                onClick={() => openDetail(v)}
                className="group flex items-center justify-between p-3.5 rounded-xl border border-border bg-card hover:bg-muted/40 hover:border-zinc-500/70 cursor-pointer transition-all gap-4"
                title="Nhấn để xem chi tiết sự cố và phóng to ảnh"
              >
                {/* Left: Type, ID, Camera, Time */}
                <div className="flex items-center gap-3 min-w-0 flex-wrap sm:flex-nowrap">
                  <span className="text-xs font-semibold text-foreground whitespace-nowrap">
                    {cfg.label}
                  </span>
                  <Badge variant="outline" className="text-[10px] h-4 font-mono px-1.5 py-0 text-muted-foreground">
                    #{v.id}
                  </Badge>

                  {v.camera_id !== null && (
                    <span className="text-[11px] text-muted-foreground flex items-center gap-1 font-mono bg-muted/60 px-2 py-0.5 rounded border border-border/40 whitespace-nowrap">
                      <Camera className="w-3 h-3 opacity-60" />
                      Cam #{v.camera_id}
                    </span>
                  )}

                  <div className="text-[11px] text-muted-foreground flex items-center gap-1.5 font-mono whitespace-nowrap">
                    <Clock className="w-3 h-3 opacity-60" />
                    <span>{time}</span>
                  </div>
                </div>

                {/* Right: Confidence Score & Direction Arrow */}
                <div className="flex items-center gap-3 sm:gap-4 flex-shrink-0">
                  <div className="flex items-center gap-1.5 text-xs font-mono">
                    <span className="text-muted-foreground text-[11px] hidden sm:inline">Độ tin cậy:</span>
                    <span className={`font-semibold px-2 py-0.5 rounded text-[11px] ${
                      v.confidence >= 80
                        ? 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/30'
                        : 'text-amber-400 bg-amber-500/10 border border-amber-500/30'
                    }`}>
                      {v.confidence}%
                    </span>
                  </div>

                  <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground group-hover:translate-x-0.5 transition-all" />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Pagination Controls */}
      {totalCount > 0 && (
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-3 border-t border-border/50 text-xs">
          {/* Summary & Per Page Selector */}
          <div className="flex items-center gap-3 text-muted-foreground flex-wrap">
            <span>
              Hiển thị <strong className="text-foreground font-mono">{startRecord}</strong> - <strong className="text-foreground font-mono">{endRecord}</strong> trong tổng số <strong className="text-foreground font-mono">{totalCount}</strong> vi phạm
            </span>
            <div className="flex items-center gap-1.5 ml-1">
              <span className="text-[11px]">Mỗi trang:</span>
              <select
                value={perPage}
                onChange={(e) => {
                  const val = Number(e.target.value)
                  setPerPage(val)
                  setPage(1)
                }}
                className="bg-muted border border-border rounded-md px-2 py-0.5 text-xs text-foreground focus:outline-none cursor-pointer"
              >
                <option value={10}>10</option>
                <option value={15}>15</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
              </select>
            </div>
          </div>

          {/* Page Buttons */}
          <div className="flex items-center gap-1">
            {/* First Page */}
            <Button
              variant="outline"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage(1)}
              className="h-8 w-8 p-0 text-xs cursor-pointer"
              title="Trang đầu"
            >
              <ChevronsLeft className="w-3.5 h-3.5" />
            </Button>

            {/* Previous Page */}
            <Button
              variant="outline"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="h-8 px-2.5 text-xs cursor-pointer gap-1"
              title="Trang trước"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Trước</span>
            </Button>

            {/* Numeric Page Buttons */}
            <div className="flex items-center gap-1">
              {getPageNumbers(page, totalPages).map((p, idx) => {
                if (p === '...') {
                  return (
                    <span key={`dots-${idx}`} className="px-1 text-muted-foreground select-none">
                      ...
                    </span>
                  )
                }
                const isCurrent = p === page
                return (
                  <button
                    key={`page-${p}`}
                    onClick={() => setPage(p)}
                    className={`h-8 min-w-[2rem] px-2 rounded-lg text-xs font-medium font-mono transition-colors cursor-pointer ${
                      isCurrent
                        ? 'bg-primary text-primary-foreground font-bold shadow-xs'
                        : 'text-muted-foreground hover:bg-muted hover:text-foreground border border-transparent hover:border-border'
                    }`}
                  >
                    {p}
                  </button>
                )
              })}
            </div>

            {/* Next Page */}
            <Button
              variant="outline"
              size="sm"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="h-8 px-2.5 text-xs cursor-pointer gap-1"
              title="Trang sau"
            >
              <span className="hidden sm:inline">Sau</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </Button>

            {/* Last Page */}
            <Button
              variant="outline"
              size="sm"
              disabled={page >= totalPages}
              onClick={() => setPage(totalPages)}
              className="h-8 w-8 p-0 text-xs cursor-pointer"
              title="Trang cuối"
            >
              <ChevronsRight className="w-3.5 h-3.5" />
            </Button>
          </div>
        </div>
      )}

      {/* Detail & Image Zoom Modal Dialog ("Khi ấn vào sẽ ra chi tiết lỗi và phóng to ảnh") */}
      {selectedViolation && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-md flex items-center justify-center z-50 p-4 animate-in fade-in-0 duration-150"
          onClick={closeDetail}
        >
          <div
            className="bg-card border border-border rounded-2xl max-w-4xl w-full max-h-[92vh] flex flex-col shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b border-border/70 flex-shrink-0">
              <div className="flex items-center gap-2.5">
                <h3 className="text-sm font-semibold text-foreground">
                  Chi tiết sự cố: {TYPE_CONFIG[selectedViolation.type]?.label || selectedViolation.type}
                </h3>
                <Badge variant="outline" className="text-xs font-mono h-5">
                  ID #{selectedViolation.id}
                </Badge>
              </div>

              <div className="flex items-center gap-2">
                {/* Download Image Button */}
                <a
                  href={selectedViolation.image}
                  download={`violation_${selectedViolation.id}.jpg`}
                  target="_blank"
                  rel="noreferrer"
                  className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                  title="Tải ảnh bằng chứng"
                >
                  <Download className="w-4 h-4" />
                </a>

                {/* Close Button */}
                <button
                  onClick={closeDetail}
                  className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors cursor-pointer"
                  title="Đóng (Esc)"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Modal Body: Image Viewport + Zoom Controls */}
            <div className="flex-1 min-h-0 flex flex-col lg:flex-row overflow-hidden">
              {/* Image Preview with Interactive Zoom */}
              <div className="flex-1 min-h-[300px] lg:min-h-0 bg-black/95 relative overflow-hidden flex items-center justify-center p-3 select-none">
                <div
                  className="w-full h-full flex items-center justify-center overflow-auto cursor-zoom-in"
                  onClick={() => setZoomLevel((z) => (z >= 2.5 ? 1 : z + 0.5))}
                >
                  <img
                    src={selectedViolation.image}
                    alt="Evidence Snapshot"
                    style={{ transform: `scale(${zoomLevel})` }}
                    className="max-h-[60vh] max-w-full object-contain transition-transform duration-200 block rounded shadow-lg"
                  />
                </div>

                {/* Floating Zoom Controls Bar */}
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/80 border border-zinc-700/80 backdrop-blur-md rounded-full px-3 py-1 flex items-center gap-2 text-xs text-white z-10 shadow-lg">
                  <button
                    onClick={() => setZoomLevel((z) => Math.max(1, Number((z - 0.5).toFixed(1))))}
                    disabled={zoomLevel <= 1}
                    className="p-1 hover:text-blue-400 disabled:opacity-30 cursor-pointer"
                    title="Thu nhỏ"
                  >
                    <ZoomOut className="w-3.5 h-3.5" />
                  </button>

                  <span className="font-mono text-[11px] min-w-[3rem] text-center font-semibold">
                    {Math.round(zoomLevel * 100)}%
                  </span>

                  <button
                    onClick={() => setZoomLevel((z) => Math.min(3, Number((z + 0.5).toFixed(1))))}
                    disabled={zoomLevel >= 3}
                    className="p-1 hover:text-blue-400 disabled:opacity-30 cursor-pointer"
                    title="Phóng to"
                  >
                    <ZoomIn className="w-3.5 h-3.5" />
                  </button>

                  <span className="w-px h-3 bg-zinc-700 mx-1" />

                  <button
                    onClick={() => setZoomLevel(zoomLevel === 1 ? 2 : 1)}
                    className="p-1 hover:text-blue-400 cursor-pointer text-[10px] font-medium"
                    title="Đặt lại hoặc phóng to"
                  >
                    <Maximize2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Sidebar Metadata */}
              <div className="w-full lg:w-72 bg-card border-t lg:border-t-0 lg:border-l border-border/70 p-4 space-y-4 overflow-y-auto">
                <div>
                  <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                    Thông tin bằng chứng
                  </h4>

                  <div className="space-y-2.5 text-xs">
                    <div>
                      <span className="text-muted-foreground block text-[11px]">Loại vi phạm:</span>
                      <strong className="text-foreground text-sm font-semibold">
                        {TYPE_CONFIG[selectedViolation.type]?.label || selectedViolation.type}
                      </strong>
                    </div>

                    <div>
                      <span className="text-muted-foreground block text-[11px]">Độ tin cậy AI:</span>
                      <div className="flex items-center gap-2 mt-0.5">
                        <div className="flex-1 bg-muted rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-blue-500 h-full rounded-full"
                            style={{ width: `${selectedViolation.confidence}%` }}
                          />
                        </div>
                        <span className="font-mono font-bold text-foreground tabular-nums">
                          {selectedViolation.confidence}%
                        </span>
                      </div>
                    </div>

                    <div>
                      <span className="text-muted-foreground block text-[11px]">Thời điểm ghi nhận:</span>
                      <span className="font-mono text-foreground font-medium block mt-0.5">
                        {new Date(selectedViolation.timestamp).toLocaleString('vi-VN')}
                      </span>
                    </div>

                    <div>
                      <span className="text-muted-foreground block text-[11px]">Nguồn camera:</span>
                      <span className="text-foreground font-medium block mt-0.5">
                        {selectedViolation.camera_id !== null ? `Camera #${selectedViolation.camera_id}` : 'Webcam mặc định (Kênh 0)'}
                      </span>
                    </div>

                    <div>
                      <span className="text-muted-foreground block text-[11px]">Trạng thái xử lý:</span>
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-emerald-500/10 text-emerald-400 font-medium text-[11px] mt-1 border border-emerald-500/30">
                        <CheckCircle2 className="w-3 h-3" />
                        Đã lưu trữ dữ liệu
                      </span>
                    </div>
                  </div>
                </div>

                <div className="pt-2 border-t border-border/50">
                  <Button
                    onClick={closeDetail}
                    className="w-full text-xs h-8 cursor-pointer"
                  >
                    Đóng chi tiết
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
