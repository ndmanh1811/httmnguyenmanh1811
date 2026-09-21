import { useState, useRef, useEffect } from 'react'
import {
  Upload as UploadIcon,
  FileVideo,
  Loader2,
  CheckCircle,
  XCircle,
  HardHat,
  AlertTriangle,
  Flame,
  Layers,
  Plus,
  Trash2,
  Check,
  X,
  Eye,
  EyeOff,
  HelpCircle,
} from 'lucide-react'
import { io } from 'socket.io-client'
import api from '../api'

const ZONE_TYPE_LABELS = {
  welding: { label: 'Hàn xì / Cắt kim loại', icon: '⚡' },
  kitchen: { label: 'Bếp / Nấu nướng', icon: '🍳' },
  boiler: { label: 'Lò hơi / Ống khói', icon: '🏭' },
  smoking: { label: 'Khu hút thuốc', icon: '🚬' },
  other: { label: 'Khu vực loại trừ khác', icon: '🛡️' },
}

// Tam thoi an tinh nang khoanh vung theo yeu cau, tap trung vao phat hien khoi lua
const HIDE_EXCLUSION_ZONES = true

export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [frameSkip, setFrameSkip] = useState(2)
  const [enablePpe, setEnablePpe] = useState(true)
  const [enableFall, setEnableFall] = useState(true)
  const [enableFire, setEnableFire] = useState(true)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [progress, setProgress] = useState(null)
  const [cancelled, setCancelled] = useState(false)

  // Exclusion Zone (ROI) for Uploaded Video
  const [exclusionZones, setExclusionZones] = useState([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [currentPoints, setCurrentPoints] = useState([])
  const [cursorPos, setCursorPos] = useState(null)
  const [showZoneModal, setShowZoneModal] = useState(false)
  const [newZoneName, setNewZoneName] = useState('')
  const [newZoneType, setNewZoneType] = useState('welding')

  const inputRef = useRef(null)
  const uploadIdRef = useRef(null)
  const socketRef = useRef(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    // Load default settings
    api.get('/settings').then((r) => {
      if (r.data.enable_ppe) setEnablePpe(String(r.data.enable_ppe.value).toLowerCase() === 'true')
      if (r.data.enable_fall) setEnableFall(String(r.data.enable_fall.value).toLowerCase() === 'true')
      if (r.data.enable_fire) setEnableFire(String(r.data.enable_fire.value).toLowerCase() === 'true')
    }).catch(() => {})

    const socket = io({ transports: ['polling'] })
    socketRef.current = socket

    socket.on('upload_progress', (data) => {
      // Chỉ nhận progress của đúng upload_id đang active, bỏ qua toàn bộ progress từ tiến trình cũ
      if (!uploadIdRef.current || data.upload_id !== uploadIdRef.current) return
      setProgress(data)
    })

    socket.on('upload_done', (data) => {
      if (uploadIdRef.current && data.upload_id !== uploadIdRef.current) return
      if (data.error) {
        alert('Lỗi: ' + data.error)
        setCancelled(false)
      } else if (data.cancelled) {
        setCancelled(true)
      } else {
        setResult(data)
      }
      setProgress(null)
      setLoading(false)
      uploadIdRef.current = null
    })

    return () => { socket.disconnect() }
  }, [])

  // Manage Preview URL when file changes
  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
      setExclusionZones([])
      setIsDrawing(false)
      setCurrentPoints([])
      return () => URL.revokeObjectURL(url)
    } else {
      setPreviewUrl(null)
    }
  }, [file])

  const handleCancel = async () => {
    const currentId = uploadIdRef.current
    // Reset UI state ngay lập tức để người dùng không bị kẹt giao diện
    uploadIdRef.current = null
    setLoading(false)
    setProgress(null)
    setCancelled(true)
    try {
      await api.post('/detect/cancel', { upload_id: currentId })
    } catch {}
  }

  const handleUpload = async () => {
    if (!file) return
    if (!enablePpe && !enableFall && !enableFire) {
      alert('Vui lòng chọn ít nhất một tính năng phát hiện (PPE, Ngã hoặc Cháy/Khói)!')
      return
    }

    // Nếu đang có tiến trình cũ, gửi lệnh hủy trước khi bắt đầu phân tích mới
    if (uploadIdRef.current) {
      try {
        await api.post('/detect/cancel', { upload_id: uploadIdRef.current })
      } catch {}
    }

    setLoading(true)
    setProgress(null)
    setResult(null)
    setCancelled(false)
    uploadIdRef.current = null

    const formData = new FormData()
    formData.append('video', file)
    formData.append('frame_skip', frameSkip)
    formData.append('enable_ppe', enablePpe)
    formData.append('enable_fall', enableFall)
    formData.append('enable_fire', enableFire)

    // Append Exclusion Zones if any
    if (exclusionZones.length > 0) {
      formData.append('exclusion_zones', JSON.stringify(exclusionZones))
    }

    try {
      const res = await api.post('/detect/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      uploadIdRef.current = res.data.upload_id
    } catch (err) {
      alert('Lỗi: ' + (err.response?.data?.error || err.message))
      setLoading(false)
      uploadIdRef.current = null
    }
  }

  // --- Canvas Drawing Logic for Video Preview ---
  const handleCanvasClick = (e) => {
    if (!isDrawing) return
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const xNorm = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const yNorm = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
    setCurrentPoints((prev) => [...prev, [Number(xNorm.toFixed(4)), Number(yNorm.toFixed(4))]])
  }

  const handleCanvasMouseMove = (e) => {
    if (!isDrawing) return
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const xNorm = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const yNorm = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
    setCursorPos({ x: xNorm, y: yNorm })
  }

  const handleStartDrawing = () => {
    if (videoRef.current) {
      videoRef.current.pause()
    }
    setIsDrawing(true)
    setCurrentPoints([])
    setCursorPos(null)
  }

  const handleCancelDrawing = () => {
    setIsDrawing(false)
    setCurrentPoints([])
    setCursorPos(null)
  }

  const handleFinishDrawing = () => {
    if (currentPoints.length < 3) {
      alert('Vui lòng chọn ít nhất 3 điểm góc để tạo thành đa giác khép kín!')
      return
    }
    setNewZoneName(`Vùng loại trừ #${exclusionZones.length + 1}`)
    setShowZoneModal(true)
  }

  const handleSaveZone = () => {
    if (!newZoneName.trim()) {
      alert('Vui lòng nhập tên vùng')
      return
    }
    const newZone = {
      id: Date.now(),
      name: newZoneName.trim(),
      zone_type: newZoneType,
      polygon_points: currentPoints,
      is_active: true,
    }
    setExclusionZones((prev) => [...prev, newZone])
    setShowZoneModal(false)
    setIsDrawing(false)
    setCurrentPoints([])
    setCursorPos(null)
  }

  const handleDeleteZone = (zoneId) => {
    setExclusionZones((prev) => prev.filter((z) => z.id !== zoneId))
  }

  // Render Polygons onto Canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.offsetWidth || 640
    const height = canvas.offsetHeight || 360
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }

    ctx.clearRect(0, 0, width, height)

    // 1. Draw Saved Exclusion Zones
    exclusionZones.forEach((zone) => {
      const points = zone.polygon_points || []
      if (points.length < 3) return

      ctx.beginPath()
      points.forEach(([xNorm, yNorm], idx) => {
        const px = xNorm * width
        const py = yNorm * height
        if (idx === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.closePath()

      ctx.fillStyle = 'rgba(245, 158, 11, 0.22)'
      ctx.strokeStyle = '#f59e0b'
      ctx.lineWidth = 2
      ctx.fill()
      ctx.stroke()

      // Zone Label Badge
      const [firstX, firstY] = points[0]
      const lx = firstX * width
      const ly = Math.max(22, firstY * height - 8)
      const typeIcon = ZONE_TYPE_LABELS[zone.zone_type]?.icon || '🛡️'
      const labelText = `${typeIcon} ${zone.name}`

      ctx.font = 'bold 11px sans-serif'
      const textMetrics = ctx.measureText(labelText)
      const textWidth = textMetrics.width
      const pad = 6

      ctx.fillStyle = 'rgba(30, 41, 59, 0.9)'
      ctx.strokeStyle = '#f59e0b'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.roundRect ? ctx.roundRect(lx - 2, ly - 14, textWidth + pad * 2, 20, 4) : ctx.rect(lx - 2, ly - 14, textWidth + pad * 2, 20)
      ctx.fill()
      ctx.stroke()

      ctx.fillStyle = '#fef3c7'
      ctx.fillText(labelText, lx + pad - 2, ly)
    })

    // 2. Draw Currently Drawn Polygon
    if (isDrawing && currentPoints.length > 0) {
      ctx.beginPath()
      currentPoints.forEach(([xNorm, yNorm], idx) => {
        const px = xNorm * width
        const py = yNorm * height
        if (idx === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })

      if (cursorPos) {
        ctx.lineTo(cursorPos.x * width, cursorPos.y * height)
      }

      ctx.strokeStyle = '#38bdf8'
      ctx.lineWidth = 2.5
      ctx.setLineDash([4, 4])
      ctx.stroke()
      ctx.setLineDash([])

      if (currentPoints.length >= 3) {
        ctx.fillStyle = 'rgba(56, 189, 248, 0.2)'
        ctx.fill()
      }

      currentPoints.forEach(([xNorm, yNorm], idx) => {
        const px = xNorm * width
        const py = yNorm * height
        ctx.beginPath()
        ctx.arc(px, py, 5, 0, Math.PI * 2)
        ctx.fillStyle = idx === 0 ? '#10b981' : '#38bdf8'
        ctx.fill()
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 2
        ctx.stroke()
      })
    }
  }, [exclusionZones, isDrawing, currentPoints, cursorPos])

  const stats = result?.stats
  const total = stats?.processed_frames || 1
  const violationRate = stats ? ((stats.violation_frames / total) * 100).toFixed(1) : 0

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Phân tích video</h1>

      {/* File Upload Drop Area */}
      <div
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer ${
          file ? 'border-sky-500 bg-sky-500/5' : 'border-gray-700 hover:border-gray-500'
        }`}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/avi,video/mov,video/mkv"
          className="hidden"
          onChange={(e) => setFile(e.target.files[0])}
        />
        {file ? (
          <div className="flex items-center justify-center gap-3">
            <FileVideo className="w-8 h-8 text-sky-400" />
            <div>
              <span className="text-lg font-medium text-gray-200">{file.name}</span>
              <p className="text-xs text-gray-400 mt-0.5">Nhấp vào đây nếu muốn chọn video khác</p>
            </div>
          </div>
        ) : (
          <div>
            <UploadIcon className="w-12 h-12 text-gray-500 mx-auto mb-3" />
            <p className="text-gray-400">Kéo thả video hoặc click để chọn</p>
            <p className="text-xs text-gray-600 mt-1">MP4, AVI, MOV, MKV (tối đa 200MB)</p>
          </div>
        )}
      </div>

      {/* Video Preview (Hien thi video don gian, an tinh nang khoanh vung theo yeu cau) */}
      {file && previewUrl && HIDE_EXCLUSION_ZONES && (
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-gray-200 flex items-center gap-2">
              <FileVideo className="w-4 h-4 text-sky-400" />
              Xem trước video: <span className="text-sky-300 font-medium">{file.name}</span>
            </span>
          </div>
          <div className="relative rounded-lg overflow-hidden bg-black border border-gray-800 max-h-[400px] flex items-center justify-center">
            <video
              ref={videoRef}
              src={previewUrl}
              className="max-h-[400px] w-full object-contain block"
              controls
              muted
            />
          </div>
        </div>
      )}

      {/* Video Preview & Exclusion Zone Drawer (Shown when file is selected and HIDE_EXCLUSION_ZONES is false) */}
      {file && previewUrl && !HIDE_EXCLUSION_ZONES && (
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-amber-400" />
              <span className="text-sm font-semibold text-gray-200">
                Xem trước & Khoanh vùng Loại trừ (ROI) cho video
              </span>
              <span className="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded-full font-medium">
                {exclusionZones.length} vùng
              </span>
            </div>

            <div className="flex items-center gap-2">
              {!isDrawing ? (
                <button
                  type="button"
                  onClick={handleStartDrawing}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 border border-amber-500/40 text-xs font-semibold cursor-pointer transition-colors"
                >
                  <Plus className="w-3.5 h-3.5" />
                  + Vẽ vùng loại trừ trên video
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleCancelDrawing}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 text-xs font-medium cursor-pointer"
                >
                  <X className="w-3.5 h-3.5" />
                  Hủy vẽ
                </button>
              )}
            </div>
          </div>

          <div className="text-xs text-gray-400 flex items-center gap-1.5 bg-gray-800/60 p-2.5 rounded-lg border border-gray-800">
            <HelpCircle className="w-4 h-4 text-sky-400 flex-shrink-0" />
            <span>
              Tạm dừng video tại khung cảnh bạn muốn, sau đó nhấn <strong>"+ Vẽ vùng loại trừ"</strong> và nhấp chuột lên video để khoanh vùng (hàn xì, bếp gas). Lửa/khói trong vùng này sẽ được bỏ qua khi phân tích.
            </span>
          </div>

          {/* Interactive Video Preview Player with Canvas Overlay */}
          <div className="relative rounded-lg overflow-hidden bg-black border border-gray-800 max-h-[440px] flex items-center justify-center select-none">
            <video
              src={previewUrl}
              className="max-h-[440px] w-full object-contain block"
              controls={!isDrawing}
              muted
            />

            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              onMouseMove={handleCanvasMouseMove}
              onDoubleClick={handleFinishDrawing}
              className={`absolute inset-0 w-full h-full ${
                isDrawing ? 'cursor-crosshair z-10' : 'pointer-events-none z-0'
              }`}
            />

            {/* Drawing Mode Floating Bar */}
            {isDrawing && (
              <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-gray-900/95 backdrop-blur-md border border-amber-500/60 rounded-xl px-4 py-2 flex items-center gap-3 z-20 shadow-2xl">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
                <span className="text-xs text-amber-300 font-medium">
                  Đã chấm {currentPoints.length} điểm
                </span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    disabled={currentPoints.length < 3}
                    onClick={handleFinishDrawing}
                    className={`px-3 py-1 rounded text-xs font-semibold flex items-center gap-1 transition-colors ${
                      currentPoints.length >= 3
                        ? 'bg-amber-500 hover:bg-amber-600 text-black cursor-pointer shadow-md'
                        : 'bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700'
                    }`}
                  >
                    <Check className="w-3.5 h-3.5" />
                    Hoàn tất vùng
                  </button>
                  <button
                    type="button"
                    onClick={handleCancelDrawing}
                    className="px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs font-medium cursor-pointer flex items-center gap-1 border border-gray-700"
                  >
                    <X className="w-3.5 h-3.5" />
                    Hủy
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* List of Drawn Zones for this Video */}
          {exclusionZones.length > 0 && (
            <div className="flex flex-wrap gap-2 pt-1">
              {exclusionZones.map((zone) => {
                const typeInfo = ZONE_TYPE_LABELS[zone.zone_type] || ZONE_TYPE_LABELS.other
                return (
                  <div
                    key={zone.id}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-800/80 border border-amber-500/40 text-xs text-gray-200"
                  >
                    <span>{typeInfo.icon}</span>
                    <span className="font-medium">{zone.name}</span>
                    <span className="text-[11px] text-gray-400">({zone.polygon_points.length} điểm)</span>
                    <button
                      type="button"
                      onClick={() => handleDeleteZone(zone.id)}
                      className="text-gray-400 hover:text-red-400 ml-1 cursor-pointer"
                      title="Xóa vùng"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* Feature Selector Pills */}
      <div className="bg-gray-900/80 border border-gray-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-300">Tính năng áp dụng:</span>
          <span className="text-xs text-gray-500">(Bật/tắt tùy biến cho video này)</span>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => setEnablePpe(!enablePpe)}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium border transition-all cursor-pointer select-none ${
              enablePpe
                ? 'bg-sky-500/20 border-sky-500/60 text-sky-300 shadow-sm'
                : 'bg-gray-800/60 border-gray-700 text-gray-400 opacity-60 hover:opacity-100'
            }`}
          >
            <HardHat className="w-4 h-4" />
            <span>Bảo hộ PPE {enablePpe ? '✓' : ''}</span>
          </button>

          <button
            type="button"
            onClick={() => setEnableFall(!enableFall)}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium border transition-all cursor-pointer select-none ${
              enableFall
                ? 'bg-rose-500/20 border-rose-500/60 text-rose-300 shadow-sm'
                : 'bg-gray-800/60 border-gray-700 text-gray-400 opacity-60 hover:opacity-100'
            }`}
          >
            <AlertTriangle className="w-4 h-4" />
            <span>Theo dõi Ngã {enableFall ? '✓' : ''}</span>
          </button>

          <button
            type="button"
            onClick={() => setEnableFire(!enableFire)}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium border transition-all cursor-pointer select-none ${
              enableFire
                ? 'bg-amber-500/20 border-amber-500/60 text-amber-300 shadow-sm'
                : 'bg-gray-800/60 border-gray-700 text-gray-400 opacity-60 hover:opacity-100'
            }`}
          >
            <Flame className="w-4 h-4" />
            <span>Cháy & Khói {enableFire ? '✓' : ''}</span>
          </button>
        </div>
      </div>

      {/* Speed & Action Buttons */}
      <div className="flex items-center gap-4">
        <div>
          <label className="text-sm text-gray-400 mr-2">Tốc độ:</label>
          <select
            value={frameSkip}
            onChange={(e) => setFrameSkip(Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
          >
            <option value={1}>Chính xác nhất</option>
            <option value={2}>Cân bằng</option>
            <option value={5}>Nhanh</option>
          </select>
        </div>
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="px-6 py-2 bg-sky-600 hover:bg-sky-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-medium flex items-center gap-2 cursor-pointer shadow-lg shadow-sky-600/20 transition-all"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Đang phân tích...
            </>
          ) : (
            <>
              <CheckCircle className="w-4 h-4" />
              Bắt đầu phân tích
            </>
          )}
        </button>
        {loading && (
          <button
            onClick={handleCancel}
            className="px-6 py-2 bg-red-600 hover:bg-red-700 rounded-lg font-medium flex items-center gap-2 cursor-pointer transition-all"
          >
            <XCircle className="w-4 h-4" />
            Hủy
          </button>
        )}
      </div>

      {cancelled && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4 text-yellow-400 text-center">
          Đã hủy phân tích
        </div>
      )}

      {loading && !progress && (
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <div className="flex items-center justify-between text-sm mb-3">
            <span className="flex items-center gap-2 text-sky-400 font-medium">
              <Loader2 className="w-4 h-4 animate-spin" />
              Đang khởi tạo luồng phân tích video...
            </span>
            <span className="text-gray-400 text-xs">Vui lòng chờ trong giây lát</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2.5 overflow-hidden">
            <div className="bg-sky-500 h-2.5 w-1/2 rounded-full animate-pulse" />
          </div>
        </div>
      )}

      {loading && progress && (
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2 text-sky-400 font-medium">
              <Loader2 className="w-4 h-4 animate-spin" />
              Đang xử lý: {progress.processed} / {progress.total} frames ({progress.percent}%)
            </span>
            <span className="text-gray-400 text-xs">
              Còn lại ~{Math.round(progress.remaining)}s
            </span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-3 overflow-hidden">
            <div
              className="bg-sky-500 h-3 rounded-full transition-all duration-300"
              style={{ width: `${progress.percent}%` }}
            />
          </div>
        </div>
      )}

      {/* Results Section */}
      {result && (
        <div className="space-y-6 animate-in fade-in duration-300">
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h2 className="text-lg font-bold mb-4">Kết quả phân tích</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-gray-800/50 p-4 rounded-lg">
                <div className="text-xs text-gray-400">Tổng số frame</div>
                <div className="text-2xl font-bold mt-1">{stats?.processed_frames}</div>
              </div>
              <div className="bg-gray-800/50 p-4 rounded-lg">
                <div className="text-xs text-gray-400">Tỷ lệ vi phạm PPE</div>
                <div className="text-2xl font-bold mt-1 text-amber-400">{violationRate}%</div>
              </div>
              <div className="bg-gray-800/50 p-4 rounded-lg">
                <div className="text-xs text-gray-400">Phát hiện ngã</div>
                <div className="text-2xl font-bold mt-1 text-rose-400">{stats?.fall_count || 0}</div>
              </div>
              <div className="bg-gray-800/50 p-4 rounded-lg">
                <div className="text-xs text-gray-400">Sự cố Cháy / Khói</div>
                <div className="text-2xl font-bold mt-1 text-red-400">
                  {(stats?.fire_count || 0) + (stats?.smoke_count || 0)}
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-300">Video đã xử lý (kèm nhãn AI & ROI):</h3>
                <a
                  href={result.video_url || (result.output_video ? `/api/detect/output/${result.output_video}` : '#')}
                  download="result_analyzed.mp4"
                  className="text-xs text-sky-400 hover:text-sky-300 flex items-center gap-1 font-medium"
                >
                  Tải video về máy ↓
                </a>
              </div>
              <div className="rounded-xl overflow-hidden border border-gray-800 bg-black shadow-lg">
                <video
                  src={result.video_url || (result.output_video ? `/api/detect/output/${result.output_video}` : '')}
                  controls
                  autoPlay
                  className="w-full max-h-[500px]"
                />
              </div>
            </div>

            {enablePpe && stats && (
              <div className="mt-6 pt-4 border-t border-gray-800">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">Chi tiết tuân thủ bảo hộ lao động (PPE):</h3>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {[
                    { key: 'helmet', name: 'Mũ bảo hộ' },
                    { key: 'vest', name: 'Áo phản quang' },
                    { key: 'mask', name: 'Khẩu trang' },
                  ].map(({ key, name }) => (
                    <div key={key} className="bg-gray-800/50 border border-gray-800 rounded-lg p-3">
                      <div className="text-xs text-gray-400 font-medium mb-1.5">{name}</div>
                      <div className="flex justify-between text-xs font-semibold">
                        <span className="text-emerald-400">Đúng: {stats[`${key}_ok`] || 0}</span>
                        <span className="text-rose-400">Vi phạm: {stats[`${key}_violation`] || 0}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Save Zone Modal Dialog */}
      {!HIDE_EXCLUSION_ZONES && showZoneModal && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Layers className="w-5 h-5 text-amber-400" />
                Lưu Vùng Loại Trừ Cho Video
              </h3>
              <button
                type="button"
                onClick={() => setShowZoneModal(false)}
                className="text-gray-400 hover:text-white p-1 cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="text-xs text-gray-400">
              Đa giác đã tạo có <strong>{currentPoints.length} điểm</strong> góc. Vùng này sẽ được áp dụng khi phân tích video.
            </div>

            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1">
                  Tên vùng loại trừ *
                </label>
                <input
                  type="text"
                  value={newZoneName}
                  onChange={(e) => setNewZoneName(e.target.value)}
                  placeholder="Ví dụ: Khu vực hàn xì xưởng 1"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-amber-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1">
                  Loại khu vực
                </label>
                <select
                  value={newZoneType}
                  onChange={(e) => setNewZoneType(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-amber-500"
                >
                  <option value="welding">⚡ Hàn xì / Cắt kim loại (Có tia lửa thường xuyên)</option>
                  <option value="kitchen">🍳 Bếp ăn / Nấu nướng (Có khói thức ăn)</option>
                  <option value="boiler">🏭 Lò hơi / Ống khói kỹ thuật</option>
                  <option value="smoking">🚬 Khu vực hút thuốc</option>
                  <option value="other">🛡️ Khu vực đặc thù khác</option>
                </select>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2.5 pt-2">
              <button
                type="button"
                onClick={() => setShowZoneModal(false)}
                className="px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs font-medium cursor-pointer"
              >
                Hủy
              </button>
              <button
                type="button"
                onClick={handleSaveZone}
                className="px-5 py-2 rounded-lg bg-amber-500 hover:bg-amber-600 text-black text-xs font-bold cursor-pointer transition-colors shadow-lg shadow-amber-500/20"
              >
                Lưu vùng này
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
