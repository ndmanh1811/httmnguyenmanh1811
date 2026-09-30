import { useState, useEffect, useRef, useCallback } from 'react'
import { io } from 'socket.io-client'
import {
  Volume2,
  VolumeX,
  Camera,
  CameraOff,
  X,
  ShieldCheck,
} from 'lucide-react'
import api from '../api'
import soundEngine from '../utils/soundEngine'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

const VIOLATION_STYLES = {
  no_helmet: { label: 'Thiếu mũ bảo hiểm', border: 'border-l-amber-400', dot: 'bg-amber-400' },
  no_vest: { label: 'Thiếu áo bảo hộ', border: 'border-l-orange-400', dot: 'bg-orange-400' },
  no_mask: { label: 'Thiếu khẩu trang', border: 'border-l-purple-400', dot: 'bg-purple-400' },
  fall_detected: { label: 'Phát hiện ngã', border: 'border-l-rose-500', dot: 'bg-rose-500 animate-pulse' },
  fire_detected: { label: 'Phát hiện cháy', border: 'border-l-red-500', dot: 'bg-red-500 animate-ping' },
  smoke_detected: { label: 'Phát hiện khói', border: 'border-l-amber-500', dot: 'bg-amber-500 animate-pulse' },
}

const HIDE_EXCLUSION_ZONES = true

export default function Monitor() {
  const [alerts, setAlerts] = useState([])
  const [soundEnabled, setSoundEnabled] = useState(true)
  const [cameraOn, setCameraOn] = useState(false)
  const [frameSkip, setFrameSkip] = useState(1)
  const [enablePpe, setEnablePpe] = useState(true)
  const [enableFall, setEnableFall] = useState(true)
  const [enableFire, setEnableFire] = useState(true)
  const [liveStats, setLiveStats] = useState({ total: 0, helmet: 0, vest: 0, mask: 0, fall: 0, fire: 0, smoke: 0 })

  const [cameras, setCameras] = useState([])
  const [selectedCameraId, setSelectedCameraId] = useState(0)

  // Exclusion Zone (ROI) State
  const [exclusionZones, setExclusionZones] = useState([])
  const [showZonePanel, setShowZonePanel] = useState(false)
  const [isDrawing, setIsDrawing] = useState(false)
  const [currentPoints, setCurrentPoints] = useState([])
  const [cursorPos, setCursorPos] = useState(null)
  const [showZoneModal, setShowZoneModal] = useState(false)
  const [newZoneName, setNewZoneName] = useState('')
  const [newZoneType, setNewZoneType] = useState('welding')
  const [hoveredZoneId, setHoveredZoneId] = useState(null)

  const camKeyRef = useRef(null)
  const canvasRef = useRef(null)
  const videoContainerRef = useRef(null)

  const loadExclusionZones = useCallback(async (camId = selectedCameraId) => {
    try {
      const res = await api.get(`/cameras/${camId}/exclusion_zones`)
      setExclusionZones(res.data)
    } catch (e) {
      console.error('Failed to load exclusion zones:', e)
    }
  }, [selectedCameraId])

  useEffect(() => {
    api.get('/settings').then(r => {
      if (r.data.enable_ppe) setEnablePpe(String(r.data.enable_ppe.value).toLowerCase() === 'true')
      if (r.data.enable_fall) setEnableFall(String(r.data.enable_fall.value).toLowerCase() === 'true')
      if (r.data.enable_fire) setEnableFire(String(r.data.enable_fire.value).toLowerCase() === 'true')
    }).catch(() => {})

    api.get('/cameras').then(r => setCameras(r.data || [])).catch(() => {})

    loadExclusionZones()
  }, [loadExclusionZones])

  useEffect(() => {
    const socket = io({ transports: ['polling'] })

    const handleAlert = (data) => {
      setAlerts((prev) => [data, ...prev].slice(0, 25))
      setLiveStats((prev) => ({
        ...prev,
        total: prev.total + 1,
        helmet: data.type === 'no_helmet' ? prev.helmet + 1 : prev.helmet,
        vest: data.type === 'no_vest' ? prev.vest + 1 : prev.vest,
        mask: data.type === 'no_mask' ? prev.mask + 1 : prev.mask,
        fall: data.type === 'fall_detected' ? prev.fall + 1 : prev.fall,
        fire: data.type === 'fire_detected' ? prev.fire + 1 : prev.fire,
        smoke: data.type === 'smoke_detected' ? prev.smoke + 1 : prev.smoke,
      }))
      if (soundEnabled) {
        soundEngine.playAlert(data.type)
      }
    }

    socket.on('violation_alert', handleAlert)
    socket.on('fall_alert', handleAlert)
    socket.on('fire_alert', handleAlert)

    return () => socket.disconnect()
  }, [soundEnabled])

  const stopCamera = useCallback(async () => {
    if (camKeyRef.current) {
      try {
        await api.post('/webcam_stop', { cam_key: camKeyRef.current })
      } catch (e) {}
      camKeyRef.current = null
    }
  }, [])

  const toggleCamera = useCallback(async () => {
    if (cameraOn) {
      await stopCamera()
      setCameraOn(false)
    } else {
      const cam = cameras.find((c) => c.id === selectedCameraId)
      const src = cam ? cam.source : '0'
      camKeyRef.current = `${src}_${frameSkip}`
      setCameraOn(true)
    }
  }, [cameraOn, frameSkip, stopCamera, cameras, selectedCameraId])

  const changeFrameSkip = useCallback(async (newSkip) => {
    setFrameSkip(newSkip)
    if (cameraOn) {
      await stopCamera()
      const cam = cameras.find((c) => c.id === selectedCameraId)
      const src = cam ? cam.source : '0'
      camKeyRef.current = `${src}_${newSkip}`
    }
  }, [cameraOn, stopCamera, cameras, selectedCameraId])

  const handleToggleModule = async (mod) => {
    let newPpe = enablePpe
    let newFall = enableFall
    let newFire = enableFire
    if (mod === 'ppe') { newPpe = !enablePpe; setEnablePpe(newPpe) }
    if (mod === 'fall') { newFall = !enableFall; setEnableFall(newFall) }
    if (mod === 'fire') { newFire = !enableFire; setEnableFire(newFire) }

    try {
      await api.post('/webcam_toggles', {
        enable_ppe: newPpe,
        enable_fall: newFall,
        enable_fire: newFire,
      })
    } catch (e) {}
  }

  // --- Exclusion Zone Canvas Drawing Logic ---
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

  const handleCanvasDoubleClick = () => {
    if (isDrawing && currentPoints.length >= 3) {
      handleFinishDrawing()
    }
  }

  const handleFinishDrawing = () => {
    if (currentPoints.length < 3) {
      alert('Vui lòng chọn ít nhất 3 điểm để tạo thành một đa giác khép kín!')
      return
    }
    setNewZoneName(`Vùng loại trừ #${exclusionZones.length + 1}`)
    setShowZoneModal(true)
  }

  const handleSaveZone = async () => {
    if (!newZoneName.trim()) {
      alert('Vui lòng nhập tên cho vùng loại trừ!')
      return
    }

    try {
      await api.post(`/cameras/${selectedCameraId}/exclusion_zones`, {
        name: newZoneName.trim(),
        zone_type: newZoneType,
        polygon: currentPoints,
      })
      setShowZoneModal(false)
      setIsDrawing(false)
      setCurrentPoints([])
      setCursorPos(null)
      await loadExclusionZones()
    } catch (e) {
      alert('Lỗi lưu vùng loại trừ: ' + (e.response?.data?.error || e.message))
    }
  }

  // Draw overlay canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const container = videoContainerRef.current
    if (!container) return

    canvas.width = container.clientWidth
    canvas.height = container.clientHeight
    const width = canvas.width
    const height = canvas.height
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, width, height)

    if (!HIDE_EXCLUSION_ZONES) {
      exclusionZones.forEach((zone) => {
        const points = zone.polygon || []
        if (points.length < 3) return
        const isActive = zone.is_active
        const isHovered = hoveredZoneId === zone.id

        ctx.beginPath()
        points.forEach(([xNorm, yNorm], idx) => {
          const px = xNorm * width
          const py = yNorm * height
          if (idx === 0) ctx.moveTo(px, py)
          else ctx.lineTo(px, py)
        })
        ctx.closePath()

        ctx.fillStyle = isActive
          ? isHovered ? 'rgba(245, 158, 11, 0.35)' : 'rgba(245, 158, 11, 0.18)'
          : 'rgba(107, 114, 128, 0.1)'
        ctx.strokeStyle = isActive ? (isHovered ? '#fbbf24' : '#f59e0b') : '#6b7280'
        ctx.lineWidth = isHovered ? 2.5 : 1.5
        ctx.fill()
        ctx.stroke()
      })
    }

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

      ctx.strokeStyle = '#60a5fa'
      ctx.lineWidth = 2
      ctx.setLineDash([4, 4])
      ctx.stroke()
      ctx.setLineDash([])

      currentPoints.forEach(([xNorm, yNorm], idx) => {
        const px = xNorm * width
        const py = yNorm * height
        ctx.beginPath()
        ctx.arc(px, py, 4, 0, Math.PI * 2)
        ctx.fillStyle = idx === 0 ? '#34d399' : '#60a5fa'
        ctx.fill()
      })
    }
  }, [exclusionZones, isDrawing, currentPoints, cursorPos, hoveredZoneId])

  const streamUrl = cameraOn
    ? `/api/webcam_stream?camera_id=${selectedCameraId}&frame_skip=${frameSkip}&enable_ppe=${enablePpe}&enable_fall=${enableFall}&enable_fire=${enableFire}`
    : null

  const ppeCount = liveStats.helmet + liveStats.vest + liveStats.mask

  return (
    <div className="space-y-4">
      {/* Top Header Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-foreground">Giám sát trực tiếp</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Luồng camera thời gian thực với AI đa nhiệm (PPE, Ngã & Khói Lửa YOLO26s)
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* AI Feature Toggle Pills */}
          <div className="flex items-center bg-card border border-border rounded-lg p-0.5 gap-0.5">
            <button
              onClick={() => handleToggleModule('ppe')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-colors cursor-pointer select-none ${
                enablePpe
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${enablePpe ? 'bg-blue-400' : 'bg-muted-foreground'}`} />
              PPE
            </button>

            <button
              onClick={() => handleToggleModule('fall')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-colors cursor-pointer select-none ${
                enableFall
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${enableFall ? 'bg-rose-400' : 'bg-muted-foreground'}`} />
              Ngã
            </button>

            <button
              onClick={() => handleToggleModule('fire')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-colors cursor-pointer select-none ${
                enableFire
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${enableFire ? 'bg-red-400' : 'bg-muted-foreground'}`} />
              Cháy/Khói
            </button>
          </div>

          {/* Sound Toggle */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSoundEnabled(!soundEnabled)}
            className="h-8 text-xs cursor-pointer"
          >
            {soundEnabled ? (
              <Volume2 className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <VolumeX className="w-3.5 h-3.5 text-muted-foreground" />
            )}
          </Button>
        </div>
      </div>

      {/* Main Grid: Stream (Left) + Realtime Alerts (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Left Column: Stream Container & Controls */}
        <div className="lg:col-span-3 space-y-3">
          <div
            ref={videoContainerRef}
            className="relative bg-card rounded-xl border border-border overflow-hidden select-none aspect-video flex items-center justify-center"
          >
            {cameraOn ? (
              <img
                key={streamUrl}
                src={streamUrl}
                alt="Webcam stream"
                className="w-full h-full object-contain block"
                onError={(e) => {
                  e.target.alt = 'Không thể kết nối webcam.'
                }}
              />
            ) : (
              <div className="flex flex-col items-center justify-center text-muted-foreground">
                <CameraOff className="w-10 h-10 mb-2 opacity-30" />
                <p className="text-xs font-medium">Camera đang tắt</p>
                <p className="text-[11px] text-muted-foreground/60 mt-0.5">
                  Nhấn nút &quot;Bật Camera&quot; bên dưới để bắt đầu nhận diện
                </p>
              </div>
            )}

            {/* Drawing Canvas Overlay */}
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              onMouseMove={handleCanvasMouseMove}
              onDoubleClick={handleCanvasDoubleClick}
              className={`absolute inset-0 w-full h-full ${
                isDrawing ? 'cursor-crosshair pointer-events-auto' : 'pointer-events-none'
              }`}
            />
          </div>

          {/* Camera Controls Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-card rounded-xl border border-border">
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                onClick={toggleCamera}
                className={`h-8 text-xs font-medium cursor-pointer ${
                  cameraOn
                    ? 'bg-red-500/20 text-red-400 border border-red-500/40 hover:bg-red-500/30'
                    : 'bg-primary text-primary-foreground hover:bg-primary/90'
                }`}
              >
                {cameraOn ? (
                  <>
                    <span className="w-2 h-2 rounded-full bg-red-400 animate-ping mr-1.5" />
                    Tắt Camera
                  </>
                ) : (
                  <>
                    <Camera className="w-3.5 h-3.5 mr-1.5" />
                    Bật Camera
                  </>
                )}
              </Button>

              {/* Camera Source Selector */}
              {cameras.length > 1 && (
                <select
                  value={selectedCameraId}
                  onChange={(e) => {
                    setSelectedCameraId(Number(e.target.value))
                    if (cameraOn) {
                      stopCamera()
                      setCameraOn(false)
                    }
                  }}
                  className="bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground focus:outline-none"
                >
                  {cameras.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              )}
            </div>

            {/* Frame Skip Selector */}
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <span className="text-[11px] mr-1">Tốc độ:</span>
              {[1, 2, 5].map((skip) => (
                <button
                  key={skip}
                  onClick={() => changeFrameSkip(skip)}
                  className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors cursor-pointer ${
                    frameSkip === skip
                      ? 'bg-accent text-accent-foreground font-semibold'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {skip === 1 ? 'Chuẩn' : `${skip}x`}
                </button>
              ))}
            </div>
          </div>

          {/* Live Stats Summary Bar */}
          <div className="grid grid-cols-4 gap-2">
            <div className="bg-card border border-border rounded-lg p-3 text-center">
              <div className="text-xl font-bold text-foreground tabular-nums">{liveStats.total}</div>
              <div className="text-[11px] text-muted-foreground mt-0.5">Tổng sự cố</div>
            </div>
            <div className="bg-card border border-border rounded-lg p-3 text-center">
              <div className="text-xl font-bold text-amber-400 tabular-nums">{ppeCount}</div>
              <div className="text-[11px] text-muted-foreground mt-0.5">Vi phạm PPE</div>
            </div>
            <div className="bg-card border border-border rounded-lg p-3 text-center">
              <div className="text-xl font-bold text-rose-400 tabular-nums">{liveStats.fall}</div>
              <div className="text-[11px] text-muted-foreground mt-0.5">Phát hiện ngã</div>
            </div>
            <div className="bg-card border border-border rounded-lg p-3 text-center">
              <div className="text-xl font-bold text-red-400 tabular-nums">
                {liveStats.fire + liveStats.smoke}
              </div>
              <div className="text-[11px] text-muted-foreground mt-0.5">Cháy & Khói</div>
            </div>
          </div>
        </div>

        {/* Right Column: Real-time Alert Feed */}
        <div className="bg-card rounded-xl border border-border p-4 flex flex-col h-[560px]">
          <div className="flex items-center justify-between pb-3 border-b border-border/50 flex-shrink-0">
            <span className="text-xs font-semibold text-foreground uppercase tracking-wider">
              Cảnh báo trực tiếp
            </span>
            {alerts.length > 0 && (
              <Badge variant="outline" className="text-[10px] h-5 font-mono">
                {alerts.length}
              </Badge>
            )}
          </div>

          {alerts.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground text-xs py-8">
              <ShieldCheck className="w-8 h-8 mb-2 opacity-30 text-emerald-400" />
              <p>Chưa phát hiện vi phạm</p>
              <p className="text-[10px] text-muted-foreground/60 mt-0.5">Khu vực làm việc an toàn</p>
            </div>
          ) : (
            <div className="space-y-2 overflow-y-auto flex-1 pr-1 pt-3">
              {alerts.map((a, i) => {
                const style = VIOLATION_STYLES[a.type] || { label: a.type, border: 'border-l-zinc-500', dot: 'bg-zinc-500' }
                const time = new Date(a.timestamp)
                const isBreakout = a.is_breakout || false
                return (
                  <div
                    key={i}
                    className={`pl-3 py-2 pr-2 rounded-r-md border-l-2 bg-muted/30 transition-all ${
                      isBreakout
                        ? 'border-l-red-500 bg-red-950/20'
                        : style.border
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-foreground">
                        {a.type_label || style.label}
                      </span>
                      <span className="text-[10px] font-mono text-muted-foreground">
                        {time.toLocaleTimeString('vi-VN')}
                      </span>
                    </div>
                    {isBreakout && (
                      <div className="text-[10px] text-red-400 font-medium mt-0.5">
                        Cháy lan ngoài vùng loại trừ
                      </div>
                    )}
                    <div className="text-[10px] text-muted-foreground mt-1">
                      Độ tin cậy: <strong className="text-foreground">{a.confidence}%</strong>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Save Exclusion Zone Modal Dialog */}
      {showZoneModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-card border border-border rounded-xl max-w-sm w-full p-5 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-foreground">
                Lưu vùng loại trừ mới
              </h3>
              <button
                onClick={() => setShowZoneModal(false)}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">
                  Tên vùng loại trừ
                </label>
                <input
                  type="text"
                  value={newZoneName}
                  onChange={(e) => setNewZoneName(e.target.value)}
                  placeholder="Ví dụ: Khu vực hàn xì xưởng B"
                  className="w-full bg-muted border border-border rounded-lg px-3 py-1.5 text-xs text-foreground focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">
                  Loại khu vực
                </label>
                <select
                  value={newZoneType}
                  onChange={(e) => setNewZoneType(e.target.value)}
                  className="w-full bg-muted border border-border rounded-lg px-3 py-1.5 text-xs text-foreground focus:outline-none"
                >
                  <option value="welding">Hàn xì / Cắt kim loại</option>
                  <option value="kitchen">Bếp ăn / Nấu nướng</option>
                  <option value="boiler">Lò hơi / Ống khói</option>
                  <option value="smoking">Khu vực hút thuốc</option>
                  <option value="other">Khu vực đặc thù khác</option>
                </select>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowZoneModal(false)}
                className="text-xs h-8"
              >
                Hủy
              </Button>
              <Button
                size="sm"
                onClick={handleSaveZone}
                className="text-xs h-8"
              >
                Lưu vùng
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

