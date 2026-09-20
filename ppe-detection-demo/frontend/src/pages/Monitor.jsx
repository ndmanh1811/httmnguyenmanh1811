import { useState, useEffect, useRef, useCallback } from 'react'
import { io } from 'socket.io-client'
import {
  HardHat,
  Shirt,
  ShieldAlert,
  AlertTriangle,
  Flame,
  Volume2,
  VolumeX,
  Camera,
  CameraOff,
  Settings2,
  Layers,
  Plus,
  Trash2,
  Check,
  X,
  Eye,
  EyeOff,
  ShieldCheck,
  HelpCircle,
} from 'lucide-react'
import api from '../api'
import soundEngine from '../utils/soundEngine'

const VIOLATION_STYLES = {
  no_helmet: { label: 'Thiếu mũ bảo hiểm', icon: HardHat, bg: 'bg-red-500/20', border: 'border-red-500/50', text: 'text-red-400', dot: 'bg-red-500' },
  no_vest: { label: 'Thiếu áo bảo hộ', icon: Shirt, bg: 'bg-orange-500/20', border: 'border-orange-500/50', text: 'text-orange-400', dot: 'bg-orange-500' },
  no_mask: { label: 'Thiếu khẩu trang', icon: ShieldAlert, bg: 'bg-yellow-500/20', border: 'border-yellow-500/50', text: 'text-yellow-400', dot: 'bg-yellow-500' },
  fall_detected: { label: 'CẢNH BÁO: PHÁT HIỆN NGÃ!', icon: AlertTriangle, bg: 'bg-rose-600/30', border: 'border-rose-500', text: 'text-rose-400 font-bold', dot: 'bg-rose-500 animate-ping' },
  fire_detected: { label: 'NGUY HIỂM: PHÁT HIỆN CHÁY!', icon: Flame, bg: 'bg-red-600/40', border: 'border-red-500', text: 'text-red-400 font-bold', dot: 'bg-red-500 animate-ping' },
  smoke_detected: { label: 'CẢNH BÁO: PHÁT HIỆN KHÓI!', icon: Flame, bg: 'bg-amber-600/30', border: 'border-amber-500', text: 'text-amber-400 font-bold', dot: 'bg-amber-500 animate-pulse' },
}

const ZONE_TYPE_LABELS = {
  welding: { label: 'Hàn xì / Cắt kim loại', icon: '⚡', color: 'text-amber-400' },
  kitchen: { label: 'Bếp / Nấu nướng', icon: '🍳', color: 'text-orange-400' },
  boiler: { label: 'Lò hơi / Ống khói', icon: '🏭', color: 'text-sky-400' },
  smoking: { label: 'Khu hút thuốc', icon: '🚬', color: 'text-gray-400' },
  other: { label: 'Khu vực loại trừ khác', icon: '🛡️', color: 'text-emerald-400' },
}

// Tam thoi an tinh nang khoanh vung theo yeu cau, tap trung vao phat hien khoi lua
const HIDE_EXCLUSION_ZONES = true

export default function Monitor() {
  const [alerts, setAlerts] = useState([])
  const [soundEnabled, setSoundEnabled] = useState(true)
  const [cameraOn, setCameraOn] = useState(false)
  const [frameSkip, setFrameSkip] = useState(2)
  const [enablePpe, setEnablePpe] = useState(true)
  const [enableFall, setEnableFall] = useState(true)
  const [enableFire, setEnableFire] = useState(true)
  const [liveStats, setLiveStats] = useState({ total: 0, helmet: 0, vest: 0, mask: 0, fall: 0, fire: 0, smoke: 0 })
  
  // Camera selection
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

  const audioRef = useRef(null)
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
    // Load default settings, cameras & exclusion zones
    api.get('/settings').then(r => {
      if (r.data.enable_ppe) setEnablePpe(String(r.data.enable_ppe.value).toLowerCase() === 'true')
      if (r.data.enable_fall) setEnableFall(String(r.data.enable_fall.value).toLowerCase() === 'true')
      if (r.data.enable_fire) setEnableFire(String(r.data.enable_fire.value).toLowerCase() === 'true')
    }).catch(() => {})

    api.get('/cameras').then(r => setCameras(r.data)).catch(() => {})

    loadExclusionZones()
  }, [loadExclusionZones])

  useEffect(() => {
    const socket = io({ transports: ['polling'] })

    const handleAlert = (data) => {
      setAlerts((prev) => [data, ...prev].slice(0, 20))
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

  const handleStartDrawing = () => {
    setIsDrawing(true)
    setCurrentPoints([])
    setCursorPos(null)
    setShowZonePanel(true)
  }

  const handleCancelDrawing = () => {
    setIsDrawing(false)
    setCurrentPoints([])
    setCursorPos(null)
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
      alert('Vui lòng nhập tên vùng')
      return
    }
    try {
      await api.post(`/cameras/${selectedCameraId}/exclusion_zones`, {
        name: newZoneName.trim(),
        zone_type: newZoneType,
        polygon_points: currentPoints,
        is_active: true,
      })
      await loadExclusionZones(selectedCameraId)
      setShowZoneModal(false)
      setIsDrawing(false)
      setCurrentPoints([])
      setCursorPos(null)
    } catch (e) {
      alert('Lỗi lưu vùng loại trừ: ' + (e.response?.data?.error || e.message))
    }
  }

  const handleToggleZoneActive = async (zone) => {
    try {
      await api.put(`/exclusion_zones/${zone.id}`, {
        is_active: !zone.is_active,
      })
      await loadExclusionZones()
    } catch (e) {
      alert('Lỗi cập nhật trạng thái vùng')
    }
  }

  const handleDeleteZone = async (zoneId) => {
    if (!confirm('Bạn có chắc chắn muốn xóa vùng loại trừ này?')) return
    try {
      await api.delete(`/exclusion_zones/${zoneId}`)
      await loadExclusionZones()
    } catch (e) {
      alert('Lỗi xóa vùng loại trừ')
    }
  }

  // Render Polygons onto Canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Sync canvas internal resolution with CSS display dimensions
    const width = canvas.offsetWidth || 640
    const height = canvas.offsetHeight || 480
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }

    ctx.clearRect(0, 0, width, height)

    // 1. Draw Saved Exclusion Zones (an khi HIDE_EXCLUSION_ZONES = true)
    if (!HIDE_EXCLUSION_ZONES) {
      exclusionZones.forEach((zone) => {
      const points = zone.polygon_points || []
      if (points.length < 3) return

      const isHovered = hoveredZoneId === zone.id
      const isActive = zone.is_active

      ctx.beginPath()
      points.forEach(([xNorm, yNorm], idx) => {
        const px = xNorm * width
        const py = yNorm * height
        if (idx === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.closePath()

      // Fill style
      if (isActive) {
        ctx.fillStyle = isHovered ? 'rgba(245, 158, 11, 0.35)' : 'rgba(245, 158, 11, 0.18)'
        ctx.strokeStyle = isHovered ? '#fbbf24' : '#f59e0b'
        ctx.lineWidth = isHovered ? 3 : 2
        ctx.setLineDash([])
      } else {
        ctx.fillStyle = 'rgba(107, 114, 128, 0.1)'
        ctx.strokeStyle = '#6b7280'
        ctx.lineWidth = 1.5
        ctx.setLineDash([5, 5])
      }

      ctx.fill()
      ctx.stroke()
      ctx.setLineDash([])

      // Draw Zone Label Badge
      const [firstX, firstY] = points[0]
      const lx = firstX * width
      const ly = Math.max(22, firstY * height - 8)

      const typeIcon = ZONE_TYPE_LABELS[zone.zone_type]?.icon || '🛡️'
      const labelText = `${typeIcon} ${zone.name} ${isActive ? '' : '(Tắt)'}`

      ctx.font = 'bold 11px sans-serif'
      const textMetrics = ctx.measureText(labelText)
      const textWidth = textMetrics.width
      const pad = 6

      ctx.fillStyle = isActive ? 'rgba(30, 41, 59, 0.9)' : 'rgba(17, 24, 39, 0.8)'
      ctx.strokeStyle = isActive ? '#f59e0b' : '#6b7280'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.roundRect ? ctx.roundRect(lx - 2, ly - 14, textWidth + pad * 2, 20, 4) : ctx.rect(lx - 2, ly - 14, textWidth + pad * 2, 20)
      ctx.fill()
      ctx.stroke()

      ctx.fillStyle = isActive ? '#fef3c7' : '#9ca3af'
      ctx.fillText(labelText, lx + pad - 2, ly)
    })
    }

    // 2. Draw Currently Drawn Polygon (if in drawing mode)
    if (isDrawing && currentPoints.length > 0) {
      ctx.beginPath()
      currentPoints.forEach(([xNorm, yNorm], idx) => {
        const px = xNorm * width
        const py = yNorm * height
        if (idx === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })

      // Draw rubberband line to cursor
      if (cursorPos) {
        ctx.lineTo(cursorPos.x * width, cursorPos.y * height)
      }

      ctx.strokeStyle = '#38bdf8'
      ctx.lineWidth = 2.5
      ctx.setLineDash([4, 4])
      ctx.stroke()
      ctx.setLineDash([])

      // Fill current path
      if (currentPoints.length >= 3) {
        ctx.fillStyle = 'rgba(56, 189, 248, 0.2)'
        ctx.fill()
      }

      // Draw point handles
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
  }, [exclusionZones, isDrawing, currentPoints, cursorPos, hoveredZoneId])

  const streamUrl = cameraOn
    ? `/api/webcam_stream?camera_id=${selectedCameraId}&frame_skip=${frameSkip}&enable_ppe=${enablePpe}&enable_fall=${enableFall}&enable_fire=${enableFire}`
    : null

  const activeZoneCount = exclusionZones.filter((z) => z.is_active).length

  return (
    <div className="space-y-6">
      {/* Top Header & AI Module Toggles */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Giám sát trực tiếp</h1>
          <p className="text-xs text-gray-400 mt-0.5">Theo dõi luồng camera thời gian thực với AI đa nhiệm & Vùng loại trừ ROI</p>
        </div>
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Quick AI feature toggles */}
          <div className="flex items-center bg-gray-900 border border-gray-800 rounded-lg p-1 gap-1">
            <button
              onClick={() => handleToggleModule('ppe')}
              title="Bật/Tắt module nhận diện bảo hộ PPE"
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all cursor-pointer select-none ${
                enablePpe
                  ? 'bg-sky-500/20 text-sky-400 border border-sky-500/40'
                  : 'text-gray-500 hover:text-gray-400 opacity-60'
              }`}
            >
              <HardHat className="w-3.5 h-3.5" />
              <span>PPE {enablePpe ? 'ON' : 'OFF'}</span>
            </button>

            <button
              onClick={() => handleToggleModule('fall')}
              title="Bật/Tắt module theo dõi ngã"
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all cursor-pointer select-none ${
                enableFall
                  ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                  : 'text-gray-500 hover:text-gray-400 opacity-60'
              }`}
            >
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Ngã {enableFall ? 'ON' : 'OFF'}</span>
            </button>

            <button
              onClick={() => handleToggleModule('fire')}
              title="Bật/Tắt module phát hiện cháy & khói"
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all cursor-pointer select-none ${
                enableFire
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                  : 'text-gray-500 hover:text-gray-400 opacity-60'
              }`}
            >
              <Flame className="w-3.5 h-3.5" />
              <span>Cháy {enableFire ? 'ON' : 'OFF'}</span>
            </button>
          </div>

          {/* Exclusion Zones Toggle Button (Tam thoi an theo yeu cau) */}
          {/*
          <button
            onClick={() => setShowZonePanel(!showZonePanel)}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer border ${
              showZonePanel || isDrawing
                ? 'bg-amber-500/20 text-amber-400 border-amber-500/50 shadow-sm shadow-amber-500/20'
                : 'bg-gray-800 hover:bg-gray-700 text-gray-300 border-gray-700'
            }`}
          >
            <Layers className="w-4 h-4 text-amber-400" />
            <span>Vùng loại trừ ({activeZoneCount})</span>
          </button>
          */}

          {/* Audio toggle button */}
          <button
            onClick={() => setSoundEnabled(!soundEnabled)}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-xs font-medium cursor-pointer"
          >
            {soundEnabled ? (
              <Volume2 className="w-4 h-4 text-green-400" />
            ) : (
              <VolumeX className="w-4 h-4 text-red-400" />
            )}
            {soundEnabled ? 'Chuông BẬT' : 'Chuông TẮT'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3 space-y-4">
          {/* Main Video Stream Container with Interactive Canvas */}
          <div
            ref={videoContainerRef}
            className="relative bg-gray-900 rounded-xl border border-gray-800 overflow-hidden select-none"
          >
            {cameraOn ? (
              <img
                key={streamUrl}
                src={streamUrl}
                alt="Webcam stream"
                className="w-full block"
                onError={(e) => {
                  e.target.alt = 'Không thể kết nối webcam.'
                }}
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-[480px] text-gray-500">
                <CameraOff className="w-16 h-16 mb-4 opacity-30" />
                <p>Camera đang tắt</p>
                <p className="text-sm mt-1">Nhấn nút bên dưới để bật webcam</p>
              </div>
            )}

            {/* Drawing Canvas Overlay */}
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              onMouseMove={handleCanvasMouseMove}
              onDoubleClick={handleCanvasDoubleClick}
              className={`absolute inset-0 w-full h-full ${
                isDrawing ? 'cursor-crosshair z-10' : 'pointer-events-none z-0'
              }`}
            />

            {/* Active Drawing Floating HUD */}
            {isDrawing && (
              <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-gray-900/95 backdrop-blur-md border border-amber-500/60 rounded-xl px-4 py-2.5 flex items-center gap-4 z-20 shadow-2xl">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-ping" />
                  <div className="text-left">
                    <p className="text-xs font-semibold text-amber-300">
                      Chế độ vẽ vùng: {currentPoints.length} điểm đã chấm
                    </p>
                    <p className="text-[10px] text-gray-400">
                      Nhấp chuột để thêm điểm góc (tối thiểu 3 điểm). Nhấp đúp để chốt.
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    disabled={currentPoints.length < 3}
                    onClick={handleFinishDrawing}
                    className={`px-3 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                      currentPoints.length >= 3
                        ? 'bg-amber-500 hover:bg-amber-600 text-black cursor-pointer shadow-md'
                        : 'bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700'
                    }`}
                  >
                    <Check className="w-3.5 h-3.5" />
                    Hoàn tất ({currentPoints.length})
                  </button>
                  <button
                    onClick={handleCancelDrawing}
                    className="px-2.5 py-1.5 rounded-md bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs font-medium cursor-pointer flex items-center gap-1 border border-gray-700"
                  >
                    <X className="w-3.5 h-3.5" />
                    Hủy
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Camera Controls & Settings */}
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <button
                onClick={toggleCamera}
                className={`flex items-center gap-2 px-6 py-2.5 rounded-lg font-medium transition-colors cursor-pointer ${
                  cameraOn
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-green-600 hover:bg-green-700 text-white'
                }`}
              >
                {cameraOn ? (
                  <>
                    <CameraOff className="w-4 h-4" />
                    Tắt camera
                  </>
                ) : (
                  <>
                    <Camera className="w-4 h-4" />
                    Bật camera
                  </>
                )}
              </button>

              {cameras.length > 0 && (
                <div className="flex items-center gap-2 bg-gray-800 rounded-lg px-3.5 py-2 border border-gray-700">
                  <Camera className="w-4 h-4 text-gray-400" />
                  <span className="text-xs text-gray-400">Camera:</span>
                  <select
                    value={selectedCameraId}
                    onChange={async (e) => {
                      const newId = Number(e.target.value)
                      setSelectedCameraId(newId)
                      if (cameraOn) {
                        await stopCamera()
                        setCameraOn(false)
                      }
                      loadExclusionZones(newId)
                    }}
                    className="bg-gray-700 border border-gray-600 rounded px-2 py-0.5 text-xs text-white"
                  >
                    <option value={0}>Webcam mặc định (0)</option>
                    {cameras.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name} ({c.source})
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <div className="flex items-center gap-2 bg-gray-800 rounded-lg px-3.5 py-2 border border-gray-700">
                <Settings2 className="w-4 h-4 text-gray-400" />
                <span className="text-xs text-gray-400">Bỏ qua frame:</span>
                <select
                  value={frameSkip}
                  onChange={(e) => changeFrameSkip(Number(e.target.value))}
                  className="bg-gray-700 border border-gray-600 rounded px-2 py-0.5 text-xs text-white"
                >
                  <option value={1}>Mỗi frame (chính xác cao)</option>
                  <option value={2}>Mỗi 2 frame (cân bằng)</option>
                  <option value={3}>Mỗi 3 frame (nhanh)</option>
                  <option value={5}>Mỗi 5 frame (rất nhanh)</option>
                </select>
              </div>
            </div>

            {!HIDE_EXCLUSION_ZONES && (
              <div className="flex items-center gap-2">
                {!isDrawing ? (
                  <button
                    onClick={handleStartDrawing}
                    className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 border border-amber-500/40 text-xs font-semibold cursor-pointer transition-colors"
                  >
                    <Plus className="w-4 h-4" />
                    + Vẽ vùng loại trừ mới
                  </button>
                ) : (
                  <button
                    onClick={handleCancelDrawing}
                    className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 text-xs font-medium cursor-pointer"
                  >
                    <X className="w-4 h-4" />
                    Hủy vẽ
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Exclusion Zones Drawer Panel */}
          {!HIDE_EXCLUSION_ZONES && showZonePanel && (
            <div className="bg-gray-900/90 rounded-xl border border-gray-800 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-amber-400" />
                  <h3 className="font-semibold text-sm text-gray-200">
                    Danh sách Vùng Loại trừ (Exclusion ROI)
                  </h3>
                  <span className="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded-full font-medium">
                    {exclusionZones.length} vùng
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {!isDrawing && (
                    <button
                      onClick={handleStartDrawing}
                      className="text-xs text-amber-400 hover:text-amber-300 flex items-center gap-1 font-medium cursor-pointer"
                    >
                      <Plus className="w-3.5 h-3.5" />
                      Vẽ thêm vùng
                    </button>
                  )}
                  <button
                    onClick={() => setShowZonePanel(false)}
                    className="text-gray-400 hover:text-white p-1"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="text-xs text-gray-400 flex items-start gap-2 bg-gray-800/60 p-2.5 rounded-lg border border-gray-800">
                <HelpCircle className="w-4 h-4 text-sky-400 flex-shrink-0 mt-0.5" />
                <p>
                  Các ngọn lửa hoặc đám khói phát sinh bên trong vùng loại trừ (hàn xì, bếp gas) sẽ được
                  hệ thống <strong>bỏ qua</strong>. Nếu đám cháy bùng phát và <strong>lan ra ngoài vùng quá 40%</strong>,
                  cơ chế Breakout Guard sẽ tự động kích hoạt báo động khẩn cấp!
                </p>
              </div>

              {exclusionZones.length === 0 ? (
                <div className="text-center py-6 text-gray-500 text-xs">
                  Chưa có vùng loại trừ nào được thiết lập. Nhấn nút <strong>"+ Vẽ vùng mới"</strong> để bắt đầu khoanh vùng.
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2.5 pt-1">
                  {exclusionZones.map((zone) => {
                    const typeInfo = ZONE_TYPE_LABELS[zone.zone_type] || ZONE_TYPE_LABELS.other
                    const pointCount = zone.polygon_points?.length || 0
                    return (
                      <div
                        key={zone.id}
                        onMouseEnter={() => setHoveredZoneId(zone.id)}
                        onMouseLeave={() => setHoveredZoneId(null)}
                        className={`p-3 rounded-lg border transition-all ${
                          zone.is_active
                            ? 'bg-gray-800/80 border-amber-500/40 hover:border-amber-500'
                            : 'bg-gray-900/60 border-gray-800 opacity-60'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="flex items-center gap-1.5 font-medium text-xs text-gray-200 truncate">
                              <span>{typeInfo.icon}</span>
                              <span className="truncate">{zone.name}</span>
                            </div>
                            <div className="text-[11px] text-gray-400 mt-0.5">
                              {typeInfo.label} • {pointCount} điểm
                            </div>
                          </div>

                          <div className="flex items-center gap-1 flex-shrink-0">
                            <button
                              onClick={() => handleToggleZoneActive(zone)}
                              title={zone.is_active ? 'Tắt vùng này' : 'Bật vùng này'}
                              className={`p-1.5 rounded transition-colors cursor-pointer ${
                                zone.is_active
                                  ? 'text-amber-400 hover:bg-amber-500/20'
                                  : 'text-gray-500 hover:bg-gray-700 hover:text-gray-300'
                              }`}
                            >
                              {zone.is_active ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                            </button>
                            <button
                              onClick={() => handleDeleteZone(zone.id)}
                              title="Xóa vùng này"
                              className="p-1.5 rounded text-gray-500 hover:text-red-400 hover:bg-red-500/20 transition-colors cursor-pointer"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )}

          {/* Live Summary Badges */}
          <div className="grid grid-cols-7 gap-2">
            <LiveStatBadge label="Tổng" value={liveStats.total} color="bg-sky-500/20 text-sky-400" />
            <LiveStatBadge label="Mũ" value={liveStats.helmet} color="bg-red-500/20 text-red-400" />
            <LiveStatBadge label="Áo" value={liveStats.vest} color="bg-orange-500/20 text-orange-400" />
            <LiveStatBadge label="Khẩu trang" value={liveStats.mask} color="bg-yellow-500/20 text-yellow-400" />
            <LiveStatBadge label="Ngã" value={liveStats.fall} color="bg-rose-500/20 text-rose-500 font-bold" />
            <LiveStatBadge label="Cháy" value={liveStats.fire} color="bg-red-600/30 text-red-400 font-bold" />
            <LiveStatBadge label="Khói" value={liveStats.smoke} color="bg-amber-600/30 text-amber-400 font-bold" />
          </div>
        </div>

        {/* Right Sidebar: Realtime Alert Log */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 flex flex-col h-full max-h-[750px]">
          <h2 className="font-semibold mb-3 flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-red-400" />
              <span>Cảnh báo realtime</span>
            </div>
            {alerts.length > 0 && (
              <span className="text-[11px] bg-red-500/20 text-red-400 px-2 py-0.5 rounded-full font-bold">
                {alerts.length}
              </span>
            )}
          </h2>

          {alerts.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-gray-500 text-xs py-10">
              <ShieldCheck className="w-10 h-10 mb-2 opacity-40 text-emerald-400" />
              <p>Chưa phát hiện vi phạm nào</p>
              <p className="text-[11px] text-gray-600 mt-0.5">Khu vực làm việc an toàn</p>
            </div>
          ) : (
            <div className="space-y-2 overflow-y-auto flex-1 pr-1">
              {alerts.map((a, i) => {
                const style = VIOLATION_STYLES[a.type] || {}
                const Icon = style.icon || AlertTriangle
                const time = new Date(a.timestamp)
                const isBreakout = a.is_breakout || false
                return (
                  <div
                    key={i}
                    className={`p-3 rounded-lg border transition-all ${
                      isBreakout
                        ? 'bg-red-950/60 border-red-500 text-red-300'
                        : `${style.bg} ${style.border}`
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${style.dot}`} />
                      <Icon className={`w-4 h-4 ${style.text}`} />
                      <span className="text-xs font-semibold leading-tight">
                        {a.type_label || style.label}
                      </span>
                    </div>
                    {isBreakout && (
                      <div className="mt-1 text-[11px] text-red-400 font-medium">
                        ⚠️ Cháy lan ngoài vùng: {a.breakout_zone || 'Khu vực bỏ qua'}
                      </div>
                    )}
                    <div className="flex items-center justify-between text-[11px] text-gray-400 mt-1.5">
                      <span>{time.toLocaleTimeString('vi-VN')}</span>
                      <span className="font-medium text-gray-300">{a.confidence}%</span>
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
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl animate-in fade-in zoom-in-95">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Layers className="w-5 h-5 text-amber-400" />
                Lưu Vùng Loại Trừ Mới
              </h3>
              <button
                onClick={() => setShowZoneModal(false)}
                className="text-gray-400 hover:text-white p-1"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="text-xs text-gray-400">
              Đa giác đã tạo có <strong>{currentPoints.length} điểm</strong> góc. Vui lòng đặt tên và chọn loại khu vực.
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
                  placeholder="Ví dụ: Khu vực hàn xì xưởng B"
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
                  <option value="smoking">🚬 Khu vực hút thuốc ngoài trời</option>
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

      {/* Audio notification player */}
      <audio ref={audioRef} preload="auto">
        <source
          src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVggoKOeF1QSF1+kJpwTEdMW3+QmmxIRUxZf5CaakdESVh8j5xuRkRIWHuPnG5GREhYe4+cbkZESFh7j5xuRkRIWHuPnG5GREhYe4+cbkZESFh7j5xuRg=="
          type="audio/wav"
        />
      </audio>
    </div>
  )
}

function LiveStatBadge({ label, value, color }) {
  return (
    <div className={`rounded-lg p-3 text-center ${color}`}>
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-xs opacity-75">{label}</div>
    </div>
  )
}
