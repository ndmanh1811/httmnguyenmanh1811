import { useState, useRef, useEffect } from 'react'
import {
  Upload as UploadIcon,
  FileVideo,
  Loader2,
  CheckCircle,
  XCircle,
  Download,
  X,
} from 'lucide-react'
import { io } from 'socket.io-client'
import api from '../api'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [frameSkip, setFrameSkip] = useState(2)
  const [imgsz, setImgsz] = useState(960)
  const [enablePpe, setEnablePpe] = useState(true)
  const [enableFall, setEnableFall] = useState(true)
  const [enableFire, setEnableFire] = useState(true)
  const [useClahe, setUseClahe] = useState(true)
  const [useSahi, setUseSahi] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [progress, setProgress] = useState(null)
  const [cancelled, setCancelled] = useState(false)

  const [exclusionZones, setExclusionZones] = useState([])

  const inputRef = useRef(null)
  const uploadIdRef = useRef(null)
  const socketRef = useRef(null)

  useEffect(() => {
    const socket = io({ transports: ['polling'] })
    socketRef.current = socket

    socket.on('upload_progress', (data) => {
      if (data.upload_id === uploadIdRef.current) {
        setProgress(data)
      }
    })

    socket.on('upload_done', (data) => {
      if (data.upload_id === uploadIdRef.current) {
        if (data.error) {
          alert('Lỗi phân tích video: ' + data.error)
          setLoading(false)
          setProgress(null)
          uploadIdRef.current = null
          return
        }
        if (data.cancelled) {
          setLoading(false)
          setProgress(null)
          setCancelled(true)
          uploadIdRef.current = null
          return
        }
        setResult(data)
        setLoading(false)
        setProgress(null)
        uploadIdRef.current = null
      }
    })

    return () => socket.disconnect()
  }, [])

  const handleCancel = async () => {
    const currentId = uploadIdRef.current
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
    formData.append('imgsz', imgsz)
    formData.append('enable_ppe', enablePpe)
    formData.append('enable_fall', enableFall)
    formData.append('enable_fire', enableFire)
    formData.append('use_clahe', useClahe)
    formData.append('use_sahi', useSahi)

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

  const stats = result?.stats
  const violationRate = stats?.total_detections > 0
    ? Math.round((stats.violation_count / stats.total_detections) * 100)
    : 0

  return (
    <div className="space-y-6 w-full">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-foreground">Phân tích video ngoại tuyến</h1>
        <p className="text-xs text-muted-foreground mt-0.5">
          Tải lên video giám sát định dạng MP4/AVI để AI quét toàn diện sự cố và xuất video bằng chứng
        </p>
      </div>

      {/* Upload Zone */}
      {!file && (
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            const f = e.dataTransfer.files[0]
            if (f && f.type.startsWith('video/')) setFile(f)
          }}
          className="border border-dashed border-border hover:border-zinc-500 bg-card rounded-xl p-10 flex flex-col items-center justify-center cursor-pointer transition-colors"
        >
          <UploadIcon className="w-8 h-8 text-muted-foreground mb-3 opacity-60" />
          <p className="text-xs font-medium text-foreground">
            Kéo thả video vào đây hoặc <span className="text-blue-400 underline underline-offset-2">chọn tệp</span>
          </p>
          <p className="text-[11px] text-muted-foreground mt-1">
            Hỗ trợ định dạng MP4, AVI, MOV, MKV (Tối đa 200MB)
          </p>
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            onChange={(e) => e.target.files[0] && setFile(e.target.files[0])}
            className="hidden"
          />
        </div>
      )}

      {/* Selected Video Preview Bar */}
      {file && (
        <Card className="bg-card border-border shadow-none">
          <CardContent className="p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FileVideo className="w-5 h-5 text-blue-400" />
              <div>
                <p className="text-xs font-semibold text-foreground truncate max-w-sm">
                  {file.name}
                </p>
                <p className="text-[11px] text-muted-foreground font-mono">
                  {(file.size / (1024 * 1024)).toFixed(1)} MB
                </p>
              </div>
            </div>
            {!loading && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setFile(null)
                  setResult(null)
                  setProgress(null)
                }}
                className="h-8 text-xs text-muted-foreground hover:text-red-400 cursor-pointer"
              >
                <X className="w-4 h-4 mr-1" /> Đổi video
              </Button>
            )}
          </CardContent>
        </Card>
      )}

      {/* Feature Selector Pills & Config */}
      <Card className="bg-card border-border shadow-none">
        <CardHeader className="p-4 pb-2 border-b border-border/50">
          <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Tính năng & Thông số xử lý
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 space-y-4">
          {/* AI Module Pills */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setEnablePpe(!enablePpe)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer select-none ${
                enablePpe
                  ? 'bg-accent text-accent-foreground border border-border'
                  : 'text-muted-foreground hover:bg-muted/40 hover:text-foreground'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${enablePpe ? 'bg-amber-400' : 'bg-muted-foreground'}`} />
              Bảo hộ PPE
            </button>

            <button
              type="button"
              onClick={() => setEnableFall(!enableFall)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer select-none ${
                enableFall
                  ? 'bg-accent text-accent-foreground border border-border'
                  : 'text-muted-foreground hover:bg-muted/40 hover:text-foreground'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${enableFall ? 'bg-rose-400' : 'bg-muted-foreground'}`} />
              Theo dõi Ngã
            </button>

            <button
              type="button"
              onClick={() => setEnableFire(!enableFire)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer select-none ${
                enableFire
                  ? 'bg-accent text-accent-foreground border border-border'
                  : 'text-muted-foreground hover:bg-muted/40 hover:text-foreground'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${enableFire ? 'bg-red-400' : 'bg-muted-foreground'}`} />
              Cháy & Khói (YOLO26s)
            </button>

            <button
              type="button"
              onClick={() => setUseClahe(!useClahe)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer select-none ${
                useClahe
                  ? 'bg-accent text-accent-foreground border border-border'
                  : 'text-muted-foreground hover:bg-muted/40 hover:text-foreground'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${useClahe ? 'bg-emerald-400' : 'bg-muted-foreground'}`} />
              Kéo sáng (CLAHE)
            </button>

            <button
              type="button"
              onClick={() => setUseSahi(!useSahi)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer select-none ${
                useSahi
                  ? 'bg-accent text-accent-foreground border border-border'
                  : 'text-muted-foreground hover:bg-muted/40 hover:text-foreground'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${useSahi ? 'bg-purple-400' : 'bg-muted-foreground'}`} />
              Cắt lát xa (SAHI)
            </button>
          </div>

          <div className="border-t border-border/40" />

          {/* Speed & Resolution Dropdowns & Action Button */}
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Tốc độ:</span>
                <select
                  value={frameSkip}
                  onChange={(e) => setFrameSkip(Number(e.target.value))}
                  className="bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground focus:outline-none"
                >
                  <option value={1}>Chính xác nhất (Mọi frame)</option>
                  <option value={2}>Cân bằng (Khuyên dùng)</option>
                  <option value={5}>Nhanh (Bỏ 5 frame)</option>
                </select>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Độ phân giải:</span>
                <select
                  value={imgsz}
                  onChange={(e) => setImgsz(Number(e.target.value))}
                  className="bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground focus:outline-none"
                >
                  <option value={960}>960p (Mặc định)</option>
                  <option value={640}>640p (Tiêu chuẩn)</option>
                  <option value={1280}>1280p (Cực nét)</option>
                </select>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {loading && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleCancel}
                  className="h-8 text-xs text-red-400 border-red-500/30 hover:bg-red-500/10 cursor-pointer"
                >
                  Hủy phân tích
                </Button>
              )}
              <Button
                size="sm"
                onClick={handleUpload}
                disabled={!file || loading}
                className="h-8 text-xs font-medium cursor-pointer"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" />
                    Đang xử lý...
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-3.5 h-3.5 mr-1.5" />
                    Bắt đầu phân tích
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Cancellation Notice */}
      {cancelled && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-3 text-amber-400 text-xs text-center">
          Tiến trình phân tích video đã được hủy
        </div>
      )}

      {/* Progress Bar */}
      {loading && progress && (
        <Card className="bg-card border-border shadow-none">
          <CardContent className="p-4 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium text-foreground flex items-center gap-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-400" />
                Đang xử lý: {progress.processed} / {progress.total} frames ({progress.percent}%)
              </span>
              <span className="font-mono text-muted-foreground text-[11px]">
                Còn lại ~{Math.round(progress.remaining)}s
              </span>
            </div>
            <div className="w-full bg-muted rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-blue-400 h-1.5 rounded-full transition-all duration-300"
                style={{ width: `${progress.percent}%` }}
              />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results View */}
      {result && !result.error && (
        <div className="space-y-4 animate-in fade-in duration-300">
          <Card className="bg-card border-border shadow-none">
            <CardHeader className="p-4 border-b border-border/50">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-semibold text-foreground">
                  Kết quả phân tích video
                </CardTitle>
                <a
                  href={result.video_url || (result.output_video ? `/api/detect/output/${result.output_video}` : '#')}
                  download="result_analyzed.mp4"
                  className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 font-medium"
                >
                  <Download className="w-3.5 h-3.5" /> Tải video kết quả
                </a>
              </div>
            </CardHeader>
            <CardContent className="p-4 space-y-4">
              {/* Metric Summary */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="bg-muted/40 p-3 rounded-lg border border-border/40">
                  <div className="text-[11px] text-muted-foreground">Tổng số frame</div>
                  <div className="text-xl font-bold mt-0.5 text-foreground tabular-nums">
                    {stats?.processed_frames}
                  </div>
                </div>
                <div className="bg-muted/40 p-3 rounded-lg border border-border/40">
                  <div className="text-[11px] text-muted-foreground">Tỷ lệ vi phạm PPE</div>
                  <div className="text-xl font-bold mt-0.5 text-amber-400 tabular-nums">
                    {violationRate}%
                  </div>
                </div>
                <div className="bg-muted/40 p-3 rounded-lg border border-border/40">
                  <div className="text-[11px] text-muted-foreground">Phát hiện ngã</div>
                  <div className="text-xl font-bold mt-0.5 text-rose-400 tabular-nums">
                    {stats?.fall_count || 0}
                  </div>
                </div>
                <div className="bg-muted/40 p-3 rounded-lg border border-border/40">
                  <div className="text-[11px] text-muted-foreground">Sự cố Cháy / Khói</div>
                  <div className="text-xl font-bold mt-0.5 text-red-400 tabular-nums">
                    {(stats?.fire_count || 0) + (stats?.smoke_count || 0)}
                  </div>
                </div>
              </div>

              {/* Video Player */}
              <div className="rounded-xl overflow-hidden border border-border bg-black">
                <video
                  src={result.video_url || (result.output_video ? `/api/detect/output/${result.output_video}` : '')}
                  controls
                  autoPlay
                  className="w-full max-h-[480px]"
                />
              </div>

              {/* PPE Detail Cards */}
              {enablePpe && stats && (
                <div className="pt-3 border-t border-border/40 space-y-2">
                  <span className="text-xs font-semibold text-foreground">
                    Tuân thủ trang bị bảo hộ cá nhân:
                  </span>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {[
                      { key: 'helmet', name: 'Mũ bảo hiểm' },
                      { key: 'vest', name: 'Áo phản quang' },
                      { key: 'mask', name: 'Khẩu trang' },
                    ].map(({ key, name }) => (
                      <div key={key} className="bg-muted/40 border border-border/40 rounded-lg p-3">
                        <div className="text-xs font-medium text-foreground">{name}</div>
                        <div className="flex items-center justify-between text-[11px] mt-1 text-muted-foreground">
                          <span>Đạt: <strong className="text-emerald-400">{stats[`${key}_safe`] || 0}</strong></span>
                          <span>Vi phạm: <strong className="text-red-400">{stats[`no_${key}`] || 0}</strong></span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
