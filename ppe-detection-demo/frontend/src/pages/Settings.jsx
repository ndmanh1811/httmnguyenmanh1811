import { useState, useEffect } from 'react'
import api from '../api'
import soundEngine from '../utils/soundEngine'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Trash2, Volume2, Plus, Check } from 'lucide-react'

export default function Settings() {
  const [settings, setSettings] = useState({})
  const [cameras, setCameras] = useState([])
  const [newCamera, setNewCamera] = useState({ name: '', source: '0', location: '' })
  const [saving, setSaving] = useState(false)
  const [saveSuccess, setSaveSuccess] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      api.get('/settings').then((r) => r.data).catch(() => ({})),
      api.get('/cameras').then((r) => r.data).catch(() => []),
    ]).then(([sData, cData]) => {
      setSettings(sData)
      setCameras(cData)
      setLoading(false)
    })
  }, [])

  const isModuleEnabled = (key, defaultVal = true) => {
    const val = settings[key]?.value
    if (val === undefined || val === null) return defaultVal
    return String(val).toLowerCase() === 'true' || val === '1'
  }

  const toggleModule = (key, defaultVal = true) => {
    const current = isModuleEnabled(key, defaultVal)
    setSettings((prev) => ({
      ...prev,
      [key]: {
        ...prev[key],
        value: (!current).toString(),
      },
    }))
  }

  const handleSaveSettings = async () => {
    setSaving(true)
    const data = {}
    for (const [key, val] of Object.entries(settings)) {
      data[key] = val.value
    }
    try {
      await api.put('/settings', data)
      await api.post('/webcam_toggles', {
        enable_ppe: isModuleEnabled('enable_ppe'),
        enable_fall: isModuleEnabled('enable_fall'),
        enable_fire: isModuleEnabled('enable_fire'),
      }).catch(() => {})

      setSaveSuccess(true)
      setTimeout(() => setSaveSuccess(false), 2500)
    } catch {
      alert('Không thể lưu cài đặt. Vui lòng thử lại!')
    } finally {
      setSaving(false)
    }
  }

  const handleAddCamera = async (e) => {
    e.preventDefault()
    if (!newCamera.name.trim()) return
    try {
      const res = await api.post('/cameras', newCamera)
      setCameras((prev) => [...prev, res.data])
      setNewCamera({ name: '', source: '0', location: '' })
    } catch {
      alert('Lỗi thêm camera mới.')
    }
  }

  const handleDeleteCamera = async (id) => {
    if (!window.confirm('Xác nhận xóa camera này?')) return
    try {
      await api.delete(`/cameras/${id}`)
      setCameras((prev) => prev.filter((c) => c.id !== id))
    } catch {
      alert('Lỗi khi xóa camera.')
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-xs text-muted-foreground">
        Đang tải thông số cấu hình...
      </div>
    )
  }

  return (
    <div className="space-y-6 w-full">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-foreground">Cài đặt hệ thống</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Cấu hình module phân tích AI, tham số ngưỡng phát hiện và nguồn cấp camera
          </p>
        </div>
        <Button
          onClick={handleSaveSettings}
          disabled={saving}
          size="sm"
          className="h-8 text-xs font-medium cursor-pointer"
        >
          {saveSuccess ? (
            <span className="flex items-center gap-1.5 text-emerald-400">
              <Check className="w-3.5 h-3.5" /> Đã lưu
            </span>
          ) : saving ? (
            'Đang lưu...'
          ) : (
            'Lưu cấu hình'
          )}
        </Button>
      </div>

      {/* Section 1: AI Modules Activation */}
      <Card className="bg-card border-border shadow-none">
        <CardHeader className="p-4 pb-2 border-b border-border/50">
          <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Kích hoạt tính năng AI
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 space-y-4">
          {/* Module PPE */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Nhận diện trang bị bảo hộ (PPE)</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Kiểm tra mũ cứng, áo phản quang và khẩu trang của công nhân trên công trường
              </p>
            </div>
            <Switch
              checked={isModuleEnabled('enable_ppe')}
              onCheckedChange={() => toggleModule('enable_ppe')}
            />
          </div>

          <div className="border-t border-border/40" />

          {/* Module Fall */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Phát hiện ngã (Fall Detection)</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Nhận diện công nhân ngã té bất thường dựa trên mô hình Pose & LSTM 3-path
              </p>
            </div>
            <Switch
              checked={isModuleEnabled('enable_fall')}
              onCheckedChange={() => toggleModule('enable_fall')}
            />
          </div>

          <div className="border-t border-border/40" />

          {/* Module Fire */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Phát hiện khói & lửa (Fire & Smoke)</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Mô hình YOLO26s độ chính xác cao (mAP50 80.1%) theo dõi luồng bốc khói và tia lửa
              </p>
            </div>
            <Switch
              checked={isModuleEnabled('enable_fire')}
              onCheckedChange={() => toggleModule('enable_fire')}
            />
          </div>
        </CardContent>
      </Card>

      {/* Section 2: Audio Siren Test */}
      <Card className="bg-card border-border shadow-none">
        <CardHeader className="p-4 pb-2 border-b border-border/50">
          <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Kiểm tra âm thanh còi báo động (Web Audio API)
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => soundEngine.playAlert('fire_detected')}
            className="text-xs h-8"
          >
            <Volume2 className="w-3.5 h-3.5 mr-1 text-red-400" />
            Còi báo cháy
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => soundEngine.playAlert('fall_detected')}
            className="text-xs h-8"
          >
            <Volume2 className="w-3.5 h-3.5 mr-1 text-rose-400" />
            Còi cấp cứu ngã
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => soundEngine.playAlert('no_helmet')}
            className="text-xs h-8"
          >
            <Volume2 className="w-3.5 h-3.5 mr-1 text-amber-400" />
            Chuông nhắc PPE
          </Button>
        </CardContent>
      </Card>

      {/* Section 3: Camera Stream Sources */}
      <Card className="bg-card border-border shadow-none">
        <CardHeader className="p-4 pb-2 border-b border-border/50">
          <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Quản lý nguồn Camera ({cameras.length})
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 space-y-4">
          {/* Add Camera Form */}
          <form onSubmit={handleAddCamera} className="grid grid-cols-1 sm:grid-cols-4 gap-2">
            <input
              type="text"
              placeholder="Tên camera (vd: Cổng chính)"
              value={newCamera.name}
              onChange={(e) => setNewCamera({ ...newCamera, name: e.target.value })}
              className="bg-muted border border-border rounded-lg px-3 py-1.5 text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-border"
            />
            <input
              type="text"
              placeholder="Source (0 hoặc RTSP URL)"
              value={newCamera.source}
              onChange={(e) => setNewCamera({ ...newCamera, source: e.target.value })}
              className="bg-muted border border-border rounded-lg px-3 py-1.5 text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-border"
            />
            <input
              type="text"
              placeholder="Vị trí (vd: Xưởng gia công)"
              value={newCamera.location}
              onChange={(e) => setNewCamera({ ...newCamera, location: e.target.value })}
              className="bg-muted border border-border rounded-lg px-3 py-1.5 text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-border"
            />
            <Button type="submit" size="sm" className="h-8 text-xs">
              <Plus className="w-3.5 h-3.5 mr-1" /> Thêm camera
            </Button>
          </form>

          {/* Camera List Table */}
          {cameras.length === 0 ? (
            <p className="text-xs text-muted-foreground py-4 text-center">
              Chưa có camera nào được cấu hình
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="border-border/50 hover:bg-transparent">
                  <TableHead className="text-xs">Tên</TableHead>
                  <TableHead className="text-xs">Nguồn</TableHead>
                  <TableHead className="text-xs">Vị trí</TableHead>
                  <TableHead className="text-xs text-right w-16">Thao tác</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {cameras.map((c) => (
                  <TableRow key={c.id} className="border-border/50 hover:bg-muted/40">
                    <TableCell className="text-xs font-medium text-foreground">{c.name}</TableCell>
                    <TableCell className="text-xs font-mono text-muted-foreground">{c.source}</TableCell>
                    <TableCell className="text-xs text-muted-foreground">{c.location || '—'}</TableCell>
                    <TableCell className="text-right">
                      <button
                        onClick={() => handleDeleteCamera(c.id)}
                        className="text-muted-foreground hover:text-red-400 p-1 cursor-pointer transition-colors"
                        title="Xóa camera"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
