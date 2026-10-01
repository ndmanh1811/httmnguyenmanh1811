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
  const [alarmState, setAlarmActive] = useState(false)

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

  const updateSettingValue = (key, value) => {
    setSettings((prev) => ({
      ...prev,
      [key]: {
        ...prev[key],
        value: String(value),
      },
    }))
  }

  const handleSaveSettings = async () => {
    setSaving(true)
    const data = {}
    for (const [key, val] of Object.entries(settings)) {
      data[key] = val?.value ?? ''
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

  const handleToggleAlarm = async (targetState) => {
    soundEngine.initContext()
    try {
      const res = await api.post('/iot/alarm', { state: targetState })
      setAlarmActive(res.data.alarm_state)
    } catch {
      alert('Không thể kết nối đến dịch vụ IoT')
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
    <div className="space-y-6 w-full max-w-5xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-foreground">Cài đặt hệ thống</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Cấu hình toàn diện các tham số AI, thuật toán phát hiện, cảm biến IoT và nguồn camera
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

      {/* 1. Kích hoạt tính năng & Hệ thống */}
      <Card className="bg-card border-border shadow-none">
        <CardHeader className="p-4 pb-2 border-b border-border/50">
          <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Tính năng phân tích AI & Hệ thống
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 space-y-4">
          {/* PPE */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Nhận diện trang bị bảo hộ (PPE)</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Kiểm tra mũ bảo hộ, áo phản quang và khẩu trang của công nhân
              </p>
            </div>
            <Switch
              checked={isModuleEnabled('enable_ppe')}
              onCheckedChange={() => toggleModule('enable_ppe')}
            />
          </div>

          <div className="border-t border-border/40" />

          {/* Fall */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Phát hiện ngã (Fall Detection)</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Theo dõi tư thế và phát hiện tình huống công nhân té ngã bất thường
              </p>
            </div>
            <Switch
              checked={isModuleEnabled('enable_fall')}
              onCheckedChange={() => toggleModule('enable_fall')}
            />
          </div>

          <div className="border-t border-border/40" />

          {/* Fire */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Phát hiện khói & lửa (Fire & Smoke)</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Mô hình YOLO26s theo dõi nguy cơ cháy và luồng khói bốc lên
              </p>
            </div>
            <Switch
              checked={isModuleEnabled('enable_fire')}
              onCheckedChange={() => toggleModule('enable_fire')}
            />
          </div>

          <div className="border-t border-border/40" />

          {/* Sound Enabled */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Âm thanh cảnh báo trên hệ thống</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Bật âm thanh còi báo động trực tiếp qua loa trình duyệt khi phát hiện sự cố
              </p>
            </div>
            <Switch
              checked={isModuleEnabled('sound_enabled', true)}
              onCheckedChange={() => toggleModule('sound_enabled', true)}
            />
          </div>

          <div className="border-t border-border/40" />

          {/* Evidence Enabled */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Tự động lưu ảnh bằng chứng vi phạm</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Chụp khung hình và lưu ảnh bằng chứng vào thư mục evidence của hệ thống
              </p>
            </div>
            <Switch
              checked={isModuleEnabled('evidence_enabled', true)}
              onCheckedChange={() => toggleModule('evidence_enabled', true)}
            />
          </div>
        </CardContent>
      </Card>

      {/* 2. Tham số phát hiện & Xử lý vi phạm */}
      <Card className="bg-card border-border shadow-none">
        <CardHeader className="p-4 pb-2 border-b border-border/50">
          <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Tham số phát hiện & Xử lý vi phạm
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 space-y-4">
          {/* Confidence Threshold */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Ngưỡng độ tin cậy tối thiểu (Confidence)</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Chỉ ghi nhận phát hiện khi độ tin cậy của AI đạt từ ngưỡng này trở lên (0.1 - 1.0)
              </p>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="number"
                step="0.05"
                min="0.1"
                max="1.0"
                placeholder="0.35"
                value={settings['confidence_threshold']?.value ?? ''}
                onChange={(e) => updateSettingValue('confidence_threshold', e.target.value)}
                className="w-24 bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground font-mono text-center focus:outline-none focus:border-border"
              />
            </div>
          </div>

          <div className="border-t border-border/40" />

          {/* Frame Skip */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Tỷ lệ bỏ qua khung hình (Frame Skip)</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Bỏ qua mỗi N khung hình để tối ưu hóa hiệu năng CPU/GPU (khuyến nghị: 2 - 4)
              </p>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="number"
                step="1"
                min="1"
                max="20"
                placeholder="4"
                value={settings['frame_skip']?.value ?? ''}
                onChange={(e) => updateSettingValue('frame_skip', e.target.value)}
                className="w-24 bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground font-mono text-center focus:outline-none focus:border-border"
              />
              <span className="text-xs text-muted-foreground">frame</span>
            </div>
          </div>

          <div className="border-t border-border/40" />

          {/* Cooldown Seconds */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Thời gian chống trùng lặp vi phạm (Cooldown)</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Khoảng thời gian tối thiểu trước khi ghi nhận lại cùng một vi phạm (tránh spam cảnh báo)
              </p>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="number"
                step="1"
                min="1"
                max="60"
                placeholder="5"
                value={settings['cooldown_seconds']?.value ?? ''}
                onChange={(e) => updateSettingValue('cooldown_seconds', e.target.value)}
                className="w-24 bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground font-mono text-center focus:outline-none focus:border-border"
              />
              <span className="text-xs text-muted-foreground">giây</span>
            </div>
          </div>

          <div className="border-t border-border/40" />

          {/* Tick Interval */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Chu kỳ đánh giá tick (Tick Interval)</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Khoảng thời gian định kỳ giữa các lần quét và đánh giá lại trạng thái vi phạm
              </p>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="number"
                step="1"
                min="1"
                max="60"
                placeholder="10"
                value={settings['tick_interval']?.value ?? ''}
                onChange={(e) => updateSettingValue('tick_interval', e.target.value)}
                className="w-24 bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground font-mono text-center focus:outline-none focus:border-border"
              />
              <span className="text-xs text-muted-foreground">giây</span>
            </div>
          </div>

          <div className="border-t border-border/40" />

          {/* Confirm Ticks */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Số tick xác nhận vi phạm (Confirm Ticks)</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Số lần kiểm tra liên tiếp phát hiện vi phạm trước khi chính thức ghi nhận vào cơ sở dữ liệu
              </p>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="number"
                step="1"
                min="1"
                max="10"
                placeholder="3"
                value={settings['confirm_ticks']?.value ?? ''}
                onChange={(e) => updateSettingValue('confirm_ticks', e.target.value)}
                className="w-24 bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground font-mono text-center focus:outline-none focus:border-border"
              />
              <span className="text-xs text-muted-foreground">tick</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 3. Thuật toán nhận diện ngã */}
      <Card className="bg-card border-border shadow-none">
        <CardHeader className="p-4 pb-2 border-b border-border/50">
          <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Thuật toán nhận diện ngã (Fall Detection)
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 space-y-4">
          {/* Fall Angle Threshold */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Góc nghiêng tối thiểu phát hiện ngã</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Độ nghiêng của trục cơ thể người so với phương thẳng đứng để kích hoạt nghi ngờ ngã
              </p>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="number"
                step="1"
                min="30"
                max="90"
                placeholder="60.0"
                value={settings['fall_angle_threshold']?.value ?? ''}
                onChange={(e) => updateSettingValue('fall_angle_threshold', e.target.value)}
                className="w-24 bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground font-mono text-center focus:outline-none focus:border-border"
              />
              <span className="text-xs text-muted-foreground">độ (°)</span>
            </div>
          </div>

          <div className="border-t border-border/40" />

          {/* Fall Consecutive Frames */}
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Số frame liên tiếp xác nhận ngã</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Số khung hình duy trì trạng thái ngã liên tiếp để xác thực sự cố (loại bỏ nhiễu cúi người, nhặt đồ)
              </p>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="number"
                step="1"
                min="1"
                max="30"
                placeholder="8"
                value={settings['fall_consecutive_frames']?.value ?? ''}
                onChange={(e) => updateSettingValue('fall_consecutive_frames', e.target.value)}
                className="w-24 bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground font-mono text-center focus:outline-none focus:border-border"
              />
              <span className="text-xs text-muted-foreground">frame</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 4. Cảm biến IoT (ESP8266) & Ngưỡng nhiệt độ */}
      <Card className="bg-card border-border shadow-none">
        <CardHeader className="p-4 pb-2 border-b border-border/50 flex flex-row items-center justify-between">
          <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Cảm biến nhiệt độ IoT (ESP8266)
          </CardTitle>
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-muted-foreground">Kiểm tra còi/đèn:</span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleToggleAlarm(!alarmState)}
              className={`h-7 px-2.5 text-xs font-medium cursor-pointer transition-colors ${
                alarmState
                  ? 'border-rose-500/50 text-rose-400 bg-rose-500/10'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {alarmState ? 'Tắt còi' : 'Bật còi'}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Ngưỡng nhiệt độ cảnh báo (°C)</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Kích hoạt cảnh báo trên thanh điều khiển toàn hệ thống khi nhiệt độ môi trường vượt quá mức này
              </p>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="number"
                step="0.5"
                min="0"
                max="100"
                placeholder="40.0"
                value={settings['temp_limit']?.value ?? ''}
                onChange={(e) => updateSettingValue('temp_limit', e.target.value)}
                className="w-24 bg-muted border border-border rounded-lg px-2.5 py-1 text-xs text-foreground font-mono text-center focus:outline-none focus:border-border"
              />
              <span className="text-xs text-muted-foreground font-medium">°C</span>
            </div>
          </div>

          <div className="border-t border-border/40" />

          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-foreground">Tự động kích hoạt còi/đèn khi quá nhiệt</div>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Tự động gửi lệnh ON đến thiết bị khi nhiệt độ vượt ngưỡng (mặc định tắt: người trực bấm bật thủ công)
              </p>
            </div>
            <Switch
              checked={isModuleEnabled('auto_alarm_on_overheat', false)}
              onCheckedChange={() => toggleModule('auto_alarm_on_overheat', false)}
            />
          </div>
        </CardContent>
      </Card>

      {/* 5. Kiểm tra âm thanh còi báo động */}
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
            className="text-xs h-8 cursor-pointer"
          >
            <Volume2 className="w-3.5 h-3.5 mr-1 text-muted-foreground" />
            Còi báo cháy
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => soundEngine.playAlert('fall_detected')}
            className="text-xs h-8 cursor-pointer"
          >
            <Volume2 className="w-3.5 h-3.5 mr-1 text-muted-foreground" />
            Còi cấp cứu ngã
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => soundEngine.playAlert('no_helmet')}
            className="text-xs h-8 cursor-pointer"
          >
            <Volume2 className="w-3.5 h-3.5 mr-1 text-muted-foreground" />
            Chuông nhắc PPE
          </Button>
        </CardContent>
      </Card>

      {/* 6. Quản lý nguồn Camera */}
      <Card className="bg-card border-border shadow-none">
        <CardHeader className="p-4 pb-2 border-b border-border/50">
          <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Quản lý nguồn Camera ({cameras.length})
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 space-y-4">
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
            <Button type="submit" size="sm" className="h-8 text-xs cursor-pointer">
              <Plus className="w-3.5 h-3.5 mr-1" /> Thêm camera
            </Button>
          </form>

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
