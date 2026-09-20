import { useState, useEffect } from 'react'
import {
  Settings as SettingsIcon,
  Save,
  Plus,
  Trash2,
  HardHat,
  AlertTriangle,
  Flame,
  Cpu,
  CheckCircle2,
  Bell,
  HelpCircle,
} from 'lucide-react'
import api from '../api'
import soundEngine from '../utils/soundEngine'

export default function Settings() {
  const [settings, setSettings] = useState({})
  const [cameras, setCameras] = useState([])
  const [newCamera, setNewCamera] = useState({ name: '', source: '0', location: '' })
  const [saving, setSaving] = useState(false)
  const [saveSuccess, setSaveSuccess] = useState(false)

  useEffect(() => {
    api.get('/settings').then(r => setSettings(r.data)).catch(() => {})
    api.get('/cameras').then(r => setCameras(r.data)).catch(() => {})
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
      // Cap nhat dong thoi camera stream dang chay (neu co)
      await api.post('/webcam_toggles', {
        enable_ppe: isModuleEnabled('enable_ppe'),
        enable_fall: isModuleEnabled('enable_fall'),
        enable_fire: isModuleEnabled('enable_fire'),
      }).catch(() => {})

      setSaveSuccess(true)
      setTimeout(() => setSaveSuccess(false), 3000)
    } catch {
      alert('Lỗi khi lưu cài đặt!')
    } finally {
      setSaving(false)
    }
  }

  const handleAddCamera = async () => {
    if (!newCamera.name.trim()) return
    try {
      const res = await api.post('/cameras', newCamera)
      setCameras([...cameras, res.data])
      setNewCamera({ name: '', source: '0', location: '' })
    } catch {
      alert('Lỗi thêm camera')
    }
  }

  const handleDeleteCamera = async (id) => {
    if (!confirm('Xóa camera này?')) return
    try {
      await api.delete(`/cameras/${id}`)
      setCameras(cameras.filter(c => c.id !== id))
    } catch {
      alert('Lỗi xóa camera')
    }
  }

  const ppeEnabled = isModuleEnabled('enable_ppe')
  const fallEnabled = isModuleEnabled('enable_fall')
  const fireEnabled = isModuleEnabled('enable_fire')

  // Loc cac tham so chi tiet (bo qua cac toggle da duoc hien thi o the tren)
  const detailSettings = Object.entries(settings).filter(
    ([key]) => !['enable_ppe', 'enable_fall', 'enable_fire'].includes(key)
  )

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Cài đặt hệ thống</h1>
          <p className="text-sm text-gray-400 mt-1">
            Quản lý các module phát hiện AI, thông số thuật toán và danh sách camera giám sát
          </p>
        </div>
        <button
          onClick={handleSaveSettings}
          disabled={saving}
          className="px-6 py-2.5 bg-sky-600 hover:bg-sky-500 active:bg-sky-700 text-white rounded-xl font-medium flex items-center gap-2 shadow-lg shadow-sky-600/20 transition-all cursor-pointer"
        >
          {saveSuccess ? (
            <>
              <CheckCircle2 className="w-5 h-5 text-emerald-300" />
              <span>Đã lưu thành công!</span>
            </>
          ) : (
            <>
              <Save className="w-5 h-5" />
              <span>{saving ? 'Đang lưu...' : 'Lưu tất cả thay đổi'}</span>
            </>
          )}
        </button>
      </div>

      {/* AI MODULE TOGGLES */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold text-lg flex items-center gap-2">
            <Cpu className="w-5 h-5 text-sky-400" />
            Bật / Tắt Các Module Phân Tích AI
          </h2>
          <span className="text-xs text-gray-400 bg-gray-800 px-3 py-1 rounded-full">
            Tắt bớt module để tăng FPS & tránh rối mắt khi quan sát cháy/khói
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          {/* Module 1: PPE */}
          <div
            onClick={() => toggleModule('enable_ppe')}
            className={`p-5 rounded-xl border transition-all cursor-pointer select-none flex flex-col justify-between ${
              ppeEnabled
                ? 'bg-sky-950/20 border-sky-600/50 hover:border-sky-500 shadow-md shadow-sky-950/30'
                : 'bg-gray-800/40 border-gray-800 hover:border-gray-700 opacity-70'
            }`}
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className={`p-2.5 rounded-lg ${ppeEnabled ? 'bg-sky-500/20 text-sky-400' : 'bg-gray-800 text-gray-400'}`}>
                  <HardHat className="w-6 h-6" />
                </div>
                <div className={`w-12 h-6 flex items-center rounded-full p-1 duration-300 ease-in-out cursor-pointer ${ppeEnabled ? 'bg-sky-500 justify-end' : 'bg-gray-700 justify-start'}`}>
                  <div className="bg-white w-4 h-4 rounded-full shadow-md transform duration-300"></div>
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-base">Phát hiện PPE (Bảo hộ)</h3>
                <p className="text-xs text-gray-400 mt-1 leading-relaxed">
                  Nhận diện Mũ bảo hộ, Áo phản quang, Khẩu trang và theo dõi liên kết theo từng công nhân.
                </p>
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-gray-800/60 flex items-center justify-between text-xs">
              <span className="text-gray-400">Trạng thái:</span>
              <span className={`font-medium px-2 py-0.5 rounded ${ppeEnabled ? 'bg-sky-500/20 text-sky-400' : 'bg-gray-800 text-gray-500'}`}>
                {ppeEnabled ? 'Đang hoạt động' : 'Đã tạm tắt'}
              </span>
            </div>
          </div>

          {/* Module 2: Fall */}
          <div
            onClick={() => toggleModule('enable_fall')}
            className={`p-5 rounded-xl border transition-all cursor-pointer select-none flex flex-col justify-between ${
              fallEnabled
                ? 'bg-rose-950/20 border-rose-600/50 hover:border-rose-500 shadow-md shadow-rose-950/30'
                : 'bg-gray-800/40 border-gray-800 hover:border-gray-700 opacity-70'
            }`}
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className={`p-2.5 rounded-lg ${fallEnabled ? 'bg-rose-500/20 text-rose-400' : 'bg-gray-800 text-gray-400'}`}>
                  <AlertTriangle className="w-6 h-6" />
                </div>
                <div className={`w-12 h-6 flex items-center rounded-full p-1 duration-300 ease-in-out cursor-pointer ${fallEnabled ? 'bg-rose-500 justify-end' : 'bg-gray-700 justify-start'}`}>
                  <div className="bg-white w-4 h-4 rounded-full shadow-md transform duration-300"></div>
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-base">Theo dõi & Cảnh báo Ngã</h3>
                <p className="text-xs text-gray-400 mt-1 leading-relaxed">
                  Phân tích khung xương tư thế dáng người và kích hoạt chuông cấp cứu khi có sự cố ngã bất động.
                </p>
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-gray-800/60 flex items-center justify-between text-xs">
              <span className="text-gray-400">Trạng thái:</span>
              <span className={`font-medium px-2 py-0.5 rounded ${fallEnabled ? 'bg-rose-500/20 text-rose-400' : 'bg-gray-800 text-gray-500'}`}>
                {fallEnabled ? 'Đang hoạt động' : 'Đã tạm tắt'}
              </span>
            </div>
          </div>

          {/* Module 3: Fire & Smoke */}
          <div
            onClick={() => toggleModule('enable_fire')}
            className={`p-5 rounded-xl border transition-all cursor-pointer select-none flex flex-col justify-between ${
              fireEnabled
                ? 'bg-amber-950/20 border-amber-600/50 hover:border-amber-500 shadow-md shadow-amber-950/30'
                : 'bg-gray-800/40 border-gray-800 hover:border-gray-700 opacity-70'
            }`}
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className={`p-2.5 rounded-lg ${fireEnabled ? 'bg-amber-500/20 text-amber-400' : 'bg-gray-800 text-gray-400'}`}>
                  <Flame className="w-6 h-6" />
                </div>
                <div className={`w-12 h-6 flex items-center rounded-full p-1 duration-300 ease-in-out cursor-pointer ${fireEnabled ? 'bg-amber-500 justify-end' : 'bg-gray-700 justify-start'}`}>
                  <div className="bg-white w-4 h-4 rounded-full shadow-md transform duration-300"></div>
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-base">Phát hiện Cháy & Khói</h3>
                <p className="text-xs text-gray-400 mt-1 leading-relaxed">
                  Giám sát đốm lửa, khói bốc lên theo thời gian thực và tự động lưu video/ảnh bằng chứng hỏa hoạn.
                </p>
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-gray-800/60 flex items-center justify-between text-xs">
              <span className="text-gray-400">Trạng thái:</span>
              <span className={`font-medium px-2 py-0.5 rounded ${fireEnabled ? 'bg-amber-500/20 text-amber-400' : 'bg-gray-800 text-gray-500'}`}>
                {fireEnabled ? 'Đang hoạt động' : 'Đã tạm tắt'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* EMERGENCY SOUND PREVIEW CARD */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold text-lg flex items-center gap-2">
            <Bell className="w-5 h-5 text-amber-400" />
            Kiểm Tra Còi Hú Cảnh Báo Trình Duyệt (Web Audio)
          </h2>
          <span className="text-xs text-gray-400 bg-gray-800 px-3 py-1 rounded-full">
            Tự tạo âm thanh lập trình, không phụ thuộc file mạng
          </span>
        </div>
        <p className="text-xs text-gray-400 leading-relaxed">
          Bấm các nút dưới đây để nghe thử âm lượng và giai điệu còi hú cảnh báo khẩn cấp khi camera phát hiện sự cố:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
          <button
            type="button"
            onClick={() => soundEngine.playAlert('fire')}
            className="p-3 bg-red-950/20 hover:bg-red-950/40 border border-red-800/40 hover:border-red-700 text-red-300 rounded-xl text-xs font-medium flex items-center justify-center gap-2 transition-all cursor-pointer"
          >
            <Flame className="w-4 h-4 text-red-400" />
            Nghe thử Còi Báo Cháy (Siren)
          </button>
          <button
            type="button"
            onClick={() => soundEngine.playAlert('fall')}
            className="p-3 bg-rose-950/20 hover:bg-rose-950/40 border border-rose-800/40 hover:border-rose-700 text-rose-300 rounded-xl text-xs font-medium flex items-center justify-center gap-2 transition-all cursor-pointer"
          >
            <AlertTriangle className="w-4 h-4 text-rose-400" />
            Nghe thử Báo Ngã (Medical Alert)
          </button>
          <button
            type="button"
            onClick={() => soundEngine.playAlert('no_helmet')}
            className="p-3 bg-sky-950/20 hover:bg-sky-950/40 border border-sky-800/40 hover:border-sky-700 text-sky-300 rounded-xl text-xs font-medium flex items-center justify-center gap-2 transition-all cursor-pointer"
          >
            <Bell className="w-4 h-4 text-sky-400" />
            Nghe thử Chuông PPE (Chime)
          </button>
        </div>
      </div>

      {/* Detection Detail Settings */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
        <h2 className="font-semibold text-lg mb-4 flex items-center gap-2">
          <SettingsIcon className="w-5 h-5 text-sky-400" />
          Thông số phát hiện chi tiết
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {detailSettings.map(([key, val]) => (
            <div key={key} className="space-y-1.5 bg-gray-800/40 p-3.5 rounded-xl border border-gray-800">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium text-gray-200">{val.description || key}</label>
                <code className="text-xs text-gray-500">{key}</code>
              </div>
              <input
                type="text"
                value={val.value}
                onChange={(e) => setSettings({
                  ...settings,
                  [key]: { ...val, value: e.target.value }
                })}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:border-sky-500 focus:outline-none transition-colors"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Camera Management */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
        <h2 className="font-semibold text-lg mb-4">Quản lý Camera</h2>

        <div className="flex gap-3 mb-4">
          <input
            type="text"
            placeholder="Tên camera"
            value={newCamera.name}
            onChange={(e) => setNewCamera({ ...newCamera, name: e.target.value })}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm flex-1 focus:border-sky-500 focus:outline-none"
          />
          <input
            type="text"
            placeholder="Source (0 = webcam, URL = RTSP)"
            value={newCamera.source}
            onChange={(e) => setNewCamera({ ...newCamera, source: e.target.value })}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm flex-1 focus:border-sky-500 focus:outline-none"
          />
          <input
            type="text"
            placeholder="Vị trí"
            value={newCamera.location}
            onChange={(e) => setNewCamera({ ...newCamera, location: e.target.value })}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm flex-1 focus:border-sky-500 focus:outline-none"
          />
          <button
            onClick={handleAddCamera}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center gap-2 font-medium cursor-pointer"
          >
            <Plus className="w-4 h-4" />
            Thêm
          </button>
        </div>

        <div className="space-y-2">
          {cameras.map(c => (
            <div key={c.id} className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg border border-gray-800">
              <div>
                <span className="font-medium">{c.name}</span>
                <span className="text-sm text-gray-500 ml-2">Source: {c.source}</span>
                {c.location && <span className="text-sm text-gray-500 ml-2">| {c.location}</span>}
              </div>
              <button
                onClick={() => handleDeleteCamera(c.id)}
                className="p-2 text-red-400 hover:bg-red-500/20 rounded-lg cursor-pointer transition-colors"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
          {cameras.length === 0 && (
            <p className="text-gray-500 text-sm">Chưa có camera nào</p>
          )}
        </div>
      </div>
    </div>
  )
}


