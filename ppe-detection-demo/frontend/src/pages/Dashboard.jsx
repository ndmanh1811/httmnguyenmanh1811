import { useState, useEffect } from 'react'
import { HardHat, Shirt, ShieldAlert, AlertTriangle, Activity, Camera, Flame } from 'lucide-react'
import api from '../api'

const TYPE_NAMES = {
  no_helmet: 'Thiếu mũ bảo hiểm',
  no_vest: 'Thiếu áo bảo hộ',
  no_mask: 'Thiếu khẩu trang',
  fall_detected: 'Phát hiện ngã',
  fire_detected: 'Phát hiện cháy 🔥',
  smoke_detected: 'Phát hiện khói 💨',
}

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [recent, setRecent] = useState([])

  useEffect(() => {
    api.get('/stats/summary').then(r => setStats(r.data)).catch(() => {})
    api.get('/stats/recent?limit=5').then(r => setRecent(r.data)).catch(() => {})
  }, [])

  if (!stats) return <div className="text-gray-500">Đang tải...</div>

  const cards = [
    { label: 'Tổng vi phạm hôm nay', value: stats.total_today, icon: AlertTriangle, color: 'text-red-400', bg: 'bg-red-500/10' },
    { label: 'Thiếu mũ bảo hiểm', value: stats.helmet_today, icon: HardHat, color: 'text-orange-400', bg: 'bg-orange-500/10' },
    { label: 'Thiếu áo bảo hộ', value: stats.vest_today, icon: Shirt, color: 'text-yellow-400', bg: 'bg-yellow-500/10' },
    { label: 'Thiếu khẩu trang', value: stats.mask_today, icon: ShieldAlert, color: 'text-purple-400', bg: 'bg-purple-500/10' },
    { label: 'Phát hiện ngã', value: stats.fall_today, icon: Activity, color: 'text-rose-400', bg: 'bg-rose-500/10' },
    { label: 'Phát hiện cháy', value: stats.fire_today || 0, icon: Flame, color: 'text-red-500 font-bold', bg: 'bg-red-600/20' },
    { label: 'Phát hiện khói', value: stats.smoke_today || 0, icon: Flame, color: 'text-amber-500 font-bold', bg: 'bg-amber-600/20' },
    { label: 'Camera hoạt động', value: stats.active_cameras, icon: Camera, color: 'text-sky-400', bg: 'bg-sky-500/10' },
  ]

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {cards.map(c => (
          <div key={c.label} className={`${c.bg} rounded-xl p-4 border border-gray-800`}>
            <c.icon className={`w-6 h-6 ${c.color} mb-2`} />
            <p className={`text-2xl font-bold ${c.color}`}>{c.value}</p>
            <p className="text-xs text-gray-400 mt-1">{c.label}</p>
          </div>
        ))}
      </div>

      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
        <h2 className="font-semibold mb-3">Sự cố & Vi phạm gần đây</h2>
        {recent.length === 0 ? (
          <p className="text-gray-500 text-sm">Chưa có sự cố nào</p>
        ) : (
          <div className="space-y-2">
            {recent.map(v => (
              <div key={v.id} className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                <div>
                  <span className="text-sm font-medium">{TYPE_NAMES[v.type] || v.type}</span>
                  <span className="text-xs text-gray-500 ml-2">{v.confidence}%</span>
                </div>
                <span className="text-xs text-gray-500">
                  {new Date(v.timestamp).toLocaleString('vi-VN')}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
