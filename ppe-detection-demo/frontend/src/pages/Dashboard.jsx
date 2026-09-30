import { useState, useEffect } from 'react'
import api from '../api'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'


const TYPE_CONFIG = {
  no_helmet: { label: 'Thiếu mũ bảo hiểm', variant: 'outline', dot: 'bg-amber-400' },
  no_vest: { label: 'Thiếu áo bảo hộ', variant: 'outline', dot: 'bg-orange-400' },
  no_mask: { label: 'Thiếu khẩu trang', variant: 'outline', dot: 'bg-purple-400' },
  fall_detected: { label: 'Phát hiện ngã', variant: 'destructive', dot: 'bg-rose-500' },
  fire_detected: { label: 'Phát hiện cháy', variant: 'destructive', dot: 'bg-red-500 animate-pulse' },
  smoke_detected: { label: 'Phát hiện khói', variant: 'destructive', dot: 'bg-amber-500' },
}

export default function Dashboard() {
  const [stats, setStats] = useState(() => {
    try {
      const cached = localStorage.getItem('ppe_dashboard_stats')
      return cached ? JSON.parse(cached) : null
    } catch (e) {
      return null
    }
  })
  const [recent, setRecent] = useState(() => {
    try {
      const cached = localStorage.getItem('ppe_dashboard_recent')
      return cached ? JSON.parse(cached) : []
    } catch (e) {
      return []
    }
  })
  const [loading, setLoading] = useState(!stats)

  const fetchDashboardData = () => {
    Promise.all([
      api.get('/stats/summary').then(r => r.data).catch(() => null),
      api.get('/stats/recent?limit=8').then(r => r.data).catch(() => []),
    ]).then(([summaryData, recentData]) => {
      if (summaryData) {
        setStats(summaryData)
        try {
          localStorage.setItem('ppe_dashboard_stats', JSON.stringify(summaryData))
        } catch (e) {}
      }
      if (recentData) {
        setRecent(recentData)
        try {
          localStorage.setItem('ppe_dashboard_recent', JSON.stringify(recentData))
        } catch (e) {}
      }
      setLoading(false)
    })
  }

  useEffect(() => {
    fetchDashboardData()
    const timer = setInterval(fetchDashboardData, 8000)
    return () => clearInterval(timer)
  }, [])

  if (loading && !stats) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
        Đang tải dữ liệu tổng quan...
      </div>
    )
  }

  const defaultStats = {
    total_today: 0,
    helmet_today: 0,
    vest_today: 0,
    mask_today: 0,
    fall_today: 0,
    fire_today: 0,
    smoke_today: 0,
    active_cameras: 0,
  }
  const safeStats = stats || defaultStats
  const ppeTotal = (safeStats.helmet_today || 0) + (safeStats.vest_today || 0) + (safeStats.mask_today || 0)
  const emergencyTotal = (safeStats.fall_today || 0) + (safeStats.fire_today || 0) + (safeStats.smoke_today || 0)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-foreground">Tổng quan an toàn</h1>
        <p className="text-xs text-muted-foreground mt-0.5">
          Giám sát tuân thủ bảo hộ lao động và phát hiện sự cố nguy hiểm thời gian thực
        </p>
      </div>

      {/* 4 Core Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Violations */}
        <Card className="bg-card border-border shadow-none">
          <CardHeader className="p-4 pb-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Tổng vi phạm hôm nay
              </span>
              <div className="w-2 h-2 rounded-full bg-red-400" />
            </div>
          </CardHeader>
          <CardContent className="p-4 pt-0">
            <div className="text-3xl font-bold tracking-tight text-foreground tabular-nums">
              {safeStats.total_today || 0}
            </div>
            <p className="text-[11px] text-muted-foreground mt-1">
              Bao gồm vi phạm PPE và cảnh báo nguy cơ
            </p>
          </CardContent>
        </Card>

        {/* PPE Violations */}
        <Card className="bg-card border-border shadow-none">
          <CardHeader className="p-4 pb-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Vi phạm trang bị PPE
              </span>
              <div className="w-2 h-2 rounded-full bg-amber-400" />
            </div>
          </CardHeader>
          <CardContent className="p-4 pt-0">
            <div className="text-3xl font-bold tracking-tight text-foreground tabular-nums">
              {ppeTotal}
            </div>
            <div className="flex items-center gap-2 text-[11px] text-muted-foreground mt-1">
              <span>Mũ: <strong className="text-foreground">{safeStats.helmet_today || 0}</strong></span>
              <span>•</span>
              <span>Áo: <strong className="text-foreground">{safeStats.vest_today || 0}</strong></span>
              <span>•</span>
              <span>Khẩu trang: <strong className="text-foreground">{safeStats.mask_today || 0}</strong></span>
            </div>
          </CardContent>
        </Card>

        {/* Emergency Hazards */}
        <Card className="bg-card border-border shadow-none">
          <CardHeader className="p-4 pb-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Sự cố khẩn cấp
              </span>
              <div className={`w-2 h-2 rounded-full ${emergencyTotal > 0 ? 'bg-red-500 animate-ping' : 'bg-emerald-400'}`} />
            </div>
          </CardHeader>
          <CardContent className="p-4 pt-0">
            <div className={`text-3xl font-bold tracking-tight tabular-nums ${emergencyTotal > 0 ? 'text-red-400' : 'text-foreground'}`}>
              {emergencyTotal}
            </div>
            <div className="flex items-center gap-2 text-[11px] text-muted-foreground mt-1">
              <span>Cháy: <strong className="text-foreground">{safeStats.fire_today || 0}</strong></span>
              <span>•</span>
              <span>Khói: <strong className="text-foreground">{safeStats.smoke_today || 0}</strong></span>
              <span>•</span>
              <span>Ngã: <strong className="text-foreground">{safeStats.fall_today || 0}</strong></span>
            </div>
          </CardContent>
        </Card>

        {/* Active Cameras */}
        <Card className="bg-card border-border shadow-none">
          <CardHeader className="p-4 pb-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Camera kết nối
              </span>
              <div className="w-2 h-2 rounded-full bg-emerald-400" />
            </div>
          </CardHeader>
          <CardContent className="p-4 pt-0">
            <div className="text-3xl font-bold tracking-tight text-foreground tabular-nums">
              {safeStats.active_cameras || 0}
            </div>
            <p className="text-[11px] text-muted-foreground mt-1">
              Luồng giám sát trực tuyến ổn định
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Recent Incidents Table */}
      <Card className="bg-card border-border shadow-none">
        <CardHeader className="p-4 border-b border-border/50">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-semibold text-foreground">
              Sự cố & Vi phạm gần đây
            </CardTitle>
            <span className="text-xs text-muted-foreground">
              {recent.length} bản ghi mới nhất
            </span>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {recent.length === 0 ? (
            <div className="py-12 text-center text-xs text-muted-foreground">
              Chưa ghi nhận sự cố hoặc vi phạm an toàn nào trong ngày
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="border-border/50 hover:bg-transparent">
                  <TableHead className="w-[180px] text-xs font-medium text-muted-foreground">Thời gian</TableHead>
                  <TableHead className="text-xs font-medium text-muted-foreground">Loại sự cố</TableHead>
                  <TableHead className="w-[120px] text-xs font-medium text-muted-foreground text-right">Độ tin cậy</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recent.map((item) => {
                  const cfg = TYPE_CONFIG[item.type] || { label: item.type, variant: 'outline', dot: 'bg-zinc-500' }
                  const time = new Date(item.timestamp).toLocaleString('vi-VN', {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    day: '2-digit',
                    month: '2-digit',
                  })
                  return (
                    <TableRow key={item.id} className="border-border/50 hover:bg-muted/40 transition-colors">
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {time}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
                          <span className="text-xs font-medium text-foreground">
                            {cfg.label}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell className="text-right font-mono text-xs tabular-nums text-muted-foreground">
                        {item.confidence}%
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
