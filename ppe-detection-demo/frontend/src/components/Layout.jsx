import { Outlet, NavLink } from 'react-router-dom'
import { LayoutGrid, Monitor, FileText, Upload, Settings, Sun, Moon } from 'lucide-react'
import { TooltipProvider, Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip'
import { useState, useEffect } from 'react'
import { io } from 'socket.io-client'
import api from '../api'
import soundEngine from '../utils/soundEngine'

const NAV_ITEMS = [
  { to: '/', icon: LayoutGrid, label: 'Dashboard' },
  { to: '/monitor', icon: Monitor, label: 'Giám sát' },
  { to: '/violations', icon: FileText, label: 'Vi phạm' },
  { to: '/upload', icon: Upload, label: 'Phân tích video' },
  { to: '/settings', icon: Settings, label: 'Cài đặt' },
]

export default function Layout() {
  const [time, setTime] = useState(new Date())
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('theme') || 'dark'
    } catch {
      return 'dark'
    }
  })

  // Global IoT Sensor & Alarm State
  const [iotData, setIotData] = useState({
    temp: null,
    hum: null,
    temp_limit: 40.0,
    is_over_limit: false,
    alarm_state: false,
  })
  const [alarmState, setAlarmState] = useState(false)

  useEffect(() => {
    api.get('/iot/status').then((r) => {
      if (r.data) {
        setIotData(r.data)
        setAlarmState(r.data.alarm_state)
      }
    }).catch(() => {})

    const socket = io({ transports: ['polling'] })

    socket.on('iot_sensor_data', (data) => {
      setIotData(data)
      setAlarmState(data.alarm_state)
    })

    socket.on('iot_alarm_state', (data) => {
      setAlarmState(data.alarm_state)
    })

    return () => socket.disconnect()
  }, [])

  // Bật / tắt còi hú báo cháy liên tục theo trạng thái alarmState
  useEffect(() => {
    if (alarmState) {
      soundEngine.startContinuousFireAlarm()
    } else {
      soundEngine.stopContinuousFireAlarm()
    }
    return () => {
      soundEngine.stopContinuousFireAlarm()
    }
  }, [alarmState])

  const toggleAlarm = async () => {
    soundEngine.initContext()
    try {
      const next = !alarmState
      const res = await api.post('/iot/alarm', { state: next })
      setAlarmState(res.data.alarm_state)
    } catch (e) {
      console.error('Failed to toggle IoT alarm:', e)
    }
  }

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    try {
      if (theme === 'dark') {
        document.documentElement.classList.add('dark')
      } else {
        document.documentElement.classList.remove('dark')
      }
      localStorage.setItem('theme', theme)
    } catch {}
  }, [theme])

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))
  }

  return (
    <TooltipProvider delayDuration={200}>
      <div className="min-h-screen bg-background text-foreground flex transition-colors duration-200">
        {/* Sidebar — icon-only, 60px */}
        <aside className="w-[60px] bg-card border-r border-border flex flex-col items-center py-4 flex-shrink-0">
          {/* Brand Logo */}
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary font-bold text-sm mb-6 select-none">
            S
          </div>

          {/* Navigation */}
          <nav className="flex-1 flex flex-col items-center gap-1">
            {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
              <Tooltip key={to}>
                <TooltipTrigger asChild>
                  <NavLink
                    to={to}
                    end={to === '/'}
                    className={({ isActive }) =>
                      `w-9 h-9 flex items-center justify-center rounded-lg transition-colors ${
                        isActive
                          ? 'bg-accent text-accent-foreground'
                          : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
                      }`
                    }
                  >
                    <Icon className="w-[18px] h-[18px]" />
                  </NavLink>
                </TooltipTrigger>
                <TooltipContent side="right" className="text-xs">
                  {label}
                </TooltipContent>
              </Tooltip>
            ))}
          </nav>

          {/* Version */}
          <div className="text-[9px] text-muted-foreground/50 select-none">v2.0</div>
        </aside>

        {/* Main Area */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Top Header */}
          <header className="h-12 border-b border-border px-6 flex items-center justify-between flex-shrink-0 bg-card/60 backdrop-blur-sm">
            <div className="flex items-center gap-4">
              <span className="text-sm font-semibold text-foreground tracking-tight">
                SafeGuard AI
              </span>

              {/* Clean Global IoT Indicator & Overheat Warning */}
              <div className="flex items-center gap-2">
                {iotData.temp === null || !iotData.connected ? (
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono text-muted-foreground/60 bg-muted/20 border border-border/40 select-none">
                    <span className="w-1.5 h-1.5 rounded-full bg-zinc-500/50" />
                    <span>Không có dữ liệu</span>
                  </div>
                ) : iotData.is_over_limit ? (
                  <div className="flex items-center gap-2 px-2.5 py-1 rounded-md text-xs bg-rose-500/10 border border-rose-500/30 text-rose-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-ping" />
                    <span className="font-mono font-medium">{iotData.temp.toFixed(1)}°C</span>
                    <span className="text-[11px] text-rose-400/90 font-sans">Quá nhiệt (&gt;{iotData.temp_limit}°C)</span>
                    <button
                      onClick={toggleAlarm}
                      className={`ml-1 px-2 py-0.5 rounded text-[11px] font-sans font-medium transition-colors cursor-pointer border ${
                        alarmState
                          ? 'bg-rose-500 text-white border-rose-500 hover:bg-rose-600'
                          : 'bg-zinc-800 text-zinc-300 border-zinc-700 hover:bg-zinc-700'
                      }`}
                    >
                      {alarmState ? 'Tắt còi' : 'Bật còi'}
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 px-2.5 py-1 rounded-md text-xs font-mono text-muted-foreground bg-muted/40 border border-border/50">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500/60" />
                    <span>{iotData.temp.toFixed(1)}°C</span>
                    <span className="text-border">·</span>
                    <span>{iotData.hum.toFixed(0)}%</span>
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs text-muted-foreground font-mono tabular-nums">
                {time.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
              </span>

              {/* Theme Toggle Button */}
              <button
                onClick={toggleTheme}
                className="w-7 h-7 rounded-md flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-accent/70 transition-colors cursor-pointer"
                title={theme === 'dark' ? 'Chuyển sang nền sáng' : 'Chuyển sang nền tối'}
              >
                {theme === 'dark' ? (
                  <Sun className="w-3.5 h-3.5 text-amber-400" />
                ) : (
                  <Moon className="w-3.5 h-3.5 text-zinc-600" />
                )}
              </button>
            </div>
          </header>

          {/* Page Content */}
          <main className="flex-1 p-6 overflow-auto">
            <Outlet />
          </main>
        </div>
      </div>
    </TooltipProvider>
  )
}
