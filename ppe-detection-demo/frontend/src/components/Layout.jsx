import { Outlet, NavLink } from 'react-router-dom'
import { LayoutGrid, Monitor, FileText, Upload, Settings, Sun, Moon } from 'lucide-react'
import { TooltipProvider, Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip'
import { useState, useEffect } from 'react'

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
            <span className="text-sm font-semibold text-foreground tracking-tight">
              SafeGuard AI
            </span>
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
