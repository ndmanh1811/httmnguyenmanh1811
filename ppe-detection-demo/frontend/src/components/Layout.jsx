import { Outlet, NavLink } from 'react-router-dom'
import { LayoutDashboard, Monitor, AlertTriangle, Upload, Settings, HardHat } from 'lucide-react'

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/monitor', icon: Monitor, label: 'Giám sát' },
  { to: '/violations', icon: AlertTriangle, label: 'Vi phạm' },
  { to: '/upload', icon: Upload, label: 'Phân tích video' },
  { to: '/settings', icon: Settings, label: 'Cài đặt' },
]

export default function Layout() {
  return (
    <div className="min-h-screen bg-gray-950 text-white flex">
      <aside className="w-64 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800 flex items-center gap-3">
          <HardHat className="w-8 h-8 text-yellow-400" />
          <div>
            <h1 className="font-bold text-lg">PPE Detection</h1>
            <p className="text-xs text-gray-500">Construction Safety AI</p>
          </div>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                  isActive
                    ? 'bg-sky-600/20 text-sky-400 font-medium'
                    : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                }`
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="p-4 border-t border-gray-800 text-xs text-gray-600">
          v2.0 — 3-Path Hybrid Fall Detection
        </div>
      </aside>
      <main className="flex-1 p-6 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
