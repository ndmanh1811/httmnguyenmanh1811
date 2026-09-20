import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Monitor from './pages/Monitor'
import Violations from './pages/Violations'
import Upload from './pages/Upload'
import Settings from './pages/Settings'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="/violations" element={<Violations />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
