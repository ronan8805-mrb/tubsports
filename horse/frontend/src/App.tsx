import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Dashboard } from './pages/Dashboard';
import { RaceDetail } from './pages/RaceDetail';
import { Performance } from './pages/Performance';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 2, refetchOnWindowFocus: false },
  },
});

function NavBar() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
      isActive
        ? 'bg-emerald-600/20 text-emerald-400'
        : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
    }`;

  return (
    <nav className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 flex items-center justify-between h-14">
        <div className="flex items-center gap-3">
          <span className="text-lg font-bold bg-gradient-to-r from-emerald-400 to-blue-500 bg-clip-text text-transparent">
            13D Engine
          </span>
          <span className="text-[10px] text-gray-600 uppercase tracking-widest">
            Horse Racing
          </span>
        </div>
        <div className="flex items-center gap-1">
          <NavLink to="/" className={linkClass} end>
            Races
          </NavLink>
          <NavLink to="/performance" className={linkClass}>
            Performance
          </NavLink>
        </div>
      </div>
    </nav>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-[#0a0e1a]">
          <NavBar />
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/race/:raceId" element={<RaceDetail />} />
            <Route path="/performance" element={<Performance />} />
          </Routes>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
