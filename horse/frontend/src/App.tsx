import { useState } from 'react';
import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { Login } from './pages/Login';
import { Dashboard } from './pages/Dashboard';
import { RaceDetail } from './pages/RaceDetail';
import { Performance } from './pages/Performance';
import { BestBets } from './pages/BestBets';
import { AdminDashboard } from './pages/AdminDashboard';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 2, refetchOnWindowFocus: false },
  },
});

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0e1a] flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function AdminRoute({ children }: { children: React.ReactNode }) {
  const { user, loading, isAdmin } = useAuth();
  if (loading) return null;
  if (!user) return <Navigate to="/login" replace />;
  if (!isAdmin) return <Navigate to="/" replace />;
  return <>{children}</>;
}

function NavBar() {
  const { user, logout, isAdmin } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);

  if (!user) return null;

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
      isActive
        ? 'bg-emerald-600/20 text-emerald-400'
        : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
    }`;

  const mobileLinkClass = ({ isActive }: { isActive: boolean }) =>
    `block px-4 py-3 text-sm font-medium transition-colors ${
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
          <span className="text-[10px] text-gray-600 uppercase tracking-widest hidden sm:inline">
            Horse Racing
          </span>
        </div>

        {/* Desktop nav */}
        <div className="hidden md:flex items-center gap-1">
          <NavLink to="/" className={linkClass} end>
            Races
          </NavLink>
          <NavLink to="/best-bets" className={linkClass}>
            Best Bets
          </NavLink>
          <NavLink to="/performance" className={linkClass}>
            Performance
          </NavLink>
          {isAdmin && (
            <NavLink to="/admin" className={linkClass}>
              Admin
            </NavLink>
          )}
          <button
            onClick={logout}
            className="ml-3 px-3 py-1.5 rounded-lg text-sm font-medium text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
          >
            Logout
          </button>
        </div>

        {/* Mobile hamburger */}
        <button
          onClick={() => setMenuOpen(!menuOpen)}
          className="md:hidden p-2 text-gray-400 hover:text-white"
        >
          {menuOpen ? (
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          )}
        </button>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden border-t border-gray-800 bg-gray-900">
          <NavLink to="/" className={mobileLinkClass} end onClick={() => setMenuOpen(false)}>
            Races
          </NavLink>
          <NavLink to="/best-bets" className={mobileLinkClass} onClick={() => setMenuOpen(false)}>
            Best Bets
          </NavLink>
          <NavLink to="/performance" className={mobileLinkClass} onClick={() => setMenuOpen(false)}>
            Performance
          </NavLink>
          {isAdmin && (
            <NavLink to="/admin" className={mobileLinkClass} onClick={() => setMenuOpen(false)}>
              Admin
            </NavLink>
          )}
          <button
            onClick={() => { logout(); setMenuOpen(false); }}
            className="w-full text-left px-4 py-3 text-sm font-medium text-red-400 hover:bg-red-500/10 transition-colors"
          >
            Logout
          </button>
        </div>
      )}
    </nav>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <div className="min-h-screen bg-[#0a0e1a]">
            <NavBar />
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
              <Route path="/best-bets" element={<ProtectedRoute><BestBets /></ProtectedRoute>} />
              <Route path="/race/:raceId" element={<ProtectedRoute><RaceDetail /></ProtectedRoute>} />
              <Route path="/performance" element={<ProtectedRoute><Performance /></ProtectedRoute>} />
              <Route path="/admin" element={<AdminRoute><AdminDashboard /></AdminRoute>} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}
