import { useState, type FormEvent } from 'react';
import { useAuth } from '../contexts/AuthContext';

type Panel = 'member' | 'admin';

export function Login() {
  const { login } = useAuth();
  const [panel, setPanel] = useState<Panel>('member');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setSubmitting(true);
    try {
      await login(username, password);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setSubmitting(false);
    }
  };

  const switchPanel = (p: Panel) => {
    setPanel(p);
    setUsername('');
    setPassword('');
    setError('');
  };

  const isMember = panel === 'member';

  return (
    <div
      className="min-h-screen flex items-center justify-center p-4"
      style={{
        backgroundImage: 'url(/login-bg.jpg)',
        backgroundSize: 'cover',
        backgroundPosition: 'center center',
        backgroundRepeat: 'no-repeat',
        backgroundColor: '#0a0a0a',
      }}
    >
      <div className="w-full max-w-xs">
        {/* Logo */}
        <div className="text-center mb-5">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-emerald-400 to-blue-500 bg-clip-text text-transparent drop-shadow-lg">
            13D Engine
          </h1>
          <p className="text-[10px] text-gray-300 uppercase tracking-[0.3em] mt-1 drop-shadow">
            Horse Racing Intelligence
          </p>
        </div>

        {/* Panel Switcher */}
        <div className="flex mb-3 bg-black/40 backdrop-blur-md p-1 rounded-xl">
          <button
            onClick={() => switchPanel('member')}
            className={`flex-1 py-2 rounded-lg text-xs font-medium transition-all ${
              isMember
                ? 'bg-emerald-600/30 text-emerald-400 shadow-md'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Member
          </button>
          <button
            onClick={() => switchPanel('admin')}
            className={`flex-1 py-2 rounded-lg text-xs font-medium transition-all ${
              !isMember
                ? 'bg-amber-600/30 text-amber-400 shadow-md'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Admin
          </button>
        </div>

        {/* Login Card */}
        <form
          onSubmit={handleSubmit}
          className="bg-black/40 backdrop-blur-md border border-white/10 rounded-2xl p-5 space-y-4"
        >
          <div className="flex items-center gap-2">
            <div className={`w-1.5 h-1.5 rounded-full ${isMember ? 'bg-emerald-500' : 'bg-amber-500'}`} />
            <span className={`text-[10px] font-medium uppercase tracking-wider ${isMember ? 'text-emerald-400' : 'text-amber-400'}`}>
              {isMember ? 'Member Access' : 'Admin Access'}
            </span>
          </div>

          <div>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              required
              className="w-full bg-white/10 backdrop-blur border border-white/15 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
              placeholder="Username"
            />
          </div>

          <div>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
              className="w-full bg-white/10 backdrop-blur border border-white/15 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
              placeholder="Password"
            />
          </div>

          {error && (
            <div className="bg-red-500/20 border border-red-500/30 rounded-lg px-3 py-2 text-xs text-red-400">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting}
            className={`w-full font-medium py-2 rounded-lg text-sm transition-colors disabled:opacity-40 ${
              isMember
                ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                : 'bg-amber-600 hover:bg-amber-500 text-black font-bold'
            }`}
          >
            {submitting ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <p className="text-center text-[10px] text-gray-300/60 mt-4 drop-shadow">
          Private access only
        </p>
      </div>
    </div>
  );
}
