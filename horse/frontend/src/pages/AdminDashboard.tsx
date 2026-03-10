import { useState, useEffect, useRef, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { api } from '../services/api';

const API_BASE = import.meta.env.VITE_API_URL || '';

interface UserInfo {
  user_id: number;
  username: string;
  role: string;
  created_at: string;
}

export function AdminDashboard() {
  const { authHeader } = useAuth();
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [newUser, setNewUser] = useState('');
  const [newPass, setNewPass] = useState('');
  const [newRole, setNewRole] = useState('member');
  const [creating, setCreating] = useState(false);
  const [msg, setMsg] = useState<{ text: string; type: 'ok' | 'err' } | null>(null);

  // Scrape state
  const [scraping, setScraping] = useState(false);
  const [scrapeMsg, setScrapeMsg] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Odds refresh state
  const [refreshingOdds, setRefreshingOdds] = useState(false);
  const [oddsMsg, setOddsMsg] = useState<string | null>(null);
  const oddsPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // DB stats
  const [health, setHealth] = useState<any>(null);

  // Security events
  const [securityEvents, setSecurityEvents] = useState<any[]>([]);

  const loadSecurityEvents = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/auth/security-events`, { headers: authHeader() });
      if (res.ok) setSecurityEvents(await res.json());
    } catch {}
  }, [authHeader]);

  const handleResetDevice = async (uid: number, uname: string) => {
    if (!confirm(`Reset device lock for "${uname}"? They will be able to log in from a new device.`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/auth/users/${uid}/reset-device`, {
        method: 'POST',
        headers: authHeader(),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || 'Failed');
      }
      setMsg({ text: `Device reset for "${uname}"`, type: 'ok' });
      loadSecurityEvents();
    } catch (e) {
      setMsg({ text: e instanceof Error ? e.message : 'Failed', type: 'err' });
    }
    setTimeout(() => setMsg(null), 4000);
  };

  const loadUsers = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/auth/users`, { headers: authHeader() });
      if (res.ok) setUsers(await res.json());
    } catch {}
  }, [authHeader]);

  useEffect(() => {
    loadUsers();
    loadSecurityEvents();
    api.getHealth().then(setHealth).catch(() => {});
  }, [loadUsers, loadSecurityEvents]);

  const handleCreate = async () => {
    if (!newUser.trim() || !newPass.trim()) return;
    setCreating(true);
    setMsg(null);
    try {
      const res = await fetch(`${API_BASE}/api/auth/users`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeader() },
        body: JSON.stringify({ username: newUser, password: newPass, role: newRole }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || 'Failed');
      }
      setMsg({ text: `User "${newUser}" created`, type: 'ok' });
      setNewUser('');
      setNewPass('');
      loadUsers();
    } catch (e) {
      setMsg({ text: e instanceof Error ? e.message : 'Failed', type: 'err' });
    } finally {
      setCreating(false);
      setTimeout(() => setMsg(null), 4000);
    }
  };

  const handleDelete = async (uid: number, uname: string) => {
    if (!confirm(`Delete user "${uname}"?`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/auth/users/${uid}`, {
        method: 'DELETE',
        headers: authHeader(),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || 'Failed');
      }
      setMsg({ text: `User "${uname}" deleted`, type: 'ok' });
      loadUsers();
    } catch (e) {
      setMsg({ text: e instanceof Error ? e.message : 'Failed', type: 'err' });
    }
    setTimeout(() => setMsg(null), 4000);
  };

  const stopPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);

  const stopOddsPolling = useCallback(() => {
    if (oddsPollRef.current) { clearInterval(oddsPollRef.current); oddsPollRef.current = null; }
  }, []);

  useEffect(() => () => { stopPolling(); stopOddsPolling(); }, [stopPolling, stopOddsPolling]);

  const handleScrape = useCallback(async () => {
    try {
      setScraping(true);
      setScrapeMsg(null);
      const res = await fetch(`${API_BASE}/api/scrape-racecards`, {
        method: 'POST',
        headers: authHeader(),
      });
      if (!res.ok) throw new Error('Failed to start scrape');
      const data = await res.json();
      if (data.status === 'already_running') setScrapeMsg('Already fetching...');

      pollRef.current = setInterval(async () => {
        try {
          const status = await api.getScrapeStatus();
          if (!status.running) {
            stopPolling();
            setScraping(false);
            setScrapeMsg(status.result === 'success' ? 'Races loaded!' : status.error || 'Done');
            setTimeout(() => setScrapeMsg(null), 5000);
          }
        } catch {
          stopPolling();
          setScraping(false);
        }
      }, 3000);
    } catch (err) {
      setScraping(false);
      setScrapeMsg(err instanceof Error ? err.message : 'Failed');
      setTimeout(() => setScrapeMsg(null), 5000);
    }
  }, [authHeader, stopPolling]);

  const handleRefreshOdds = useCallback(async () => {
    try {
      setRefreshingOdds(true);
      setOddsMsg(null);
      const res = await fetch(`${API_BASE}/api/refresh-odds`, {
        method: 'POST',
        headers: authHeader(),
      });
      if (!res.ok) throw new Error('Failed');
      oddsPollRef.current = setInterval(async () => {
        try {
          const status = await api.getOddsStatus();
          if (!status.running) {
            stopOddsPolling();
            setRefreshingOdds(false);
            setOddsMsg(status.result === 'success' ? 'Odds refreshed!' : status.error || 'Done');
            setTimeout(() => setOddsMsg(null), 5000);
          }
        } catch {
          stopOddsPolling();
          setRefreshingOdds(false);
        }
      }, 3000);
    } catch (err) {
      setRefreshingOdds(false);
      setOddsMsg(err instanceof Error ? err.message : 'Failed');
      setTimeout(() => setOddsMsg(null), 5000);
    }
  }, [authHeader, stopOddsPolling]);

  return (
    <div className="max-w-4xl mx-auto px-4 py-6">
      <h1 className="text-2xl font-bold text-white mb-1">Admin Dashboard</h1>
      <p className="text-sm text-gray-500 mb-6">System controls & user management</p>

      {/* System Stats */}
      {health && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          {[
            { label: 'Races', value: health.races?.toLocaleString() },
            { label: 'Results', value: health.results?.toLocaleString() },
            { label: 'Meetings', value: health.meetings?.toLocaleString() },
            { label: 'Form', value: health.horse_form?.toLocaleString() },
          ].map((s) => (
            <div key={s.label} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <p className="text-xs text-gray-500">{s.label}</p>
              <p className="text-lg font-bold text-white mt-1">{s.value ?? '—'}</p>
            </div>
          ))}
        </div>
      )}

      {/* System Controls */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 mb-6">
        <h2 className="text-sm font-semibold text-gray-300 mb-4">System Controls</h2>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={handleScrape}
            disabled={scraping}
            className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
              scraping
                ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white'
            }`}
          >
            {scraping ? (
              <>
                <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Fetching...
              </>
            ) : 'Fetch New Races'}
          </button>
          <button
            onClick={handleRefreshOdds}
            disabled={refreshingOdds}
            className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
              refreshingOdds
                ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-500 text-white'
            }`}
          >
            {refreshingOdds ? 'Refreshing...' : 'Refresh Odds'}
          </button>
        </div>
        {scrapeMsg && (
          <p className={`mt-3 text-sm ${scrapeMsg.includes('loaded') ? 'text-emerald-400' : 'text-amber-400'}`}>
            {scrapeMsg}
          </p>
        )}
        {oddsMsg && (
          <p className={`mt-3 text-sm ${oddsMsg.includes('refreshed') ? 'text-blue-400' : 'text-amber-400'}`}>
            {oddsMsg}
          </p>
        )}
      </div>

      {/* Create User */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 mb-6">
        <h2 className="text-sm font-semibold text-gray-300 mb-4">Create User</h2>
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            placeholder="Username"
            value={newUser}
            onChange={(e) => setNewUser(e.target.value)}
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
          />
          <input
            type="password"
            placeholder="Password"
            value={newPass}
            onChange={(e) => setNewPass(e.target.value)}
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
          />
          <select
            value={newRole}
            onChange={(e) => setNewRole(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-gray-200 focus:outline-none"
          >
            <option value="member">Member</option>
            <option value="admin">Admin</option>
          </select>
          <button
            onClick={handleCreate}
            disabled={creating || !newUser.trim() || !newPass.trim()}
            className="bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 disabled:text-gray-500 text-white font-medium px-5 py-2.5 rounded-lg text-sm transition-colors whitespace-nowrap"
          >
            {creating ? 'Creating...' : 'Create'}
          </button>
        </div>
        {msg && (
          <p className={`mt-3 text-sm ${msg.type === 'ok' ? 'text-emerald-400' : 'text-red-400'}`}>
            {msg.text}
          </p>
        )}
      </div>

      {/* Users List */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h2 className="text-sm font-semibold text-gray-300 mb-4">
          Users ({users.length})
        </h2>
        <div className="space-y-2">
          {users.map((u) => (
            <div
              key={u.user_id}
              className="flex items-center justify-between bg-gray-800/50 rounded-lg px-4 py-3"
            >
              <div className="flex items-center gap-3">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
                    u.role === 'admin'
                      ? 'bg-amber-500/20 text-amber-400'
                      : 'bg-emerald-500/20 text-emerald-400'
                  }`}
                >
                  {u.username[0].toUpperCase()}
                </div>
                <div>
                  <p className="text-sm font-medium text-white">{u.username}</p>
                  <p className="text-[11px] text-gray-500">
                    {u.role === 'admin' ? 'Admin' : 'Member'} &middot; {u.created_at.split('T')[0]}
                  </p>
                </div>
              </div>
              {u.role !== 'admin' && (
                <div className="flex gap-2">
                  <button
                    onClick={() => handleResetDevice(u.user_id, u.username)}
                    className="text-xs text-amber-400 hover:text-amber-300 px-3 py-1 rounded hover:bg-amber-500/10 transition-colors"
                  >
                    Reset Device
                  </button>
                  <button
                    onClick={() => handleDelete(u.user_id, u.username)}
                    className="text-xs text-red-400 hover:text-red-300 px-3 py-1 rounded hover:bg-red-500/10 transition-colors"
                  >
                    Remove
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Security Alerts */}
      <div className="bg-gray-900/60 rounded-xl p-5 border border-gray-700/50">
        <h2 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
          <span className="text-red-400">&#9888;</span> Security Alerts
        </h2>
        {securityEvents.length === 0 ? (
          <p className="text-sm text-gray-500">No security events</p>
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {securityEvents.map((ev) => (
              <div
                key={ev.event_id}
                className={`flex items-start justify-between rounded-lg px-4 py-3 text-sm ${
                  ev.event_type === 'device_mismatch'
                    ? 'bg-red-500/10 border border-red-500/30'
                    : ev.event_type === 'failed_login'
                    ? 'bg-amber-500/10 border border-amber-500/30'
                    : 'bg-gray-800/50 border border-gray-700/30'
                }`}
              >
                <div>
                  <p className="font-medium text-white">
                    {ev.event_type === 'device_mismatch' && 'Device Mismatch'}
                    {ev.event_type === 'failed_login' && 'Failed Login'}
                    {ev.event_type === 'rate_limited' && 'Rate Limited'}
                    {ev.event_type === 'device_reset' && 'Device Reset'}
                    <span className="text-gray-400 font-normal ml-2">@{ev.username}</span>
                  </p>
                  <p className="text-xs text-gray-400 mt-0.5">{ev.detail}</p>
                </div>
                <span className="text-[11px] text-gray-500 whitespace-nowrap ml-4">
                  {ev.created_at?.split('.')[0]}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
