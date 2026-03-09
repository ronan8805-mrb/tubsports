import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';

interface AuthUser {
  username: string;
  role: 'admin' | 'member';
  token: string;
}

interface AuthContextType {
  user: AuthUser | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  isAdmin: boolean;
  authHeader: () => Record<string, string>;
}

const AuthContext = createContext<AuthContextType | null>(null);

const STORAGE_KEY = 'horse_auth';

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as AuthUser;
        fetch('/api/auth/me', {
          headers: { Authorization: `Bearer ${parsed.token}` },
        })
          .then((r) => {
            if (r.ok) {
              setUser(parsed);
            } else {
              localStorage.removeItem(STORAGE_KEY);
            }
          })
          .catch(() => localStorage.removeItem(STORAGE_KEY))
          .finally(() => setLoading(false));
      } catch {
        localStorage.removeItem(STORAGE_KEY);
        setLoading(false);
      }
    } else {
      setLoading(false);
    }
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Login failed');
    }
    const data = await res.json();
    const authUser: AuthUser = {
      username: data.username,
      role: data.role,
      token: data.token,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(authUser));
    setUser(authUser);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setUser(null);
  }, []);

  const authHeader = useCallback((): Record<string, string> => {
    if (!user) return {} as Record<string, string>;
    return { Authorization: `Bearer ${user.token}` };
  }, [user]);

  return (
    <AuthContext.Provider
      value={{ user, loading, login, logout, isAdmin: user?.role === 'admin', authHeader }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
