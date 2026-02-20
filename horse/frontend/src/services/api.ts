/**
 * API service -- typed fetch wrappers for the Horse 13D backend.
 */

const BASE = '/api';

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

// ---- Types ----

export interface RaceInfo {
  race_id: number;
  meeting_date: string;
  course: string;
  race_number: number | null;
  race_name: string | null;
  race_time: string | null;
  distance_furlongs: number | null;
  race_type: string | null;
  race_class: string | null;
  going: string | null;
  surface: string | null;
  num_runners: number;
  region_code: string | null;
}

export interface RaceCard {
  races: RaceInfo[];
  meeting_date: string;
  course: string | null;
  total_races: number;
  timestamp: string;
}

export interface DimensionScore {
  name: string;
  score: number;
  features: Record<string, number>;
  tooltip: string;
  color: string;
}

export interface DriverItem {
  feature: string;
  value: string;
  impact: 'positive' | 'negative';
  description: string;
}

export interface RunnerPrediction {
  horse_name: string;
  draw: number | null;
  jockey: string | null;
  trainer: string | null;
  age: number | null;
  weight_lbs: number | null;
  official_rating: number | null;
  win_prob: number;
  place_prob: number;
  win_rank: number;
  place_rank: number;
  fair_odds: number;
  back_odds: number | null;
  value_flag: string | null;
  odds_updated_at: string | null;
  silk_url: string | null;
  dimensions: DimensionScore[];
  positive_drivers: DriverItem[];
  negative_drivers: DriverItem[];
}

export interface ThirteenDRace {
  race_info: RaceInfo;
  runners: RunnerPrediction[];
  model_version: string;
  timestamp: string;
  predictions_status: string;
}

export interface DateInfo {
  date: string;
  meetings: number;
  races: number;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  database: string;
  meetings: number;
  races: number;
  results: number;
  horse_form: number;
  engine: string;
}

// Performance / Feedback types

export interface TopPickStats {
  total: number;
  wins: number;
  places: number;
  win_rate: number;
  place_rate: number;
}

export interface Top3Stats {
  total: number;
  placed: number;
  place_rate: number;
}

export interface RoiStats {
  bets: number;
  returns: number;
  roi_pct: number;
  profit_loss: number;
}

export interface CalibrationBucket {
  bucket: string;
  count: number;
  predicted_avg: number;
  actual_win_rate: number;
  gap: number;
}

export interface DailyPerf {
  date: string;
  races: number;
  top_pick_wins: number;
  top_pick_places: number;
  top_pick_total: number;
  win_rate: number;
  roi_pct: number;
}

export interface CoursePerf {
  course: string;
  races: number;
  wins: number;
  win_rate: number;
}

export interface PerformanceSummary {
  total_races: number;
  total_predictions: number;
  days_back: number;
  top_pick: TopPickStats;
  top_3_picks: Top3Stats;
  roi: RoiStats;
  calibration: CalibrationBucket[];
  daily: DailyPerf[];
  courses: CoursePerf[];
  timestamp: string;
  message?: string;
}

export interface PredictionCount {
  total_predictions: number;
  reconciled: number;
  unresolved: number;
  races_predicted: number;
}

// Scrape status

export interface ScrapeStatus {
  running: boolean;
  started_at: string | null;
  finished_at: string | null;
  result: string | null;
  error: string | null;
}

// ---- API calls ----

export const api = {
  getHealth: () => fetchJson<HealthResponse>('/health'),

  getRaceCard: (date?: string, course?: string, region?: string) => {
    const params = new URLSearchParams();
    if (date) params.set('date', date);
    if (course) params.set('course', course);
    if (region && region !== 'ALL') params.set('region', region);
    const qs = params.toString();
    return fetchJson<RaceCard>(`/races${qs ? `?${qs}` : ''}`);
  },

  getAvailableDates: (limit = 30) =>
    fetchJson<{ dates: DateInfo[]; timestamp: string }>(`/races/dates?limit=${limit}`),

  getThirteenD: (raceId: number) =>
    fetchJson<ThirteenDRace>(`/race/${raceId}/thirteen_d`),

  getPerformance: (days = 30) =>
    fetchJson<PerformanceSummary>(`/performance?days=${days}`),

  getPerformanceSummary: () =>
    fetchJson<PredictionCount>('/performance/summary'),

  triggerScrape: async (): Promise<{ status: string; started_at?: string }> => {
    const res = await fetch(`${BASE}/scrape-racecards`, { method: 'POST' });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API ${res.status}: ${text}`);
    }
    return res.json();
  },

  getScrapeStatus: () => fetchJson<ScrapeStatus>('/scrape-status'),

  triggerOddsRefresh: async (): Promise<{ status: string; started_at?: string }> => {
    const res = await fetch(`${BASE}/refresh-odds`, { method: 'POST' });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API ${res.status}: ${text}`);
    }
    return res.json();
  },

  getOddsStatus: () => fetchJson<ScrapeStatus>('/odds-status'),
};
