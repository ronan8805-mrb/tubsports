import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { api } from '../services/api';
import type { PerformanceSummary } from '../services/api';

export function Performance() {
  const [days, setDays] = useState(30);

  const { data, isLoading, error } = useQuery({
    queryKey: ['performance', days],
    queryFn: () => api.getPerformance(days),
    staleTime: 60_000,
  });

  const { data: counts } = useQuery({
    queryKey: ['perfSummary'],
    queryFn: () => api.getPerformanceSummary(),
    staleTime: 60_000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-10">
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400 text-sm">
          {(error as Error).message}
        </div>
      </div>
    );
  }

  if (!data || data.total_races === 0) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-10">
        <h1 className="text-2xl font-bold text-white mb-4">Performance</h1>
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8 text-center">
          <p className="text-gray-400 text-lg mb-2">No reconciled predictions yet</p>
          <p className="text-gray-500 text-sm">
            1. View races in the dashboard to generate predictions<br />
            2. After the races run, scrape results (BACKFILL_HORSE.bat Phase 2)<br />
            3. Come back here to see how the model performed
          </p>
          {counts && (
            <div className="mt-4 flex justify-center gap-6 text-sm">
              <div className="text-center">
                <div className="text-2xl font-mono text-emerald-400">{counts.total_predictions}</div>
                <div className="text-gray-500">Predictions saved</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-mono text-amber-400">{counts.unresolved}</div>
                <div className="text-gray-500">Awaiting results</div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  const { top_pick, top_3_picks, roi, calibration, daily, courses } = data;

  return (
    <div className="max-w-5xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">Performance</h1>
          <p className="text-sm text-gray-500">
            Prediction track record -- {data.total_races} races, {data.total_predictions} predictions
          </p>
        </div>
        <select
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          className="bg-gray-800 border border-gray-700 text-gray-200 rounded-lg px-3 py-2 text-sm"
        >
          <option value={7}>Last 7 days</option>
          <option value={14}>Last 14 days</option>
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
          <option value={365}>Last year</option>
        </select>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <StatCard
          label="Top Pick Win Rate"
          value={`${(top_pick.win_rate * 100).toFixed(1)}%`}
          sub={`${top_pick.wins}/${top_pick.total} wins`}
          color={top_pick.win_rate > 0.2 ? 'emerald' : top_pick.win_rate > 0.1 ? 'amber' : 'red'}
        />
        <StatCard
          label="Top Pick Place Rate"
          value={`${(top_pick.place_rate * 100).toFixed(1)}%`}
          sub={`${top_pick.places}/${top_pick.total} placed`}
          color={top_pick.place_rate > 0.5 ? 'emerald' : top_pick.place_rate > 0.3 ? 'amber' : 'red'}
        />
        <StatCard
          label="Top 3 Place Rate"
          value={`${(top_3_picks.place_rate * 100).toFixed(1)}%`}
          sub={`${top_3_picks.placed}/${top_3_picks.total}`}
          color={top_3_picks.place_rate > 0.4 ? 'emerald' : 'amber'}
        />
        <StatCard
          label="ROI (flat stake)"
          value={`${roi.roi_pct >= 0 ? '+' : ''}${roi.roi_pct.toFixed(1)}%`}
          sub={`P/L: ${roi.profit_loss >= 0 ? '+' : ''}${roi.profit_loss.toFixed(2)}`}
          color={roi.roi_pct > 0 ? 'emerald' : roi.roi_pct > -10 ? 'amber' : 'red'}
        />
      </div>

      {/* Daily Breakdown */}
      {daily.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 mb-6">
          <h2 className="text-sm font-semibold text-white mb-3 uppercase tracking-wider">
            Daily Breakdown
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-xs uppercase tracking-wider">
                  <th className="text-left py-2 pr-4">Date</th>
                  <th className="text-center py-2 px-2">Races</th>
                  <th className="text-center py-2 px-2">Wins</th>
                  <th className="text-center py-2 px-2">Places</th>
                  <th className="text-center py-2 px-2">Win Rate</th>
                  <th className="text-right py-2 pl-2">ROI</th>
                </tr>
              </thead>
              <tbody>
                {daily.map((d) => (
                  <motion.tr
                    key={d.date}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="border-t border-gray-800/50"
                  >
                    <td className="py-2 pr-4 text-gray-300 font-mono text-xs">{d.date}</td>
                    <td className="py-2 px-2 text-center text-gray-400">{d.races}</td>
                    <td className="py-2 px-2 text-center">
                      <span className={d.top_pick_wins > 0 ? 'text-emerald-400' : 'text-gray-500'}>
                        {d.top_pick_wins}/{d.top_pick_total}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-center text-gray-400">
                      {d.top_pick_places}/{d.top_pick_total}
                    </td>
                    <td className="py-2 px-2 text-center">
                      <WinRateBar rate={d.win_rate} />
                    </td>
                    <td className={`py-2 pl-2 text-right font-mono text-xs ${
                      d.roi_pct > 0 ? 'text-emerald-400' : d.roi_pct < -20 ? 'text-red-400' : 'text-gray-400'
                    }`}>
                      {d.roi_pct >= 0 ? '+' : ''}{d.roi_pct.toFixed(1)}%
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Calibration + Courses side by side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Calibration */}
        {calibration.length > 0 && (
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
            <h2 className="text-sm font-semibold text-white mb-3 uppercase tracking-wider">
              Calibration
            </h2>
            <p className="text-xs text-gray-500 mb-3">
              When we predict X%, how often does the horse actually win?
            </p>
            <div className="space-y-2">
              {calibration.map((c) => (
                <div key={c.bucket} className="flex items-center gap-3 text-xs">
                  <span className="text-gray-400 w-16 text-right font-mono">{c.bucket}</span>
                  <div className="flex-1 relative h-5">
                    {/* Predicted bar */}
                    <div
                      className="absolute inset-y-0 left-0 bg-blue-500/20 border border-blue-500/30 rounded"
                      style={{ width: `${Math.min(c.predicted_avg * 100, 100) * 2}%` }}
                    />
                    {/* Actual bar */}
                    <div
                      className="absolute inset-y-0 left-0 bg-emerald-500/40 rounded"
                      style={{ width: `${Math.min(c.actual_win_rate * 100, 100) * 2}%` }}
                    />
                  </div>
                  <span className="text-gray-500 w-12 text-right">{c.count}</span>
                  <span className={`w-14 text-right font-mono ${
                    Math.abs(c.gap) < 0.05 ? 'text-emerald-400' : 'text-amber-400'
                  }`}>
                    {c.gap >= 0 ? '+' : ''}{(c.gap * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
              <div className="flex items-center gap-3 text-[10px] text-gray-600 mt-2">
                <span className="w-16" />
                <div className="flex gap-3">
                  <span className="flex items-center gap-1">
                    <span className="w-3 h-2 bg-blue-500/30 rounded" /> Predicted
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-3 h-2 bg-emerald-500/40 rounded" /> Actual
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Top courses */}
        {courses.length > 0 && (
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
            <h2 className="text-sm font-semibold text-white mb-3 uppercase tracking-wider">
              Top Courses
            </h2>
            <p className="text-xs text-gray-500 mb-3">
              Where the model performs best (min 3 races)
            </p>
            <div className="space-y-2">
              {courses.map((c, i) => (
                <div key={c.course} className="flex items-center gap-3 text-xs">
                  <span className={`w-5 text-center font-bold ${
                    i === 0 ? 'text-amber-400' : 'text-gray-500'
                  }`}>
                    {i + 1}
                  </span>
                  <span className="text-gray-300 flex-1 truncate">{c.course}</span>
                  <span className="text-gray-500">{c.wins}/{c.races}</span>
                  <span className="font-mono w-12 text-right text-emerald-400">
                    {(c.win_rate * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Prediction counts */}
      {counts && (
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 text-center text-xs text-gray-500">
          {counts.total_predictions} total predictions saved | {counts.reconciled} reconciled | {counts.unresolved} awaiting results
        </div>
      )}
    </div>
  );
}


function StatCard({ label, value, sub, color }: {
  label: string; value: string; sub: string; color: 'emerald' | 'amber' | 'red';
}) {
  const colors = {
    emerald: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
    amber: 'bg-amber-500/10 border-amber-500/20 text-amber-400',
    red: 'bg-red-500/10 border-red-500/20 text-red-400',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl border p-4 ${colors[color]}`}
    >
      <div className="text-[10px] uppercase tracking-wider opacity-70 mb-1">{label}</div>
      <div className="text-2xl font-bold font-mono">{value}</div>
      <div className="text-xs opacity-60 mt-0.5">{sub}</div>
    </motion.div>
  );
}


function WinRateBar({ rate }: { rate: number }) {
  const pct = rate * 100;
  return (
    <div className="flex items-center gap-1.5 justify-center">
      <div className="w-16 h-2 bg-gray-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(pct, 100)}%` }}
          transition={{ duration: 0.5 }}
          className={`h-full rounded-full ${
            pct >= 25 ? 'bg-emerald-500' : pct >= 15 ? 'bg-amber-500' : 'bg-red-500'
          }`}
        />
      </div>
      <span className="text-[10px] font-mono text-gray-400 w-8 text-right">
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}
