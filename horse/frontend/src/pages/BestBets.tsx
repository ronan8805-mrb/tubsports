import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { api, BestBetRunner } from '../services/api';
import { useAvailableDates } from '../hooks/useRaceCard';

type BetTab = 'nap' | 'trixie' | 'yankee' | 'lucky15';

const TABS: { id: BetTab; label: string; desc: string }[] = [
  { id: 'nap', label: 'NAP', desc: 'Best single pick' },
  { id: 'trixie', label: 'E/W Trixie', desc: '3 picks \u00b7 4 bets' },
  { id: 'yankee', label: 'E/W Yankee', desc: '4 picks \u00b7 11 bets' },
  { id: 'lucky15', label: 'Lucky 15', desc: '4 picks \u00b7 15 bets' },
];

const BET_INFO: Record<BetTab, { picks: number; summary: string }> = {
  nap: { picks: 1, summary: 'Your single strongest selection of the day. Back it to win or each way.' },
  trixie: { picks: 3, summary: '3 doubles + 1 treble, each way. All 3 must place for full payout.' },
  yankee: { picks: 4, summary: '6 doubles + 4 trebles + 1 fourfold, each way. 2+ must place to see returns.' },
  lucky15: { picks: 4, summary: '4 singles + 6 doubles + 4 trebles + 1 fourfold, each way. 1 winner returns.' },
};

function PickCard({ pick, rank, isNap }: { pick: BestBetRunner; rank: number; isNap?: boolean }) {
  const navigate = useNavigate();

  return (
    <div
      onClick={() => navigate(`/race/${pick.race_id}`)}
      className={`relative border rounded-2xl p-6 cursor-pointer transition-all group ${
        isNap
          ? 'bg-gradient-to-br from-amber-900/30 to-gray-900/80 border-amber-600/40 hover:border-amber-400/60 hover:shadow-lg hover:shadow-amber-500/10'
          : 'bg-gradient-to-br from-gray-800/80 to-gray-900/80 border-gray-700/50 hover:border-emerald-500/50 hover:shadow-lg hover:shadow-emerald-500/10'
      }`}
    >
      {isNap && (
        <div className="absolute top-4 right-4 bg-amber-600 text-black text-[10px] font-bold uppercase tracking-wider px-3 py-1 rounded-full">
          NAP of the Day
        </div>
      )}

      <div className="flex items-start gap-4">
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-lg shrink-0 ${
          isNap ? 'bg-amber-600/20 text-amber-400' : 'bg-emerald-600/20 text-emerald-400'
        }`}>
          {rank}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className={`text-xl font-bold transition-colors truncate ${
            isNap ? 'text-white group-hover:text-amber-400' : 'text-white group-hover:text-emerald-400'
          }`}>
            {pick.horse_name}
          </h3>
          <div className="flex items-center gap-2 mt-1 text-sm text-gray-400">
            <span className="font-medium text-blue-400">{pick.course}</span>
            {pick.race_time && (
              <>
                <span className="text-gray-600">&middot;</span>
                <span>{pick.race_time}</span>
              </>
            )}
          </div>
          {pick.race_name && (
            <p className="text-xs text-gray-500 mt-1 truncate">{pick.race_name}</p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5">
        <div className="bg-gray-900/60 rounded-xl p-3 text-center">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">Win</div>
          <div className="text-lg font-bold text-emerald-400">
            {(pick.win_prob * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-gray-900/60 rounded-xl p-3 text-center">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">Place</div>
          <div className="text-lg font-bold text-blue-400">
            {(pick.place_prob * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-gray-900/60 rounded-xl p-3 text-center">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">% Gap</div>
          <div className={`text-lg font-bold ${pick.pct_gap > 0.25 ? 'text-purple-400' : 'text-gray-400'}`}>
            {(pick.pct_gap * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-gray-900/60 rounded-xl p-3 text-center">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">Fair Odds</div>
          <div className="text-lg font-bold text-amber-400">
            {pick.fair_odds.toFixed(1)}
          </div>
        </div>
      </div>

      {pick.back_odds && (
        <div className="flex items-center gap-3 mt-4 px-3 py-2 bg-gray-900/60 rounded-xl flex-wrap">
          <span className={`text-xs font-bold px-2 py-0.5 rounded-md ${
            pick.value_flag === 'VALUE'
              ? 'bg-emerald-600/20 text-emerald-400'
              : pick.value_flag === 'SHORT'
              ? 'bg-orange-600/20 text-orange-400'
              : 'bg-gray-700/50 text-gray-400'
          }`}>
            {pick.value_flag}
          </span>
          <span className="text-sm font-semibold text-white">
            @ {pick.back_odds.toFixed(2)}
          </span>
          {pick.morning_price && pick.morning_price !== pick.back_odds && (
            <span className={`text-xs font-medium ${
              (pick.market_move_pct ?? 0) > 0 ? 'text-emerald-400' : 'text-red-400'
            }`}>
              {(pick.market_move_pct ?? 0) > 0 ? '▼' : '▲'} from {pick.morning_price.toFixed(2)}
              {pick.steam_flag && (
                <span className="ml-1 bg-emerald-500/20 text-emerald-300 text-[10px] font-bold px-1.5 py-0.5 rounded-full">
                  STEAM
                </span>
              )}
            </span>
          )}
          <span className="text-gray-600 text-xs">·</span>
          <span className="text-xs text-gray-500">
            Harmony{' '}
            <span className={`font-semibold ${
              (pick.harmony_score ?? 0) >= 80
                ? 'text-emerald-400'
                : (pick.harmony_score ?? 0) >= 60
                ? 'text-yellow-400'
                : 'text-red-400'
            }`}>
              {pick.harmony_score != null ? `${pick.harmony_score.toFixed(0)}%` : '—'}
            </span>
          </span>
        </div>
      )}

      <div className="flex items-center gap-4 mt-4 text-xs text-gray-400">
        {pick.jockey && (
          <span><span className="text-gray-600">J:</span> {pick.jockey}</span>
        )}
        {pick.trainer && (
          <span><span className="text-gray-600">T:</span> {pick.trainer}</span>
        )}
        {pick.official_rating && (
          <span><span className="text-gray-600">OR:</span> {pick.official_rating}</span>
        )}
      </div>

      <div className="mt-3 px-3 py-1.5 bg-emerald-600/10 border border-emerald-600/20 rounded-lg">
        <p className="text-xs text-emerald-400">{pick.reason}</p>
      </div>
    </div>
  );
}

function SelectionBanner({ tab, picks }: { tab: BetTab; picks: BestBetRunner[] }) {
  const info = BET_INFO[tab];
  const shown = picks.slice(0, info.picks);

  if (shown.length === 0) return null;

  const colors: Record<BetTab, string> = {
    nap: 'from-amber-900/30 to-amber-800/10 border-amber-700/30',
    trixie: 'from-emerald-900/30 to-blue-900/30 border-emerald-700/30',
    yankee: 'from-blue-900/30 to-purple-900/30 border-blue-700/30',
    lucky15: 'from-purple-900/30 to-pink-900/30 border-purple-700/30',
  };

  const labelColors: Record<BetTab, string> = {
    nap: 'text-amber-400',
    trixie: 'text-emerald-400',
    yankee: 'text-blue-400',
    lucky15: 'text-purple-400',
  };

  return (
    <div className={`bg-gradient-to-r ${colors[tab]} border rounded-2xl p-5 mb-6`}>
      <h2 className={`text-sm font-semibold ${labelColors[tab]} uppercase tracking-wider mb-2`}>
        {TABS.find(t => t.id === tab)?.label}
      </h2>
      <div className="flex items-center gap-3 flex-wrap">
        {shown.map((p, i) => (
          <span key={i} className="text-white font-bold text-lg">
            {p.horse_name}
            {i < shown.length - 1 && (
              <span className="text-gray-600 mx-2">&times;</span>
            )}
          </span>
        ))}
      </div>
      <p className="text-xs text-gray-500 mt-2">{info.summary}</p>
    </div>
  );
}

export function BestBets() {
  const { data: datesData } = useAvailableDates(30);
  const dates = datesData?.dates || [];

  const [selectedDate, setSelectedDate] = useState('');
  const [activeTab, setActiveTab] = useState<BetTab>('trixie');
  const activeDate = selectedDate || (dates.length > 0 ? dates[0].date : '');

  const { data, isLoading, error } = useQuery({
    queryKey: ['bestBets', activeDate],
    queryFn: () => api.getBestBets(activeDate),
    enabled: !!activeDate,
    staleTime: 120_000,
  });

  const picks = data?.picks || [];
  const tabInfo = BET_INFO[activeTab];
  const visiblePicks = picks.slice(0, tabInfo.picks);

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-blue-500 bg-clip-text text-transparent">
          Best Bets
        </h1>
        <p className="text-gray-500 mt-1">
          AI-powered selections across all meetings
        </p>
      </div>

      <div className="mb-6">
        <select
          value={activeDate}
          onChange={(e) => setSelectedDate(e.target.value)}
          className="bg-gray-800 border border-gray-700 text-gray-200 rounded-lg px-4 py-2.5 text-sm focus:ring-emerald-500 focus:border-emerald-500"
        >
          {dates.map((d) => (
            <option key={d.date} value={d.date}>
              {new Date(d.date + 'T12:00:00').toLocaleDateString('en-GB', {
                weekday: 'short',
                day: 'numeric',
                month: 'short',
                year: 'numeric',
              })}{' '}
              — {d.meetings} meeting{d.meetings !== 1 ? 's' : ''}, {d.races} races
            </option>
          ))}
        </select>
      </div>

      <div className="flex gap-1 mb-6 bg-gray-800/50 p-1 rounded-xl overflow-x-auto">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 min-w-[80px] px-3 py-2.5 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
              activeTab === tab.id
                ? 'bg-gray-700 text-white shadow-md'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700/40'
            }`}
          >
            <div>{tab.label}</div>
            <div className="text-[10px] text-gray-500 mt-0.5 hidden sm:block">{tab.desc}</div>
          </button>
        ))}
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-20">
          <div className="animate-spin rounded-full h-10 w-10 border-2 border-emerald-500 border-t-transparent" />
          <span className="ml-3 text-gray-400">Analysing all races...</span>
        </div>
      )}

      {error && (
        <div className="bg-red-900/20 border border-red-700/50 rounded-xl p-4 text-red-400 text-sm">
          Failed to load best bets. Is the API running?
        </div>
      )}

      {data && !isLoading && (
        <>
          <div className="mb-4 text-sm text-gray-500">
            Scanned {data.total_races_scanned} races across all meetings
          </div>

          {visiblePicks.length === 0 ? (
            <div className="text-center py-16 text-gray-500">
              No races found for this date.
            </div>
          ) : (
            <>
              <SelectionBanner tab={activeTab} picks={picks} />

              <div className="space-y-4">
                {visiblePicks.map((pick, i) => (
                  <PickCard
                    key={pick.race_id}
                    pick={pick}
                    rank={i + 1}
                    isNap={activeTab === 'nap'}
                  />
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
