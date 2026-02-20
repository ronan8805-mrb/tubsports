import { useState, useCallback, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { useThirteenD } from '../hooks/useRaceCard';
import { RunnerRow } from '../components/RunnerRow';
import { api } from '../services/api';

export function RaceDetail() {
  const { raceId } = useParams<{ raceId: string }>();
  const id = Number(raceId) || 0;
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useThirteenD(id);

  const [refreshingOdds, setRefreshingOdds] = useState(false);
  const [oddsMsg, setOddsMsg] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const handleRefreshOdds = useCallback(async () => {
    try {
      setRefreshingOdds(true);
      setOddsMsg(null);
      await api.triggerOddsRefresh();
      pollRef.current = setInterval(async () => {
        try {
          const status = await api.getOddsStatus();
          if (!status.running) {
            stopPolling();
            setRefreshingOdds(false);
            if (status.result === 'success') {
              setOddsMsg('Odds updated!');
              queryClient.invalidateQueries({ queryKey: ['thirteenD', id] });
            } else {
              setOddsMsg(status.error ? `Error: ${status.error}` : 'Done');
            }
            setTimeout(() => setOddsMsg(null), 4000);
          }
        } catch {
          stopPolling();
          setRefreshingOdds(false);
        }
      }, 3000);
    } catch {
      setRefreshingOdds(false);
      setOddsMsg('Failed to refresh odds');
      setTimeout(() => setOddsMsg(null), 4000);
    }
  }, [id, queryClient, stopPolling]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-10">
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400 text-sm">
          {(error as Error).message}
        </div>
      </div>
    );
  }

  if (!data) return null;

  const { race_info, runners } = data;

  return (
    <div className="max-w-5xl mx-auto px-4 py-6">
      {/* Back link */}
      <button
        onClick={() => navigate(-1)}
        className="inline-flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 mb-4 transition-colors"
      >
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back to card
      </button>

      {/* Race header */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 mb-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-white flex items-center gap-2">
              {race_info.race_time && (
                <span className="inline-flex items-center px-2 py-0.5 rounded bg-amber-500/15 text-amber-400 text-sm font-bold tabular-nums border border-amber-500/20">
                  {race_info.race_time}
                </span>
              )}
              {race_info.course} - Race {race_info.race_number}
            </h1>
            {race_info.race_name && (
              <p className="text-sm text-gray-400 mt-1">{race_info.race_name}</p>
            )}
            <p className="text-xs text-gray-500 mt-1">{race_info.meeting_date}</p>
          </div>

          <div className="flex flex-wrap gap-2">
            {race_info.distance_furlongs && (
              <span className="px-2.5 py-1 rounded-lg text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
                {race_info.distance_furlongs}f
              </span>
            )}
            {race_info.race_type && (
              <span className="px-2.5 py-1 rounded-lg text-xs font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20">
                {race_info.race_type}
              </span>
            )}
            {race_info.race_class && (
              <span className="px-2.5 py-1 rounded-lg text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                {race_info.race_class}
              </span>
            )}
            {race_info.going && (
              <span className="px-2.5 py-1 rounded-lg text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                {race_info.going}
              </span>
            )}
            {race_info.surface && (
              <span className="px-2.5 py-1 rounded-lg text-xs font-medium bg-gray-500/10 text-gray-400 border border-gray-500/20">
                {race_info.surface}
              </span>
            )}
            <span className="px-2.5 py-1 rounded-lg text-xs font-medium bg-sky-500/10 text-sky-400 border border-sky-500/20">
              {race_info.num_runners} runners
            </span>
          </div>
        </div>

        {/* Refresh Odds button */}
        <div className="mt-3 flex items-center gap-3">
          <button
            onClick={handleRefreshOdds}
            disabled={refreshingOdds}
            className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              refreshingOdds
                ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                : 'bg-sky-600 hover:bg-sky-500 text-white shadow-lg shadow-sky-500/20'
            }`}
          >
            {refreshingOdds ? (
              <>
                <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Refreshing...
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Refresh Odds
              </>
            )}
          </button>
          {oddsMsg && (
            <span className={`text-xs font-medium ${
              oddsMsg.startsWith('Error') || oddsMsg.startsWith('Failed')
                ? 'text-red-400' : 'text-emerald-400'
            }`}>
              {oddsMsg}
            </span>
          )}
        </div>

      </div>

      {/* Runners */}
      {runners.length === 0 ? (
        <div className="text-center py-16">
          <p className="text-gray-500">No runner data available for this race</p>
        </div>
      ) : (
        <div className="space-y-2">
          {runners.map((runner, i) => (
            <RunnerRow key={runner.horse_name} runner={runner} rank={i + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
