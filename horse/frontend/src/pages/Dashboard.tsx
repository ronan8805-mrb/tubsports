import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { useRaceCard, useAvailableDates } from '../hooks/useRaceCard';
import { api } from '../services/api';

type RegionFilter = 'ALL' | 'GB' | 'IE' | 'FR' | 'US' | 'AU' | 'HK' | 'JP' | 'INTL';

export function Dashboard() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedDate, setSelectedDate] = useState(searchParams.get('date') || '');
  const [selectedCourse, setSelectedCourse] = useState('');
  const [region, setRegion] = useState<RegionFilter>((searchParams.get('region') as RegionFilter) || 'ALL');

  // Scrape state
  const [scraping, setScraping] = useState(false);
  const [scrapeMsg, setScrapeMsg] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const handleScrape = useCallback(async () => {
    try {
      setScraping(true);
      setScrapeMsg(null);
      const res = await api.triggerScrape();
      if (res.status === 'already_running') {
        setScrapeMsg('Already fetching...');
      }
      pollRef.current = setInterval(async () => {
        try {
          const status = await api.getScrapeStatus();
          if (!status.running) {
            stopPolling();
            setScraping(false);
            if (status.result === 'success') {
              setScrapeMsg('New races loaded!');
              queryClient.invalidateQueries({ queryKey: ['raceCard'] });
              queryClient.invalidateQueries({ queryKey: ['dates'] });
            } else {
              setScrapeMsg(status.error ? `Error: ${status.error}` : 'Scrape finished');
            }
            setTimeout(() => setScrapeMsg(null), 5000);
          }
        } catch {
          stopPolling();
          setScraping(false);
          setScrapeMsg('Failed to check status');
          setTimeout(() => setScrapeMsg(null), 5000);
        }
      }, 3000);
    } catch (err) {
      setScraping(false);
      setScrapeMsg(`Failed: ${err instanceof Error ? err.message : 'unknown error'}`);
      setTimeout(() => setScrapeMsg(null), 5000);
    }
  }, [queryClient, stopPolling]);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  // Persist date & region in URL so they survive navigation
  useEffect(() => {
    const params = new URLSearchParams();
    if (selectedDate) params.set('date', selectedDate);
    if (region !== 'ALL') params.set('region', region);
    setSearchParams(params, { replace: true });
  }, [selectedDate, region, setSearchParams]);

  const { data: datesData } = useAvailableDates(5);
  const { data: cardData, isLoading, error } = useRaceCard(
    selectedDate || undefined,
    selectedCourse || undefined,
    region,
  );

  const dates = datesData?.dates ?? [];
  const races = cardData?.races ?? [];

  const courseGroups = races.reduce<Record<string, typeof races>>((acc, race) => {
    const key = race.course || 'Unknown';
    (acc[key] ??= []).push(race);
    return acc;
  }, {});

  const uniqueCourses = [...new Set(races.map((r) => r.course))];

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-1">Race Card</h1>
        <p className="text-sm text-gray-500">
          UK/IRE Horse Racing - 13D Predictive Intelligence
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 mb-6">
        {/* Date */}
        <select
          value={selectedDate}
          onChange={(e) => setSelectedDate(e.target.value)}
          className="bg-gray-800 border border-gray-700 text-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
        >
          <option value="">Latest date</option>
          {dates.map((d) => (
            <option key={d.date} value={d.date}>
              {d.date} ({d.races} races)
            </option>
          ))}
        </select>

        {/* Course */}
        <select
          value={selectedCourse}
          onChange={(e) => setSelectedCourse(e.target.value)}
          className="bg-gray-800 border border-gray-700 text-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
        >
          <option value="">All courses</option>
          {uniqueCourses.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>

        {/* Region */}
        <div className="flex flex-wrap rounded-lg overflow-hidden border border-gray-700">
          {(['ALL', 'GB', 'IE', 'FR', 'US', 'AU', 'HK', 'JP', 'INTL'] as RegionFilter[]).map((r) => {
            const labels: Record<string, string> = {
              ALL: 'All', GB: 'GB', IE: 'IE', FR: 'FR', US: 'US',
              AU: 'AU', HK: 'HK', JP: 'JP', INTL: 'Intl',
            };
            return (
              <button
                key={r}
                onClick={() => setRegion(r)}
                className={`px-2.5 py-2 text-xs font-medium transition-colors ${
                  region === r
                    ? 'bg-emerald-600/30 text-emerald-400'
                    : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                }`}
              >
                {labels[r] ?? r}
              </button>
            );
          })}
        </div>

        {/* Stats */}
        <span className="text-xs text-gray-500 ml-auto">
          {cardData?.total_races ?? 0} races
        </span>

        {/* Fetch New Races button */}
        <button
          onClick={handleScrape}
          disabled={scraping}
          className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            scraping
              ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
              : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-500/20'
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
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Fetch New Races
            </>
          )}
        </button>
      </div>

      {/* Scrape status message */}
      {scrapeMsg && (
        <div className={`mb-4 px-4 py-2 rounded-lg text-sm font-medium ${
          scrapeMsg.startsWith('Error') || scrapeMsg.startsWith('Failed')
            ? 'bg-red-500/10 border border-red-500/30 text-red-400'
            : scrapeMsg === 'New races loaded!'
              ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400'
              : 'bg-blue-500/10 border border-blue-500/30 text-blue-400'
        }`}>
          {scrapeMsg}
        </div>
      )}

      {/* Loading / Error */}
      {isLoading && (
        <div className="flex items-center justify-center py-20">
          <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400 text-sm">
          {(error as Error).message}
        </div>
      )}

      {/* No races */}
      {!isLoading && !error && races.length === 0 && (
        <div className="text-center py-20">
          <p className="text-gray-500 text-lg">No races found for this date</p>
          <p className="text-gray-600 text-sm mt-1">Try selecting a different date above</p>
        </div>
      )}

      {/* Race cards grouped by course */}
      {Object.entries(courseGroups).map(([course, courseRaces]) => (
        <div key={course} className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            {course}
            <span className="text-xs text-gray-500 font-normal">
              ({courseRaces.length} races)
            </span>
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {courseRaces.map((race) => (
              <button
                key={race.race_id}
                onClick={() => navigate(`/race/${race.race_id}`)}
                className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-left hover:border-emerald-500/30 hover:bg-emerald-500/5 transition-all group"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {race.race_time && (
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400 text-[11px] font-bold tabular-nums border border-amber-500/20">
                        {race.race_time}
                      </span>
                    )}
                    <span className="text-sm font-semibold text-white group-hover:text-emerald-400 transition-colors">
                      Race {race.race_number ?? '?'}
                    </span>
                  </div>
                  <span className="text-xs text-gray-500">
                    {race.num_runners} runners
                  </span>
                </div>

                {race.race_name && (
                  <p className="text-xs text-gray-400 mb-2 truncate">
                    {race.race_name}
                  </p>
                )}

                <div className="flex flex-wrap gap-1.5">
                  {race.distance_furlongs && (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
                      {race.distance_furlongs}f
                    </span>
                  )}
                  {race.race_type && (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20">
                      {race.race_type}
                    </span>
                  )}
                  {race.race_class && (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                      {race.race_class}
                    </span>
                  )}
                  {race.going && (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      {race.going}
                    </span>
                  )}
                  {race.surface && (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-gray-500/10 text-gray-400 border border-gray-500/20">
                      {race.surface}
                    </span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
