import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useRaceCard, useAvailableDates } from '../hooks/useRaceCard';

type RegionFilter = 'ALL' | 'GB' | 'IE' | 'FR' | 'US' | 'AU' | 'HK' | 'JP' | 'INTL';

export function Dashboard() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedDate, setSelectedDate] = useState(searchParams.get('date') || '');
  const [selectedCourse, setSelectedCourse] = useState('');
  const [region, setRegion] = useState<RegionFilter>((searchParams.get('region') as RegionFilter) || 'ALL');

  // Persist date & region in URL so they survive navigation
  useEffect(() => {
    const params = new URLSearchParams();
    if (selectedDate) params.set('date', selectedDate);
    if (region !== 'ALL') params.set('region', region);
    setSearchParams(params, { replace: true });
  }, [selectedDate, region, setSearchParams]);

  const { data: datesData } = useAvailableDates(30);
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

  const [monitoring, setMonitoring] = useState<any>(null);
  useEffect(() => {
    import('../services/api').then(({ api }) =>
      api.getModelMonitoring().then(setMonitoring).catch(() => {})
    );
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-1">Race Card</h1>
        <p className="text-sm text-gray-500">
          UK/IRE Horse Racing - Stacked ML Intelligence Platform
        </p>
      </div>

      {/* Model Health Bar */}
      {monitoring?.history?.length > 0 && (() => {
        const latest = monitoring.history[monitoring.history.length - 1];
        return (
          <div className="mb-6 bg-gray-900 border border-gray-800 rounded-xl p-4">
            <div className="flex items-center gap-4 flex-wrap">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${latest.auc == null ? 'bg-gray-500' : latest.auc >= 0.72 ? 'bg-emerald-500' : latest.auc >= 0.68 ? 'bg-amber-500' : 'bg-red-500'}`} />
                <span className="text-xs text-gray-400">Model</span>
              </div>
              <div className="text-xs">
                <span className="text-gray-500">AUC </span>
                <span className="text-white font-mono">{latest.auc?.toFixed(4) ?? '—'}</span>
              </div>
              <div className="text-xs">
                <span className="text-gray-500">Brier </span>
                <span className="text-white font-mono">{latest.brier?.toFixed(4) ?? '—'}</span>
              </div>
              <div className="text-xs">
                <span className="text-gray-500">Features </span>
                <span className="text-white font-mono">{latest.feature_count ?? '—'}</span>
              </div>
              <div className="text-xs">
                <span className="text-gray-500">Stacked </span>
                <span className={`font-mono ${latest.stacked ? 'text-emerald-400' : 'text-gray-500'}`}>
                  {latest.stacked ? 'Yes' : 'No'}
                </span>
              </div>
              <div className="text-xs">
                <span className="text-gray-500">Train </span>
                <span className="text-white font-mono">{latest.train_size?.toLocaleString() ?? '—'}</span>
              </div>
              <div className="text-xs ml-auto">
                <span className="text-gray-600">{latest.date}</span>
              </div>
            </div>
          </div>
        );
      })()}

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2 sm:gap-3 mb-6">
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
      </div>

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
                    {race.full_field === true && (
                      <span style={{background:'#16a34a',color:'#fff',fontWeight:'bold',padding:'2px 6px',borderRadius:'4px',fontSize:'11px',whiteSpace:'nowrap'}}>FULL DATA</span>
                    )}
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

                {race.top_gap != null && (
                  <div className="mt-2 flex items-center gap-1.5">
                    <span className="text-[10px] text-gray-600">Gap #1 vs #2:</span>
                    <span className={`text-[11px] font-bold ${
                      race.top_gap >= 0.15 ? 'text-emerald-400' :
                      race.top_gap >= 0.08 ? 'text-amber-400' : 'text-gray-500'
                    }`}>
                      {(race.top_gap * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
