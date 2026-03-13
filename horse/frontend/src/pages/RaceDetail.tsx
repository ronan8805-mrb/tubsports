import { useParams, useNavigate } from 'react-router-dom';
import { useThirteenD } from '../hooks/useRaceCard';
import { RunnerRow } from '../components/RunnerRow';

export function RaceDetail() {
  const { raceId } = useParams<{ raceId: string }>();
  const id = Number(raceId) || 0;
  const navigate = useNavigate();

  const { data, isLoading, error } = useThirteenD(id);

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
    <div className="max-w-5xl mx-auto px-3 py-4 sm:px-4 sm:py-6">
      {/* Back link */}
      <button
        onClick={() => navigate(-1)}
        className="inline-flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 mb-3 sm:mb-4 transition-colors"
      >
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back to card
      </button>

      {/* Race header */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl sm:rounded-2xl p-3 sm:p-5 mb-4 sm:mb-6">
        <div>
          <h1 className="text-base sm:text-xl font-bold text-white flex items-center gap-2 flex-wrap">
            {race_info.race_time && (
              <span className="inline-flex items-center px-2 py-0.5 rounded bg-amber-500/15 text-amber-400 text-xs sm:text-sm font-bold tabular-nums border border-amber-500/20">
                {race_info.race_time}
              </span>
            )}
            <span>{race_info.course} - Race {race_info.race_number}</span>
          </h1>
          {race_info.race_name && (
            <p className="text-xs sm:text-sm text-gray-400 mt-1">{race_info.race_name}</p>
          )}
          <p className="text-[10px] sm:text-xs text-gray-500 mt-1">{race_info.meeting_date}</p>
        </div>

        <div className="flex flex-wrap gap-1.5 sm:gap-2 mt-3">
          {race_info.distance_furlongs && (
            <span className="px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-lg text-[10px] sm:text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
              {race_info.distance_furlongs}f
            </span>
          )}
          {race_info.race_type && (
            <span className="px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-lg text-[10px] sm:text-xs font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20">
              {race_info.race_type}
            </span>
          )}
          {race_info.race_class && (
            <span className="px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-lg text-[10px] sm:text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
              {race_info.race_class}
            </span>
          )}
          {race_info.going && (
            <span className="px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-lg text-[10px] sm:text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              {race_info.going}
            </span>
          )}
          {race_info.surface && (
            <span className="px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-lg text-[10px] sm:text-xs font-medium bg-gray-500/10 text-gray-400 border border-gray-500/20">
              {race_info.surface}
            </span>
          )}
          <span className="px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-lg text-[10px] sm:text-xs font-medium bg-sky-500/10 text-sky-400 border border-sky-500/20">
            {race_info.num_runners} runners
          </span>
          {race_info.top_gap != null && (
            <span className={`px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-lg text-[10px] sm:text-xs font-medium border ${
              race_info.top_gap >= 0.15
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                : race_info.top_gap >= 0.08
                  ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                  : 'bg-gray-500/10 text-gray-400 border-gray-500/20'
            }`}>
              Gap {(race_info.top_gap * 100).toFixed(1)}%
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
