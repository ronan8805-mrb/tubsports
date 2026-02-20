import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { RunnerPrediction } from '../services/api';
import { FactorBreakdown } from './FactorBreakdown';

interface Props {
  runner: RunnerPrediction;
  rank: number;
}

function toFractional(decimal: number): string {
  if (decimal <= 1.01) return 'EVS';
  const frac = decimal - 1;
  const common: [number, string][] = [
    [0.2, '1/5'], [0.25, '1/4'], [0.33, '1/3'], [0.4, '2/5'],
    [0.5, '1/2'], [0.57, '4/7'], [0.6, '3/5'], [0.67, '2/3'],
    [0.73, '8/11'], [0.8, '4/5'], [0.83, '5/6'], [0.91, '10/11'],
    [1.0, 'EVS'], [1.1, '11/10'], [1.2, '6/5'], [1.25, '5/4'],
    [1.33, '4/3'], [1.5, '6/4'], [1.67, '5/3'], [1.8, '9/5'],
    [2.0, '2/1'], [2.25, '9/4'], [2.5, '5/2'], [2.75, '11/4'],
    [3.0, '3/1'], [3.5, '7/2'], [4.0, '4/1'], [4.5, '9/2'],
    [5.0, '5/1'], [5.5, '11/2'], [6.0, '6/1'], [7.0, '7/1'],
    [8.0, '8/1'], [9.0, '9/1'], [10.0, '10/1'], [11.0, '11/1'],
    [12.0, '12/1'], [14.0, '14/1'], [16.0, '16/1'], [20.0, '20/1'],
    [25.0, '25/1'], [33.0, '33/1'], [40.0, '40/1'], [50.0, '50/1'],
    [66.0, '66/1'], [100.0, '100/1'],
  ];
  let best = common[0];
  let bestDiff = Math.abs(frac - common[0][0]);
  for (const entry of common) {
    const diff = Math.abs(frac - entry[0]);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = entry;
    }
  }
  if (bestDiff > 0.3) {
    return `${Math.round(frac)}/1`;
  }
  return best[1];
}

function oddsLabel(decimal: number): string {
  if (decimal <= 2.0) return 'Strong Fav';
  if (decimal <= 3.5) return 'Fav';
  if (decimal <= 6.0) return 'Contender';
  if (decimal <= 10.0) return 'Each-Way';
  if (decimal <= 20.0) return 'Outsider';
  return 'Long Shot';
}

function oddsColor(decimal: number): string {
  if (decimal <= 2.0) return 'text-emerald-400';
  if (decimal <= 3.5) return 'text-green-400';
  if (decimal <= 6.0) return 'text-blue-400';
  if (decimal <= 10.0) return 'text-amber-400';
  if (decimal <= 20.0) return 'text-orange-400';
  return 'text-red-400';
}

export function RunnerRow({ runner, rank }: Props) {
  const [expanded, setExpanded] = useState(false);

  const isTopPick = rank <= 2;
  const fractional = toFractional(runner.fair_odds);
  const label = oddsLabel(runner.fair_odds);
  const color = oddsColor(runner.fair_odds);
  const bookieFrac = runner.back_odds ? toFractional(runner.back_odds) : null;

  const valueBadge = runner.value_flag === 'VALUE'
    ? { text: 'VALUE', cls: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' }
    : runner.value_flag === 'SHORT'
      ? { text: 'SHORT', cls: 'bg-red-500/20 text-red-400 border-red-500/30' }
      : runner.value_flag === 'FAIR'
        ? { text: 'FAIR', cls: 'bg-amber-500/20 text-amber-400 border-amber-500/30' }
        : null;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: rank * 0.04 }}
      className={`rounded-xl border transition-colors ${
        isTopPick
          ? 'border-emerald-500/30 bg-emerald-500/5'
          : 'border-gray-800 bg-gray-900/50'
      }`}
    >
      {/* Main row */}
      <div
        className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-white/[0.02] transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        {/* Rank */}
        <div
          className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
            rank === 1
              ? 'bg-amber-500/20 text-amber-400'
              : rank === 2
                ? 'bg-gray-600/30 text-gray-300'
                : rank === 3
                  ? 'bg-orange-500/20 text-orange-400'
                  : 'bg-gray-800 text-gray-500'
          }`}
        >
          {rank}
        </div>

        {/* Draw */}
        {runner.draw != null && (
          <div className="w-7 h-7 rounded-md flex items-center justify-center text-xs font-bold bg-sky-500/10 text-sky-400 border border-sky-500/20">
            {runner.draw}
          </div>
        )}

        {/* Silk */}
        {runner.silk_url && (
          <img
            src={runner.silk_url}
            alt=""
            className="w-8 h-8 rounded object-contain bg-white/10 flex-shrink-0"
            loading="lazy"
            onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
          />
        )}

        {/* Name + badges */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span
              className={`font-semibold text-sm ${isTopPick ? 'text-white' : 'text-gray-200'}`}
            >
              {runner.horse_name}
            </span>
            {rank === 1 && (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider bg-amber-500/20 text-amber-400 border border-amber-500/30">
                TOP PICK
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            {runner.jockey && (
              <span className="text-[10px] text-gray-500 truncate max-w-[120px]">
                J: {runner.jockey}
              </span>
            )}
            {runner.trainer && (
              <span className="text-[10px] text-gray-500 truncate max-w-[120px]">
                T: {runner.trainer}
              </span>
            )}
          </div>
        </div>

        {/* OR */}
        {runner.official_rating != null && (
          <div className="text-right w-12">
            <div className="text-sm font-mono text-gray-300">
              {runner.official_rating}
            </div>
            <div className="text-[10px] text-gray-500">OR</div>
          </div>
        )}

        {/* Win % */}
        <div className="text-right w-16">
          <div className="text-sm font-mono text-blue-400">
            {(runner.win_prob * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-gray-500">Win</div>
        </div>

        {/* Place % */}
        <div className="text-right w-16">
          <div className="text-sm font-mono text-emerald-400">
            {(runner.place_prob * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-gray-500">Place</div>
        </div>

        {/* Model Price (fractional + decimal) */}
        <div className="text-right w-20">
          <div className={`text-sm font-bold font-mono ${color}`}>
            {fractional}
          </div>
          <div className="text-[10px] text-gray-500">
            ({runner.fair_odds.toFixed(1)}) Model
          </div>
        </div>

        {/* Bookie Odds + Value Flag */}
        <div className="text-right w-20">
          {bookieFrac ? (
            <>
              <div className="text-sm font-bold font-mono text-white">
                {bookieFrac}
              </div>
              <div className="text-[10px] text-gray-500">
                ({runner.back_odds!.toFixed(1)}) Bookie
              </div>
            </>
          ) : (
            <>
              <div className="text-sm font-mono text-gray-600">--</div>
              <div className="text-[10px] text-gray-600">Bookie</div>
            </>
          )}
        </div>

        {/* Value badge */}
        {valueBadge && (
          <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider border ${valueBadge.cls}`}>
            {valueBadge.text}
          </span>
        )}

        {/* Expand icon */}
        <svg
          className={`w-4 h-4 text-gray-500 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </div>

      {/* Expanded: 13D breakdown */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-1 border-t border-gray-800/50">
              {/* Odds explanation card */}
              <div className="bg-gray-800/50 rounded-lg p-3 mb-3">
                <div className="flex items-center gap-4 flex-wrap">
                  <div>
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-0.5">Model Price</div>
                    <div className={`text-lg font-bold font-mono ${color}`}>{fractional}</div>
                    <div className="text-[10px] text-gray-500">({runner.fair_odds.toFixed(2)} decimal)</div>
                  </div>
                  <div className="h-8 w-px bg-gray-700" />
                  <div>
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-0.5">Win Chance</div>
                    <div className="text-lg font-bold font-mono text-blue-400">{(runner.win_prob * 100).toFixed(1)}%</div>
                    <div className="text-[10px] text-gray-500">1 in {Math.round(1 / runner.win_prob)} races</div>
                  </div>
                  <div className="h-8 w-px bg-gray-700" />
                  <div>
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-0.5">Place Chance</div>
                    <div className="text-lg font-bold font-mono text-emerald-400">{(runner.place_prob * 100).toFixed(1)}%</div>
                    <div className="text-[10px] text-gray-500">Top 3 finish</div>
                  </div>
                  <div className="h-8 w-px bg-gray-700" />
                  {runner.back_odds != null && (
                    <>
                      <div className="h-8 w-px bg-gray-700" />
                      <div>
                        <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-0.5">Bookie Price</div>
                        <div className="text-lg font-bold font-mono text-white">{bookieFrac}</div>
                        <div className="text-[10px] text-gray-500">({runner.back_odds.toFixed(2)} decimal)</div>
                      </div>
                      <div className="h-8 w-px bg-gray-700" />
                      <div>
                        <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-0.5">Verdict</div>
                        {valueBadge && (
                          <div className={`text-sm font-bold ${
                            runner.value_flag === 'VALUE' ? 'text-emerald-400' :
                            runner.value_flag === 'SHORT' ? 'text-red-400' : 'text-amber-400'
                          }`}>{valueBadge.text}</div>
                        )}
                        <div className="text-[10px] text-gray-500">
                          {runner.value_flag === 'VALUE'
                            ? 'Bookie offering more than model price'
                            : runner.value_flag === 'SHORT'
                              ? 'Bookie offering less than model price'
                              : 'Prices roughly aligned'}
                        </div>
                      </div>
                    </>
                  )}
                  {runner.back_odds == null && (
                    <>
                      <div className="h-8 w-px bg-gray-700" />
                      <div>
                        <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-0.5">Category</div>
                        <div className={`text-sm font-bold ${color}`}>{label}</div>
                        <div className="text-[10px] text-gray-500">
                          {runner.fair_odds <= 3.5
                            ? 'Should be in the mix'
                            : runner.fair_odds <= 6.0
                              ? 'Solid chance'
                              : runner.fair_odds <= 10.0
                                ? 'Could place'
                                : 'Needs things to fall right'}
                        </div>
                      </div>
                    </>
                  )}
                </div>
                <div className="mt-2 pt-2 border-t border-gray-700/50">
                  <p className="text-[10px] text-gray-500 leading-relaxed">
                    <span className="text-gray-400 font-medium">How to read: </span>
                    The model says {runner.horse_name} has a <span className="text-blue-400 font-medium">{(runner.win_prob * 100).toFixed(1)}%</span> chance of winning.
                    {' '}That converts to fair odds of <span className={`font-medium ${color}`}>{fractional}</span> ({runner.fair_odds.toFixed(1)} decimal).
                    {runner.back_odds != null ? (
                      <>
                        {' '}The bookie is offering <span className="text-white font-medium">{bookieFrac}</span> ({runner.back_odds.toFixed(1)} decimal).
                        {runner.value_flag === 'VALUE' ? (
                          <>{' '}<span className="text-emerald-400 font-medium">That's VALUE</span> -- the bookie is offering bigger odds than the model thinks are fair. Edge in your favour.</>
                        ) : runner.value_flag === 'SHORT' ? (
                          <>{' '}<span className="text-red-400 font-medium">That's SHORT</span> -- the bookie is offering less than the model price. The market fancies this horse more than the model does.</>
                        ) : (
                          <>{' '}<span className="text-amber-400 font-medium">Prices are FAIR</span> -- model and bookie roughly agree.</>
                        )}
                      </>
                    ) : (
                      <>
                        {' '}No live bookie odds available yet. Hit <span className="text-gray-400 font-medium">Refresh Odds</span> to fetch latest prices.
                      </>
                    )}
                  </p>
                  {runner.odds_updated_at && (
                    <p className="text-[9px] text-gray-600 mt-1">Odds updated at {runner.odds_updated_at}</p>
                  )}
                </div>
              </div>

              {/* Runner details */}
              <div className="flex flex-wrap gap-3 mb-3 text-xs text-gray-400">
                {runner.age != null && <span>Age: {runner.age}</span>}
                {runner.weight_lbs != null && <span>Weight: {runner.weight_lbs}lbs</span>}
                {runner.draw != null && <span>Draw: {runner.draw}</span>}
              </div>

              {/* 13D Factor bars */}
              <FactorBreakdown
                dimensions={runner.dimensions}
                positiveDrivers={runner.positive_drivers}
                negativeDrivers={runner.negative_drivers}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
