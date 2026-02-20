import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DimensionScore, DriverItem } from '../services/api';

interface Props {
  dimensions: DimensionScore[];
  positiveDrivers: DriverItem[];
  negativeDrivers: DriverItem[];
}

export function FactorBreakdown({ dimensions, positiveDrivers, negativeDrivers }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <div className="space-y-1.5">
      {/* 13 dimension bars */}
      {dimensions.map((dim) => {
        const isOpen = expanded === dim.name;
        return (
          <div key={dim.name}>
            <button
              onClick={() => setExpanded(isOpen ? null : dim.name)}
              className="w-full flex items-center gap-2.5 group cursor-pointer hover:bg-white/5 rounded-lg px-2 py-1 transition-colors"
            >
              <span className="text-[11px] font-medium text-gray-400 w-20 text-left shrink-0 truncate">
                {dim.name}
              </span>
              <div className="flex-1 h-2.5 bg-gray-800 rounded-full overflow-hidden relative">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${dim.score}%` }}
                  transition={{ duration: 0.6, ease: 'easeOut' }}
                  className="h-full rounded-full"
                  style={{ backgroundColor: dim.color }}
                />
              </div>
              <span
                className="text-[11px] font-mono w-8 text-right"
                style={{ color: dim.color }}
              >
                {dim.score.toFixed(0)}
              </span>
              <svg
                className={`w-3 h-3 text-gray-500 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            <AnimatePresence>
              {isOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="pl-24 pr-4 pb-2 text-xs text-gray-400">
                    <p className="mb-1.5 text-gray-500">{dim.tooltip}</p>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
                      {Object.entries(dim.features).map(([feat, score]) => (
                        <div key={feat} className="flex justify-between">
                          <span className="truncate">{feat.replace(/_/g, ' ')}</span>
                          <span className="font-mono ml-2" style={{ color: dim.color }}>
                            {score.toFixed(0)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );
      })}

      {/* Drivers */}
      {(positiveDrivers.length > 0 || negativeDrivers.length > 0) && (
        <div className="mt-3 pt-3 border-t border-gray-800">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <h4 className="text-xs font-semibold text-emerald-500 mb-1.5 uppercase tracking-wider">
                Strengths
              </h4>
              {positiveDrivers.map((d, i) => (
                <div key={i} className="text-xs text-gray-400 mb-1 flex items-start gap-1.5">
                  <span className="text-emerald-500 mt-0.5">+</span>
                  <div>
                    <span className="text-gray-300">{d.feature}:</span> {d.value}
                    <span className="text-gray-600 ml-1">({d.description})</span>
                  </div>
                </div>
              ))}
              {positiveDrivers.length === 0 && (
                <p className="text-xs text-gray-600">None identified</p>
              )}
            </div>
            <div>
              <h4 className="text-xs font-semibold text-red-500 mb-1.5 uppercase tracking-wider">
                Concerns
              </h4>
              {negativeDrivers.map((d, i) => (
                <div key={i} className="text-xs text-gray-400 mb-1 flex items-start gap-1.5">
                  <span className="text-red-500 mt-0.5">-</span>
                  <div>
                    <span className="text-gray-300">{d.feature}:</span> {d.value}
                    <span className="text-gray-600 ml-1">({d.description})</span>
                  </div>
                </div>
              ))}
              {negativeDrivers.length === 0 && (
                <p className="text-xs text-gray-600">None identified</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
