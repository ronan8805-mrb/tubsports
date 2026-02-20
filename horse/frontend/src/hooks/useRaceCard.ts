import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';

export function useRaceCard(date?: string, course?: string, region?: string) {
  return useQuery({
    queryKey: ['raceCard', date, course, region],
    queryFn: () => api.getRaceCard(date, course, region),
    enabled: true,
    staleTime: 60_000,
  });
}

export function useAvailableDates(limit = 30) {
  return useQuery({
    queryKey: ['dates', limit],
    queryFn: () => api.getAvailableDates(limit),
    staleTime: 60_000,
  });
}

export function useThirteenD(raceId: number) {
  return useQuery({
    queryKey: ['thirteenD', raceId],
    queryFn: () => api.getThirteenD(raceId),
    enabled: raceId > 0,
    staleTime: 120_000,
  });
}


