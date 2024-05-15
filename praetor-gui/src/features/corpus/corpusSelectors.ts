import type { RootState } from '../../app/store';
import { ScsynthEntry, SelectedAxes } from '../../types';

export const selectors = {
  getDigests: (state: RootState): string[] => {
    return state.corpus.digests;
  },
  getDurations: (state: RootState): [number, number] => {
    return state.corpus.durations;
  },
  getEntries: (state: RootState): { [key: string]: ScsynthEntry[] } => {
    return state.corpus.entries;
  },
  getIsLoading: (state: RootState): boolean => {
    return state.corpus.isLoading;
  },
  getSelectedAxes: (state: RootState): SelectedAxes => {
    return state.corpus.selectedAxes;
  },
  getSelectedDigests: (state: RootState): string[] => {
    return state.corpus.selectedDigests;
  },
  getSelectedEntries: (state: RootState): ScsynthEntry[] => {
    return state.corpus.selectedDigests
      .map((x) => (state.corpus.entries[x] ? state.corpus.entries[x] : []))
      .reduce((acc: ScsynthEntry[], val: ScsynthEntry[]) => [...acc, ...val], [])
      .filter((entry) => {
        const duration = entry.count / 48000.0;
        return state.corpus.durations[0] <= duration && duration <= state.corpus.durations[1];
      });
  },
};
