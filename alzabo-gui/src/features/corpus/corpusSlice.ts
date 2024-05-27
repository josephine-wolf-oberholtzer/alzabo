import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { call, put, SagaReturnType, takeEvery, takeLatest } from 'redux-saga/effects';

import * as api from '../../api';
import { Axis, Feature, ScsynthEntry, SelectedAxes } from '../../types';

interface CorpusState {
  digests: string[];
  durations: [number, number];
  entries: { [key: string]: ScsynthEntry[] };
  error: Error | null;
  isLoading: boolean;
  selectedAxes: SelectedAxes;
  selectedDigests: string[];
}

const initialState: CorpusState = {
  digests: [],
  durations: [0, 10.0],
  entries: {},
  error: null,
  isLoading: false,
  selectedAxes: { x: 'r:f0:mean', y: 'r:f0:std', z: 'r:rms:mean' },
  selectedDigests: [],
};

export const corpusSlice = createSlice({
  name: 'corpus',
  initialState,
  reducers: {
    getDigestEntries(state: CorpusState, _action: PayloadAction<string>) {
      state.error = null;
      state.isLoading = false;
    },
    getDigestEntriesSuccess(state: CorpusState, action: PayloadAction<[string, ScsynthEntry[]]>) {
      const [digest, entries] = action.payload;
      state.error = null;
      state.isLoading = false;
      state.entries[digest] = entries;
    },
    getDigestEntriesFailure(state: CorpusState, action: PayloadAction<Error>) {
      state.error = action.payload;
      state.isLoading = false;
    },
    listDigests(state: CorpusState) {
      state.error = null;
      state.isLoading = false;
    },
    listDigestsSuccess(state: CorpusState, action: PayloadAction<string[]>) {
      state.digests = action.payload;
      state.error = null;
      state.isLoading = false;
      state.selectedDigests = [];
    },
    listDigestsFailure(state: CorpusState, action: PayloadAction<Error>) {
      state.error = action.payload;
      state.isLoading = false;
    },
    selectAxis(state: CorpusState, action: PayloadAction<[Axis, Feature]>) {
      const [axis, feature] = action.payload;
      state.selectedAxes[axis] = feature;
    },
    selectDigests(state: CorpusState, action: PayloadAction<string[]>) {
      state.selectedDigests = action.payload;
    },
    selectDurations(state: CorpusState, action: PayloadAction<[number, number]>) {
      state.durations = action.payload.sort();
    },
  },
});

export const {
  getDigestEntries,
  getDigestEntriesSuccess,
  getDigestEntriesFailure,
  listDigests,
  listDigestsSuccess,
  listDigestsFailure,
  selectAxis,
  selectDigests,
  selectDurations,
} = corpusSlice.actions;

export default corpusSlice.reducer;

export function* sagaGetDigestEntries(action: PayloadAction<string>) {
  try {
    const entries: SagaReturnType<typeof api.getDigestEntries> = yield call(api.getDigestEntries, action.payload);
    yield put(getDigestEntriesSuccess([action.payload, entries]));
  } catch (err) {
    yield put(getDigestEntriesFailure(err as Error));
  }
}

export function* sagaListDigests() {
  try {
    const digests: SagaReturnType<typeof api.listDigests> = yield call(api.listDigests);
    for (const digest of digests) {
      yield put(getDigestEntries(digest));
    }
    yield put(listDigestsSuccess(digests));
  } catch (err) {
    yield put(listDigestsFailure(err as Error));
  }
}

export const sagas = [
  takeEvery(getDigestEntries.type, sagaGetDigestEntries),
  takeLatest(listDigests.type, sagaListDigests),
];
