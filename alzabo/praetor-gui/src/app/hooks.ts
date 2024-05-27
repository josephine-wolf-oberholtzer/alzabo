import * as React from 'react';
import type { TypedUseSelectorHook } from 'react-redux';
import { useDispatch, useSelector } from 'react-redux';

import type { RootState, AppDispatch } from './store';

// Use throughout your app instead of plain `useDispatch` and `useSelector`
export const useAppDispatch: () => AppDispatch = useDispatch;
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;

export const useAudio = () => {
  const audioContext = React.useMemo(() => new AudioContext(), []);
  const play = React.useCallback(
    async (arrayBuffer: ArrayBuffer) => {
      const bufferSource = audioContext.createBufferSource();
      bufferSource.buffer = await audioContext.decodeAudioData(arrayBuffer);
      bufferSource.connect(audioContext.destination);
      bufferSource.start();
      bufferSource.onended = () => bufferSource.disconnect();
    },
    [audioContext],
  );
  return play;
};
