import { configureStore } from '@reduxjs/toolkit';
import createSagaMiddleware from 'redux-saga';
import { all } from 'redux-saga/effects';

import corpusReducer, { sagas as corpusSagas } from '../features/corpus/corpusSlice';

export function* rootSaga() {
  yield all([...corpusSagas]);
}

const createStore = () => {
  const sagaMiddleware = createSagaMiddleware();
  const store = configureStore({
    devTools: true,
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({
        serializableCheck: false,
      }).concat(sagaMiddleware),
    reducer: { corpus: corpusReducer },
  });
  sagaMiddleware.run(rootSaga);
  return store;
};

export const store = createStore();

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<typeof store.getState>;
// Inferred type: {posts: PostsState, comments: CommentsState, users: UsersState}
export type AppDispatch = typeof store.dispatch;
