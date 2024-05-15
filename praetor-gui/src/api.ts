import axios from 'axios';

import { config } from './config';
import { ScsynthEntry } from './types';

const serverAPI = axios.create({ baseURL: config.serverURL });

export function fetchAudio(digest: string, start: number, count: number): Promise<ArrayBuffer> {
  return serverAPI
    .get(`/audio/fetch/${digest}`, { params: { count, start }, responseType: 'arraybuffer' })
    .then((response) => response.data);
}

export function getDigestEntries(digest: string): Promise<ScsynthEntry[]> {
  return serverAPI.get(`/query/scsynth/data/${digest}`).then((response) => response.data.entries);
}

export function listDigests(): Promise<string[]> {
  return serverAPI.get(`/audio/partitions`).then((response) => response.data.partitions);
}
