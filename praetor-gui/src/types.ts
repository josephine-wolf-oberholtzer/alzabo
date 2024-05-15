export interface ScsynthEntry {
  count: number;
  digest: string;
  start: number;
  features: {
    'centroid:mean': number;
    'centroid:std': number;
    'chroma:0': number;
    'chroma:1': number;
    'chroma:2': number;
    'chroma:3': number;
    'chroma:4': number;
    'chroma:5': number;
    'chroma:6': number;
    'chroma:7': number;
    'chroma:8': number;
    'chroma:9': number;
    'chroma:10': number;
    'chroma:11': number;
    'f0:mean': number;
    'f0:std': number;
    'flatness:mean': number;
    'flatness:std': number;
    is_onset: number;
    is_voiced: number;
    'mfcc:0': number;
    'mfcc:1': number;
    'mfcc:2': number;
    'mfcc:3': number;
    'mfcc:4': number;
    'mfcc:5': number;
    'mfcc:6': number;
    'mfcc:7': number;
    'mfcc:8': number;
    'mfcc:9': number;
    'mfcc:10': number;
    'mfcc:11': number;
    'mfcc:12': number;
    'peak:mean': number;
    'peak:std': number;
    'rms:mean': number;
    'rms:std': number;
    'rolloff:mean': number;
    'rolloff:std': number;
  };
}

export type Axis = 'x' | 'y' | 'z';

export type Feature =
  | 'centroid:mean'
  | 'centroid:std'
  | 'chroma:0'
  | 'chroma:1'
  | 'chroma:2'
  | 'chroma:3'
  | 'chroma:4'
  | 'chroma:5'
  | 'chroma:6'
  | 'chroma:7'
  | 'chroma:8'
  | 'chroma:9'
  | 'chroma:10'
  | 'chroma:11'
  | 'f0:mean'
  | 'f0:std'
  | 'flatness:mean'
  | 'flatness:std'
  | 'is_onset'
  | 'is_voiced'
  | 'mfcc:0'
  | 'mfcc:1'
  | 'mfcc:2'
  | 'mfcc:3'
  | 'mfcc:4'
  | 'mfcc:5'
  | 'mfcc:6'
  | 'mfcc:7'
  | 'mfcc:8'
  | 'mfcc:9'
  | 'mfcc:10'
  | 'mfcc:11'
  | 'mfcc:12'
  | 'peak:mean'
  | 'peak:std'
  | 'rms:mean'
  | 'rms:std'
  | 'rolloff:mean'
  | 'rolloff:std';

export interface SelectedAxes {
  x: Feature;
  y: Feature;
  z: Feature;
}
