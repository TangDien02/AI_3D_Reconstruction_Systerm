import { Platform } from 'react-native';

export const C = {
  // Backgrounds
  bg: '#080C18',
  bgCard: '#0F1623',
  bgCardAlt: '#151E2E',
  bgSurface: '#1A2235',

  // Borders
  border: 'rgba(91,143,239,0.12)',
  borderActive: 'rgba(91,143,239,0.45)',
  borderSubtle: 'rgba(91,143,239,0.07)',

  // Accent — electric blue
  accent: '#2D6BE4',
  accentLight: '#5B8FEF',
  accentMid: '#3D7CF4',
  accentGlow: 'rgba(45,107,228,0.2)',
  accentGlowStrong: 'rgba(45,107,228,0.35)',

  // Status
  green: '#10D98A',
  greenDim: 'rgba(16,217,138,0.12)',
  greenBorder: 'rgba(16,217,138,0.3)',
  amber: '#F59E0B',
  amberDim: 'rgba(245,158,11,0.12)',
  red: '#EF4444',
  redDim: 'rgba(239,68,68,0.12)',

  // Text
  textPrimary: '#EEF2FF',
  textSecondary: '#7B91B8',
  textMuted: '#3D4F6A',
  textDim: '#2A3850',

  // Misc
  scanBox: 'rgba(91,143,239,0.85)',
  white: '#FFFFFF',
  overlay: 'rgba(8,12,24,0.92)',
};

const defaultApiBaseUrl = Platform.select({
  android: 'http://10.0.2.2:8000',
  ios: 'http://localhost:8000',
  default: 'http://localhost:8000',
});

export const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL || defaultApiBaseUrl;
export const STORAGE_KEY = '@3DRecon_History';

export const CONFIG = {
  DETECT_FRAME_WIDTH: 640,
  DETECT_CAPTURE_QUALITY: 0.5,
  DETECT_UPLOAD_COMPRESS: 0.65,
  DETECT_COOLDOWN_MS: 350,
  DETECT_EMPTY_HOLD_MS: 900,
  RECON_CAPTURE_QUALITY: 0.92,
  RECON_POLL_INTERVAL_MS: 5000,
  RECON_POLL_TIMEOUT_MS: 45 * 60 * 1000,
  TEXTURE_POLL_INTERVAL_MS: 5000,
  TEXTURE_POLL_TIMEOUT_MS: 45 * 60 * 1000,
};