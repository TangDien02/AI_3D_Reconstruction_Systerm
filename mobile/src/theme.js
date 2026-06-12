import { Platform } from 'react-native';

export const C = {
  bg: '#0A0E1A',
  bgCard: '#111827',
  bgCardAlt: '#1A2233',
  border: 'rgba(99,179,237,0.15)',
  borderActive: 'rgba(99,179,237,0.5)',
  accent: '#2D6BE4',
  accentLight: '#5B8FEF',
  accentGlow: 'rgba(45,107,228,0.25)',
  green: '#10D98A',
  greenDim: 'rgba(16,217,138,0.15)',
  amber: '#F59E0B',
  amberDim: 'rgba(245,158,11,0.15)',
  red: '#EF4444',
  textPrimary: '#F0F4FF',
  textSecondary: '#8A9DC4',
  textMuted: '#4A5A78',
  scanBox: 'rgba(99,179,237,0.9)',
  white: '#FFFFFF',
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
