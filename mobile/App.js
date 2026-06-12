import { useEffect, useRef, useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import {
  Animated, Dimensions, Image, Linking, Platform,
  ScrollView, StyleSheet, Text, TouchableOpacity, View, Modal, ActivityIndicator,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Internal Imports
import { C, API_BASE_URL, STORAGE_KEY, CONFIG } from './src/theme';
import { delay, shortErrorMessage, clamp01 } from './src/utils';
import { LogoMark } from './src/components/LogoMark';
import { ProcessingTimeline } from './src/components/ProcessingTimeline';
import { Viewer3DModal } from './src/components/Viewer3DModal';
import SaveModal from './src/components/SaveModal';
import HistoryModal from './src/components/HistoryModal';

const { width: SCREEN_W } = Dimensions.get('window');

// ─── Decorative hex grid for hero ────────────────────────────────────────────
const HEX_POSITIONS = [
  { x: 0.08, y: 0.12, s: 28 }, { x: 0.85, y: 0.08, s: 20 },
  { x: 0.92, y: 0.32, s: 34 }, { x: 0.04, y: 0.42, s: 18 },
  { x: 0.78, y: 0.55, s: 26 }, { x: 0.14, y: 0.68, s: 22 },
  { x: 0.88, y: 0.72, s: 16 }, { x: 0.50, y: 0.05, s: 14 },
  { x: 0.62, y: 0.78, s: 20 }, { x: 0.30, y: 0.85, s: 15 },
];

const FloatingHex = ({ x, y, size, delay: d }) => {
  const anim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.delay(d),
        Animated.timing(anim, { toValue: 1, duration: 3000 + d * 0.5, useNativeDriver: true }),
        Animated.timing(anim, { toValue: 0, duration: 3000 + d * 0.5, useNativeDriver: true }),
      ])
    ).start();
  }, []);
  return (
    <Animated.Text
      style={{
        position: 'absolute',
        left: `${x * 100}%`,
        top: `${y * 100}%`,
        fontSize: size,
        color: C.borderActive,
        opacity: anim.interpolate({ inputRange: [0, 1], outputRange: [0.08, 0.3] }),
        transform: [
          { translateY: anim.interpolate({ inputRange: [0, 1], outputRange: [0, -8] }) },
          { rotate: anim.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '15deg'] }) },
        ],
      }}
    >
      ⬡
    </Animated.Text>
  );
};

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const cameraRef = useRef(null);
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const heroScale = useRef(new Animated.Value(0.92)).current;

  // UI State
  const [screen, setScreen] = useState('intro');
  const [cameraStatus, setCameraStatus] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState('');
  const [permission, requestPermission] = useCameraPermissions();
  const [show3DViewer, setShow3DViewer] = useState(false);
  const [viewerModelUrl, setViewerModelUrl] = useState(null);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [saveName, setSaveName] = useState('');

  // Data State
  const [capturedPhoto, setCapturedPhoto] = useState(null);
  const [cropAreaLayout, setCropAreaLayout] = useState({ width: 0, height: 0 });
  const [manualBbox, setManualBbox] = useState(null);
  const [segmentResult, setSegmentResult] = useState(null);
  const [reconstructionResult, setReconstructionResult] = useState(null);
  const [isPaintingTexture, setIsPaintingTexture] = useState(false);
  const cropDragStartRef = useRef(null);

  // Load history
  useEffect(() => {
    AsyncStorage.getItem(STORAGE_KEY)
      .then(s => s && setHistory(JSON.parse(s)))
      .catch(() => {});
  }, []);

  // Screen fade + hero entrance
  useEffect(() => {
    fadeAnim.setValue(0);
    heroScale.setValue(0.94);
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 380, useNativeDriver: true }),
      Animated.spring(heroScale, { toValue: 1, tension: 60, friction: 9, useNativeDriver: true }),
    ]).start();
  }, [screen]);

  // ─── Helpers ──────────────────────────────────────────────────────────────
  const getServerFileUrl = (path) =>
    path ? (path.startsWith('http') ? path : `${API_BASE_URL}${path}`) : null;

  const clearObjectState = () => {
    setCapturedPhoto(null); setManualBbox(null); setSegmentResult(null);
    setReconstructionResult(null); setIsPaintingTexture(false);
    setProcessingStage(''); setCameraStatus('');
  };

  const saveToHistory = async (reconstruction, segment, label) => {
    try {
      const isUpdate = history.some(i => i.id === reconstruction.job_id);
      const newItem = {
        id: reconstruction.job_id || Date.now().toString(),
        label: label || 'Untitled Object',
        timestamp: Date.now(),
        meshPath: reconstruction.files?.mesh_textured_glb || reconstruction.files?.mesh_glb,
        thumbPath: segment?.files?.clean_image || segment?.files?.crop
          || reconstruction.files?.input_image || reconstruction.files?.input_original,
        isTextured: !!reconstruction.files?.mesh_textured_glb,
        backend: reconstruction.backend,
        meshSummary: reconstruction.mesh || {},
      };
      const updated = isUpdate
        ? history.map(i => i.id === newItem.id ? { ...i, ...newItem, label: i.label } : i)
        : [newItem, ...history.filter(i => i.id !== newItem.id)].slice(0, 20);
      setHistory(updated);
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch {}
  };

  const textureHistoryItem = (item) => {
    setReconstructionResult({ job_id: item.id, backend: item.backend, files: { mesh_glb: item.meshPath } });
    setSegmentResult({ files: { clean_image: item.thumbPath } });
    setScreen('camera');
    setShowHistory(false);
    paintTexture();
  };

  const deleteHistoryItem = async (id) => {
    const updated = history.filter(i => i.id !== id);
    setHistory(updated);
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  };

  const exportHistoryItem = async (item) => {
    const url = getServerFileUrl(item.meshPath);
    if (url) Platform.OS === 'web' ? window.open(url, '_blank') : Linking.openURL(url);
  };

  const openCamera = async () => {
    if (!permission) return;
    if (!permission.granted) {
      const next = await requestPermission();
      if (!next.granted) { setScreen('permission'); return; }
    }
    clearObjectState(); setScreen('camera');
  };

  const open3DViewer = (path) => {
    const url = getServerFileUrl(path);
    if (url) { setViewerModelUrl(url); setShow3DViewer(true); }
  };

  // ─── API ──────────────────────────────────────────────────────────────────
  const waitForReconstructionJob = async (payload) => {
    if (payload.status === 'done') return payload;
    const statusPath = payload.status_url || `/reconstruction-jobs/${payload.job_id}`;
    const startedAt = Date.now(); let current = payload;
    while (current.status !== 'done') {
      if (['failed', 'error'].includes(current.status))
        throw new Error(typeof current.error === 'string' ? current.error : JSON.stringify(current.error));
      if (Date.now() - startedAt > CONFIG.RECON_POLL_TIMEOUT_MS) throw new Error('Reconstruction timed out.');
      setProcessingStage(current.stage || current.status || 'processing');
      await delay(CONFIG.RECON_POLL_INTERVAL_MS);
      const res = await fetch(`${API_BASE_URL}${statusPath}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      current = await res.json();
    }
    return current;
  };

  const reconstructManualBbox = async () => {
    if (!capturedPhoto?.uri || !manualBbox) return;
    setIsProcessing(true); setProcessingStage('preprocess'); setCameraStatus('Processing…');
    try {
      const { x, y, width, height } = manualBbox;
      const fd = new FormData();
      fd.append('image', { uri: capturedPhoto.uri, name: 'manual.jpg', type: 'image/jpeg' });
      fd.append('bbox_x', String(x * capturedPhoto.width));
      fd.append('bbox_y', String(y * capturedPhoto.height));
      fd.append('bbox_width', String(width * capturedPhoto.width));
      fd.append('bbox_height', String(height * capturedPhoto.height));
      const res = await fetch(`${API_BASE_URL}/reconstruct-bbox`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload = await waitForReconstructionJob(await res.json());
      setSegmentResult(payload.preprocess);
      setReconstructionResult(payload.reconstruction);
      setProcessingStage('done'); setCameraStatus('3D model ready!');
    } catch (e) {
      setCameraStatus(`Error: ${shortErrorMessage(e.message)}`);
      setProcessingStage('error');
    } finally { setIsProcessing(false); }
  };

  const captureAndReconstructFull = async () => {
    try {
      setIsProcessing(true); setProcessingStage('capturing'); setCameraStatus('Capturing…');
      const photo = await cameraRef.current.takePictureAsync({ quality: CONFIG.RECON_CAPTURE_QUALITY });
      setCapturedPhoto(photo);
      const fd = new FormData();
      fd.append('image', { uri: photo.uri, name: 'obj.jpg', type: 'image/jpeg' });
      setCameraStatus('Generating mesh…');
      const res = await fetch(`${API_BASE_URL}/reconstruct-image`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload = await res.json();
      setReconstructionResult(payload.reconstruction || payload);
      setProcessingStage('done'); setCameraStatus('Reconstruction ready!');
    } catch (e) {
      setCameraStatus(`Error: ${shortErrorMessage(e.message)}`);
      setProcessingStage('error');
    } finally { setIsProcessing(false); }
  };

  const paintTexture = async () => {
    const jobId = reconstructionResult?.job_id; if (!jobId) return;
    setIsPaintingTexture(true); setProcessingStage('texturing'); setCameraStatus('Painting texture…');
    try {
      const fd = new FormData(); fd.append('job_id', jobId);
      const res = await fetch(`${API_BASE_URL}/paint-texture`, { method: 'POST', body: fd });
      let payload = await res.json();
      const statusPath = payload.status_url || `/texture-jobs/${jobId}`;
      const startAt = Date.now();
      while (payload.status !== 'done') {
        if (payload.status === 'error') throw new Error(payload.error);
        if (Date.now() - startAt > CONFIG.TEXTURE_POLL_TIMEOUT_MS) throw new Error('Timeout');
        setProcessingStage(`texturing · ${payload.status}`);
        await delay(CONFIG.TEXTURE_POLL_INTERVAL_MS);
        const sRes = await fetch(`${API_BASE_URL}${statusPath}`);
        payload = await sRes.json();
      }
      setReconstructionResult(payload.reconstruction || payload);
      setProcessingStage('done'); setCameraStatus('Textured model ready!');
    } catch (e) {
      setCameraStatus(`Texture error: ${shortErrorMessage(e.message)}`);
      setProcessingStage('error');
    } finally { setIsPaintingTexture(false); }
  };

  // ─── Bounding box helpers ──────────────────────────────────────────────────
  const imageRect = () => {
    if (!capturedPhoto || !cropAreaLayout.width) return null;
    const s = Math.min(cropAreaLayout.width / capturedPhoto.width, cropAreaLayout.height / capturedPhoto.height);
    const w = capturedPhoto.width * s, h = capturedPhoto.height * s;
    return {
      left: (cropAreaLayout.width - w) / 2,
      top: (cropAreaLayout.height - h) / 2,
      width: w, height: h,
    };
  };

  const statusColor = processingStage === 'done'
    ? C.green
    : processingStage === 'error'
    ? C.red
    : C.accentLight;

  // ══════════════════════════════════════════════════════════════════════════
  // CAMERA SCREEN
  // ══════════════════════════════════════════════════════════════════════════
  if (screen === 'camera') {
    const rect = imageRect();
    const manualStyle = rect && manualBbox ? {
      left: rect.left + manualBbox.x * rect.width,
      top: rect.top + manualBbox.y * rect.height,
      width: manualBbox.width * rect.width,
      height: manualBbox.height * rect.height,
    } : null;
    const meshPath = reconstructionResult?.files?.mesh_textured_glb || reconstructionResult?.files?.mesh_glb;
    const isDone = processingStage === 'done';
    const isError = processingStage === 'error';

    return (
      <View style={S.camScreen}>
        <StatusBar style="light" />

        <SaveModal
          visible={showSaveModal}
          saveName={saveName}
          setSaveName={setSaveName}
          onCancel={() => setShowSaveModal(false)}
          onSave={() => {
            if (saveName.trim()) {
              saveToHistory(reconstructionResult, segmentResult, saveName.trim());
              setShowSaveModal(false); setSaveName('');
              setCameraStatus('Saved to history!');
            }
          }}
        />

        <Viewer3DModal visible={show3DViewer} modelUrl={viewerModelUrl} onClose={() => setShow3DViewer(false)} />

        {/* Viewfinder */}
        {capturedPhoto ? (
          <View style={S.cropArea} onLayout={e => setCropAreaLayout(e.nativeEvent.layout)}>
            <Image source={{ uri: capturedPhoto.uri }} style={StyleSheet.absoluteFill} resizeMode="contain" />
            {!reconstructionResult && (
              <View
                style={StyleSheet.absoluteFill}
                onStartShouldSetResponder={() => !isProcessing}
                onMoveShouldSetResponder={() => !isProcessing}
                onResponderGrant={e => {
                  const px = clamp01((e.nativeEvent.locationX - (rect?.left || 0)) / (rect?.width || 1));
                  const py = clamp01((e.nativeEvent.locationY - (rect?.top || 0)) / (rect?.height || 1));
                  cropDragStartRef.current = { x: px, y: py };
                }}
                onResponderMove={e => {
                  const px = clamp01((e.nativeEvent.locationX - (rect?.left || 0)) / (rect?.width || 1));
                  const py = clamp01((e.nativeEvent.locationY - (rect?.top || 0)) / (rect?.height || 1));
                  setManualBbox({
                    x: Math.min(cropDragStartRef.current.x, px),
                    y: Math.min(cropDragStartRef.current.y, py),
                    width: Math.max(0.01, Math.abs(px - cropDragStartRef.current.x)),
                    height: Math.max(0.01, Math.abs(py - cropDragStartRef.current.y)),
                  });
                }}
              >
                {manualStyle && (
                  <View style={[S.cropBox, manualStyle]}>
                    <View style={[S.corner, S.cornerTL]} />
                    <View style={[S.corner, S.cornerTR]} />
                    <View style={[S.corner, S.cornerBL]} />
                    <View style={[S.corner, S.cornerBR]} />
                  </View>
                )}
              </View>
            )}
          </View>
        ) : (
          <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />
        )}

        {/* Overlay UI */}
        <View style={S.camUI} pointerEvents="box-none">
          {/* Top bar */}
          <View style={S.camTopBar}>
            <TouchableOpacity
              onPress={() => { clearObjectState(); setScreen('intro'); }}
              style={S.topBtn}
            >
              <Text style={S.topBtnText}>‹</Text>
            </TouchableOpacity>

            <View style={S.camTopCenter}>
              <LogoMark size={18} showText />
            </View>

            <View style={S.modeChip}>
              <Text style={S.modeChipText}>PHOTO</Text>
            </View>
          </View>

          {/* Spacer */}
          <View style={{ flex: 1 }} pointerEvents="none" />

          {/* Bottom panel */}
          <View style={S.camPanel}>
            {/* Timeline */}
            {processingStage !== '' && (
              <ProcessingTimeline stage={processingStage} />
            )}

            {/* Status message */}
            {cameraStatus !== '' && (
              <View style={[S.statusPill, { borderColor: `${statusColor}40` }]}>
                <View style={[S.statusDot, { backgroundColor: statusColor }]} />
                <Text style={[S.statusText, { color: statusColor }]}>{cameraStatus}</Text>
              </View>
            )}

            {/* Result actions */}
            {reconstructionResult && meshPath && (
              <View style={S.resultRow}>
                <TouchableOpacity style={S.actionBtn} onPress={() => open3DViewer(meshPath)} activeOpacity={0.8}>
                  <Text style={S.actionBtnIcon}>⬡</Text>
                  <Text style={S.actionBtnLabel}>Preview</Text>
                </TouchableOpacity>

                {reconstructionResult.backend === 'hunyuan_remote' && !reconstructionResult.files?.mesh_textured_glb && (
                  <TouchableOpacity
                    style={[S.actionBtn, S.actionBtnGreen]}
                    onPress={paintTexture}
                    disabled={isPaintingTexture}
                    activeOpacity={0.8}
                  >
                    {isPaintingTexture
                      ? <ActivityIndicator size="small" color={C.green} />
                      : <Text style={[S.actionBtnIcon, { color: C.green }]}>◈</Text>
                    }
                    <Text style={[S.actionBtnLabel, { color: C.green }]}>Texture</Text>
                  </TouchableOpacity>
                )}

                <TouchableOpacity
                  style={[S.actionBtn, S.actionBtnSave]}
                  onPress={() => setShowSaveModal(true)}
                  activeOpacity={0.8}
                >
                  <Text style={[S.actionBtnIcon, { color: C.accentLight }]}>↓</Text>
                  <Text style={[S.actionBtnLabel, { color: C.accentLight }]}>Save</Text>
                </TouchableOpacity>
              </View>
            )}

            {/* Primary actions */}
            <View style={S.camBtnRow}>
              {capturedPhoto ? (
                <>
                  <TouchableOpacity style={S.btnSec} onPress={clearObjectState} activeOpacity={0.8}>
                    <Text style={S.btnSecText}>Retake</Text>
                  </TouchableOpacity>
                  {!reconstructionResult && (
                    <TouchableOpacity
                      style={[S.btnPri, isProcessing && S.btnPriDisabled]}
                      onPress={manualBbox ? reconstructManualBbox : captureAndReconstructFull}
                      disabled={isProcessing}
                      activeOpacity={0.85}
                    >
                      {isProcessing
                        ? <ActivityIndicator color="#fff" />
                        : <Text style={S.btnPriText}>{manualBbox ? 'Reconstruct crop' : 'Reconstruct full'}</Text>
                      }
                    </TouchableOpacity>
                  )}
                </>
              ) : (
                <TouchableOpacity
                  style={S.btnPri}
                  onPress={async () => {
                    const p = await cameraRef.current.takePictureAsync({ quality: CONFIG.RECON_CAPTURE_QUALITY });
                    setCapturedPhoto(p);
                  }}
                  activeOpacity={0.85}
                >
                  <View style={S.captureCircle} />
                  <Text style={S.btnPriText}>Capture</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
        </View>
      </View>
    );
  }

  // ══════════════════════════════════════════════════════════════════════════
  // INTRO SCREEN
  // ══════════════════════════════════════════════════════════════════════════
  return (
    <View style={S.intro}>
      <StatusBar style="light" />

      <HistoryModal
        visible={showHistory}
        history={history}
        onClose={() => setShowHistory(false)}
        onTexture={textureHistoryItem}
        onExport={exportHistoryItem}
        onPreview={(item) => { setShowHistory(false); open3DViewer(item.meshPath); }}
        onDelete={deleteHistoryItem}
        getServerFileUrl={getServerFileUrl}
      />

      <Viewer3DModal visible={show3DViewer} modelUrl={viewerModelUrl} onClose={() => setShow3DViewer(false)} />

      <ScrollView
        contentContainerStyle={S.introScroll}
        showsVerticalScrollIndicator={false}
      >
        {/* Hero */}
        <Animated.View style={[S.hero, { opacity: fadeAnim, transform: [{ scale: heroScale }] }]}>
          {/* Ambient hexes */}
          <View style={S.hexBg} pointerEvents="none">
            {HEX_POSITIONS.map((h, i) => (
              <FloatingHex key={i} x={h.x} y={h.y} size={h.s} delay={i * 320} />
            ))}
          </View>

          {/* Logo */}
          <View style={S.logoRow}>
            <LogoMark size={40} showText pulse />
          </View>

          {/* Tagline */}
          <Text style={S.heroTitle}>Photo → 3D mesh</Text>
          <Text style={S.heroSub}>
            Point. Tap. Get a textured 3D model ready for Blender or Unity.
          </Text>

          {/* Stats row */}
          <View style={S.statsRow}>
            {[
              { v: 'GLB', l: 'format' },
              { v: 'GPU', l: 'backend' },
              { v: '3D', l: 'output' },
            ].map(({ v, l }) => (
              <View key={l} style={S.stat}>
                <Text style={S.statVal}>{v}</Text>
                <Text style={S.statLabel}>{l}</Text>
              </View>
            ))}
          </View>
        </Animated.View>

        {/* Feature cards */}
        <Animated.View style={[S.cards, { opacity: fadeAnim }]}>
          {[
            {
              icon: '◉',
              title: 'Crop or full frame',
              desc: 'Drag to select a region, or send the full image.',
              color: C.accentLight,
            },
            {
              icon: '◈',
              title: 'AI texturing',
              desc: 'Paint realistic surface materials after mesh generation.',
              color: C.green,
            },
            {
              icon: '↓',
              title: 'Export GLB',
              desc: 'Download and use anywhere — Blender, Unity, Three.js.',
              color: C.amber,
            },
          ].map(({ icon, title, desc, color }) => (
            <View key={title} style={S.featureCard}>
              <View style={[S.featureIcon, { backgroundColor: `${color}18` }]}>
                <Text style={[S.featureIconText, { color }]}>{icon}</Text>
              </View>
              <View style={S.featureMeta}>
                <Text style={S.featureTitle}>{title}</Text>
                <Text style={S.featureDesc}>{desc}</Text>
              </View>
            </View>
          ))}
        </Animated.View>

        {/* History entry */}
        {history.length > 0 && (
          <Animated.View style={{ opacity: fadeAnim }}>
            <TouchableOpacity style={S.historyEntry} onPress={() => setShowHistory(true)} activeOpacity={0.8}>
              <View style={S.historyEntryLeft}>
                <View style={S.historyEntryIcon}>
                  <Text style={{ color: C.accentLight, fontSize: 16 }}>⟳</Text>
                </View>
                <View>
                  <Text style={S.historyEntryTitle}>Recent scans</Text>
                  <Text style={S.historyEntrySub}>{history.length} saved</Text>
                </View>
              </View>
              <Text style={S.historyChevron}>›</Text>
            </TouchableOpacity>
          </Animated.View>
        )}

        <View style={{ height: 160 }} />
      </ScrollView>

      {/* Sticky CTA */}
      <View style={S.stickyFooter}>
        <TouchableOpacity style={S.ctaBtn} onPress={openCamera} activeOpacity={0.88}>
          <View style={S.ctaBtnInner}>
            <View style={S.ctaIconWrap}>
              <Text style={S.ctaIcon}>◉</Text>
            </View>
            <Text style={S.ctaText}>Open camera</Text>
            <Text style={S.ctaArrow}>→</Text>
          </View>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// STYLES
// ══════════════════════════════════════════════════════════════════════════════
const S = StyleSheet.create({
  // ── Intro ────────────────────────────────────────────────────────────────
  intro: { flex: 1, backgroundColor: C.bg },
  introScroll: {
    paddingTop: Platform.OS === 'ios' ? 58 : 32,
    paddingHorizontal: 20,
  },

  // Hero
  hero: {
    borderRadius: 28,
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    padding: 28,
    marginBottom: 16,
    overflow: 'hidden',
    minHeight: 260,
  },
  hexBg: { ...StyleSheet.absoluteFillObject },
  logoRow: { marginBottom: 20 },
  heroTitle: {
    color: C.textPrimary,
    fontSize: 28,
    fontWeight: '900',
    letterSpacing: -0.8,
    marginBottom: 8,
  },
  heroSub: {
    color: C.textSecondary,
    fontSize: 14,
    lineHeight: 21,
    maxWidth: '85%',
    marginBottom: 24,
  },
  statsRow: {
    flexDirection: 'row',
    gap: 12,
  },
  stat: {
    backgroundColor: C.bgCardAlt,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderWidth: 1,
    borderColor: C.borderSubtle,
    alignItems: 'center',
  },
  statVal: {
    color: C.accentLight,
    fontSize: 13,
    fontWeight: '900',
    letterSpacing: 0.5,
  },
  statLabel: {
    color: C.textMuted,
    fontSize: 9,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginTop: 1,
  },

  // Feature cards
  cards: { gap: 10, marginBottom: 16 },
  featureCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.bgCard,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: C.borderSubtle,
    padding: 16,
    gap: 14,
  },
  featureIcon: {
    width: 44,
    height: 44,
    borderRadius: 13,
    alignItems: 'center',
    justifyContent: 'center',
  },
  featureIconText: { fontSize: 20 },
  featureMeta: { flex: 1 },
  featureTitle: {
    color: C.textPrimary,
    fontWeight: '700',
    fontSize: 14,
    marginBottom: 3,
  },
  featureDesc: {
    color: C.textMuted,
    fontSize: 12,
    lineHeight: 17,
  },

  // History entry
  historyEntry: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.bgCard,
    borderRadius: 18,
    padding: 16,
    borderWidth: 1,
    borderColor: C.borderActive,
    marginBottom: 8,
  },
  historyEntryLeft: { flexDirection: 'row', alignItems: 'center', gap: 12, flex: 1 },
  historyEntryIcon: {
    width: 38,
    height: 38,
    borderRadius: 11,
    backgroundColor: C.accentGlow,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: C.borderActive,
  },
  historyEntryTitle: { color: C.textPrimary, fontWeight: '700', fontSize: 15 },
  historyEntrySub: { color: C.textMuted, fontSize: 12, marginTop: 1 },
  historyChevron: { color: C.textMuted, fontSize: 22 },

  // Sticky footer CTA
  stickyFooter: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: Platform.OS === 'ios' ? 36 : 20,
    backgroundColor: `${C.bg}F0`,
  },
  ctaBtn: {
    borderRadius: 22,
    overflow: 'hidden',
    backgroundColor: C.accent,
    shadowColor: C.accent,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.45,
    shadowRadius: 16,
    elevation: 10,
  },
  ctaBtnInner: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingVertical: 18,
    gap: 12,
  },
  ctaIconWrap: {
    width: 32,
    height: 32,
    borderRadius: 9,
    backgroundColor: 'rgba(255,255,255,0.18)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  ctaIcon: { color: '#fff', fontSize: 16 },
  ctaText: { flex: 1, color: '#fff', fontSize: 17, fontWeight: '800' },
  ctaArrow: { color: 'rgba(255,255,255,0.6)', fontSize: 20 },

  // ── Camera screen ────────────────────────────────────────────────────────
  camScreen: { flex: 1, backgroundColor: '#000' },
  cropArea: { flex: 1, backgroundColor: '#000' },
  camUI: { ...StyleSheet.absoluteFillObject },

  // Top bar
  camTopBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 58 : 38,
    paddingBottom: 12,
  },
  topBtn: {
    width: 42,
    height: 42,
    borderRadius: 14,
    backgroundColor: 'rgba(0,0,0,0.55)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  topBtnText: { color: '#fff', fontSize: 22 },
  camTopCenter: { flex: 1, alignItems: 'center' },
  modeChip: {
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  modeChipText: { color: '#fff', fontSize: 10, fontWeight: '800', letterSpacing: 1.2 },

  // Bounding box
  cropBox: {
    position: 'absolute',
    borderWidth: 1.5,
    borderColor: C.accentLight,
    backgroundColor: `${C.accentGlow}50`,
  },
  corner: {
    position: 'absolute',
    width: 14,
    height: 14,
    borderColor: C.accentLight,
  },
  cornerTL: { top: -1, left: -1, borderTopWidth: 3, borderLeftWidth: 3 },
  cornerTR: { top: -1, right: -1, borderTopWidth: 3, borderRightWidth: 3 },
  cornerBL: { bottom: -1, left: -1, borderBottomWidth: 3, borderLeftWidth: 3 },
  cornerBR: { bottom: -1, right: -1, borderBottomWidth: 3, borderRightWidth: 3 },

  // Bottom panel
  camPanel: {
    backgroundColor: `${C.bgCard}FA`,
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: Platform.OS === 'ios' ? 36 : 20,
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    borderTopWidth: 1,
    borderTopColor: C.borderSubtle,
  },

  // Status
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'center',
    backgroundColor: C.bgCardAlt,
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 8,
    marginBottom: 14,
    gap: 7,
    borderWidth: 1,
  },
  statusDot: { width: 6, height: 6, borderRadius: 3 },
  statusText: { fontSize: 12, fontWeight: '700' },

  // Result row
  resultRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 16,
  },
  actionBtn: {
    flex: 1,
    height: 52,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: C.border,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
    backgroundColor: C.bgCardAlt,
  },
  actionBtnGreen: {
    borderColor: C.greenBorder,
    backgroundColor: C.greenDim,
  },
  actionBtnSave: {
    borderColor: C.borderActive,
    backgroundColor: C.accentGlow,
  },
  actionBtnIcon: { fontSize: 17, color: C.accentLight },
  actionBtnLabel: { fontSize: 12, fontWeight: '700', color: C.textPrimary },

  // Buttons
  camBtnRow: { flexDirection: 'row', gap: 12 },
  btnPri: {
    flex: 2,
    height: 58,
    backgroundColor: C.accent,
    borderRadius: 18,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    shadowColor: C.accent,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.35,
    shadowRadius: 10,
    elevation: 6,
  },
  btnPriDisabled: { opacity: 0.55 },
  btnPriText: { color: '#fff', fontSize: 15, fontWeight: '800' },
  captureCircle: {
    width: 10, height: 10, borderRadius: 5,
    backgroundColor: 'rgba(255,255,255,0.85)',
  },
  btnSec: {
    flex: 1,
    height: 58,
    backgroundColor: C.bgCardAlt,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: C.border,
  },
  btnSecText: { color: C.textSecondary, fontSize: 15, fontWeight: '700' },
});