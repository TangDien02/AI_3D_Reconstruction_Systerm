import { useEffect, useRef, useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import {
  Animated, Dimensions, Image, Linking, Platform,
  ScrollView, StyleSheet, Text, TouchableOpacity, View, Modal, ActivityIndicator,
  TextInput
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

export default function App() {
  const cameraRef = useRef(null);
  const fadeAnim = useRef(new Animated.Value(0)).current;

  // --- UI State ---
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

  // --- Data State ---
  const [capturedPhoto, setCapturedPhoto] = useState(null);
  const [cropAreaLayout, setCropAreaLayout] = useState({ width: 0, height: 0 });
  const [manualBbox, setManualBbox] = useState(null);
  const [segmentResult, setSegmentResult] = useState(null);
  const [reconstructionResult, setReconstructionResult] = useState(null);
  const [isPaintingTexture, setIsPaintingTexture] = useState(false);
  const cropDragStartRef = useRef(null);

  // --- Effects ---
  useEffect(() => {
    const loadHistory = async () => {
      try {
        const stored = await AsyncStorage.getItem(STORAGE_KEY);
        if (stored) setHistory(JSON.parse(stored));
      } catch (e) { console.error('Failed to load history', e); }
    };
    loadHistory();
  }, []);

  useEffect(() => {
    Animated.timing(fadeAnim, { toValue: 1, duration: 320, useNativeDriver: true }).start();
  }, [screen]);

  // --- Logic Helpers ---
  const getServerFileUrl = (path) => path ? (path.startsWith('http') ? path : `${API_BASE_URL}${path}`) : null;
  
  const clearObjectState = () => {
    setCapturedPhoto(null); setManualBbox(null); setSegmentResult(null);
    setReconstructionResult(null); setIsPaintingTexture(false); setProcessingStage('');
    setCameraStatus('');
  };

  const saveToHistory = async (reconstruction, segment, label) => {
    try {
      const isUpdate = history.some(i => i.id === reconstruction.job_id);
      const newItem = {
        id: reconstruction.job_id || Date.now().toString(),
        label: label || 'Untitled Object', timestamp: Date.now(),
        meshPath: reconstruction.files?.mesh_textured_glb || reconstruction.files?.mesh_glb,
        thumbPath: segment?.files?.clean_image || segment?.files?.crop || reconstruction.files?.input_image || reconstruction.files?.input_original,
        isTextured: !!reconstruction.files?.mesh_textured_glb,
        backend: reconstruction.backend, meshSummary: reconstruction.mesh || {},
      };
      let updated;
      if (isUpdate) {
        updated = history.map(item => item.id === newItem.id ? { ...item, ...newItem, label: item.label } : item);
      } else {
        updated = [newItem, ...history.filter(i => i.id !== newItem.id)].slice(0, 20);
      }
      setHistory(updated);
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch (e) { console.error('Failed to save history', e); }
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
    if (url) {
      if (Platform.OS === 'web') { window.open(url, '_blank'); }
      else { Linking.openURL(url); }
    }
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

  // --- API Actions ---
  const waitForReconstructionJob = async (payload) => {
    if (payload.status === 'done') return payload;
    const statusPath = payload.status_url || `/reconstruction-jobs/${payload.job_id}`;
    const startedAt = Date.now(); let current = payload;
    while (current.status !== 'done') {
      if (['failed', 'error'].includes(current.status)) throw new Error(typeof current.error === 'string' ? current.error : JSON.stringify(current.error));
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
    setIsProcessing(true); setProcessingStage('preprocess'); setCameraStatus('Processing...');
    try {
      const padded = manualBbox;
      const bbox = { x: padded.x * capturedPhoto.width, y: padded.y * capturedPhoto.height, width: padded.width * capturedPhoto.width, height: padded.height * capturedPhoto.height };
      const fd = new FormData();
      fd.append('image', { uri: capturedPhoto.uri, name: 'manual.jpg', type: 'image/jpeg' });
      fd.append('bbox_x', String(bbox.x)); fd.append('bbox_y', String(bbox.y));
      fd.append('bbox_width', String(bbox.width)); fd.append('bbox_height', String(bbox.height));
      const res = await fetch(`${API_BASE_URL}/reconstruct-bbox`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload = await waitForReconstructionJob(await res.json());
      setSegmentResult(payload.preprocess); setReconstructionResult(payload.reconstruction);
      setProcessingStage('done'); setCameraStatus('3D model ready!');
    } catch (e) { setCameraStatus(`Error: ${shortErrorMessage(e.message)}`); setProcessingStage('error'); }
    finally { setIsProcessing(false); }
  };

  const captureAndReconstructFull = async () => {
    try {
      setIsProcessing(true); setProcessingStage('capturing'); setCameraStatus('Capturing...');
      const photo = await cameraRef.current.takePictureAsync({ quality: CONFIG.RECON_CAPTURE_QUALITY });
      setCapturedPhoto(photo);
      
      const fd = new FormData(); 
      fd.append('image', { uri: photo.uri, name: 'obj.jpg', type: 'image/jpeg' });
      
      setCameraStatus('Generating Mesh...');
      const res = await fetch(`${API_BASE_URL}/reconstruct-image`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      
      const payload = await res.json();
      setReconstructionResult(payload.reconstruction || payload); 
      setProcessingStage('done'); 
      setCameraStatus('Reconstruction ready!');
    } catch (e) {
      setCameraStatus(`Error: ${shortErrorMessage(e.message)}`);
      setProcessingStage('error');
    } finally {
      setIsProcessing(false);
    }
  };

  const paintTexture = async () => {
    const jobId = reconstructionResult?.job_id; if (!jobId) return;
    setIsPaintingTexture(true); setProcessingStage('texturing'); setCameraStatus('Painting texture...');
    try {
      const fd = new FormData(); fd.append('job_id', jobId);
      const res = await fetch(`${API_BASE_URL}/paint-texture`, { method: 'POST', body: fd });
      let payload = await res.json();
      const statusPath = payload.status_url || `/texture-jobs/${jobId}`;
      const startAt = Date.now();
      while (payload.status !== 'done') {
        if (payload.status === 'error') throw new Error(payload.error);
        if (Date.now() - startAt > CONFIG.TEXTURE_POLL_TIMEOUT_MS) throw new Error('Timeout');
        setProcessingStage(`texturing · ${payload.status}`); await delay(CONFIG.TEXTURE_POLL_INTERVAL_MS);
        const sRes = await fetch(`${API_BASE_URL}${statusPath}`);
        payload = await sRes.json();
      }
      setReconstructionResult(payload.reconstruction || payload); setProcessingStage('done'); setCameraStatus('Textured model ready!');
    } catch (e) { setCameraStatus(`Texture error: ${shortErrorMessage(e.message)}`); setProcessingStage('error'); }
    finally { setIsPaintingTexture(false); }
  };

  // --- Render Helpers ---
  const imageRect = () => {
    if (!capturedPhoto || !cropAreaLayout.width) return null;
    const s = Math.min(cropAreaLayout.width / capturedPhoto.width, cropAreaLayout.height / capturedPhoto.height);
    const w = capturedPhoto.width * s, h = capturedPhoto.height * s;
    return { left: (cropAreaLayout.width - w) / 2, top: (cropAreaLayout.height - h) / 2, width: w, height: h };
  };

  // --- Main Render ---
  if (screen === 'camera') {
    const rect = imageRect();
    const manualStyle = rect && manualBbox ? { left: rect.left + manualBbox.x * rect.width, top: rect.top + manualBbox.y * rect.height, width: manualBbox.width * rect.width, height: manualBbox.height * rect.height } : null;
    const meshPath = reconstructionResult?.files?.mesh_textured_glb || reconstructionResult?.files?.mesh_glb;

    return (
      <View style={S.camScreen}>
        <SaveModal
          visible={showSaveModal}
          saveName={saveName}
          setSaveName={setSaveName}
          onCancel={() => setShowSaveModal(false)}
          onSave={() => {
            if (saveName.trim()) {
              saveToHistory(reconstructionResult, segmentResult, saveName.trim());
              setShowSaveModal(false);
              setSaveName('');
              setCameraStatus('Saved to history!');
            }
          }}
          styles={S}
        />

        <Viewer3DModal visible={show3DViewer} modelUrl={viewerModelUrl} onClose={() => setShow3DViewer(false)} />
        
        {capturedPhoto ? (
          <View style={S.cropArea} onLayout={e => setCropAreaLayout(e.nativeEvent.layout)}>
            <Image source={{ uri: capturedPhoto.uri }} style={StyleSheet.absoluteFill} resizeMode="contain" />
            {!reconstructionResult && (
              <View style={StyleSheet.absoluteFill}
                onStartShouldSetResponder={() => !isProcessing} onMoveShouldSetResponder={() => !isProcessing}
                onResponderGrant={e => { const p = clamp01((e.nativeEvent.locationX - (rect?.left || 0)) / (rect?.width || 1)); cropDragStartRef.current = { x: p, y: clamp01((e.nativeEvent.locationY - (rect?.top || 0)) / (rect?.height || 1)) }; }}
                onResponderMove={e => { const p = clamp01((e.nativeEvent.locationX - (rect?.left || 0)) / (rect?.width || 1)); const py = clamp01((e.nativeEvent.locationY - (rect?.top || 0)) / (rect?.height || 1)); setManualBbox({ x: Math.min(cropDragStartRef.current.x, p), y: Math.min(cropDragStartRef.current.y, py), width: Math.max(0.01, Math.abs(p - cropDragStartRef.current.x)), height: Math.max(0.01, Math.abs(py - cropDragStartRef.current.y)) }); }}
              >
                {manualStyle && (
                  <View style={[S.manualBox, manualStyle]}>
                    <View style={S.cropCorners}>
                      <View style={[S.corner, { top: -2, left: -2, borderTopWidth: 3, borderLeftWidth: 3 }]} />
                      <View style={[S.corner, { top: -2, right: -2, borderTopWidth: 3, borderRightWidth: 3 }]} />
                      <View style={[S.corner, { bottom: -2, left: -2, borderBottomWidth: 3, borderLeftWidth: 3 }]} />
                      <View style={[S.corner, { bottom: -2, right: -2, borderBottomWidth: 3, borderRightWidth: 3 }]} />
                    </View>
                  </View>
                )}
              </View>
            )}
          </View>
        ) : (
          <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />
        )}

        <View style={S.camUI} pointerEvents="box-none">
          <View style={S.camTop}>
            <TouchableOpacity onPress={() => { clearObjectState(); setScreen('intro'); }} style={S.roundBtn}><Text style={{ color: '#fff', fontSize: 24 }}>‹</Text></TouchableOpacity>
            <View style={S.camTitleBox}>
              <LogoMark size={20} showText />
            </View>
            <View style={[S.badge]}>
              <Text style={S.badgeText}>PHOTO</Text>
            </View>
          </View>
          <View style={{ flex: 1 }} pointerEvents="none" />
          <View style={S.camPanel}>
            {processingStage !== '' && <ProcessingTimeline stage={processingStage} />}
            {cameraStatus !== '' && (
              <View style={S.statusRow}>
                <Text style={S.statusText}>{cameraStatus}</Text>
              </View>
            )}
            {reconstructionResult && meshPath && (
              <View style={S.resultRow}>
                <TouchableOpacity style={S.actionBtn} onPress={() => open3DViewer(meshPath)}>
                  <Text style={S.actionBtnIcon}>⬡</Text>
                  <Text style={S.actionBtnText}>Preview</Text>
                </TouchableOpacity>
                {reconstructionResult.backend === 'hunyuan_remote' && !reconstructionResult.files?.mesh_textured_glb && (
                  <TouchableOpacity style={[S.actionBtn, S.actionBtnGreen]} onPress={paintTexture} disabled={isPaintingTexture}>
                    <Text style={[S.actionBtnIcon, { color: C.green }]}>◈</Text>
                    <Text style={[S.actionBtnText, { color: C.green }]}>Texture</Text>
                  </TouchableOpacity>
                )}
                <TouchableOpacity style={[S.actionBtn, { borderColor: C.accentActive }]} onPress={() => setShowSaveModal(true)}>
                  <Text style={[S.actionBtnIcon, { color: C.accentLight }]}>💾</Text>
                  <Text style={[S.actionBtnText, { color: C.white }]}>Save</Text>
                </TouchableOpacity>
              </View>
            )}
            <View style={S.btnRow}>
              {capturedPhoto ? (
                <>
                  <TouchableOpacity style={S.secBtn} onPress={clearObjectState}><Text style={S.secBtnText}>Retake</Text></TouchableOpacity>
                  {!reconstructionResult && (
                    <TouchableOpacity style={S.priBtn} onPress={manualBbox ? reconstructManualBbox : captureAndReconstructFull} disabled={isProcessing}>
                      {isProcessing ? <ActivityIndicator color="#fff" /> : <Text style={S.priBtnText}>{manualBbox ? 'Reconstruct Crop' : 'Reconstruct Full'}</Text>}
                    </TouchableOpacity>
                  )}
                </>
              ) : (
                <TouchableOpacity style={S.priBtn} onPress={async () => {
                  const p = await cameraRef.current.takePictureAsync({ quality: CONFIG.RECON_CAPTURE_QUALITY });
                  setCapturedPhoto(p);
                }}>
                  <Text style={S.priBtnText}>Capture Image</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
        </View>
      </View>
    );
  }

  return (
    <View style={S.intro}>
      <HistoryModal
        visible={showHistory}
        history={history}
        onClose={() => setShowHistory(false)}
        onTexture={textureHistoryItem}
        onExport={exportHistoryItem}
        onPreview={(item) => { setShowHistory(false); open3DViewer(item.meshPath); }}
        onDelete={deleteHistoryItem}
        getServerFileUrl={getServerFileUrl}
        styles={S}
      />

      <Viewer3DModal visible={show3DViewer} modelUrl={viewerModelUrl} onClose={() => setShow3DViewer(false)} />
      
      <View style={{ flex: 1, paddingTop: Platform.OS === 'ios' ? 50 : 0 }}>
        <ScrollView contentContainerStyle={{ padding: 24, paddingBottom: 40 }}>
          <View style={S.hero}>
            <LogoMark size={48} showText />
            <Text style={S.heroSub}>Capture an object to generate a 3D model.</Text>
          </View>

          <View style={S.cardGrid}>
            <View style={S.glassCard}>
              <Text style={S.cardIcon}>📷</Text>
              <Text style={S.cardTitle}>Capture</Text>
              <Text style={S.cardDesc}>Take a photo and crop the object you want.</Text>
            </View>
            <View style={S.glassCard}>
              <Text style={S.cardIcon}>⬡</Text>
              <Text style={S.cardTitle}>Export</Text>
              <Text style={S.cardDesc}>Download GLB files for Blender or Unity.</Text>
            </View>
          </View>

          {history.length > 0 && (
            <TouchableOpacity style={S.historyEntry} onPress={() => setShowHistory(true)}>
              <View style={S.historyEntryIcon}><Text style={{ color: C.accentLight, fontSize: 18 }}>⟳</Text></View>
              <View style={{ flex: 1 }}>
                <Text style={S.historyEntryText}>Recent Scans</Text>
                <Text style={S.historyEntrySub}>{history.length} items saved</Text>
              </View>
              <Text style={{ color: C.textMuted, fontSize: 24 }}>›</Text>
            </TouchableOpacity>
          )}
        </ScrollView>

        <View style={S.footer}>
          <TouchableOpacity style={S.mainBtn} onPress={openCamera} activeOpacity={0.8}>
            <Text style={S.mainBtnText}>Open Camera</Text>
            <Text style={S.mainBtnArrow}>→</Text>
          </TouchableOpacity>
        </View>
      </View>
      <StatusBar style="light" />
    </View>
  );
}

const S = StyleSheet.create({
  intro: { flex: 1, backgroundColor: C.bg },
  hero: { marginTop: 60, marginBottom: 40, alignItems: 'center' },
  heroSub: { fontSize: 15, color: C.textSecondary, textAlign: 'center', marginTop: 16, lineHeight: 22, maxWidth: '80%' },
  cardGrid: { flexDirection: 'row', gap: 12, marginBottom: 24 },
  glassCard: { flex: 1, backgroundColor: C.bgCardAlt, padding: 20, borderRadius: 24, borderWidth: 1, borderColor: C.border },
  cardIcon: { fontSize: 24, marginBottom: 12 },
  cardTitle: { color: C.textPrimary, fontWeight: '700', fontSize: 14, marginBottom: 6 },
  cardDesc: { color: C.textMuted, fontSize: 11, lineHeight: 16 },
  footer: { padding: 24, backgroundColor: C.bg },
  mainBtn: { backgroundColor: C.accent, height: 64, borderRadius: 20, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 12, elevation: 8, shadowColor: C.accent, shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 8 },
  mainBtnText: { color: '#fff', fontSize: 18, fontWeight: '800' },
  mainBtnArrow: { color: 'rgba(255,255,255,0.6)', fontSize: 20 },
  historyEntry: { backgroundColor: C.bgCardAlt, padding: 16, borderRadius: 20, flexDirection: 'row', alignItems: 'center', gap: 16, borderWidth: 1, borderColor: C.borderActive },
  historyEntryIcon: { width: 40, height: 40, borderRadius: 12, backgroundColor: C.accentGlow, alignItems: 'center', justifyContent: 'center' },
  historyEntryText: { color: C.textPrimary, fontWeight: '700', fontSize: 16 },
  historyEntrySub: { color: C.textMuted, fontSize: 12 },

  camScreen: { flex: 1, backgroundColor: '#000' },
  cropArea: { flex: 1, backgroundColor: '#000' },
  camUI: { ...StyleSheet.absoluteFillObject },
  camTop: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingTop: Platform.OS === 'ios' ? 60 : 40 },
  roundBtn: { width: 44, height: 44, borderRadius: 22, backgroundColor: 'rgba(0,0,0,0.6)', alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)' },
  camTitleBox: { flex: 1, alignItems: 'center' },
  badge: { backgroundColor: 'rgba(0,0,0,0.6)', paddingHorizontal: 10, paddingVertical: 6, borderRadius: 10, borderWidth: 1, borderColor: C.border },
  badgeText: { color: C.white, fontSize: 10, fontWeight: '800', letterSpacing: 1 },
  camPanel: { backgroundColor: 'rgba(10,14,26,0.98)', padding: 20, borderTopLeftRadius: 32, borderTopRightRadius: 32, borderTopWidth: 1, borderTopColor: C.border },
  statusRow: { backgroundColor: C.bgCardAlt, borderRadius: 12, paddingVertical: 8, paddingHorizontal: 12, marginBottom: 16, alignSelf: 'center' },
  statusText: { color: C.accentLight, fontSize: 12, fontWeight: '600' },
  btnRow: { flexDirection: 'row', gap: 12 },
  priBtn: { flex: 1.5, height: 58, backgroundColor: C.accent, borderRadius: 18, alignItems: 'center', justifyContent: 'center' },
  priBtnText: { color: '#fff', fontSize: 16, fontWeight: '800' },
  secBtn: { flex: 1, height: 58, backgroundColor: C.bgCardAlt, borderRadius: 18, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: C.border },
  secBtnText: { color: C.textSecondary, fontSize: 15, fontWeight: '700' },
  manualBox: { position: 'absolute', borderWidth: 1.5, borderColor: C.accentLight, backgroundColor: 'rgba(45,107,228,0.08)' },
  cropCorners: { flex: 1, position: 'relative' },
  corner: { position: 'absolute', width: 15, height: 15, borderColor: C.accentLight },
  resultRow: { flexDirection: 'row', gap: 10, marginBottom: 20 },
  actionBtn: { flex: 1, height: 48, borderRadius: 14, borderWidth: 1, borderColor: C.border, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: C.bgCardAlt },
  actionBtnGreen: { borderColor: 'rgba(16,217,138,0.3)', backgroundColor: 'rgba(16,217,138,0.05)' },
  actionBtnIcon: { fontSize: 18, color: C.accentLight },
  actionBtnText: { fontSize: 13, fontWeight: '700', color: C.textPrimary },

  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.8)', justifyContent: 'flex-end' },
  modalContent: { backgroundColor: C.bg, height: '85%', borderTopLeftRadius: 32, borderTopRightRadius: 32, overflow: 'hidden' },
  modalHeader: { flexDirection: 'row', alignItems: 'center', padding: 24, borderBottomWidth: 1, borderBottomColor: C.border },
  modalTitle: { flex: 1, marginLeft: 12, fontSize: 20, fontWeight: '800', color: C.white },
  closeBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: C.bgCardAlt, alignItems: 'center', justifyContent: 'center' },
  closeTxt: { color: C.textSecondary, fontSize: 18 },
  historyCard: { flexDirection: 'row', alignItems: 'center', backgroundColor: C.bgCardAlt, padding: 14, borderRadius: 24, marginBottom: 12, borderWidth: 1, borderColor: C.border },
  historyThumb: { width: 56, height: 56, borderRadius: 12, backgroundColor: '#000' },
  historyLabel: { color: C.textPrimary, fontWeight: '700', fontSize: 15, marginBottom: 2 },
  historyDate: { color: C.textMuted, fontSize: 11 },
  editInput: { color: C.white, backgroundColor: C.bg, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 4, fontSize: 15, fontWeight: '700', borderWidth: 1, borderColor: C.accent },
  historyActions: { flexDirection: 'row', gap: 8 },
  historyBtn: { width: 38, height: 38, borderRadius: 12, backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: C.border },
  emptyBox: { alignItems: 'center', marginTop: 100 },
  emptyIcon: { fontSize: 60, color: C.textMuted, marginBottom: 20 },
  emptyHistory: { color: C.textMuted, fontSize: 16, fontWeight: '600' },
});
