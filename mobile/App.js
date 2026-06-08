import { useEffect, useRef, useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import {
  Alert,
  Animated,
  Dimensions,
  Image,
  Linking,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Modal,
  ActivityIndicator,
} from 'react-native';
import { WebView } from 'react-native-webview';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

const defaultApiBaseUrl = Platform.select({
  android: 'http://10.0.2.2:8000',
  ios: 'http://localhost:8000',
  default: 'http://localhost:8000',
});
const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL || defaultApiBaseUrl;
const DETECT_FRAME_WIDTH = 640;
const DETECT_CAPTURE_QUALITY = 0.5;
const DETECT_UPLOAD_COMPRESS = 0.65;
const DETECT_COOLDOWN_MS = 350;
const DETECT_EMPTY_HOLD_MS = 900;
const RECON_CAPTURE_QUALITY = 0.92;
const RECON_POLL_INTERVAL_MS = 5000;
const RECON_POLL_TIMEOUT_MS = 45 * 60 * 1000;
const TEXTURE_POLL_INTERVAL_MS = 5000;
const TEXTURE_POLL_TIMEOUT_MS = 45 * 60 * 1000;
const AR_MODEL_TITLE = 'Recon 3D Model';

// ─── COLORS ────────────────────────────────────────────────────────────────────
const C = {
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

// ─── HELPERS ────────────────────────────────────────────────────────────────────
const buildAndroidSceneViewerUrl = (modelUrl, title = AR_MODEL_TITLE) => {
  const encodedModelUrl = encodeURIComponent(modelUrl);
  return (
    `intent://arvr.google.com/scene-viewer/1.0?file=${encodedModelUrl}`
    + `&mode=ar_preferred&title=${encodeURIComponent(title)}`
    + '#Intent;scheme=https;package=com.google.android.googlequicksearchbox;'
    + `action=android.intent.action.VIEW;S.browser_fallback_url=${encodedModelUrl};end;`
  );
};

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const shortErrorMessage = (message) => {
  if (!message) return 'Unknown error';
  return String(message).replace(/\s+/g, ' ').slice(0, 240);
};

// ─── LOGO SVG COMPONENT ─────────────────────────────────────────────────────────
const LogoMark = ({ size = 28 }) => (
  <View style={{ width: size, height: size, borderRadius: 8, backgroundColor: C.accent, alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
    <Text style={{ color: '#fff', fontWeight: '900', fontSize: size * 0.45, letterSpacing: -1 }}>3D</Text>
  </View>
);

// ─── INLINE 3D VIEWER HTML ────────────────────────────────────────────────────
const build3DViewerHTML = (modelUrl) => `<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;background:#0A0E1A;overflow:hidden}
canvas{display:block;width:100%!important;height:100%!important}
#info{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);
  background:rgba(0,0,0,0.6);color:#8A9DC4;font-size:11px;font-family:sans-serif;
  padding:6px 14px;border-radius:20px;pointer-events:none;white-space:nowrap}
#loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  color:#5B8FEF;font-size:13px;font-family:sans-serif;text-align:center}
.spinner{width:36px;height:36px;border:2px solid rgba(91,143,239,0.2);
  border-top-color:#5B8FEF;border-radius:50%;animation:spin 0.9s linear infinite;margin:0 auto 12px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div id="loading"><div class="spinner"></div>Loading 3D model...</div>
<div id="info">Drag to rotate · Pinch to zoom</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function(){
  var loader=document.createElement('script');
  loader.src='https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js';
  loader.onload=function(){
    var controls=document.createElement('script');
    controls.src='https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js';
    controls.onload=initScene;
    document.head.appendChild(controls);
  };
  document.head.appendChild(loader);
})();

function initScene(){
  var loadEl=document.getElementById('loading');
  var scene=new THREE.Scene();
  scene.background=new THREE.Color(0x0A0E1A);

  var w=window.innerWidth,h=window.innerHeight;
  var camera=new THREE.PerspectiveCamera(45,w/h,0.01,100);
  camera.position.set(0,0,2.5);

  var renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.setSize(w,h);
  renderer.outputEncoding=THREE.sRGBEncoding;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure=1.2;
  document.body.appendChild(renderer.domElement);

  var controls=new THREE.OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true;
  controls.dampingFactor=0.08;
  controls.autoRotate=true;
  controls.autoRotateSpeed=1.2;

  var ambientLight=new THREE.AmbientLight(0xffffff,0.6);
  scene.add(ambientLight);
  var dirLight1=new THREE.DirectionalLight(0x6699ff,1.2);
  dirLight1.position.set(2,3,2);
  scene.add(dirLight1);
  var dirLight2=new THREE.DirectionalLight(0xffffff,0.4);
  dirLight2.position.set(-2,-1,-1);
  scene.add(dirLight2);

  var loader=new THREE.GLTFLoader();
  loader.load(
    '${modelUrl}',
    function(gltf){
      loadEl.style.display='none';
      var obj=gltf.scene;
      var box=new THREE.Box3().setFromObject(obj);
      var center=box.getCenter(new THREE.Vector3());
      var size=box.getSize(new THREE.Vector3());
      var maxDim=Math.max(size.x,size.y,size.z);
      var scale=1.6/maxDim;
      obj.scale.setScalar(scale);
      obj.position.sub(center.multiplyScalar(scale));
      scene.add(obj);
    },
    function(xhr){
      if(xhr.total>0&&loadEl){
        var pct=Math.round(xhr.loaded/xhr.total*100);
        loadEl.innerHTML='<div class="spinner"></div>'+pct+'%';
      }
    },
    function(err){
      loadEl.innerHTML='<div style="color:#EF4444">Failed to load model</div>';
    }
  );

  window.addEventListener('resize',function(){
    camera.aspect=window.innerWidth/window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth,window.innerHeight);
  });

  (function animate(){
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene,camera);
  })();
}
</script>
</body>
</html>`;

// ─── MAIN APP ────────────────────────────────────────────────────────────────────
export default function App() {
  const cameraRef = useRef(null);
  const scanActiveRef = useRef(false);
  const detectingRef = useRef(false);
  const detectSequenceRef = useRef(0);
  const lastStableDetectionRef = useRef(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;

  const [screen, setScreen] = useState('intro');
  const [cameraStatus, setCameraStatus] = useState('');
  const [isScanning, setIsScanning] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [detectedImageSize, setDetectedImageSize] = useState(null);
  const [cameraLayout, setCameraLayout] = useState({ width: 0, height: 0 });
  const [latestDetectImageUri, setLatestDetectImageUri] = useState(null);
  const [selectedObject, setSelectedObject] = useState(null);
  const [selectedFrameUri, setSelectedFrameUri] = useState(null);
  const [selectedDetectionSize, setSelectedDetectionSize] = useState(null);
  const [capturedPhoto, setCapturedPhoto] = useState(null);
  const [cropAreaLayout, setCropAreaLayout] = useState({ width: 0, height: 0 });
  const [manualBbox, setManualBbox] = useState(null);
  const [segmentResult, setSegmentResult] = useState(null);
  const [reconstructionResult, setReconstructionResult] = useState(null);
  const [isPaintingTexture, setIsPaintingTexture] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const [show3DViewer, setShow3DViewer] = useState(false);
  const [viewerModelUrl, setViewerModelUrl] = useState(null);
  const [processingStage, setProcessingStage] = useState('');
  const cropDragStartRef = useRef(null);

  // Pulse animation for scan button
  useEffect(() => {
    if (isScanning) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.06, duration: 900, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 900, useNativeDriver: true }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [isScanning]);

  // Fade in on screen change
  useEffect(() => {
    Animated.timing(fadeAnim, { toValue: 1, duration: 320, useNativeDriver: true }).start();
  }, [screen]);

  const clearObjectState = () => {
    detectSequenceRef.current += 1;
    lastStableDetectionRef.current = null;
    setDetectedObjects([]);
    setDetectedImageSize(null);
    setLatestDetectImageUri(null);
    setSelectedObject(null);
    setSelectedFrameUri(null);
    setSelectedDetectionSize(null);
    setCapturedPhoto(null);
    setManualBbox(null);
    setSegmentResult(null);
    setReconstructionResult(null);
    setIsPaintingTexture(false);
    setProcessingStage('');
  };

  const getServerFileUrl = (path) => {
    if (!path) return null;
    return path.startsWith('http') ? path : `${API_BASE_URL}${path}`;
  };

  const openCamera = async () => {
    if (!permission) return;
    if (!permission.granted) {
      const nextPermission = await requestPermission();
      if (!nextPermission.granted) { setScreen('permission'); return; }
    }
    setCameraStatus('');
    scanActiveRef.current = false;
    detectingRef.current = false;
    setIsScanning(false);
    setIsSegmenting(false);
    clearObjectState();
    setScreen('camera');
  };

  const imageDisplayRect = () => {
    if (!capturedPhoto?.width || !capturedPhoto?.height || !cropAreaLayout.width || !cropAreaLayout.height) return null;
    const scale = Math.min(cropAreaLayout.width / capturedPhoto.width, cropAreaLayout.height / capturedPhoto.height);
    const width = capturedPhoto.width * scale;
    const height = capturedPhoto.height * scale;
    return {
      left: (cropAreaLayout.width - width) / 2,
      top: (cropAreaLayout.height - height) / 2,
      width, height,
    };
  };

  const clamp01 = (value) => Math.max(0, Math.min(1, value));
  const defaultBbox = () => ({ x: 0.15, y: 0.15, width: 0.7, height: 0.7 });

  const bboxToImagePixels = (bbox) => {
    if (!bbox || !capturedPhoto?.width || !capturedPhoto?.height) throw new Error('No valid bbox.');
    return {
      x: bbox.x * capturedPhoto.width,
      y: bbox.y * capturedPhoto.height,
      width: bbox.width * capturedPhoto.width,
      height: bbox.height * capturedPhoto.height,
    };
  };

  const cropPointFromEvent = (event) => {
    const rect = imageDisplayRect();
    if (!rect) return null;
    const { locationX, locationY } = event.nativeEvent;
    return {
      x: clamp01((locationX - rect.left) / rect.width),
      y: clamp01((locationY - rect.top) / rect.height),
    };
  };

  const updateManualBboxFromDrag = (event) => {
    const start = cropDragStartRef.current;
    const point = cropPointFromEvent(event);
    if (!start || !point) return;
    const x1 = Math.min(start.x, point.x), y1 = Math.min(start.y, point.y);
    const x2 = Math.max(start.x, point.x), y2 = Math.max(start.y, point.y);
    setManualBbox({ x: x1, y: y1, width: Math.max(0.02, x2 - x1), height: Math.max(0.02, y2 - y1) });
  };

  const capturePhotoForCrop = async () => {
    if (!cameraRef.current) return;
    setIsSegmenting(false); setSegmentResult(null); setReconstructionResult(null); setIsPaintingTexture(false);
    setCameraStatus('Capturing...');
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: RECON_CAPTURE_QUALITY, base64: false, shutterSound: false, skipProcessing: false,
      });
      setCapturedPhoto(photo);
      setSelectedFrameUri(photo.uri);
      setManualBbox(defaultBbox());
      setCameraStatus('Drag to select object, then tap Reconstruct');
    } catch (error) {
      setCameraStatus(`Error: ${error.message}`);
    }
  };

  const retakePhoto = () => {
    setCapturedPhoto(null); setSelectedFrameUri(null); setManualBbox(null);
    setSegmentResult(null); setReconstructionResult(null); setIsPaintingTexture(false);
    setCameraStatus('');
  };

  const waitForReconstructionJob = async (payload) => {
    if (payload.status === 'done') return payload;
    const statusPath = payload.status_url || `/reconstruction-jobs/${payload.job_id}`;
    const startedAt = Date.now();
    let current = payload;
    while (current.status !== 'done') {
      if (current.status === 'failed' || current.status === 'error') {
        throw new Error(typeof current.error === 'string' ? current.error : JSON.stringify(current.error));
      }
      if (Date.now() - startedAt > RECON_POLL_TIMEOUT_MS) throw new Error('Reconstruction timed out.');
      setProcessingStage(current.stage || current.status || 'processing');
      await delay(RECON_POLL_INTERVAL_MS);
      const statusResponse = await fetch(`${API_BASE_URL}${statusPath}`);
      if (!statusResponse.ok) {
        const errorText = await statusResponse.text();
        throw new Error(errorText || `HTTP ${statusResponse.status}`);
      }
      current = await statusResponse.json();
    }
    return current;
  };

  const reconstructManualBbox = async () => {
    if (!capturedPhoto?.uri || !manualBbox) { setCameraStatus('Capture a photo and draw a bounding box first.'); return; }
    setIsSegmenting(true); setSegmentResult(null); setReconstructionResult(null); setIsPaintingTexture(false);
    setProcessingStage('cropping');
    setCameraStatus('Processing...');
    try {
      const bbox = bboxToImagePixels(manualBbox);
      const formData = new FormData();
      formData.append('image', { uri: capturedPhoto.uri, name: 'manual-bbox-fullres.jpg', type: 'image/jpeg' });
      formData.append('bbox_x', String(bbox.x));
      formData.append('bbox_y', String(bbox.y));
      formData.append('bbox_width', String(bbox.width));
      formData.append('bbox_height', String(bbox.height));
      const response = await fetch(`${API_BASE_URL}/reconstruct-bbox`, { method: 'POST', body: formData });
      if (!response.ok) { const errorText = await response.text(); throw new Error(errorText || `HTTP ${response.status}`); }
      const startedPayload = await response.json();
      const payload = await waitForReconstructionJob(startedPayload);
      setSegmentResult(payload.preprocess || null);
      setReconstructionResult(payload.reconstruction || null);
      setCameraStatus('3D model ready!');
      setProcessingStage('done');
    } catch (error) {
      setSegmentResult(null); setReconstructionResult(null);
      setCameraStatus(`Error: ${shortErrorMessage(error.message)}`);
      setProcessingStage('error');
    } finally {
      setIsSegmenting(false);
    }
  };

  const waitForDetectionIdle = async () => {
    for (let i = 0; i < 12 && detectingRef.current; i++) await new Promise((r) => setTimeout(r, 80));
  };

  const scaleBboxToImage = (bbox, sourceSize, targetSize) => {
    if (!bbox || !sourceSize?.width || !sourceSize?.height || !targetSize?.width || !targetSize?.height)
      throw new Error('Cannot scale bbox.');
    const scaleX = targetSize.width / sourceSize.width, scaleY = targetSize.height / sourceSize.height;
    const x = Math.max(0, bbox.x * scaleX), y = Math.max(0, bbox.y * scaleY);
    const width = Math.min(targetSize.width - x, bbox.width * scaleX);
    const height = Math.min(targetSize.height - y, bbox.height * scaleY);
    return { x, y, width: Math.max(1, width), height: Math.max(1, height) };
  };

  const scanCurrentFrame = async () => {
    if (detectingRef.current || !scanActiveRef.current || !cameraRef.current) return;
    detectingRef.current = true;
    const requestId = detectSequenceRef.current + 1;
    detectSequenceRef.current = requestId;
    const requestStartedAt = Date.now();
    setIsDetecting(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: DETECT_CAPTURE_QUALITY, base64: false, shutterSound: false, skipProcessing: false,
      });
      const detectImage = await manipulateAsync(photo.uri, [{ resize: { width: DETECT_FRAME_WIDTH } }], {
        compress: DETECT_UPLOAD_COMPRESS, format: SaveFormat.JPEG,
      });
      const formData = new FormData();
      formData.append('image', { uri: detectImage.uri, name: 'camera-frame.jpg', type: 'image/jpeg' });
      const response = await fetch(`${API_BASE_URL}/detect-frame`, { method: 'POST', body: formData });
      if (!response.ok) { const errorText = await response.text(); throw new Error(errorText || `HTTP ${response.status}`); }
      const payload = await response.json();
      const objects = Array.isArray(payload.objects) ? payload.objects : [];
      const imageSize = {
        width: payload.image_width || detectImage.width || photo.width || 0,
        height: payload.image_height || detectImage.height || photo.height || 0,
      };
      const serverMs = Number(payload.processing_ms || 0);
      const roundTripMs = Date.now() - requestStartedAt;
      if (scanActiveRef.current && requestId === detectSequenceRef.current) {
        if (objects.length > 0) {
          lastStableDetectionRef.current = { objects, imageUri: detectImage.uri, imageSize, updatedAt: Date.now() };
        }
        const stableDetection = lastStableDetectionRef.current;
        const stableAgeMs = stableDetection ? Date.now() - stableDetection.updatedAt : Infinity;
        const shouldHold = objects.length === 0 && stableDetection && stableAgeMs <= DETECT_EMPTY_HOLD_MS;
        setDetectedObjects(shouldHold ? stableDetection.objects : objects);
        setLatestDetectImageUri(shouldHold ? stableDetection.imageUri : detectImage.uri);
        setDetectedImageSize(shouldHold ? stableDetection.imageSize : imageSize);
        if (objects.length > 0) {
          setCameraStatus(`${objects.length} object${objects.length > 1 ? 's' : ''} detected · ${serverMs}ms`);
        } else {
          setCameraStatus(`Scanning... · ${serverMs}ms`);
        }
      }
    } catch (error) {
      if (scanActiveRef.current && requestId === detectSequenceRef.current) {
        setDetectedObjects([]); setDetectedImageSize(null); setLatestDetectImageUri(null);
        setCameraStatus(`Detection error: ${error.message}`);
      }
    } finally {
      detectingRef.current = false;
      setIsDetecting(false);
    }
  };

  const toggleScanning = () => {
    if (scanActiveRef.current) {
      scanActiveRef.current = false; setIsScanning(false); setIsDetecting(false);
      clearObjectState(); setCameraStatus('');
      return;
    }
    scanActiveRef.current = true; setIsScanning(true);
    clearObjectState(); setCameraStatus('Scanning...');
  };

  const selectDetectedObject = (object) => {
    if (!latestDetectImageUri || !object?.bbox) { setCameraStatus('No valid frame to select object.'); return; }
    setSelectedObject(object); setSelectedFrameUri(latestDetectImageUri);
    setSelectedDetectionSize(detectedImageSize); setSegmentResult(null); setReconstructionResult(null);
    setCameraStatus(`Selected: ${object.label}`);
  };

  const reconstructSelectedObject = async () => {
    if (!selectedObject?.bbox) { setCameraStatus('Tap on a detected object first.'); return; }
    if (!cameraRef.current) { setCameraStatus('Camera not ready.'); return; }
    setIsSegmenting(true); setReconstructionResult(null);
    scanActiveRef.current = false; detectSequenceRef.current += 1;
    setIsScanning(false); setIsDetecting(false);
    setProcessingStage('capturing');
    setCameraStatus('Capturing high-res frame...');
    try {
      await waitForDetectionIdle();
      const reconstructionPhoto = await cameraRef.current.takePictureAsync({
        quality: RECON_CAPTURE_QUALITY, base64: false, shutterSound: false, skipProcessing: false,
      });
      const sourceSize = selectedDetectionSize || detectedImageSize;
      const targetSize = { width: reconstructionPhoto.width || 0, height: reconstructionPhoto.height || 0 };
      const scaledBbox = scaleBboxToImage(selectedObject.bbox, sourceSize, targetSize);
      setSelectedFrameUri(reconstructionPhoto.uri);
      setProcessingStage('reconstructing');
      setCameraStatus('Generating 3D mesh...');
      const formData = new FormData();
      formData.append('image', { uri: reconstructionPhoto.uri, name: 'selected-object-fullres.jpg', type: 'image/jpeg' });
      formData.append('bbox_x', String(scaledBbox.x));
      formData.append('bbox_y', String(scaledBbox.y));
      formData.append('bbox_width', String(scaledBbox.width));
      formData.append('bbox_height', String(scaledBbox.height));
      const response = await fetch(`${API_BASE_URL}/reconstruct-object`, { method: 'POST', body: formData });
      if (!response.ok) { const errorText = await response.text(); throw new Error(errorText || `HTTP ${response.status}`); }
      const payload = await response.json();
      setSegmentResult(payload.segmentation || null);
      setReconstructionResult(payload.reconstruction || null);
      setCameraStatus(`${payload.selected?.label || selectedObject.label} — 3D model ready!`);
      setProcessingStage('done');
    } catch (error) {
      setSegmentResult(null); setReconstructionResult(null); setIsPaintingTexture(false);
      setCameraStatus(`Error: ${error.message}`);
      setProcessingStage('error');
    } finally {
      setIsSegmenting(false);
    }
  };

  const paintTexture = async () => {
    const jobId = reconstructionResult?.job_id;
    if (!jobId) { setCameraStatus('Missing job ID for texture.'); return; }
    setIsPaintingTexture(true);
    setProcessingStage('texturing');
    setCameraStatus('Painting texture...');
    try {
      const formData = new FormData();
      formData.append('job_id', jobId);
      const response = await fetch(`${API_BASE_URL}/paint-texture`, { method: 'POST', body: formData });
      if (!response.ok) { const errorText = await response.text(); throw new Error(errorText || `HTTP ${response.status}`); }
      let payload = await response.json();
      if (payload.status !== 'done') {
        const statusPath = payload.status_url || `/texture-jobs/${jobId}`;
        const startedAt = Date.now();
        while (payload.status !== 'done') {
          if (payload.status === 'error') throw new Error(typeof payload.error === 'string' ? payload.error : JSON.stringify(payload.error));
          if (Date.now() - startedAt > TEXTURE_POLL_TIMEOUT_MS) throw new Error('Texture paint timed out.');
          setProcessingStage(`texturing · ${payload.status || 'running'}`);
          await delay(TEXTURE_POLL_INTERVAL_MS);
          const statusResponse = await fetch(`${API_BASE_URL}${statusPath}`);
          if (!statusResponse.ok) { const errorText = await statusResponse.text(); throw new Error(errorText || `HTTP ${statusResponse.status}`); }
          payload = await statusResponse.json();
        }
      }
      setReconstructionResult(payload.reconstruction || null);
      setCameraStatus('Textured model ready!');
      setProcessingStage('done');
    } catch (error) {
      setCameraStatus(`Texture error: ${shortErrorMessage(error.message)}`);
      setProcessingStage('error');
    } finally {
      setIsPaintingTexture(false);
    }
  };

  useEffect(() => {
    if (screen !== 'camera' || !isScanning) return undefined;
    let timer = null;
    let cancelled = false;
    const runLoop = async () => {
      if (cancelled || !scanActiveRef.current) return;
      await scanCurrentFrame();
      if (!cancelled && scanActiveRef.current) timer = setTimeout(runLoop, DETECT_COOLDOWN_MS);
    };
    runLoop();
    return () => { cancelled = true; detectSequenceRef.current += 1; if (timer) clearTimeout(timer); };
  }, [screen, isScanning]);

  const mapDetectionBox = (object) => {
    if (!detectedImageSize || !cameraLayout.width || !cameraLayout.height || !object?.bbox) return null;
    const scale = Math.max(cameraLayout.width / detectedImageSize.width, cameraLayout.height / detectedImageSize.height);
    const renderedWidth = detectedImageSize.width * scale, renderedHeight = detectedImageSize.height * scale;
    const offsetX = (cameraLayout.width - renderedWidth) / 2, offsetY = (cameraLayout.height - renderedHeight) / 2;
    const left = object.bbox.x * scale + offsetX, top = object.bbox.y * scale + offsetY;
    const right = left + object.bbox.width * scale, bottom = top + object.bbox.height * scale;
    const clampedLeft = Math.max(0, Math.min(cameraLayout.width, left));
    const clampedTop = Math.max(0, Math.min(cameraLayout.height, top));
    const clampedRight = Math.max(0, Math.min(cameraLayout.width, right));
    const clampedBottom = Math.max(0, Math.min(cameraLayout.height, bottom));
    const w = clampedRight - clampedLeft, h = clampedBottom - clampedTop;
    if (w <= 1 || h <= 1) return null;
    return { left: clampedLeft, top: clampedTop, width: w, height: h };
  };

  const renderDetectionBox = (object) => {
    const box = mapDetectionBox(object);
    if (!box) return null;
    const confidence = Math.round((object.confidence || 0) * 100);
    const isSelected = selectedObject?.id === object.id;
    return (
      <Pressable key={object.id} style={[S.detectionBox, isSelected && S.selectedDetectionBox, box]}
        onPress={() => selectDetectedObject(object)}>
        <View style={S.detectionLabel}>
          <Text style={S.detectionLabelText}>{object.label} {confidence}%</Text>
        </View>
      </Pressable>
    );
  };

  const meshFilePath = (
    reconstructionResult?.files?.mesh_textured_glb ||
    reconstructionResult?.files?.mesh_textured ||
    reconstructionResult?.files?.mesh_glb ||
    reconstructionResult?.files?.mesh_obj ||
    reconstructionResult?.files?.mesh
  );
  const hasTexturedMesh = Boolean(
    reconstructionResult?.files?.mesh_textured_glb || reconstructionResult?.files?.mesh_textured
  );
  const canPaintTexture = Boolean(
    reconstructionResult?.backend === 'hunyuan_remote' &&
    reconstructionResult?.job_id &&
    reconstructionResult?.files?.mesh_glb &&
    !hasTexturedMesh
  );
  const meshFileLabel = hasTexturedMesh ? 'Textured GLB' : reconstructionResult?.files?.mesh_glb ? 'GLB' :
    reconstructionResult?.files?.mesh_obj ? 'OBJ' : 'MESH';
  const coloredMeshPath = reconstructionResult?.files?.mesh_colored_ply;
  const pointCloudPath = reconstructionResult?.files?.pointcloud_ply;
  const reconstructionInputPath = reconstructionResult?.files?.input_image;
  const arGlbPath = reconstructionResult?.files?.mesh_textured_glb || reconstructionResult?.files?.mesh_glb;
  const arUsdzPath = (
    reconstructionResult?.files?.mesh_textured_usdz || reconstructionResult?.files?.textured_usdz ||
    reconstructionResult?.files?.ar_textured_usdz || reconstructionResult?.files?.mesh_usdz ||
    reconstructionResult?.files?.ar_usdz || reconstructionResult?.files?.usdz
  );
  const meshSummary = reconstructionResult?.mesh || {};
  const cropRect = imageDisplayRect();
  const manualBboxStyle = cropRect && manualBbox ? {
    left: cropRect.left + manualBbox.x * cropRect.width,
    top: cropRect.top + manualBbox.y * cropRect.height,
    width: manualBbox.width * cropRect.width,
    height: manualBbox.height * cropRect.height,
  } : null;

  const openServerFile = (path) => {
    const url = getServerFileUrl(path);
    if (url) Linking.openURL(url);
  };

  const open3DViewer = (path) => {
    const url = getServerFileUrl(path);
    if (url) { setViewerModelUrl(url); setShow3DViewer(true); }
  };

  const openArPreview = async () => {
    const glbUrl = getServerFileUrl(arGlbPath);
    const usdzUrl = getServerFileUrl(arUsdzPath);
    try {
      if (Platform.OS === 'android' && glbUrl) {
        await Linking.openURL(buildAndroidSceneViewerUrl(glbUrl, selectedObject?.label || AR_MODEL_TITLE));
        return;
      }
      if (Platform.OS === 'ios' && usdzUrl) { await Linking.openURL(usdzUrl); return; }
      if (Platform.OS === 'ios') {
        Alert.alert('AR not ready', 'iOS Quick Look requires USDZ. Backend currently exports GLB only.');
        return;
      }
      if (glbUrl) { await Linking.openURL(glbUrl); return; }
      Alert.alert('AR not ready', 'No GLB or USDZ file available.');
    } catch (error) {
      Alert.alert('Cannot open AR', error.message);
    }
  };

  const segmentPreviewPath = (
    segmentResult?.files?.clean_image || segmentResult?.files?.input ||
    segmentResult?.files?.reconstruction_input || segmentResult?.files?.crop ||
    segmentResult?.files?.input_crop || segmentResult?.files?.masked_crop
  );

  const getProcessingIcon = () => {
    switch (processingStage) {
      case 'done': return '✓';
      case 'error': return '✕';
      case 'cropping': return '⌗';
      case 'cleaning': return '◈';
      case 'reconstructing': return '⟳';
      case 'texturing': return '◉';
      default: return '…';
    }
  };

  // ── SCREEN: 3D VIEWER MODAL ──────────────────────────────────────────────────
  const Viewer3DModal = () => (
    <Modal visible={show3DViewer} animationType="slide" statusBarTranslucent>
      <View style={S.viewerContainer}>
        <View style={S.viewerHeader}>
          <View style={S.viewerHeaderLeft}>
            <LogoMark size={24} />
            <Text style={S.viewerTitle}>3D Preview</Text>
          </View>
          <TouchableOpacity style={S.viewerCloseBtn} onPress={() => setShow3DViewer(false)}>
            <Text style={S.viewerCloseTxt}>✕</Text>
          </TouchableOpacity>
        </View>
        {viewerModelUrl ? (
          <WebView
            source={{ html: build3DViewerHTML(viewerModelUrl) }}
            style={S.webview}
            originWhitelist={['*']}
            javaScriptEnabled={true}
            domStorageEnabled={true}
            allowFileAccess={true}
            mixedContentMode="always"
            onError={(e) => console.log('WebView error:', e.nativeEvent)}
          />
        ) : (
          <View style={S.viewerLoading}>
            <ActivityIndicator color={C.accentLight} size="large" />
            <Text style={S.viewerLoadingTxt}>Loading model...</Text>
          </View>
        )}
        <View style={S.viewerFooter}>
          <Text style={S.viewerHint}>Drag to rotate · Pinch to zoom</Text>
          {arGlbPath && (
            <TouchableOpacity style={S.viewerARBtn} onPress={openArPreview}>
              <Text style={S.viewerARTxt}>View in AR</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
      <StatusBar style="light" />
    </Modal>
  );

  // ── SCREEN: CAMERA ────────────────────────────────────────────────────────────
  if (screen === 'camera') {
    return (
      <View style={S.cameraScreen} onLayout={(e) => setCameraLayout(e.nativeEvent.layout)}>
        <Viewer3DModal />
        {capturedPhoto ? (
          <View style={S.cropPreviewArea} onLayout={(e) => setCropAreaLayout(e.nativeEvent.layout)}>
            <Image source={{ uri: capturedPhoto.uri }} style={StyleSheet.absoluteFill} resizeMode="contain" />
            <View
              style={StyleSheet.absoluteFill}
              onStartShouldSetResponder={() => !isSegmenting}
              onMoveShouldSetResponder={() => !isSegmenting}
              onResponderGrant={(event) => {
                const point = cropPointFromEvent(event);
                cropDragStartRef.current = point;
                if (point) setManualBbox({ x: point.x, y: point.y, width: 0.02, height: 0.02 });
              }}
              onResponderMove={updateManualBboxFromDrag}
              onResponderRelease={() => {
                cropDragStartRef.current = null;
                setCameraStatus('Selection set. Tap Reconstruct to generate 3D.');
              }}>
              {manualBboxStyle && (
                <View style={[S.manualCropBox, manualBboxStyle]}>
                  <View style={S.manualCropLabel}>
                    <Text style={S.manualCropLabelText}>Object</Text>
                  </View>
                  {/* Corner marks */}
                  <View style={[S.corner, S.cornerTL]} />
                  <View style={[S.corner, S.cornerTR]} />
                  <View style={[S.corner, S.cornerBL]} />
                  <View style={[S.corner, S.cornerBR]} />
                </View>
              )}
            </View>
          </View>
        ) : (
          <>
            <CameraView ref={cameraRef} animateShutter={false} style={StyleSheet.absoluteFill} facing="back" />
            <View style={S.cameraDimOverlay} />
            {detectedObjects.map(renderDetectionBox)}
          </>
        )}

        <SafeAreaView pointerEvents="box-none" style={S.cameraOverlay}>
          {/* TOP BAR */}
          <View style={S.camTopBar}>
            <Pressable style={S.camBackBtn} onPress={() => {
              scanActiveRef.current = false; detectingRef.current = false;
              detectSequenceRef.current += 1;
              setIsScanning(false); setIsDetecting(false); setIsSegmenting(false);
              clearObjectState(); setScreen('intro');
            }}>
              <Text style={S.camBackIcon}>‹</Text>
            </Pressable>
            <View style={S.camTitleRow}>
              <LogoMark size={20} />
              <Text style={S.camScreenTitle}>{capturedPhoto ? 'Select Object' : 'Live Scan'}</Text>
            </View>
            <View style={[S.liveBadge, isScanning && S.liveBadgeActive]}>
              <View style={[S.liveDot, isScanning && S.liveDotActive]} />
              <Text style={S.liveBadgeText}>{capturedPhoto ? 'CROP' : isScanning ? 'LIVE' : 'CAM'}</Text>
            </View>
          </View>

          {/* SCAN FRAME (live mode only) */}
          {!capturedPhoto && (
            <View style={S.scanFrameWrapper}>
              <View style={S.scanFrame}>
                <View style={[S.scanCorner, S.scanCornerTL]} />
                <View style={[S.scanCorner, S.scanCornerTR]} />
                <View style={[S.scanCorner, S.scanCornerBL]} />
                <View style={[S.scanCorner, S.scanCornerBR]} />
                {isScanning && (
                  <View style={S.scanLineContainer}>
                    <View style={S.scanLine} />
                  </View>
                )}
              </View>
              {detectedObjects.length > 0 && (
                <Text style={S.detectionHint}>Tap an object to select</Text>
              )}
            </View>
          )}

          {/* BOTTOM PANEL */}
          <View style={S.camPanel}>
            {/* STATUS */}
            {cameraStatus !== '' && (
              <View style={[S.statusRow, processingStage === 'error' && S.statusRowError, processingStage === 'done' && S.statusRowDone]}>
                {isSegmenting && <ActivityIndicator size="small" color={C.accentLight} style={{ marginRight: 8 }} />}
                <Text style={[S.statusText, processingStage === 'error' && S.statusTextError, processingStage === 'done' && S.statusTextDone]} numberOfLines={2}>
                  {cameraStatus}
                </Text>
              </View>
            )}

            {/* BBOX INFO */}
            {capturedPhoto && manualBbox && (
              <View style={S.bboxInfo}>
                <Text style={S.bboxInfoText}>
                  {Math.round(manualBbox.width * 100)}% × {Math.round(manualBbox.height * 100)}%
                </Text>
              </View>
            )}

            {/* SEGMENT PREVIEW */}
            {segmentPreviewPath && (
              <View style={S.segmentRow}>
                <Image source={{ uri: getServerFileUrl(segmentPreviewPath) }} style={S.segmentThumb} />
                <View style={S.segmentInfo}>
                  <Text style={S.segmentLabel}>Clean input sent to Hunyuan</Text>
                  <Text style={S.segmentSub}>Background removed · Ready for 3D</Text>
                </View>
              </View>
            )}

            {/* RECONSTRUCTION RESULT */}
            {reconstructionResult && meshFilePath && (
              <View style={S.reconPanel}>
                <View style={S.reconHeader}>
                  <View style={S.reconBadge}><Text style={S.reconBadgeText}>3D READY</Text></View>
                  <Text style={S.reconStats}>
                    {meshSummary.vertices ? `${meshSummary.vertices} verts` : ''}
                    {meshSummary.faces ? ` · ${meshSummary.faces} faces` : ''}
                  </Text>
                </View>
                <View style={S.reconActions}>
                  {/* PREVIEW 3D */}
                  <TouchableOpacity style={S.reconActionBtn} onPress={() => open3DViewer(meshFilePath)}>
                    <Text style={S.reconActionIcon}>⬡</Text>
                    <Text style={S.reconActionText}>Preview</Text>
                  </TouchableOpacity>
                  {/* DOWNLOAD */}
                  <TouchableOpacity style={S.reconActionBtn} onPress={() => openServerFile(meshFilePath)}>
                    <Text style={S.reconActionIcon}>↓</Text>
                    <Text style={S.reconActionText}>{meshFileLabel}</Text>
                  </TouchableOpacity>
                  {/* PAINT TEXTURE */}
                  {canPaintTexture && (
                    <TouchableOpacity style={[S.reconActionBtn, S.reconActionBtnGreen, isPaintingTexture && S.reconActionBtnDisabled]}
                      onPress={paintTexture} disabled={isPaintingTexture}>
                      {isPaintingTexture ? <ActivityIndicator size="small" color={C.green} /> : <Text style={S.reconActionIconGreen}>◈</Text>}
                      <Text style={S.reconActionTextGreen}>{isPaintingTexture ? 'Painting' : 'Texture'}</Text>
                    </TouchableOpacity>
                  )}
                  {/* AR */}
                  {(arGlbPath || arUsdzPath) && (
                    <TouchableOpacity style={[S.reconActionBtn, S.reconActionBtnAmber]} onPress={openArPreview}>
                      <Text style={S.reconActionIconAmber}>⬚</Text>
                      <Text style={S.reconActionTextAmber}>AR</Text>
                    </TouchableOpacity>
                  )}
                </View>
              </View>
            )}

            {/* MAIN ACTIONS */}
            <View style={S.mainActionRow}>
              {capturedPhoto ? (
                <>
                  <TouchableOpacity style={S.secondaryBtn} onPress={retakePhoto} disabled={isSegmenting}>
                    <Text style={S.secondaryBtnText}>Retake</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[S.primaryBtn, (!capturedPhoto || !manualBbox || isSegmenting) && S.primaryBtnDisabled]}
                    disabled={!capturedPhoto || !manualBbox || isSegmenting}
                    onPress={reconstructManualBbox}>
                    {isSegmenting
                      ? <ActivityIndicator color="#fff" size="small" />
                      : <Text style={S.primaryBtnText}>Reconstruct</Text>}
                  </TouchableOpacity>
                </>
              ) : (
                <>
                  <Animated.View style={{ transform: [{ scale: pulseAnim }], flex: 1 }}>
                    <TouchableOpacity style={[S.scanToggleBtn, isScanning && S.scanToggleBtnActive]} onPress={toggleScanning}>
                      <View style={[S.scanToggleDot, isScanning && S.scanToggleDotActive]} />
                      <Text style={[S.scanToggleTxt, isScanning && S.scanToggleTxtActive]}>
                        {isScanning ? 'Stop Scan' : 'Live Scan'}
                      </Text>
                    </TouchableOpacity>
                  </Animated.View>
                  <TouchableOpacity
                    style={[S.primaryBtn, isSegmenting && S.primaryBtnDisabled]}
                    disabled={isSegmenting}
                    onPress={selectedObject ? reconstructSelectedObject : capturePhotoForCrop}>
                    <Text style={S.primaryBtnText}>
                      {selectedObject ? 'Reconstruct' : 'Capture'}
                    </Text>
                  </TouchableOpacity>
                </>
              )}
            </View>
          </View>
        </SafeAreaView>
        <StatusBar style="light" />
      </View>
    );
  }

  // ── SCREEN: PERMISSION ────────────────────────────────────────────────────────
  if (screen === 'permission') {
    return (
      <SafeAreaView style={S.permScreen}>
        <View style={S.permCard}>
          <Text style={S.permIcon}>📷</Text>
          <Text style={S.permTitle}>Camera Access Required</Text>
          <Text style={S.permText}>3DRecon needs camera access to scan and reconstruct 3D models from real objects.</Text>
          <TouchableOpacity style={S.permBtn} onPress={openCamera}>
            <Text style={S.permBtnText}>Allow Camera</Text>
          </TouchableOpacity>
          <TouchableOpacity style={S.permGhostBtn} onPress={() => setScreen('intro')}>
            <Text style={S.permGhostBtnText}>Back</Text>
          </TouchableOpacity>
        </View>
        <StatusBar style="light" />
      </SafeAreaView>
    );
  }

  // ── SCREEN: INTRO ─────────────────────────────────────────────────────────────
  return (
    <SafeAreaView style={S.introScreen}>
      <Viewer3DModal />
      <ScrollView style={{ flex: 1 }} contentContainerStyle={S.introScroll} showsVerticalScrollIndicator={false}>
        {/* HERO */}
        <View style={S.heroSection}>
          <View style={S.logoRow}>
            <View style={S.logoBox}>
              <Text style={S.logoBoxText3D}>3D</Text>
              <Text style={S.logoBoxTextR}>RECON</Text>
            </View>
          </View>
          <Text style={S.heroTagline}>AI-Powered 3D Reconstruction</Text>
          <Text style={S.heroSub}>
            Point your camera at any object. We'll turn it into a textured 3D mesh in seconds.
          </Text>
        </View>

        {/* PIPELINE STEPS */}
        <View style={S.pipelineSection}>
          <Text style={S.sectionLabel}>HOW IT WORKS</Text>
          <View style={S.pipeline}>
            {[
              { n: '01', icon: '📷', title: 'Capture', sub: 'Point camera at object' },
              { n: '02', icon: '⬡', title: 'Detect', sub: 'YOLO locates objects' },
              { n: '03', icon: '✦', title: 'Clean', sub: 'Remove background' },
              { n: '04', icon: '◈', title: 'Mesh', sub: 'Hunyuan generates 3D' },
            ].map((step, i) => (
              <View key={step.n} style={S.pipelineStep}>
                <View style={S.pipelineIconBox}>
                  <Text style={S.pipelineIcon}>{step.icon}</Text>
                </View>
                <Text style={S.pipelineNum}>{step.n}</Text>
                <Text style={S.pipelineTitle}>{step.title}</Text>
                <Text style={S.pipelineSub}>{step.sub}</Text>
                {i < 3 && <View style={S.pipelineArrow} />}
              </View>
            ))}
          </View>
        </View>

        {/* FEATURES */}
        <View style={S.featuresSection}>
          <Text style={S.sectionLabel}>CAPABILITIES</Text>
          <View style={S.featuresGrid}>
            {[
              { icon: '⬡', title: 'GLB / OBJ Export', sub: 'Standard 3D formats ready for any tool' },
              { icon: '◉', title: 'Texture Painting', sub: 'AI-generated photorealistic textures' },
              { icon: '⬚', title: 'AR Preview', sub: 'View your model in augmented reality' },
              { icon: '⟳', title: 'Live Detection', sub: 'Real-time YOLO object scanning' },
            ].map((f) => (
              <View key={f.title} style={S.featureCard}>
                <Text style={S.featureIcon}>{f.icon}</Text>
                <Text style={S.featureTitle}>{f.title}</Text>
                <Text style={S.featureSub}>{f.sub}</Text>
              </View>
            ))}
          </View>
        </View>

        {/* STATUS NOTE */}
        <View style={S.noteBox}>
          <View style={S.noteDot} />
          <Text style={S.noteText}>
            Backend: Hunyuan3D-2 · YOLO object detection · Local image cleaner
          </Text>
        </View>
      </ScrollView>

      {/* CTA */}
      <View style={S.introFooter}>
        <TouchableOpacity
          style={[S.startBtn, !permission && S.startBtnDisabled]}
          onPress={openCamera}
          disabled={!permission}
          activeOpacity={0.85}>
          <Text style={S.startBtnText}>
            {!permission ? 'Preparing...' : 'Start Scanning'}
          </Text>
          <Text style={S.startBtnArrow}>→</Text>
        </TouchableOpacity>
      </View>
      <StatusBar style="light" />
    </SafeAreaView>
  );
}

// ─── STYLES ───────────────────────────────────────────────────────────────────
const S = StyleSheet.create({
  // ── INTRO
  introScreen: { flex: 1, backgroundColor: C.bg },
  introScroll: { paddingBottom: 24 },
  heroSection: { paddingHorizontal: 24, paddingTop: 48, paddingBottom: 32 },
  logoRow: { marginBottom: 24 },
  logoBox: { flexDirection: 'row', alignItems: 'baseline', gap: 0 },
  logoBoxText3D: { fontSize: 38, fontWeight: '900', color: C.accent, letterSpacing: -2, fontVariant: ['tabular-nums'] },
  logoBoxTextR: { fontSize: 38, fontWeight: '900', color: C.textPrimary, letterSpacing: -2 },
  heroTagline: { fontSize: 12, fontWeight: '700', color: C.accentLight, letterSpacing: 3, marginBottom: 12, textTransform: 'uppercase' },
  heroSub: { fontSize: 16, color: C.textSecondary, lineHeight: 24, maxWidth: 300 },
  sectionLabel: { fontSize: 10, fontWeight: '700', color: C.textMuted, letterSpacing: 3, marginBottom: 16, textTransform: 'uppercase' },
  pipelineSection: { paddingHorizontal: 24, marginBottom: 32 },
  pipeline: { flexDirection: 'row', alignItems: 'flex-start', position: 'relative' },
  pipelineStep: { flex: 1, alignItems: 'center', position: 'relative' },
  pipelineIconBox: { width: 44, height: 44, borderRadius: 12, backgroundColor: C.bgCard, borderWidth: 0.5, borderColor: C.border, alignItems: 'center', justifyContent: 'center', marginBottom: 8 },
  pipelineIcon: { fontSize: 20 },
  pipelineNum: { fontSize: 9, fontWeight: '700', color: C.accent, letterSpacing: 1, marginBottom: 3 },
  pipelineTitle: { fontSize: 11, fontWeight: '700', color: C.textPrimary, marginBottom: 2 },
  pipelineSub: { fontSize: 9, color: C.textMuted, textAlign: 'center', lineHeight: 12 },
  pipelineArrow: { position: 'absolute', right: -2, top: 20, width: 4, height: 4, borderRadius: 2, backgroundColor: C.accent },
  featuresSection: { paddingHorizontal: 24, marginBottom: 24 },
  featuresGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  featureCard: { width: (SCREEN_WIDTH - 58) / 2, backgroundColor: C.bgCard, borderRadius: 14, borderWidth: 0.5, borderColor: C.border, padding: 16 },
  featureIcon: { fontSize: 22, marginBottom: 10 },
  featureTitle: { fontSize: 12, fontWeight: '700', color: C.textPrimary, marginBottom: 4 },
  featureSub: { fontSize: 11, color: C.textMuted, lineHeight: 15 },
  noteBox: { marginHorizontal: 24, backgroundColor: C.bgCard, borderRadius: 10, borderWidth: 0.5, borderColor: C.border, padding: 14, flexDirection: 'row', alignItems: 'center', gap: 10 },
  noteDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: C.green },
  noteText: { flex: 1, fontSize: 11, color: C.textMuted, lineHeight: 16 },
  introFooter: { paddingHorizontal: 24, paddingBottom: 24, paddingTop: 12, backgroundColor: C.bg },
  startBtn: { backgroundColor: C.accent, borderRadius: 16, height: 58, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 10 },
  startBtnDisabled: { opacity: 0.45 },
  startBtnText: { color: C.white, fontSize: 16, fontWeight: '800', letterSpacing: 0.3 },
  startBtnArrow: { color: 'rgba(255,255,255,0.6)', fontSize: 20 },

  // ── PERMISSION
  permScreen: { flex: 1, backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center', padding: 24 },
  permCard: { backgroundColor: C.bgCard, borderRadius: 20, borderWidth: 0.5, borderColor: C.border, padding: 28, width: '100%', alignItems: 'center' },
  permIcon: { fontSize: 48, marginBottom: 16 },
  permTitle: { fontSize: 20, fontWeight: '800', color: C.textPrimary, marginBottom: 10, textAlign: 'center' },
  permText: { fontSize: 14, color: C.textSecondary, lineHeight: 21, textAlign: 'center', marginBottom: 24 },
  permBtn: { backgroundColor: C.accent, borderRadius: 12, paddingVertical: 16, paddingHorizontal: 32, width: '100%', alignItems: 'center', marginBottom: 10 },
  permBtnText: { color: C.white, fontSize: 15, fontWeight: '800' },
  permGhostBtn: { paddingVertical: 12, alignItems: 'center' },
  permGhostBtnText: { color: C.accentLight, fontSize: 14, fontWeight: '600' },

  // ── CAMERA SCREEN
  cameraScreen: { flex: 1, backgroundColor: '#000' },
  cropPreviewArea: { ...StyleSheet.absoluteFillObject, backgroundColor: '#000' },
  cameraDimOverlay: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.18)' },
  cameraOverlay: { flex: 1, justifyContent: 'space-between' },
  camTopBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingTop: 8 },
  camBackBtn: { width: 40, height: 40, borderRadius: 10, backgroundColor: 'rgba(0,0,0,0.5)', alignItems: 'center', justifyContent: 'center', borderWidth: 0.5, borderColor: 'rgba(255,255,255,0.15)' },
  camBackIcon: { color: C.textPrimary, fontSize: 24, lineHeight: 26 },
  camTitleRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  camScreenTitle: { color: C.textPrimary, fontSize: 14, fontWeight: '700' },
  liveBadge: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: 'rgba(0,0,0,0.55)', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 7, borderWidth: 0.5, borderColor: 'rgba(255,255,255,0.1)' },
  liveBadgeActive: { borderColor: 'rgba(16,217,138,0.4)', backgroundColor: 'rgba(16,217,138,0.1)' },
  liveDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: C.textMuted },
  liveDotActive: { backgroundColor: C.green },
  liveBadgeText: { color: C.textPrimary, fontSize: 10, fontWeight: '800', letterSpacing: 1.5 },

  // SCAN FRAME
  scanFrameWrapper: { alignItems: 'center' },
  scanFrame: { width: SCREEN_WIDTH * 0.78, aspectRatio: 0.75, position: 'relative' },
  scanCorner: { position: 'absolute', width: 22, height: 22, borderColor: C.scanBox },
  scanCornerTL: { left: 0, top: 0, borderLeftWidth: 3, borderTopWidth: 3 },
  scanCornerTR: { right: 0, top: 0, borderRightWidth: 3, borderTopWidth: 3 },
  scanCornerBL: { left: 0, bottom: 0, borderLeftWidth: 3, borderBottomWidth: 3 },
  scanCornerBR: { right: 0, bottom: 0, borderRightWidth: 3, borderBottomWidth: 3 },
  scanLineContainer: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, overflow: 'hidden' },
  scanLine: { height: 1.5, backgroundColor: 'rgba(99,179,237,0.6)', marginTop: '40%' },
  detectionHint: { color: 'rgba(255,255,255,0.55)', fontSize: 11, marginTop: 10, letterSpacing: 0.3 },

  // DETECTION BOXES
  detectionBox: { position: 'absolute', borderWidth: 2, borderColor: C.green, backgroundColor: 'rgba(16,217,138,0.1)' },
  selectedDetectionBox: { borderColor: C.accentLight, backgroundColor: 'rgba(91,143,239,0.18)' },
  detectionLabel: { position: 'absolute', left: -2, top: -26, backgroundColor: C.green, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 4 },
  detectionLabelText: { color: '#001A0A', fontSize: 11, fontWeight: '900' },

  // MANUAL CROP BOX
  manualCropBox: { position: 'absolute', borderWidth: 2, borderColor: C.accentLight, backgroundColor: 'rgba(91,143,239,0.12)' },
  corner: { position: 'absolute', width: 12, height: 12, borderColor: C.accentLight },
  cornerTL: { left: -2, top: -2, borderLeftWidth: 3, borderTopWidth: 3 },
  cornerTR: { right: -2, top: -2, borderRightWidth: 3, borderTopWidth: 3 },
  cornerBL: { left: -2, bottom: -2, borderLeftWidth: 3, borderBottomWidth: 3 },
  cornerBR: { right: -2, bottom: -2, borderRightWidth: 3, borderBottomWidth: 3 },
  manualCropLabel: { position: 'absolute', left: -2, top: -26, backgroundColor: C.accentLight, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 4 },
  manualCropLabelText: { color: '#001030', fontSize: 11, fontWeight: '900' },

  // BOTTOM PANEL
  camPanel: { backgroundColor: 'rgba(10,14,26,0.95)', borderTopWidth: 0.5, borderTopColor: C.border, padding: 16, gap: 10 },
  statusRow: { flexDirection: 'row', alignItems: 'center', backgroundColor: C.bgCardAlt, borderRadius: 10, paddingHorizontal: 12, paddingVertical: 10, borderWidth: 0.5, borderColor: C.border },
  statusRowError: { borderColor: 'rgba(239,68,68,0.3)', backgroundColor: 'rgba(239,68,68,0.08)' },
  statusRowDone: { borderColor: 'rgba(16,217,138,0.3)', backgroundColor: 'rgba(16,217,138,0.08)' },
  statusText: { flex: 1, fontSize: 12, color: C.textSecondary, lineHeight: 17 },
  statusTextError: { color: C.red },
  statusTextDone: { color: C.green },
  bboxInfo: { alignSelf: 'flex-start', backgroundColor: C.accentGlow, borderRadius: 8, paddingHorizontal: 10, paddingVertical: 4, borderWidth: 0.5, borderColor: C.borderActive },
  bboxInfoText: { color: C.accentLight, fontSize: 11, fontWeight: '700' },
  segmentRow: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: C.bgCardAlt, borderRadius: 12, padding: 10, borderWidth: 0.5, borderColor: C.border },
  segmentThumb: { width: 52, height: 52, borderRadius: 8, backgroundColor: C.bgCard },
  segmentInfo: { flex: 1 },
  segmentLabel: { fontSize: 12, fontWeight: '700', color: C.textPrimary, marginBottom: 3 },
  segmentSub: { fontSize: 10, color: C.textMuted },
  reconPanel: { backgroundColor: C.bgCard, borderRadius: 14, borderWidth: 0.5, borderColor: C.borderActive, padding: 12 },
  reconHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 },
  reconBadge: { backgroundColor: C.accentGlow, borderRadius: 6, paddingHorizontal: 8, paddingVertical: 4, borderWidth: 0.5, borderColor: C.borderActive },
  reconBadgeText: { color: C.accentLight, fontSize: 9, fontWeight: '900', letterSpacing: 1.5 },
  reconStats: { fontSize: 10, color: C.textMuted },
  reconActions: { flexDirection: 'row', gap: 8 },
  reconActionBtn: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: C.bgCardAlt, borderRadius: 10, paddingVertical: 10, borderWidth: 0.5, borderColor: C.border, gap: 3 },
  reconActionBtnGreen: { borderColor: 'rgba(16,217,138,0.3)', backgroundColor: 'rgba(16,217,138,0.08)' },
  reconActionBtnAmber: { borderColor: 'rgba(245,158,11,0.3)', backgroundColor: 'rgba(245,158,11,0.08)' },
  reconActionBtnDisabled: { opacity: 0.45 },
  reconActionIcon: { color: C.accentLight, fontSize: 16 },
  reconActionIconGreen: { color: C.green, fontSize: 16 },
  reconActionIconAmber: { color: C.amber, fontSize: 16 },
  reconActionText: { color: C.textSecondary, fontSize: 10, fontWeight: '700' },
  reconActionTextGreen: { color: C.green, fontSize: 10, fontWeight: '700' },
  reconActionTextAmber: { color: C.amber, fontSize: 10, fontWeight: '700' },
  mainActionRow: { flexDirection: 'row', gap: 10, marginTop: 2 },
  secondaryBtn: { flex: 1, height: 52, borderRadius: 14, alignItems: 'center', justifyContent: 'center', backgroundColor: C.bgCardAlt, borderWidth: 0.5, borderColor: C.border },
  secondaryBtnText: { color: C.textSecondary, fontSize: 15, fontWeight: '700' },
  primaryBtn: { flex: 1.4, height: 52, borderRadius: 14, alignItems: 'center', justifyContent: 'center', backgroundColor: C.accent },
  primaryBtnDisabled: { opacity: 0.45 },
  primaryBtnText: { color: C.white, fontSize: 15, fontWeight: '800' },
  scanToggleBtn: { flex: 1, height: 52, borderRadius: 14, alignItems: 'center', justifyContent: 'center', flexDirection: 'row', gap: 8, backgroundColor: C.bgCardAlt, borderWidth: 0.5, borderColor: C.border },
  scanToggleBtnActive: { borderColor: 'rgba(16,217,138,0.4)', backgroundColor: 'rgba(16,217,138,0.1)' },
  scanToggleDot: { width: 7, height: 7, borderRadius: 3.5, backgroundColor: C.textMuted },
  scanToggleDotActive: { backgroundColor: C.green },
  scanToggleTxt: { color: C.textSecondary, fontSize: 14, fontWeight: '700' },
  scanToggleTxtActive: { color: C.green },

  // ── 3D VIEWER
  viewerContainer: { flex: 1, backgroundColor: C.bg },
  viewerHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 20, paddingTop: Platform.OS === 'ios' ? 52 : 16, paddingBottom: 12, borderBottomWidth: 0.5, borderBottomColor: C.border, backgroundColor: C.bgCard },
  viewerHeaderLeft: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  viewerTitle: { color: C.textPrimary, fontSize: 16, fontWeight: '700' },
  viewerCloseBtn: { width: 36, height: 36, borderRadius: 10, backgroundColor: C.bgCardAlt, alignItems: 'center', justifyContent: 'center', borderWidth: 0.5, borderColor: C.border },
  viewerCloseTxt: { color: C.textSecondary, fontSize: 15 },
  webview: { flex: 1, backgroundColor: C.bg },
  viewerLoading: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12 },
  viewerLoadingTxt: { color: C.textMuted, fontSize: 14 },
  viewerFooter: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 20, paddingVertical: 14, paddingBottom: Platform.OS === 'ios' ? 28 : 14, borderTopWidth: 0.5, borderTopColor: C.border, backgroundColor: C.bgCard },
  viewerHint: { color: C.textMuted, fontSize: 12 },
  viewerARBtn: { backgroundColor: C.accent, borderRadius: 10, paddingHorizontal: 16, paddingVertical: 8 },
  viewerARTxt: { color: C.white, fontSize: 13, fontWeight: '700' },
});