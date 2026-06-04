import { useEffect, useRef, useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import {
  Alert,
  Image,
  Linking,
  Platform,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

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
const activeWorkflowSteps = [
  'Nhan anh hoac video object',
  'Nguoi dung chon vung vat the',
  'Backend goi Hunyuan worker de sinh mesh',
  'Export GLB va cac artifact 3D',
];

const buildAndroidSceneViewerUrl = (modelUrl, title = AR_MODEL_TITLE) => {
  const encodedModelUrl = encodeURIComponent(modelUrl);
  const encodedFallbackUrl = encodeURIComponent(modelUrl);
  const encodedTitle = encodeURIComponent(title);

  return (
    `intent://arvr.google.com/scene-viewer/1.0?file=${encodedModelUrl}`
    + `&mode=ar_preferred&title=${encodedTitle}`
    + '#Intent;scheme=https;package=com.google.android.googlequicksearchbox;'
    + `action=android.intent.action.VIEW;S.browser_fallback_url=${encodedFallbackUrl};end;`
  );
};

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const shortErrorMessage = (message) => {
  if (!message) {
    return 'Unknown error';
  }
  return String(message).replace(/\s+/g, ' ').slice(0, 240);
};

export default function App() {
  const cameraRef = useRef(null);
  const scanActiveRef = useRef(false);
  const detectingRef = useRef(false);
  const detectSequenceRef = useRef(0);
  const lastStableDetectionRef = useRef(null);
  const [screen, setScreen] = useState('intro');
  const [cameraStatus, setCameraStatus] = useState('Camera đã sẵn sàng.');
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
  const cropDragStartRef = useRef(null);

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
  };

  const getServerFileUrl = (path) => {
    if (!path) {
      return null;
    }

    return path.startsWith('http') ? path : `${API_BASE_URL}${path}`;
  };

  const openCamera = async () => {
    if (!permission) {
      return;
    }

    if (!permission.granted) {
      const nextPermission = await requestPermission();
      if (!nextPermission.granted) {
        setScreen('permission');
        return;
      }
    }

    setCameraStatus('Camera đã sẵn sàng.');
    scanActiveRef.current = false;
    detectingRef.current = false;
    setIsScanning(false);
    setIsSegmenting(false);
    clearObjectState();
    setScreen('camera');
  };

  const runReconstruction = () => {
    reconstructManualBbox();
  };

  const imageDisplayRect = () => {
    if (!capturedPhoto?.width || !capturedPhoto?.height || !cropAreaLayout.width || !cropAreaLayout.height) {
      return null;
    }
    const scale = Math.min(
      cropAreaLayout.width / capturedPhoto.width,
      cropAreaLayout.height / capturedPhoto.height,
    );
    const width = capturedPhoto.width * scale;
    const height = capturedPhoto.height * scale;
    return {
      left: (cropAreaLayout.width - width) / 2,
      top: (cropAreaLayout.height - height) / 2,
      width,
      height,
    };
  };

  const clamp01 = (value) => Math.max(0, Math.min(1, value));

  const defaultBbox = () => ({
    x: 0.15,
    y: 0.15,
    width: 0.7,
    height: 0.7,
  });

  const bboxToImagePixels = (bbox) => {
    if (!bbox || !capturedPhoto?.width || !capturedPhoto?.height) {
      throw new Error('Chua co bbox hop le.');
    }
    return {
      x: bbox.x * capturedPhoto.width,
      y: bbox.y * capturedPhoto.height,
      width: bbox.width * capturedPhoto.width,
      height: bbox.height * capturedPhoto.height,
    };
  };

  const cropPointFromEvent = (event) => {
    const rect = imageDisplayRect();
    if (!rect) {
      return null;
    }
    const { locationX, locationY } = event.nativeEvent;
    return {
      x: clamp01((locationX - rect.left) / rect.width),
      y: clamp01((locationY - rect.top) / rect.height),
    };
  };

  const updateManualBboxFromDrag = (event) => {
    const start = cropDragStartRef.current;
    const point = cropPointFromEvent(event);
    if (!start || !point) {
      return;
    }
    const x1 = Math.min(start.x, point.x);
    const y1 = Math.min(start.y, point.y);
    const x2 = Math.max(start.x, point.x);
    const y2 = Math.max(start.y, point.y);
    setManualBbox({
      x: x1,
      y: y1,
      width: Math.max(0.02, x2 - x1),
      height: Math.max(0.02, y2 - y1),
    });
  };

  const capturePhotoForCrop = async () => {
    if (!cameraRef.current) {
      setCameraStatus('Camera chua san sang.');
      return;
    }

    setIsSegmenting(false);
    setSegmentResult(null);
    setReconstructionResult(null);
    setIsPaintingTexture(false);
    setCameraStatus('Dang chup anh...');

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: RECON_CAPTURE_QUALITY,
        base64: false,
        shutterSound: false,
        skipProcessing: false,
      });
      setCapturedPhoto(photo);
      setSelectedFrameUri(photo.uri);
      setManualBbox(defaultBbox());
      setCameraStatus('Keo khung quanh vat the, sau do bam Tai tao.');
    } catch (error) {
      setCameraStatus(`Khong chup duoc anh: ${error.message}`);
    }
  };

  const retakePhoto = () => {
    setCapturedPhoto(null);
    setSelectedFrameUri(null);
    setManualBbox(null);
    setSegmentResult(null);
    setReconstructionResult(null);
    setIsPaintingTexture(false);
    setCameraStatus('Camera da san sang. Bam Chup anh de chon vat the.');
  };

  const waitForReconstructionJob = async (payload) => {
    if (payload.status === 'done') {
      return payload;
    }
    const statusPath = payload.status_url || `/reconstruction-jobs/${payload.job_id}`;
    const startedAt = Date.now();
    let current = payload;
    while (current.status !== 'done') {
      if (current.status === 'failed' || current.status === 'error') {
        throw new Error(typeof current.error === 'string' ? current.error : JSON.stringify(current.error));
      }
      if (Date.now() - startedAt > RECON_POLL_TIMEOUT_MS) {
        throw new Error('Reconstruction timed out while waiting for backend status.');
      }
      setCameraStatus(`Dang xu ly... ${current.stage || current.status || 'running'}`);
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
    if (!capturedPhoto?.uri || !manualBbox) {
      setCameraStatus('Hay chup anh va keo bbox quanh vat the truoc.');
      return;
    }

    setIsSegmenting(true);
    setSegmentResult(null);
    setReconstructionResult(null);
    setIsPaintingTexture(false);
    setCameraStatus('Dang crop, local clean va tao mesh...');

    try {
      const bbox = bboxToImagePixels(manualBbox);
      const formData = new FormData();
      formData.append('image', {
        uri: capturedPhoto.uri,
        name: 'manual-bbox-fullres.jpg',
        type: 'image/jpeg',
      });
      formData.append('bbox_x', String(bbox.x));
      formData.append('bbox_y', String(bbox.y));
      formData.append('bbox_width', String(bbox.width));
      formData.append('bbox_height', String(bbox.height));

      const response = await fetch(`${API_BASE_URL}/reconstruct-bbox`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }

      const startedPayload = await response.json();
      const payload = await waitForReconstructionJob(startedPayload);
      setSegmentResult(payload.preprocess || null);
      setReconstructionResult(payload.reconstruction || null);
      setCameraStatus('Da reconstruct xong. GLB san sang.');
    } catch (error) {
      setSegmentResult(null);
      setReconstructionResult(null);
      setCameraStatus(`Loi reconstruct: ${shortErrorMessage(error.message)}`);
    } finally {
      setIsSegmenting(false);
    }
  };

  const waitForDetectionIdle = async () => {
    for (let index = 0; index < 12 && detectingRef.current; index += 1) {
      await new Promise((resolve) => setTimeout(resolve, 80));
    }
  };

  const scaleBboxToImage = (bbox, sourceSize, targetSize) => {
    if (!bbox || !sourceSize?.width || !sourceSize?.height || !targetSize?.width || !targetSize?.height) {
      throw new Error('Khong doc duoc kich thuoc frame de scale bbox.');
    }

    const scaleX = targetSize.width / sourceSize.width;
    const scaleY = targetSize.height / sourceSize.height;
    const x = Math.max(0, bbox.x * scaleX);
    const y = Math.max(0, bbox.y * scaleY);
    const width = Math.min(targetSize.width - x, bbox.width * scaleX);
    const height = Math.min(targetSize.height - y, bbox.height * scaleY);

    return {
      x,
      y,
      width: Math.max(1, width),
      height: Math.max(1, height),
    };
  };

  const scanCurrentFrame = async () => {
    if (detectingRef.current || !scanActiveRef.current || !cameraRef.current) {
      return;
    }

    detectingRef.current = true;
    const requestId = detectSequenceRef.current + 1;
    detectSequenceRef.current = requestId;
    const requestStartedAt = Date.now();
    setIsDetecting(true);

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: DETECT_CAPTURE_QUALITY,
        base64: false,
        shutterSound: false,
        skipProcessing: false,
      });
      const detectImage = await manipulateAsync(
        photo.uri,
        [{ resize: { width: DETECT_FRAME_WIDTH } }],
        {
          compress: DETECT_UPLOAD_COMPRESS,
          format: SaveFormat.JPEG,
        },
      );

      const formData = new FormData();
      formData.append('image', {
        uri: detectImage.uri,
        name: 'camera-frame.jpg',
        type: 'image/jpeg',
      });

      const response = await fetch(`${API_BASE_URL}/detect-frame`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }

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
          lastStableDetectionRef.current = {
            objects,
            imageUri: detectImage.uri,
            imageSize,
            updatedAt: Date.now(),
          };
        }

        const stableDetection = lastStableDetectionRef.current;
        const stableAgeMs = stableDetection ? Date.now() - stableDetection.updatedAt : Infinity;
        const shouldHoldLastDetection = objects.length === 0 && stableDetection && stableAgeMs <= DETECT_EMPTY_HOLD_MS;

        setDetectedObjects(objects);
        setLatestDetectImageUri(detectImage.uri);
        setDetectedImageSize(imageSize);

        if (shouldHoldLastDetection) {
          setDetectedObjects(stableDetection.objects);
          setLatestDetectImageUri(stableDetection.imageUri);
          setDetectedImageSize(stableDetection.imageSize);
          setCameraStatus(`Dang giu bbox gan nhat (${Math.round(stableAgeMs)}ms). Server ${serverMs}ms, tong ${roundTripMs}ms.`);
        } else {
          setCameraStatus(
            objects.length === 0
              ? `Dang quet lien tuc. Chua tim thay vat the. Server ${serverMs}ms, tong ${roundTripMs}ms.`
              : `Dang quet lien tuc. YOLO detect ${objects.length} vat the. Server ${serverMs}ms, tong ${roundTripMs}ms.`,
          );
        }
      }
    } catch (error) {
      if (scanActiveRef.current && requestId === detectSequenceRef.current) {
        setDetectedObjects([]);
        setDetectedImageSize(null);
        setLatestDetectImageUri(null);
        setCameraStatus(`Loi detect: ${error.message}`);
      }
    } finally {
      detectingRef.current = false;
      setIsDetecting(false);
    }
  };

  const toggleScanning = () => {
    if (scanActiveRef.current) {
      scanActiveRef.current = false;
      setIsScanning(false);
      setIsDetecting(false);
      clearObjectState();
      setCameraStatus('Da dung quet vat the.');
      return;
    }

    scanActiveRef.current = true;
    setIsScanning(true);
    clearObjectState();
    setCameraStatus('Dang quet lien tuc...');
  };

  const selectDetectedObject = (object) => {
    if (!latestDetectImageUri || !object?.bbox) {
      setCameraStatus('Chua co frame detect hop le de chon vat the.');
      return;
    }

    setSelectedObject(object);
    setSelectedFrameUri(latestDetectImageUri);
    setSelectedDetectionSize(detectedImageSize);
    setSegmentResult(null);
    setReconstructionResult(null);
    setCameraStatus(`Da chon ${object.label}. Bam Tai tao de chup full-res va chay reconstruction worker.`);
  };

  const reconstructSelectedObject = async () => {
    if (!selectedObject?.bbox) {
      setCameraStatus('Hay cham vao bbox vat the truoc khi Tai tao.');
      return;
    }
    if (!cameraRef.current) {
      setCameraStatus('Camera chua san sang de chup anh reconstruct.');
      return;
    }

    setIsSegmenting(true);
    setReconstructionResult(null);
    scanActiveRef.current = false;
    detectSequenceRef.current += 1;
    setIsScanning(false);
    setIsDetecting(false);
    setCameraStatus('Dang chup frame full-res de reconstruct...');

    try {
      await waitForDetectionIdle();
      const reconstructionPhoto = await cameraRef.current.takePictureAsync({
        quality: RECON_CAPTURE_QUALITY,
        base64: false,
        shutterSound: false,
        skipProcessing: false,
      });
      const sourceSize = selectedDetectionSize || detectedImageSize;
      const targetSize = {
        width: reconstructionPhoto.width || 0,
        height: reconstructionPhoto.height || 0,
      };
      const scaledBbox = scaleBboxToImage(selectedObject.bbox, sourceSize, targetSize);

      setSelectedFrameUri(reconstructionPhoto.uri);
      setCameraStatus('Dang YOLO crop full-res + reconstruct mesh + export GLB...');

      const formData = new FormData();
      formData.append('image', {
        uri: reconstructionPhoto.uri,
        name: 'selected-object-fullres.jpg',
        type: 'image/jpeg',
      });
      formData.append('bbox_x', String(scaledBbox.x));
      formData.append('bbox_y', String(scaledBbox.y));
      formData.append('bbox_width', String(scaledBbox.width));
      formData.append('bbox_height', String(scaledBbox.height));

      const response = await fetch(`${API_BASE_URL}/reconstruct-object`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }

      const payload = await response.json();
      setSegmentResult(payload.segmentation || null);
      setReconstructionResult(payload.reconstruction || null);
      setCameraStatus(`Da reconstruct ${payload.selected?.label || selectedObject.label}. GLB san sang.`);
    } catch (error) {
      setSegmentResult(null);
      setReconstructionResult(null);
      setIsPaintingTexture(false);
      setCameraStatus(`Loi reconstruct: ${error.message}`);
    } finally {
      setIsSegmenting(false);
    }
  };

  const paintTexture = async () => {
    const jobId = reconstructionResult?.job_id;
    if (!jobId) {
      setCameraStatus('Missing job_id for texture paint.');
      return;
    }

    setIsPaintingTexture(true);
    setCameraStatus('Painting texture on saved shape mesh...');

    try {
      const formData = new FormData();
      formData.append('job_id', jobId);

      const response = await fetch(`${API_BASE_URL}/paint-texture`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }

      let payload = await response.json();
      if (payload.status !== 'done') {
        const statusPath = payload.status_url || `/texture-jobs/${jobId}`;
        const startedAt = Date.now();
        while (payload.status !== 'done') {
          if (payload.status === 'error') {
            throw new Error(typeof payload.error === 'string' ? payload.error : JSON.stringify(payload.error));
          }
          if (Date.now() - startedAt > TEXTURE_POLL_TIMEOUT_MS) {
            throw new Error('Texture paint timed out while waiting for backend status.');
          }
          setCameraStatus(`Painting texture... ${payload.status || 'running'}`);
          await delay(TEXTURE_POLL_INTERVAL_MS);
          const statusResponse = await fetch(`${API_BASE_URL}${statusPath}`);
          if (!statusResponse.ok) {
            const errorText = await statusResponse.text();
            throw new Error(errorText || `HTTP ${statusResponse.status}`);
          }
          payload = await statusResponse.json();
        }
      }
      setReconstructionResult(payload.reconstruction || null);
      setCameraStatus('Texture painted. Textured GLB is ready.');
    } catch (error) {
      setCameraStatus(`Texture paint failed: ${shortErrorMessage(error.message)}`);
    } finally {
      setIsPaintingTexture(false);
    }
  };

  useEffect(() => {
    if (screen !== 'camera' || !isScanning) {
      return undefined;
    }

    let timer = null;
    let cancelled = false;

    const runLoop = async () => {
      if (cancelled || !scanActiveRef.current) {
        return;
      }

      await scanCurrentFrame();

      if (!cancelled && scanActiveRef.current) {
        timer = setTimeout(runLoop, DETECT_COOLDOWN_MS);
      }
    };

    runLoop();

    return () => {
      cancelled = true;
      detectSequenceRef.current += 1;
      if (timer) {
        clearTimeout(timer);
      }
    };
  }, [screen, isScanning]);

  const mapDetectionBox = (object) => {
    if (!detectedImageSize || !cameraLayout.width || !cameraLayout.height || !object?.bbox) {
      return null;
    }

    const scale = Math.max(
      cameraLayout.width / detectedImageSize.width,
      cameraLayout.height / detectedImageSize.height,
    );
    const renderedWidth = detectedImageSize.width * scale;
    const renderedHeight = detectedImageSize.height * scale;
    const offsetX = (cameraLayout.width - renderedWidth) / 2;
    const offsetY = (cameraLayout.height - renderedHeight) / 2;
    const left = object.bbox.x * scale + offsetX;
    const top = object.bbox.y * scale + offsetY;
    const width = object.bbox.width * scale;
    const height = object.bbox.height * scale;
    const right = left + width;
    const bottom = top + height;
    const clampedLeft = Math.max(0, Math.min(cameraLayout.width, left));
    const clampedTop = Math.max(0, Math.min(cameraLayout.height, top));
    const clampedRight = Math.max(0, Math.min(cameraLayout.width, right));
    const clampedBottom = Math.max(0, Math.min(cameraLayout.height, bottom));
    const clampedWidth = clampedRight - clampedLeft;
    const clampedHeight = clampedBottom - clampedTop;

    if (clampedWidth <= 1 || clampedHeight <= 1) {
      return null;
    }

    return {
      left: clampedLeft,
      top: clampedTop,
      width: clampedWidth,
      height: clampedHeight,
    };
  };

  const renderDetectionBox = (object) => {
    const box = mapDetectionBox(object);
    if (!box) {
      return null;
    }

    const confidence = Math.round((object.confidence || 0) * 100);
    const isSelected = selectedObject?.id === object.id;
    return (
      <Pressable
        key={object.id}
        style={[styles.detectionBox, isSelected && styles.selectedDetectionBox, box]}
        onPress={() => selectDetectedObject(object)}
      >
        <View style={styles.detectionLabel}>
          <Text style={styles.detectionLabelText}>
            {object.label} {confidence}%
          </Text>
        </View>
      </Pressable>
    );
  };

  const segmentPreviewPath = (
    segmentResult?.files?.clean_image
    || segmentResult?.files?.input
    || segmentResult?.files?.reconstruction_input
    || segmentResult?.files?.crop
    || segmentResult?.files?.input_crop
    || segmentResult?.files?.masked_crop
  );
  const meshFilePath = (
    reconstructionResult?.files?.mesh_textured_glb
    || reconstructionResult?.files?.mesh_textured
    || reconstructionResult?.files?.mesh_glb
    || reconstructionResult?.files?.mesh_obj
    || reconstructionResult?.files?.mesh
  );
  const hasTexturedMesh = Boolean(
    reconstructionResult?.files?.mesh_textured_glb
    || reconstructionResult?.files?.mesh_textured
  );
  const canPaintTexture = Boolean(
    reconstructionResult?.backend === 'hunyuan_remote'
    && reconstructionResult?.job_id
    && reconstructionResult?.files?.mesh_glb
    && !hasTexturedMesh
  );
  const meshFileLabel = hasTexturedMesh
    ? 'Textured GLB'
    : reconstructionResult?.files?.mesh_glb
    ? 'GLB'
    : reconstructionResult?.files?.mesh_obj
      ? 'OBJ'
      : 'MESH';
  const coloredMeshPath = reconstructionResult?.files?.mesh_colored_ply;
  const pointCloudPath = reconstructionResult?.files?.pointcloud_ply;
  const reconstructionInputPath = reconstructionResult?.files?.input_image;
  const arGlbPath = reconstructionResult?.files?.mesh_textured_glb || reconstructionResult?.files?.mesh_glb;
  const arUsdzPath = (
    reconstructionResult?.files?.mesh_textured_usdz
    || reconstructionResult?.files?.textured_usdz
    || reconstructionResult?.files?.ar_textured_usdz
    || reconstructionResult?.files?.mesh_usdz
    || reconstructionResult?.files?.ar_usdz
    || reconstructionResult?.files?.usdz
  );
  const meshSummary = reconstructionResult?.mesh || {};
  const cropRect = imageDisplayRect();
  const manualBboxStyle = cropRect && manualBbox
    ? {
        left: cropRect.left + manualBbox.x * cropRect.width,
        top: cropRect.top + manualBbox.y * cropRect.height,
        width: manualBbox.width * cropRect.width,
        height: manualBbox.height * cropRect.height,
      }
    : null;
  const openServerFile = (path) => {
    const url = getServerFileUrl(path);
    if (url) {
      Linking.openURL(url);
    }
  };
  const openArPreview = async () => {
    const glbUrl = getServerFileUrl(arGlbPath);
    const usdzUrl = getServerFileUrl(arUsdzPath);

    try {
      if (Platform.OS === 'android' && glbUrl) {
        await Linking.openURL(buildAndroidSceneViewerUrl(glbUrl, selectedObject?.label || AR_MODEL_TITLE));
        return;
      }

      if (Platform.OS === 'ios' && usdzUrl) {
        await Linking.openURL(usdzUrl);
        return;
      }

      if (Platform.OS === 'ios') {
        Alert.alert(
          'AR chua san sang',
          'iOS Quick Look can file USDZ. Backend hien moi xuat GLB, can them buoc convert GLB sang USDZ.',
        );
        return;
      }

      if (glbUrl) {
        await Linking.openURL(glbUrl);
        return;
      }

      Alert.alert('AR chua san sang', 'Chua co file GLB hoac USDZ de mo AR.');
    } catch (error) {
      Alert.alert('Khong mo duoc AR', error.message);
    }
  };

  if (screen === 'camera') {
    return (
      <View
        style={styles.cameraScreen}
        onLayout={(event) => setCameraLayout(event.nativeEvent.layout)}
      >
        {capturedPhoto ? (
          <View
            style={styles.cropPreviewArea}
            onLayout={(event) => setCropAreaLayout(event.nativeEvent.layout)}
          >
            <Image
              source={{ uri: capturedPhoto.uri }}
              style={StyleSheet.absoluteFill}
              resizeMode="contain"
            />
            <View
              style={StyleSheet.absoluteFill}
              onStartShouldSetResponder={() => !isSegmenting}
              onMoveShouldSetResponder={() => !isSegmenting}
              onResponderGrant={(event) => {
                const point = cropPointFromEvent(event);
                cropDragStartRef.current = point;
                if (point) {
                  setManualBbox({ x: point.x, y: point.y, width: 0.02, height: 0.02 });
                }
              }}
              onResponderMove={updateManualBboxFromDrag}
              onResponderRelease={() => {
                cropDragStartRef.current = null;
                setCameraStatus('Da chon bbox. Bam Tai tao de clean input va tao mesh.');
              }}
            >
              {manualBboxStyle && (
                <View style={[styles.manualCropBox, manualBboxStyle]}>
                  <View style={styles.manualCropLabel}>
                    <Text style={styles.manualCropLabelText}>Object</Text>
                  </View>
                </View>
              )}
            </View>
          </View>
        ) : (
          <>
            <CameraView
              ref={cameraRef}
              animateShutter={false}
              style={StyleSheet.absoluteFill}
              facing="back"
            />
            <View style={styles.cameraShade} />
          </>
        )}
        <SafeAreaView pointerEvents="box-none" style={styles.cameraOverlay}>
          <View style={styles.cameraTopBar}>
            <Pressable
              style={styles.backButton}
              onPress={() => {
                scanActiveRef.current = false;
                detectingRef.current = false;
                detectSequenceRef.current += 1;
                setIsScanning(false);
                setIsDetecting(false);
                setIsSegmenting(false);
                clearObjectState();
                setScreen('intro');
              }}
            >
              <Text style={styles.backButtonText}>Trở lại</Text>
            </Pressable>
            <View style={styles.liveBadge}>
              <View style={styles.liveDot} />
              <Text style={styles.liveBadgeText}>{capturedPhoto ? 'Crop' : 'Camera'}</Text>
            </View>
          </View>

          {!capturedPhoto && (
            <View style={styles.scanFrame}>
              <View style={[styles.corner, styles.cornerTopLeft]} />
              <View style={[styles.corner, styles.cornerTopRight]} />
              <View style={[styles.corner, styles.cornerBottomLeft]} />
              <View style={[styles.corner, styles.cornerBottomRight]} />
            </View>
          )}

          <View style={styles.cameraPanel}>
            <Text style={styles.panelTitle}>
              {capturedPhoto ? 'Keo khung quanh vat the' : 'Chup anh vat the'}
            </Text>
            <Text style={styles.panelText}>{cameraStatus}</Text>
            {capturedPhoto && manualBbox && (
              <Text style={styles.selectedText}>
                Bbox: {Math.round(manualBbox.width * 100)}% x {Math.round(manualBbox.height * 100)}%
              </Text>
            )}
            {segmentPreviewPath && (
              <View style={styles.segmentPreview}>
                <Image
                  source={{ uri: getServerFileUrl(segmentPreviewPath) }}
                  style={styles.segmentPreviewImage}
                />
                <Text style={styles.segmentPreviewText}>Local clean input gui sang Hunyuan worker.</Text>
              </View>
            )}
            {reconstructionResult && (reconstructionResult.files?.preview_png || meshFilePath) && (
              <View style={styles.reconstructionPanel}>
                {reconstructionResult.files?.preview_png && (
                  <Image
                    source={{ uri: getServerFileUrl(reconstructionResult.files.preview_png) }}
                    style={styles.reconstructionPreviewImage}
                  />
                )}
                <View style={styles.reconstructionInfo}>
                  <Text style={styles.reconstructionTitle}>3D mesh ready</Text>
                  <Text style={styles.reconstructionText}>
                    {reconstructionResult.num_points || 0} pts -> {meshSummary.vertices || 0} verts / {meshSummary.faces || 0} faces
                  </Text>
                  <View style={styles.linkRow}>
                    {meshFilePath && (
                      <Pressable
                        style={styles.fileLink}
                        onPress={() => openServerFile(meshFilePath)}
                      >
                        <Text style={styles.fileLinkText}>{meshFileLabel}</Text>
                      </Pressable>
                    )}
                    {canPaintTexture && (
                      <Pressable
                        style={[styles.fileLink, styles.textureLink, isPaintingTexture && styles.disabledCameraAction]}
                        onPress={paintTexture}
                        disabled={isPaintingTexture}
                      >
                        <Text style={styles.fileLinkText}>
                          {isPaintingTexture ? 'Painting' : 'Paint texture'}
                        </Text>
                      </Pressable>
                    )}
                    {(arGlbPath || arUsdzPath) && (
                      <Pressable
                        style={styles.fileLink}
                        onPress={openArPreview}
                      >
                        <Text style={styles.fileLinkText}>AR</Text>
                      </Pressable>
                    )}
                    {coloredMeshPath && (
                      <Pressable
                        style={styles.fileLink}
                        onPress={() => openServerFile(coloredMeshPath)}
                      >
                        <Text style={styles.fileLinkText}>Color PLY</Text>
                      </Pressable>
                    )}
                    {pointCloudPath && (
                      <Pressable
                        style={styles.fileLink}
                        onPress={() => openServerFile(pointCloudPath)}
                      >
                        <Text style={styles.fileLinkText}>Point PLY</Text>
                      </Pressable>
                    )}
                    {reconstructionInputPath && (
                      <Pressable
                        style={styles.fileLink}
                        onPress={() => openServerFile(reconstructionInputPath)}
                      >
                        <Text style={styles.fileLinkText}>Input</Text>
                      </Pressable>
                    )}
                  </View>
                </View>
              </View>
            )}
            <View style={styles.actionRow}>
              <Pressable
                style={[
                  styles.cameraAction,
                  styles.secondaryAction,
                  capturedPhoto && styles.scanningAction,
                ]}
                onPress={capturedPhoto ? retakePhoto : capturePhotoForCrop}
                disabled={isSegmenting}
              >
                <Text style={styles.secondaryActionText}>
                  {capturedPhoto ? 'Chup lai' : 'Chup anh'}
                </Text>
              </Pressable>
              <Pressable
                style={[
                  styles.cameraAction,
                  styles.primaryAction,
                  (!capturedPhoto || !manualBbox || isSegmenting) && styles.disabledCameraAction,
                ]}
                disabled={!capturedPhoto || !manualBbox || isSegmenting}
                onPress={runReconstruction}
              >
                <Text style={styles.primaryActionText}>
                  {isSegmenting ? 'Dang xu ly' : 'Tái tạo'}
                </Text>
              </Pressable>
            </View>
          </View>
        </SafeAreaView>
        <StatusBar style="light" />
      </View>
    );
  }

  if (screen === 'permission') {
    return (
      <SafeAreaView style={styles.permissionScreen}>
        <View style={styles.permissionCard}>
          <Text style={styles.permissionTitle}>Cần quyền camera</Text>
          <Text style={styles.permissionText}>
            Ung dung can quyen camera de chon object, gui bbox ve backend va tai tao
            mesh 3D tu object da detect.
          </Text>
          <Pressable style={styles.primaryButton} onPress={openCamera}>
            <Text style={styles.primaryButtonText}>Cho phép camera</Text>
          </Pressable>
          <Pressable style={styles.ghostButton} onPress={() => setScreen('intro')}>
            <Text style={styles.ghostButtonText}>Về giới thiệu</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.introScreen}>
      <View style={styles.introContent}>
        <View style={styles.brandRow}>
          <Image source={require('./assets/icon.png')} style={styles.logo} />
          <View>
            <Text style={styles.eyebrow}>AI 3D Reconstruction</Text>
            <Text style={styles.brandName}>Recon Mobile</Text>
          </View>
        </View>

        <View style={styles.heroBlock}>
          <Text style={styles.title}>Quét vật thể và tái tạo mô hình 3D</Text>
          <Text style={styles.subtitle}>
            Camera mobile chon bbox vat the, gui anh sang backend de crop va clean input,
            Backend crop object, reconstruct mesh va export artifact 3D.
          </Text>
        </View>

        <View style={styles.workflowPanel}>
          <Text style={styles.sectionTitle}>Luồng xử lý</Text>
          {activeWorkflowSteps.map((step, index) => (
            <View key={step} style={styles.stepRow}>
              <View style={styles.stepIndex}>
                <Text style={styles.stepIndexText}>{index + 1}</Text>
              </View>
              <Text style={styles.stepText}>{step}</Text>
            </View>
          ))}
        </View>

        <View style={styles.noteBox}>
          <Text style={styles.noteTitle}>Phiên bản hiện tại</Text>
          <Text style={styles.noteText}>
            Backend dang dung Hunyuan reconstruction worker. Input preprocessing se crop/clean vat the,
            dung mesh GLB va xuat colored PLY de kiem tra mau sac trong Blender.
          </Text>
        </View>
      </View>

      <View style={styles.bottomBar}>
        <Pressable
          style={[styles.primaryButton, !permission && styles.disabledButton]}
          onPress={openCamera}
          disabled={!permission}
        >
          <Text style={styles.primaryButtonText}>
            {!permission ? 'Đang chuẩn bị camera...' : 'Bắt đầu'}
          </Text>
        </Pressable>
      </View>
      <StatusBar style="dark" />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  introScreen: {
    flex: 1,
    backgroundColor: '#F7F8FA',
  },
  introContent: {
    flex: 1,
    paddingHorizontal: 22,
    paddingTop: 26,
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  logo: {
    width: 52,
    height: 52,
    borderRadius: 12,
  },
  eyebrow: {
    color: '#667085',
    fontSize: 13,
    fontWeight: '600',
  },
  brandName: {
    color: '#101828',
    fontSize: 20,
    fontWeight: '800',
    marginTop: 2,
  },
  heroBlock: {
    marginTop: 42,
  },
  title: {
    color: '#101828',
    fontSize: 31,
    lineHeight: 38,
    fontWeight: '800',
  },
  subtitle: {
    color: '#475467',
    fontSize: 16,
    lineHeight: 24,
    marginTop: 14,
  },
  workflowPanel: {
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    marginTop: 28,
    padding: 16,
  },
  sectionTitle: {
    color: '#101828',
    fontSize: 16,
    fontWeight: '800',
    marginBottom: 12,
  },
  stepRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    minHeight: 42,
  },
  stepIndex: {
    width: 26,
    height: 26,
    borderRadius: 13,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#E6F0FF',
  },
  stepIndexText: {
    color: '#155EEF',
    fontSize: 13,
    fontWeight: '800',
  },
  stepText: {
    flex: 1,
    color: '#344054',
    fontSize: 15,
    lineHeight: 20,
  },
  noteBox: {
    backgroundColor: '#EEF4FF',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#C7D7FE',
    marginTop: 16,
    padding: 14,
  },
  noteTitle: {
    color: '#1849A9',
    fontSize: 14,
    fontWeight: '800',
  },
  noteText: {
    color: '#344054',
    fontSize: 14,
    lineHeight: 20,
    marginTop: 6,
  },
  bottomBar: {
    paddingHorizontal: 22,
    paddingBottom: 24,
    paddingTop: 14,
    backgroundColor: '#F7F8FA',
  },
  primaryButton: {
    minHeight: 54,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#155EEF',
  },
  disabledButton: {
    backgroundColor: '#98A2B3',
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '800',
  },
  ghostButton: {
    minHeight: 48,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
  },
  ghostButtonText: {
    color: '#155EEF',
    fontSize: 15,
    fontWeight: '700',
  },
  permissionScreen: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 22,
    backgroundColor: '#F7F8FA',
  },
  permissionCard: {
    width: '100%',
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    padding: 20,
  },
  permissionTitle: {
    color: '#101828',
    fontSize: 22,
    fontWeight: '800',
  },
  permissionText: {
    color: '#475467',
    fontSize: 15,
    lineHeight: 22,
    marginBottom: 18,
    marginTop: 10,
  },
  cameraScreen: {
    flex: 1,
    backgroundColor: '#000000',
  },
  cameraShade: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.16)',
  },
  cropPreviewArea: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#000000',
  },
  manualCropBox: {
    position: 'absolute',
    minWidth: 28,
    minHeight: 28,
    borderWidth: 3,
    borderColor: '#A3E635',
    backgroundColor: 'rgba(163, 230, 53, 0.12)',
  },
  manualCropLabel: {
    position: 'absolute',
    left: -3,
    top: -30,
    minHeight: 28,
    justifyContent: 'center',
    paddingHorizontal: 8,
    backgroundColor: '#A3E635',
  },
  manualCropLabelText: {
    color: '#1A2E05',
    fontSize: 13,
    fontWeight: '900',
  },
  detectionBox: {
    position: 'absolute',
    borderWidth: 3,
    borderColor: '#A3E635',
    backgroundColor: 'rgba(163, 230, 53, 0.14)',
  },
  selectedDetectionBox: {
    borderColor: '#155EEF',
    backgroundColor: 'rgba(21, 94, 239, 0.18)',
  },
  detectionLabel: {
    position: 'absolute',
    left: -3,
    top: -30,
    minHeight: 28,
    justifyContent: 'center',
    paddingHorizontal: 8,
    backgroundColor: '#A3E635',
  },
  detectionLabelText: {
    color: '#1A2E05',
    fontSize: 13,
    fontWeight: '900',
  },
  cameraOverlay: {
    flex: 1,
    justifyContent: 'space-between',
    padding: 18,
  },
  cameraTopBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  backButton: {
    minHeight: 40,
    borderRadius: 8,
    justifyContent: 'center',
    paddingHorizontal: 14,
    backgroundColor: 'rgba(255, 255, 255, 0.92)',
  },
  backButtonText: {
    color: '#101828',
    fontSize: 14,
    fontWeight: '800',
  },
  liveBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    minHeight: 40,
    borderRadius: 8,
    paddingHorizontal: 14,
    backgroundColor: 'rgba(16, 24, 40, 0.74)',
  },
  liveDot: {
    width: 9,
    height: 9,
    borderRadius: 5,
    backgroundColor: '#12B76A',
  },
  liveBadgeText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '800',
  },
  scanFrame: {
    alignSelf: 'center',
    width: '82%',
    aspectRatio: 0.72,
    position: 'relative',
  },
  corner: {
    position: 'absolute',
    width: 42,
    height: 42,
    borderColor: '#FFFFFF',
  },
  cornerTopLeft: {
    left: 0,
    top: 0,
    borderLeftWidth: 4,
    borderTopWidth: 4,
  },
  cornerTopRight: {
    right: 0,
    top: 0,
    borderRightWidth: 4,
    borderTopWidth: 4,
  },
  cornerBottomLeft: {
    left: 0,
    bottom: 0,
    borderLeftWidth: 4,
    borderBottomWidth: 4,
  },
  cornerBottomRight: {
    right: 0,
    bottom: 0,
    borderRightWidth: 4,
    borderBottomWidth: 4,
  },
  cameraPanel: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 8,
    padding: 16,
  },
  panelTitle: {
    color: '#101828',
    fontSize: 18,
    fontWeight: '800',
  },
  panelText: {
    color: '#475467',
    fontSize: 14,
    lineHeight: 20,
    marginTop: 6,
  },
  selectedText: {
    color: '#155EEF',
    fontSize: 14,
    fontWeight: '800',
    marginTop: 8,
  },
  segmentPreview: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginTop: 10,
  },
  segmentPreviewImage: {
    width: 72,
    height: 72,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#D0D5DD',
    backgroundColor: '#FFFFFF',
  },
  segmentPreviewText: {
    flex: 1,
    color: '#344054',
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '700',
  },
  reconstructionPanel: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 10,
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#B2DDFF',
    backgroundColor: '#EFF8FF',
  },
  reconstructionPreviewImage: {
    width: 82,
    height: 82,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#D0D5DD',
    backgroundColor: '#FFFFFF',
  },
  reconstructionInfo: {
    flex: 1,
    justifyContent: 'center',
  },
  reconstructionTitle: {
    color: '#1849A9',
    fontSize: 14,
    fontWeight: '900',
  },
  reconstructionText: {
    color: '#344054',
    fontSize: 13,
    lineHeight: 18,
    marginTop: 4,
  },
  linkRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 8,
  },
  fileLink: {
    minHeight: 30,
    minWidth: 54,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#155EEF',
  },
  fileLinkText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '900',
  },
  textureLink: {
    minWidth: 104,
    backgroundColor: '#079455',
  },
  actionRow: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 14,
  },
  cameraAction: {
    flex: 1,
    minHeight: 50,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryAction: {
    borderWidth: 1,
    borderColor: '#D0D5DD',
    backgroundColor: '#FFFFFF',
  },
  scanningAction: {
    borderColor: '#A3E635',
    backgroundColor: '#ECFCCB',
  },
  disabledCameraAction: {
    opacity: 0.65,
  },
  secondaryActionText: {
    color: '#344054',
    fontSize: 15,
    fontWeight: '800',
  },
  primaryAction: {
    backgroundColor: '#155EEF',
  },
  primaryActionText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '800',
  },
});
