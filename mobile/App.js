import { useEffect, useRef, useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import {
  Image,
  Linking,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL || 'http://192.168.1.3:8000';
const DETECT_FRAME_WIDTH = 640;
const DETECT_CAPTURE_QUALITY = 0.5;
const DETECT_UPLOAD_COMPRESS = 0.65;
const DETECT_COOLDOWN_MS = 350;
const DETECT_EMPTY_HOLD_MS = 900;
const RECON_CAPTURE_QUALITY = 0.92;
const activeWorkflowSteps = [
  'Nhan anh hoac video object',
  'YOLO phat hien va crop vat the',
  'TripoSR tu tach nen va reconstruct mesh',
  'Export GLB, colored PLY va point cloud',
];

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
  const [segmentResult, setSegmentResult] = useState(null);
  const [reconstructionResult, setReconstructionResult] = useState(null);
  const [permission, requestPermission] = useCameraPermissions();

  const clearObjectState = () => {
    detectSequenceRef.current += 1;
    lastStableDetectionRef.current = null;
    setDetectedObjects([]);
    setDetectedImageSize(null);
    setLatestDetectImageUri(null);
    setSelectedObject(null);
    setSelectedFrameUri(null);
    setSelectedDetectionSize(null);
    setSegmentResult(null);
    setReconstructionResult(null);
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
    reconstructSelectedObject();
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
    setCameraStatus(`Da chon ${object.label}. Bam Tai tao de chup full-res va chay TripoSR.`);
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
      setCameraStatus('Dang YOLO crop full-res + TripoSR reconstruct + export GLB...');

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
      setCameraStatus(`Loi reconstruct: ${error.message}`);
    } finally {
      setIsSegmenting(false);
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
    segmentResult?.files?.triposr_crop
    || segmentResult?.files?.crop
    || segmentResult?.files?.masked_crop
  );
  const meshFilePath = (
    reconstructionResult?.files?.mesh_glb
    || reconstructionResult?.files?.mesh_obj
    || reconstructionResult?.files?.mesh
  );
  const meshFileLabel = reconstructionResult?.files?.mesh_glb
    ? 'GLB'
    : reconstructionResult?.files?.mesh_obj
      ? 'OBJ'
      : 'MESH';
  const coloredMeshPath = reconstructionResult?.files?.mesh_colored_ply;
  const pointCloudPath = reconstructionResult?.files?.pointcloud_ply;
  const triposrInputPath = reconstructionResult?.files?.triposr_input;
  const meshSummary = reconstructionResult?.mesh || {};
  const openServerFile = (path) => {
    const url = getServerFileUrl(path);
    if (url) {
      Linking.openURL(url);
    }
  };

  if (screen === 'camera') {
    return (
      <View
        style={styles.cameraScreen}
        onLayout={(event) => setCameraLayout(event.nativeEvent.layout)}
      >
        <CameraView
          ref={cameraRef}
          animateShutter={false}
          style={StyleSheet.absoluteFill}
          facing="back"
        />
        <View style={styles.cameraShade} />
        <View pointerEvents="box-none" style={StyleSheet.absoluteFill}>
          {isScanning && detectedObjects.map(renderDetectionBox)}
        </View>
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
              <Text style={styles.liveBadgeText}>Camera</Text>
            </View>
          </View>

          <View style={styles.scanFrame}>
            <View style={[styles.corner, styles.cornerTopLeft]} />
            <View style={[styles.corner, styles.cornerTopRight]} />
            <View style={[styles.corner, styles.cornerBottomLeft]} />
            <View style={[styles.corner, styles.cornerBottomRight]} />
          </View>

          <View style={styles.cameraPanel}>
            <Text style={styles.panelTitle}>Đưa vật thể vào khung</Text>
            <Text style={styles.panelText}>{cameraStatus}</Text>
            {selectedObject && (
              <Text style={styles.selectedText}>
                Da chon: {selectedObject.label} {Math.round((selectedObject.confidence || 0) * 100)}%
              </Text>
            )}
            {segmentPreviewPath && (
              <View style={styles.segmentPreview}>
                <Image
                  source={{ uri: getServerFileUrl(segmentPreviewPath) }}
                  style={styles.segmentPreviewImage}
                />
                <Text style={styles.segmentPreviewText}>YOLO bbox crop gui sang TripoSR.</Text>
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
                  <Text style={styles.reconstructionTitle}>TripoSR mesh ready</Text>
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
                    {triposrInputPath && (
                      <Pressable
                        style={styles.fileLink}
                        onPress={() => openServerFile(triposrInputPath)}
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
                  isScanning && styles.scanningAction,
                ]}
                onPress={toggleScanning}
              >
                <Text style={styles.secondaryActionText}>
                  {isScanning ? 'Dừng quét' : 'Quét vật thể'}
                </Text>
              </Pressable>
              <Pressable
                style={[
                  styles.cameraAction,
                  styles.primaryAction,
                  (!selectedObject || isSegmenting) && styles.disabledCameraAction,
                ]}
                disabled={!selectedObject || isSegmenting}
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
            Ung dung can quyen camera de quet object, gui YOLO bbox ve backend va tai tao
            mesh / point cloud bang TripoSR.
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
            Camera mobile detect object lien tuc, chon bbox, gui anh sang backend de YOLO crop,
            TripoSR tu tach nen, reconstruct mesh va export GLB / colored PLY.
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
            Backend dang dung TripoSR. YOLO chi chon bbox va crop vat the; TripoSR xu ly nen,
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
