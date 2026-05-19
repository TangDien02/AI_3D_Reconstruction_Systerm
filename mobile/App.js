import { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import {
  Image,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

const workflowSteps = [
  'Nhận ảnh hoặc video object',
  'Phát hiện và crop vật thể',
  'Trích xuất feature ảnh',
  'Tái tạo point cloud / model 3D',
];

export default function App() {
  const [screen, setScreen] = useState('intro');
  const [cameraStatus, setCameraStatus] = useState('Camera đã sẵn sàng.');
  const [permission, requestPermission] = useCameraPermissions();

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
    setScreen('camera');
  };

  const showPlaceholder = (action) => {
    setCameraStatus(`${action} sẽ được tích hợp ở bước AI backend.`);
  };

  if (screen === 'camera') {
    return (
      <View style={styles.cameraScreen}>
        <CameraView style={StyleSheet.absoluteFill} facing="back" />
        <View style={styles.cameraShade} />
        <SafeAreaView style={styles.cameraOverlay}>
          <View style={styles.cameraTopBar}>
            <Pressable style={styles.backButton} onPress={() => setScreen('intro')}>
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
            <View style={styles.actionRow}>
              <Pressable
                style={[styles.cameraAction, styles.secondaryAction]}
                onPress={() => showPlaceholder('Quét vật thể')}
              >
                <Text style={styles.secondaryActionText}>Quét vật thể</Text>
              </Pressable>
              <Pressable
                style={[styles.cameraAction, styles.primaryAction]}
                onPress={() => showPlaceholder('Tái tạo 3D')}
              >
                <Text style={styles.primaryActionText}>Tái tạo</Text>
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
            Ứng dụng chỉ cần quyền camera để mở màn hình quét. Phần AI scan và tái tạo
            chưa được tích hợp trong bản này.
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
          <Text style={styles.title}>Quét vật thể và chuẩn bị tái tạo mô hình 3D</Text>
          <Text style={styles.subtitle}>
            Ứng dụng mobile dùng để mở camera, định hướng người dùng quét object và
            gửi dữ liệu cho pipeline AI ở backend trong các bước tiếp theo.
          </Text>
        </View>

        <View style={styles.workflowPanel}>
          <Text style={styles.sectionTitle}>Luồng xử lý dự kiến</Text>
          {workflowSteps.map((step, index) => (
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
            Đã setup giao diện và chức năng mở camera. Các nút quét vật thể và tái
            tạo đang là placeholder cho bước tích hợp AI sau.
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
