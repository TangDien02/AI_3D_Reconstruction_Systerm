import { CameraView, useCameraPermissions } from 'expo-camera';
import { StatusBar } from 'expo-status-bar';
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native';

import { PrimaryButton } from '../components/PrimaryButton';
import { ScanFrame } from '../components/ScanFrame';
import { colors, radii } from '../constants/theme';

type ScannerScreenProps = {
  onBack: () => void;
};

export function ScannerScreen({ onBack }: ScannerScreenProps) {
  const [permission, requestPermission] = useCameraPermissions();

  if (permission?.granted !== true) {
    return (
      <View style={styles.permissionScreen}>
        <StatusBar style="dark" />
        <Text style={styles.permissionTitle}>Cần quyền camera</Text>
        <Text style={styles.permissionText}>
          3DRecon cần camera để nhận diện vật thể và quay video quét 360 độ.
        </Text>
        <View style={styles.permissionAction}>
          <PrimaryButton label="Cấp quyền camera" onPress={requestPermission} />
        </View>
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <StatusBar style="light" />
      <CameraView style={StyleSheet.absoluteFill} facing="back" mode="picture" />

      <View style={styles.overlay} pointerEvents="box-none">
        <View style={styles.topBar}>
          <Pressable
            accessibilityRole="button"
            onPress={onBack}
            style={styles.backButton}>
            <Text style={styles.backButtonText}>‹</Text>
          </Pressable>

          <View style={styles.titleBlock}>
            <Text style={styles.smallLabel}>3DRecon</Text>
            <Text style={styles.title}>Sẵn sàng quét</Text>
          </View>

          <View style={styles.livePill}>
            <Text style={styles.liveText}>CAM</Text>
          </View>
        </View>

        <ScanFrame />

        <View style={styles.controls}>
          <Pressable style={styles.secondaryButton}>
            <Text style={styles.secondaryButtonText}>Nhận diện</Text>
          </Pressable>
          <Pressable style={styles.scanButton}>
            <Text style={styles.scanButtonText}>Quét vật thể</Text>
          </Pressable>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: colors.cameraBlack,
  },
  permissionScreen: {
    flex: 1,
    justifyContent: 'center',
    padding: 24,
    backgroundColor: colors.background,
  },
  permissionTitle: {
    color: colors.text,
    fontSize: 28,
    fontWeight: '800',
    textAlign: 'center',
  },
  permissionText: {
    marginTop: 12,
    color: colors.muted,
    fontSize: 16,
    lineHeight: 23,
    textAlign: 'center',
  },
  permissionAction: {
    marginTop: 28,
  },
  overlay: {
    flex: 1,
    justifyContent: 'space-between',
  },
  topBar: {
    minHeight: Platform.OS === 'android' ? 104 : 115,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'android' ? 30 : 46,
    backgroundColor: 'rgba(0, 0, 0, 0.34)',
  },
  backButton: {
    width: 44,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radii.sm,
    backgroundColor: 'rgba(255, 255, 255, 0.18)',
  },
  backButtonText: {
    color: colors.white,
    fontSize: 36,
    lineHeight: 38,
    fontWeight: '500',
  },
  titleBlock: {
    flex: 1,
  },
  smallLabel: {
    color: '#c9f4db',
    fontSize: 12,
    fontWeight: '900',
  },
  title: {
    marginTop: 3,
    color: colors.white,
    fontSize: 22,
    fontWeight: '800',
  },
  livePill: {
    minWidth: 58,
    minHeight: 30,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radii.sm,
    backgroundColor: colors.orange,
  },
  liveText: {
    color: colors.white,
    fontSize: 12,
    fontWeight: '900',
  },
  controls: {
    flexDirection: 'row',
    gap: 10,
    paddingHorizontal: 16,
    paddingBottom: 18,
  },
  secondaryButton: {
    flex: 1,
    minHeight: 54,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radii.sm,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
  },
  secondaryButtonText: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '800',
  },
  scanButton: {
    flex: 1,
    minHeight: 54,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radii.sm,
    backgroundColor: colors.orange,
  },
  scanButtonText: {
    color: colors.white,
    fontSize: 15,
    fontWeight: '800',
  },
});
