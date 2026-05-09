import { StatusBar } from 'expo-status-bar';
import { Image, Platform, StyleSheet, Text, View } from 'react-native';

import { PipelineItem } from '../components/PipelineItem';
import { PrimaryButton } from '../components/PrimaryButton';
import { colors, radii, spacing } from '../constants/theme';
import { introSteps, pipelineSteps } from '../constants/workflow';

type IntroScreenProps = {
  onStart: () => void;
};

export function IntroScreen({ onStart }: IntroScreenProps) {
  return (
    <View style={styles.screen}>
      <StatusBar style="dark" />
      <View style={styles.hero}>
        <View style={styles.logoWrap}>
          <Image source={require('../../assets/icon.png')} style={styles.logo} />
        </View>

        <Text style={styles.title}>3DRecon</Text>
        <Text style={styles.subtitle}>
          Quét vật thể bằng camera, gửi dữ liệu lên server AI và nhận lại mô
          hình 3D.
        </Text>

        <View style={styles.flowRow}>
          {introSteps.map(step => (
            <View key={step} style={styles.flowChip}>
              <Text style={styles.flowText}>{step}</Text>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.panel}>
        <View style={styles.panelHeader}>
          <Text style={styles.panelTitle}>Luồng xử lý</Text>
          <Text style={styles.panelTag}>Expo Go</Text>
        </View>

        <View style={styles.pipeline}>
          {pipelineSteps.map(step => (
            <PipelineItem
              key={step.index}
              index={step.index}
              title={step.title}
              text={step.text}
            />
          ))}
        </View>
      </View>

      <PrimaryButton label="Bắt đầu quét" onPress={onStart} />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingTop: Platform.OS === 'android' ? 42 : 28,
    paddingBottom: spacing.xl,
    backgroundColor: colors.background,
  },
  hero: {
    alignItems: 'center',
    paddingTop: spacing.xl,
  },
  logoWrap: {
    width: 108,
    height: 108,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radii.lg,
    backgroundColor: colors.surfaceSoft,
    borderWidth: 1,
    borderColor: '#d2dfd6',
  },
  logo: {
    width: 72,
    height: 72,
    resizeMode: 'contain',
  },
  title: {
    marginTop: 24,
    color: colors.text,
    fontSize: 42,
    fontWeight: '800',
    letterSpacing: 0,
  },
  subtitle: {
    maxWidth: 330,
    marginTop: 12,
    color: colors.muted,
    fontSize: 16,
    lineHeight: 23,
    textAlign: 'center',
  },
  flowRow: {
    width: '100%',
    flexDirection: 'row',
    gap: 8,
    marginTop: 26,
  },
  flowChip: {
    flex: 1,
    minHeight: 38,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radii.sm,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  flowText: {
    color: '#26362e',
    fontSize: 12,
    fontWeight: '700',
  },
  panel: {
    padding: 18,
    borderRadius: radii.sm,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  panelHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 14,
  },
  panelTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '800',
  },
  panelTag: {
    color: colors.green,
    fontSize: 12,
    fontWeight: '800',
  },
  pipeline: {
    gap: 14,
  },
});
