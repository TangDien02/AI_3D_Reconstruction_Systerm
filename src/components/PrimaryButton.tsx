import { Pressable, StyleSheet, Text } from 'react-native';

import { colors, radii } from '../constants/theme';

type PrimaryButtonProps = {
  label: string;
  onPress: () => void;
};

export function PrimaryButton({ label, onPress }: PrimaryButtonProps) {
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      style={({ pressed }) => [
        styles.button,
        pressed && styles.pressed,
      ]}>
      <Text style={styles.label}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    minHeight: 56,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radii.sm,
    backgroundColor: colors.orange,
  },
  pressed: {
    opacity: 0.82,
  },
  label: {
    color: colors.white,
    fontSize: 17,
    fontWeight: '800',
  },
});
