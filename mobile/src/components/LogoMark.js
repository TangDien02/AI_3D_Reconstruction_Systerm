import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { C } from '../theme';

export const LogoMark = ({ size = 28, showText = false, pulse = false }) => {
  const glowAnim = useRef(new Animated.Value(0.6)).current;

  useEffect(() => {
    if (!pulse) return;
    Animated.loop(
      Animated.sequence([
        Animated.timing(glowAnim, { toValue: 1, duration: 1600, useNativeDriver: true }),
        Animated.timing(glowAnim, { toValue: 0.6, duration: 1600, useNativeDriver: true }),
      ])
    ).start();
    return () => glowAnim.stopAnimation();
  }, [pulse]);

  return (
    <View style={S.container}>
      <Animated.View
        style={[
          S.box,
          {
            width: size,
            height: size,
            borderRadius: size * 0.22,
            opacity: pulse ? glowAnim : 1,
          },
        ]}
      >
        <Text style={[S.text3D, { fontSize: size * 0.42 }]}>3D</Text>
      </Animated.View>
      {showText && (
        <Text style={[S.textRecon, { fontSize: size * 0.72, letterSpacing: size * 0.04 }]}>
          RECON
        </Text>
      )}
    </View>
  );
};

const S = StyleSheet.create({
  container: { flexDirection: 'row', alignItems: 'center', gap: 9 },
  box: {
    backgroundColor: 'rgba(45,107,228,0.15)',
    borderWidth: 1.5,
    borderColor: C.accentMid,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text3D: {
    color: C.accentLight,
    fontWeight: '900',
    letterSpacing: -0.5,
  },
  textRecon: {
    color: C.textPrimary,
    fontWeight: '900',
  },
});