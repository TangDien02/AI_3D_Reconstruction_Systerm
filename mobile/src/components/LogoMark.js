import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { C } from '../theme';

export const LogoMark = ({ size = 28, showText = false }) => (
  <View style={S.container}>
    <View style={[S.box, { width: size, height: size, borderRadius: size * 0.25 }]}>
      <Text style={[S.text3D, { fontSize: size * 0.45 }]}>3D</Text>
    </View>
    {showText && (
      <Text style={[S.textRecon, { fontSize: size * 0.8 }]}>RECON</Text>
    )}
  </View>
);

const S = StyleSheet.create({
  container: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  box: { backgroundColor: 'transparent', borderWidth: 2, borderColor: C.accent, alignItems: 'center', justifyContent: 'center' },
  text3D: { color: C.accent, fontWeight: '900', letterSpacing: -0.5 },
  textRecon: { color: C.white, fontWeight: '900', letterSpacing: -1 },
});
