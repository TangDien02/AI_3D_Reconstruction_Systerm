import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { C } from '../theme';

export const ProcessingTimeline = ({ stage }) => {
  const stages = [
    { id: 'capturing', label: 'Capture', icon: '📷' },
    { id: 'preprocess', label: 'Clean', icon: '✦' },
    { id: 'reconstructing', label: 'Mesh', icon: '⬡' },
    { id: 'texturing', label: 'Texture', icon: '◈' },
  ];

  const currentStageId = (s) => {
    if (!s) return 'idle';
    if (s === 'cropping' || s === 'cleaning' || s === 'preprocess') return 'preprocess';
    if (s === 'generating_shape' || s === 'reconstructing') return 'reconstructing';
    if (s.startsWith('texturing')) return 'texturing';
    if (s === 'done') return 'done';
    return s;
  };

  const activeId = currentStageId(stage);
  const getStageStatus = (index) => {
    const activeIndex = stages.findIndex(s => s.id === activeId);
    if (activeId === 'done') return 'completed';
    if (activeId === 'error') return 'failed';
    if (activeIndex === index) return 'active';
    if (activeIndex > index) return 'completed';
    return 'pending';
  };

  return (
    <View style={S.timelineContainer}>
      {stages.map((s, i) => {
        const status = getStageStatus(i);
        return (
          <View key={s.id} style={S.timelineStep}>
            <View style={[
              S.timelineDot,
              status === 'active' && S.timelineDotActive,
              status === 'completed' && S.timelineDotCompleted,
              status === 'failed' && S.timelineDotFailed
            ]}>
              {status === 'completed' ? (
                <Text style={S.timelineCheck}>✓</Text>
              ) : (
                <Text style={[S.timelineIcon, status === 'active' && S.timelineIconActive]}>{s.icon}</Text>
              )}
            </View>
            <Text style={[S.timelineLabel, status === 'active' && S.timelineLabelActive]}>{s.label}</Text>
            {i < stages.length - 1 && (
              <View style={[S.timelineLine, getStageStatus(i+1) !== 'pending' && S.timelineLineActive]} />
            )}
          </View>
        );
      })}
    </View>
  );
};

const S = StyleSheet.create({
  timelineContainer: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: 12, paddingHorizontal: 4, marginBottom: 8 },
  timelineStep: { flex: 1, alignItems: 'center', position: 'relative' },
  timelineDot: { width: 32, height: 32, borderRadius: 16, backgroundColor: C.bgCardAlt, borderWidth: 1, borderColor: C.border, alignItems: 'center', justifyContent: 'center', zIndex: 2 },
  timelineDotActive: { borderColor: C.accentLight, backgroundColor: C.accentGlow, transform: [{ scale: 1.1 }] },
  timelineDotCompleted: { borderColor: C.green, backgroundColor: C.greenDim },
  timelineDotFailed: { borderColor: C.red, backgroundColor: 'rgba(239,68,68,0.1)' },
  timelineIcon: { fontSize: 14, color: C.textMuted },
  timelineIconActive: { color: C.accentLight },
  timelineCheck: { fontSize: 14, color: C.green, fontWeight: '900' },
  timelineLabel: { fontSize: 9, fontWeight: '700', color: C.textMuted, marginTop: 6, textTransform: 'uppercase' },
  timelineLabelActive: { color: C.textPrimary },
  timelineLine: { position: 'absolute', height: 2, backgroundColor: C.border, top: 16, left: '50%', right: '-50%', zIndex: 1 },
  timelineLineActive: { backgroundColor: C.green },
});
