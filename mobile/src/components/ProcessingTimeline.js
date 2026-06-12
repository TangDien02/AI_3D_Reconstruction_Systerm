import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { C } from '../theme';

const STAGES = [
  { id: 'capturing',     label: 'Capture',  icon: '◉' },
  { id: 'preprocess',    label: 'Clean',    icon: '✦' },
  { id: 'reconstructing',label: 'Mesh',     icon: '⬡' },
  { id: 'texturing',     label: 'Texture',  icon: '◈' },
];

function resolveStageId(s) {
  if (!s) return 'idle';
  if (s === 'cropping' || s === 'cleaning' || s === 'preprocess') return 'preprocess';
  if (s === 'generating_shape' || s === 'reconstructing') return 'reconstructing';
  if (s.startsWith('texturing')) return 'texturing';
  if (s === 'done') return 'done';
  return s;
}

const PulsingDot = ({ color }) => {
  const anim = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(anim, { toValue: 1.25, duration: 700, useNativeDriver: true }),
        Animated.timing(anim, { toValue: 1, duration: 700, useNativeDriver: true }),
      ])
    ).start();
    return () => anim.stopAnimation();
  }, []);
  return (
    <Animated.View
      style={[
        S.pulsing,
        { backgroundColor: color, transform: [{ scale: anim }] },
      ]}
    />
  );
};

export const ProcessingTimeline = ({ stage }) => {
  const activeId = resolveStageId(stage);
  const activeIndex = STAGES.findIndex(s => s.id === activeId);

  const getStatus = (i) => {
    if (activeId === 'done') return 'completed';
    if (activeId === 'error') return 'failed';
    if (activeIndex === i) return 'active';
    if (activeIndex > i) return 'completed';
    return 'pending';
  };

  return (
    <View style={S.wrap}>
      {STAGES.map((s, i) => {
        const status = getStatus(i);
        const isLast = i === STAGES.length - 1;

        return (
          <React.Fragment key={s.id}>
            <View style={S.step}>
              <View
                style={[
                  S.dot,
                  status === 'active' && S.dotActive,
                  status === 'completed' && S.dotDone,
                  status === 'failed' && S.dotFail,
                ]}
              >
                {status === 'active' && <PulsingDot color={C.accentLight} />}
                {status === 'completed' && (
                  <Text style={S.check}>✓</Text>
                )}
                {(status === 'pending' || status === 'failed') && (
                  <Text style={[S.icon, status === 'failed' && { color: C.red }]}>
                    {s.icon}
                  </Text>
                )}
              </View>
              <Text
                style={[
                  S.label,
                  status === 'active' && S.labelActive,
                  status === 'completed' && S.labelDone,
                ]}
              >
                {s.label}
              </Text>
            </View>

            {!isLast && (
              <View
                style={[
                  S.line,
                  getStatus(i + 1) !== 'pending' && S.lineActive,
                ]}
              />
            )}
          </React.Fragment>
        );
      })}
    </View>
  );
};

const S = StyleSheet.create({
  wrap: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 14,
    paddingHorizontal: 8,
    marginBottom: 6,
  },
  step: {
    alignItems: 'center',
    width: 58,
  },
  dot: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: C.bgSurface,
    borderWidth: 1,
    borderColor: C.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  dotActive: {
    borderColor: C.accentLight,
    backgroundColor: C.accentGlow,
  },
  dotDone: {
    borderColor: C.green,
    backgroundColor: C.greenDim,
  },
  dotFail: {
    borderColor: C.red,
    backgroundColor: C.redDim,
  },
  pulsing: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  icon: {
    fontSize: 13,
    color: C.textMuted,
  },
  check: {
    fontSize: 13,
    color: C.green,
    fontWeight: '900',
  },
  label: {
    fontSize: 9,
    fontWeight: '700',
    color: C.textMuted,
    marginTop: 5,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  labelActive: { color: C.accentLight },
  labelDone: { color: C.green },
  line: {
    flex: 1,
    height: 1.5,
    backgroundColor: C.borderSubtle,
    marginBottom: 16,
    marginHorizontal: -2,
  },
  lineActive: {
    backgroundColor: C.greenBorder,
  },
});