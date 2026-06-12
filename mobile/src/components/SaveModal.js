import React from 'react';
import { Modal, View, Text, TextInput, TouchableOpacity, StyleSheet, KeyboardAvoidingView, Platform } from 'react-native';
import { C } from '../theme';

const SaveModal = ({ visible, saveName, setSaveName, onCancel, onSave }) => (
  <Modal visible={visible} animationType="fade" transparent>
    <KeyboardAvoidingView
      style={S.overlay}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <View style={S.card}>
        {/* Icon */}
        <View style={S.iconWrap}>
          <Text style={S.iconText}>⬡</Text>
        </View>

        <Text style={S.title}>Name this object</Text>
        <Text style={S.sub}>The model will be saved to your history.</Text>

        <TextInput
          style={S.input}
          placeholder="e.g. Coffee mug, Sneaker..."
          placeholderTextColor={C.textMuted}
          value={saveName}
          onChangeText={setSaveName}
          onSubmitEditing={onSave}
          returnKeyType="done"
          autoFocus
          selectionColor={C.accentLight}
        />

        <View style={S.row}>
          <TouchableOpacity style={S.btnSec} onPress={onCancel} activeOpacity={0.7}>
            <Text style={S.btnSecTxt}>Cancel</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[S.btnPri, !saveName.trim() && S.btnPriDisabled]}
            onPress={onSave}
            disabled={!saveName.trim()}
            activeOpacity={0.8}
          >
            <Text style={S.btnPriTxt}>Save</Text>
          </TouchableOpacity>
        </View>
      </View>
    </KeyboardAvoidingView>
  </Modal>
);

const S = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.78)',
    justifyContent: 'center',
    paddingHorizontal: 28,
  },
  card: {
    backgroundColor: C.bgCard,
    borderRadius: 24,
    padding: 28,
    borderWidth: 1,
    borderColor: C.border,
    alignItems: 'center',
    gap: 8,
  },
  iconWrap: {
    width: 56,
    height: 56,
    borderRadius: 16,
    backgroundColor: C.accentGlow,
    borderWidth: 1,
    borderColor: C.borderActive,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 4,
  },
  iconText: { fontSize: 26, color: C.accentLight },
  title: {
    color: C.textPrimary,
    fontSize: 18,
    fontWeight: '800',
    marginBottom: 2,
  },
  sub: {
    color: C.textMuted,
    fontSize: 13,
    textAlign: 'center',
    marginBottom: 12,
  },
  input: {
    width: '100%',
    color: C.textPrimary,
    backgroundColor: C.bgCardAlt,
    borderRadius: 14,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 15,
    fontWeight: '600',
    borderWidth: 1.5,
    borderColor: C.borderActive,
    marginBottom: 8,
  },
  row: {
    flexDirection: 'row',
    gap: 12,
    width: '100%',
    marginTop: 4,
  },
  btnSec: {
    flex: 1,
    height: 50,
    backgroundColor: C.bgCardAlt,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: C.border,
  },
  btnSecTxt: {
    color: C.textSecondary,
    fontSize: 15,
    fontWeight: '700',
  },
  btnPri: {
    flex: 1.5,
    height: 50,
    backgroundColor: C.accent,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  btnPriDisabled: { opacity: 0.45 },
  btnPriTxt: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '800',
  },
});

export default SaveModal;