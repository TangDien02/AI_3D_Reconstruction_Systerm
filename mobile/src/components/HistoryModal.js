import React, { useState } from 'react';
import {
  Modal, View, Text, TouchableOpacity,
  ScrollView, Image, TextInput, StyleSheet,
} from 'react-native';
import { LogoMark } from './LogoMark';
import { C } from '../theme';

const HistoryItem = ({ item, onTexture, onExport, onPreview, onDelete, onRename, getServerFileUrl }) => {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(item.label);

  const thumbUri = getServerFileUrl(item.thumbPath);
  const dateStr = new Date(item.timestamp).toLocaleDateString('vi-VN', {
    day: '2-digit', month: '2-digit', year: 'numeric',
  });

  const commit = () => {
    if (name.trim()) onRename?.(item.id, name.trim());
    setEditing(false);
  };

  return (
    <View style={S.card}>
      {/* Thumb */}
      <TouchableOpacity onPress={() => onPreview(item)} activeOpacity={0.8}>
        <View style={S.thumbWrap}>
          {thumbUri ? (
            <Image source={{ uri: thumbUri }} style={S.thumb} resizeMode="cover" />
          ) : (
            <View style={S.thumbEmpty}>
              <Text style={S.thumbEmptyIcon}>⬡</Text>
            </View>
          )}
          {item.isTextured && (
            <View style={S.badge}>
              <Text style={S.badgeTxt}>TEX</Text>
            </View>
          )}
          <View style={S.thumbOverlay}>
            <Text style={{ color: '#fff', fontSize: 22 }}>⬡</Text>
          </View>
        </View>
      </TouchableOpacity>

      {/* Meta */}
      <View style={S.meta}>
        {editing ? (
          <TextInput
            style={S.editInput}
            value={name}
            onChangeText={setName}
            onBlur={commit}
            onSubmitEditing={commit}
            autoFocus
            selectTextOnFocus
          />
        ) : (
          <TouchableOpacity onPress={() => setEditing(true)}>
            <Text style={S.label} numberOfLines={1}>{item.label}</Text>
          </TouchableOpacity>
        )}
        <Text style={S.date}>{dateStr}</Text>
        {item.backend && (
          <Text style={S.backend}>{item.backend.replace('_remote', '')}</Text>
        )}
      </View>

      {/* Actions */}
      <View style={S.actions}>
        {!item.isTextured && (
          <TouchableOpacity onPress={() => onTexture(item)} style={S.btn} hitSlop={{ top: 8, bottom: 8 }}>
            <Text style={[S.btnIcon, { color: C.green }]}>◈</Text>
          </TouchableOpacity>
        )}
        <TouchableOpacity onPress={() => onPreview(item)} style={S.btn} hitSlop={{ top: 8, bottom: 8 }}>
          <Text style={[S.btnIcon, { color: C.accentLight }]}>⬡</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => onExport(item)} style={S.btn} hitSlop={{ top: 8, bottom: 8 }}>
          <Text style={[S.btnIcon, { color: C.amber }]}>↓</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => onDelete(item.id)} style={[S.btn, S.btnDanger]} hitSlop={{ top: 8, bottom: 8 }}>
          <Text style={[S.btnIcon, { color: C.red }]}>✕</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const HistoryModal = ({
  visible, history, onClose,
  onTexture, onExport, onPreview, onDelete,
  getServerFileUrl,
}) => {
  const handleRename = (id, newLabel) => {
    // Bubble up if parent supports it — no-op otherwise
  };

  return (
    <Modal visible={visible} animationType="slide" statusBarTranslucent transparent>
      <View style={S.overlay}>
        <View style={S.sheet}>
          {/* Handle */}
          <View style={S.handle} />

          {/* Header */}
          <View style={S.header}>
            <LogoMark size={22} showText />
            <View style={{ flex: 1 }} />
            <Text style={S.count}>
              {history.length} {history.length === 1 ? 'scan' : 'scans'}
            </Text>
            <TouchableOpacity onPress={onClose} style={S.closeBtn}>
              <Text style={S.closeTxt}>✕</Text>
            </TouchableOpacity>
          </View>

          {/* List */}
          <ScrollView
            contentContainerStyle={S.listContent}
            showsVerticalScrollIndicator={false}
          >
            {history.length === 0 ? (
              <View style={S.empty}>
                <Text style={S.emptyHex}>⬡</Text>
                <Text style={S.emptyTitle}>No scans yet</Text>
                <Text style={S.emptySub}>Go capture your first object.</Text>
              </View>
            ) : (
              history.map(item => (
                <HistoryItem
                  key={item.id}
                  item={item}
                  onTexture={onTexture}
                  onExport={onExport}
                  onPreview={onPreview}
                  onDelete={onDelete}
                  onRename={handleRename}
                  getServerFileUrl={getServerFileUrl}
                />
              ))
            )}
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
};

const S = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.75)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: C.bgCard,
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    height: '88%',
    overflow: 'hidden',
    borderTopWidth: 1,
    borderTopColor: C.border,
  },
  handle: {
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: C.textMuted,
    alignSelf: 'center',
    marginTop: 12,
    marginBottom: 4,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: C.borderSubtle,
    gap: 10,
  },
  count: {
    color: C.textMuted,
    fontSize: 12,
    fontWeight: '600',
    marginRight: 8,
  },
  closeBtn: {
    width: 32,
    height: 32,
    borderRadius: 10,
    backgroundColor: C.bgCardAlt,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: C.border,
  },
  closeTxt: { color: C.textSecondary, fontSize: 14 },
  listContent: { padding: 16, paddingBottom: 40, gap: 12 },

  // Card
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.bgCardAlt,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: C.borderSubtle,
    padding: 12,
    gap: 12,
  },
  thumbWrap: {
    width: 64,
    height: 64,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: C.bgSurface,
    position: 'relative',
  },
  thumb: { width: '100%', height: '100%' },
  thumbEmpty: {
    width: '100%',
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  thumbEmptyIcon: { fontSize: 24, color: C.textMuted },
  thumbOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0,0,0,0)',
  },
  badge: {
    position: 'absolute',
    top: 4,
    right: 4,
    backgroundColor: C.greenDim,
    borderRadius: 4,
    paddingHorizontal: 4,
    paddingVertical: 1,
    borderWidth: 1,
    borderColor: C.greenBorder,
  },
  badgeTxt: { color: C.green, fontSize: 7, fontWeight: '800', letterSpacing: 0.5 },
  meta: { flex: 1, gap: 3 },
  label: {
    color: C.textPrimary,
    fontWeight: '700',
    fontSize: 14,
  },
  date: { color: C.textMuted, fontSize: 11 },
  backend: {
    color: C.accentLight,
    fontSize: 9,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginTop: 1,
  },
  editInput: {
    color: C.white,
    backgroundColor: C.bgSurface,
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
    fontSize: 14,
    fontWeight: '700',
    borderWidth: 1,
    borderColor: C.accentLight,
  },
  actions: { flexDirection: 'column', gap: 4 },
  btn: {
    width: 34,
    height: 34,
    borderRadius: 10,
    backgroundColor: C.bgSurface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: C.border,
  },
  btnDanger: { borderColor: C.redDim },
  btnIcon: { fontSize: 15 },

  // Empty
  empty: { alignItems: 'center', paddingTop: 80, gap: 12 },
  emptyHex: { fontSize: 56, color: C.textDim },
  emptyTitle: { color: C.textSecondary, fontSize: 18, fontWeight: '700' },
  emptySub: { color: C.textMuted, fontSize: 13 },
});

export default HistoryModal;