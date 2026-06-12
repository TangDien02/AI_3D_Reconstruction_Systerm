import React from 'react';
import { Modal, View, Text, TouchableOpacity, ScrollView, Image } from 'react-native';
import { LogoMark } from './LogoMark';
import { C } from '../theme';

const HistoryModal = ({ visible, history, onClose, onTexture, onExport, onPreview, onDelete, getServerFileUrl, styles }) => {
  return (
    <Modal visible={visible} animationType="slide" statusBarTranslucent transparent>
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <LogoMark size={24} showText />
            <Text style={styles.modalTitle}>History</Text>
            <TouchableOpacity onPress={onClose} style={styles.closeBtn}>
              <Text style={styles.closeTxt}>✕</Text>
            </TouchableOpacity>
          </View>
          <ScrollView contentContainerStyle={{ padding: 20 }}>
            {history.length === 0 ? (
              <View style={styles.emptyBox}>
                <Text style={styles.emptyIcon}>⬡</Text>
                <Text style={styles.emptyHistory}>No scans found yet</Text>
              </View>
            ) : history.map(item => (
              <View key={item.id} style={styles.historyCard}>
                <Image source={{ uri: getServerFileUrl(item.thumbPath) }} style={styles.historyThumb} />
                <View style={{ flex: 1, marginLeft: 12 }}>
                  <Text style={styles.historyLabel} numberOfLines={1}>{item.label}</Text>
                  <Text style={styles.historyDate}>{new Date(item.timestamp).toLocaleDateString()}</Text>
                </View>
                <View style={styles.historyActions}>
                  {!item.isTextured && (
                    <TouchableOpacity onPress={() => onTexture(item)} style={styles.historyBtn}>
                      <Text style={{ color: C.green, fontSize: 16 }}>◈</Text>
                    </TouchableOpacity>
                  )}
                  <TouchableOpacity onPress={() => onExport(item)} style={styles.historyBtn}>
                    <Text style={{ color: C.amber, fontSize: 16 }}>↓</Text>
                  </TouchableOpacity>
                  <TouchableOpacity onPress={() => onPreview(item)} style={styles.historyBtn}>
                    <Text style={{ color: C.accentLight, fontSize: 18 }}>⬡</Text>
                  </TouchableOpacity>
                  <TouchableOpacity onPress={() => onDelete(item.id)} style={styles.historyBtn}>
                    <Text style={{ color: C.red, fontSize: 16 }}>✕</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ))}
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
};

export default HistoryModal;
