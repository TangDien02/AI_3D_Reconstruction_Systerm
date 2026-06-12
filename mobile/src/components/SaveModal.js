import React from 'react';
import { Modal, View, Text, TextInput, TouchableOpacity } from 'react-native';
import { C } from '../theme';

const SaveModal = ({ visible, saveName, setSaveName, onCancel, onSave, styles }) => {
  return (
    <Modal visible={visible} animationType="fade" transparent>
      <View style={[styles.modalOverlay, { justifyContent: 'center', padding: 32 }]}>
        <View style={[styles.modalContent, { height: 'auto', borderRadius: 24, padding: 24 }]}>
          <Text style={{ color: C.white, fontSize: 18, fontWeight: '800', marginBottom: 16 }}>Save 3D Object</Text>
          <TextInput
            style={[styles.editInput, { height: 50, marginBottom: 20 }]}
            placeholder="Enter object name..."
            placeholderTextColor={C.textMuted}
            value={saveName}
            onChangeText={setSaveName}
            autoFocus
          />
          <View style={{ flexDirection: 'row', gap: 12 }}>
            <TouchableOpacity style={[styles.secBtn, { flex: 1 }]} onPress={onCancel}>
              <Text style={styles.secBtnText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.priBtn, { flex: 1 }]}
              onPress={onSave}
            >
              <Text style={styles.priBtnText}>Save</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );
};

export default SaveModal;
