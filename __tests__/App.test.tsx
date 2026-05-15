import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import App from '../App';

describe('ReconApp Workflow - Virtual DOM Simulation', () => {
  it('Trải qua toàn bộ quy trình quét 3D (End-to-End Workflow)', async () => {
    // ==========================================
    // [ BƯỚC 1: READY ] - Màn hình IntroScreen
    // ==========================================
    const { getByText, getByTestId, queryByText } = render(<App />);
    
    const startButton = getByText('Bắt đầu quét');
    expect(startButton).toBeTruthy();

    // ==========================================
    // [ BƯỚC 2: DETECTING ] - Chuyển sang ScannerScreen
    // ==========================================
    fireEvent.press(startButton);
    
    // Intro Screen phải biến mất
    expect(queryByText('Bắt đầu quét')).toBeNull(); 
    
    // UI báo hiệu Camera đang hoạt động và đang tìm vật thể (YOLO mock)
    const detectingHeader = getByText('Đang tìm kiếm vật thể...');
    expect(detectingHeader).toBeTruthy();

    // ==========================================
    // [ BƯỚC 3: OBJECT_SELECTED ] - Người dùng chọn Bounding Box
    // ==========================================
    // Giả lập UI render một TouchableOpacity có testID='bounding-box'
    const boundingBox = getByTestId('bounding-box');
    fireEvent.press(boundingBox);

    // ==========================================
    // [ BƯỚC 4: SCANNING ] - Chế độ quay quét 360 độ
    // ==========================================
    const scanningHeader = getByText('Vui lòng quay thiết bị xung quanh vật thể');
    expect(scanningHeader).toBeTruthy();
    
    // Giả lập người dùng bấm nút Quay Video (Record)
    const recordButton = getByTestId('record-button');
    fireEvent.press(recordButton); // Bắt đầu quay
    fireEvent.press(recordButton); // Dừng quay

    // ==========================================
    // [ BƯỚC 5: UPLOADING ] - Đẩy dữ liệu lên máy chủ
    // ==========================================
    const uploadingStatus = getByText('Đang tải dữ liệu lên máy chủ...');
    expect(uploadingStatus).toBeTruthy();

    // ==========================================
    // [ BƯỚC 6 & 7: PROCESSING -> DONE ] - Chờ kết quả 3D
    // ==========================================
    // Dùng waitFor để chờ UI thay đổi trạng thái (mô phỏng API bất đồng bộ)
    await waitFor(() => {
      const doneStatus = getByText('Hoàn tất tái tạo 3D!');
      expect(doneStatus).toBeTruthy();
      
      // Khung chứa render model 3D (ví dụ: Expo GL hoặc WebView) phải xuất hiện
      const modelViewer = getByTestId('3d-model-viewer');
      expect(modelViewer).toBeTruthy();
    }, { timeout: 3000 });
  });
});
