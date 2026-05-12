import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import App from '../App';

describe('ReconApp Workflow - DOM Test', () => {
  it('Chuyển trạng thái từ READY sang DETECTING/SCANNING theo workflow', () => {
    // 1. Render App (Trạng thái mặc định: IntroScreen - READY)
    const { getByText, queryByText } = render(<App />);

    // Kiểm tra màn hình Intro
    const startButton = getByText('Bắt đầu quét'); 
    expect(startButton).toBeTruthy();

    // 2. Chuyển sang ScannerScreen
    fireEvent.press(startButton);

    // Kiểm tra màn hình Scanner (Trạng thái: DETECTING / READY TO SCAN)
    const scanHeader = getByText('Sẵn sàng quét');
    expect(scanHeader).toBeTruthy();
    
    // Nút Bắt đầu bên Intro không còn
    expect(queryByText('Bắt đầu quét')).toBeNull();
    
    // (Tuỳ thuộc vào UI thực tế của bạn, bạn có thể viết thêm logic test cho 
    // luồng "Chụp ảnh -> Nhận diện -> Chọn vật thể -> Bắt đầu quay video")
  });
});
