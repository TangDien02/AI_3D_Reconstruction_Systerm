export const introSteps = ['Nhận diện', 'Chọn vật thể', 'Quét 360'];

export const pipelineSteps = [
  {
    index: '01',
    title: 'Camera client',
    text: 'Chụp frame và quay video scan.',
  },
  {
    index: '02',
    title: 'AI server',
    text: 'YOLO nhận diện, tracking vật thể.',
  },
  {
    index: '03',
    title: '3D model',
    text: 'Xử lý video và xuất model.',
  },
];
