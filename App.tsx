import { useState } from 'react';

import { IntroScreen } from './src/screens/IntroScreen';
import { ScannerScreen } from './src/screens/ScannerScreen';
import type { AppMode } from './src/types/app';

export default function App() {
  const [mode, setMode] = useState<AppMode>('intro');

  if (mode === 'scanner') {
    return <ScannerScreen onBack={() => setMode('intro')} />;
  }

  return <IntroScreen onStart={() => setMode('scanner')} />;
}
