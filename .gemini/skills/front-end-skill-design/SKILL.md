---
name: front-end-skill-design
description: >
  Use for ANY UI work in the 3DRecon React Native/Expo app. Trigger when:
  adding or modifying App.js screens (intro, camera, permission); creating or
  editing files in src/components/ (LogoMark, ProcessingTimeline, Viewer3DModal,
  SaveModal, HistoryModal); adjusting styling, spacing, or colors; adding
  animations or transitions; fixing camera overlay or bounding-box layout;
  updating ProcessingTimeline stage strings; handling loading/error/success states
  in the camera flow; ensuring visual consistency with the dark navy design system;
  or when asked to "improve the UI", "add a feature", "fix the layout", or
  "make it look better" in the 3DRecon app.
---

# 3DRecon Frontend Design Skill

Expert, codebase-grounded guidance for the 3DRecon React Native / Expo mobile app.

---

## Architecture at a Glance

- **Main file**: `App.js` — owns ALL state, API calls, and screen rendering. Do not split this without explicit instruction.
- **Theme**: `src/theme.js` — single source of truth for colors (`C`), `API_BASE_URL`, `STORAGE_KEY`, `CONFIG`.
- **Components**: `src/components/` — stateless/prop-driven UI pieces.
- **Navigation**: No React Navigation. A single `screen` state string controls which screen renders.
- **Storage**: `AsyncStorage` for history (max 20 items, JSON-encoded).
- **3D Viewer**: `react-native-webview` + inline Three.js r128 HTML inside `Viewer3DModal.js`.

---

## Screen Navigation

```js
// Possible values: 'intro' | 'camera' | 'permission'
const [screen, setScreen] = useState('intro');

// The camera block is an early return at the top of App's render:
if (screen === 'camera') { return <View style={S.camScreen}>...</View>; }

// The intro screen is the final return block.
// NEVER add React Navigation or expo-router.
```

---

## Color System — `src/theme.js`

**Import `C`, never hardcode hex values.**

```js
import { C } from '../theme';          // from components
import { C } from './src/theme';       // from App.js
```

| Token | Value | Primary Usage |
|---|---|---|
| `C.bg` | `#0A0E1A` | All screen backgrounds |
| `C.bgCard` | `#111827` | Modal header/footer bars, viewer chrome |
| `C.bgCardAlt` | `#1A2233` | Cards, secondary buttons, history items |
| `C.border` | `rgba(99,179,237,0.15)` | Default borders, separators |
| `C.borderActive` | `rgba(99,179,237,0.5)` | Focused/hovered borders |
| `C.accent` | `#2D6BE4` | Primary CTA, logo border, main highlight |
| `C.accentLight` | `#5B8FEF` | Icons, active state color, status text |
| `C.accentGlow` | `rgba(45,107,228,0.25)` | Active timeline dot background |
| `C.green` | `#10D98A` | Success, texture button, ✓ steps |
| `C.greenDim` | `rgba(16,217,138,0.15)` | Completed dot background |
| `C.amber` | `#F59E0B` | Export / download action icon |
| `C.amberDim` | `rgba(245,158,11,0.15)` | Amber backgrounds |
| `C.red` | `#EF4444` | Error states, delete button |
| `C.textPrimary` | `#F0F4FF` | Titles, labels, card text |
| `C.textSecondary` | `#8A9DC4` | Secondary labels, close button text |
| `C.textMuted` | `#4A5A78` | Hints, dates, empty-state copy |
| `C.scanBox` | `rgba(99,179,237,0.9)` | Camera scan overlay stroke |
| `C.white` | `#FFFFFF` | Text on accent-colored backgrounds |

> ⚠️ **Bug note**: `C.accentActive` is referenced in `App.js` (`S.actionBtn` Save button) but is **not defined** in `theme.js`, so it resolves to `undefined`. The intended value is likely `C.accentLight`. Fix by adding `accentActive: '#5B8FEF'` to `theme.js`.

**Adding a new token:**
1. Add it to the `C` object in `src/theme.js`
2. Then use it — never inline it first and refactor later.

---

## Component API Reference

### `LogoMark`
```js
import { LogoMark } from './src/components/LogoMark';

<LogoMark size={28} showText={false} />
// size    — controls box width/height AND font size proportionally
// showText — when true, renders "RECON" text to the right of the box
// Used in: camTitleBox, modalHeader, viewerHeader
```

### `ProcessingTimeline`
```js
import { ProcessingTimeline } from './src/components/ProcessingTimeline';

<ProcessingTimeline stage={processingStage} />
// stage — string. Render only when processingStage !== ''
```

**Exact stage string → visual step mapping:**

| `stage` value(s) | Step | Icon | Index |
|---|---|---|---|
| `'capturing'` | Capture | 📷 | 0 |
| `'preprocess'` · `'cropping'` · `'cleaning'` | Clean | ✦ | 1 |
| `'reconstructing'` · `'generating_shape'` | Mesh | ⬡ | 2 |
| `'texturing'` · `'texturing · <any>'` | Texture | ◈ | 3 |
| `'done'` | All steps show ✓ | — | — |
| `'error'` | Current step turns red | — | — |
| `''` / falsy | Nothing highlighted | — | — |

**How App.js drives stages:**
```js
setProcessingStage('capturing');       // just before takePictureAsync
setProcessingStage('preprocess');      // background removal / crop stage
setProcessingStage('reconstructing');  // mesh generation polling loop
setProcessingStage('texturing');       // texture paint polling loop
setProcessingStage(`texturing · ${payload.status}`); // detailed texture status
setProcessingStage('done');            // success
setProcessingStage('error');           // any caught exception
setProcessingStage('');                // reset / clearObjectState
```

### `Viewer3DModal`
```js
import { Viewer3DModal } from './src/components/Viewer3DModal';

<Viewer3DModal
  visible={show3DViewer}
  modelUrl={viewerModelUrl}          // full URL string — MUST be non-null/non-empty
  onClose={() => setShow3DViewer(false)}
/>
// Internally: Modal > View > WebView with build3DViewerHTML(modelUrl)
// Three.js version pinned to r128 — never change CDN version
// Controls: OrbitControls (drag, pinch), wireframe toggle, auto-rotate toggle
```

**Always use the helper — never open without a valid URL:**
```js
// This pattern is already in App.js — reuse it:
const open3DViewer = (path) => {
  const url = getServerFileUrl(path);
  if (url) { setViewerModelUrl(url); setShow3DViewer(true); }
};
```

### `SaveModal`
```js
import SaveModal from './src/components/SaveModal';

<SaveModal
  visible={showSaveModal}
  saveName={saveName}
  setSaveName={setSaveName}
  onCancel={() => setShowSaveModal(false)}
  onSave={handleSave}
  styles={S}   // ← passes App.js's local StyleSheet object
/>
// The modal reuses App.js styles (priBtn, secBtn, editInput, modalOverlay, etc.)
// This pattern avoids duplicating style definitions
```

### `HistoryModal`
```js
import HistoryModal from './src/components/HistoryModal';

<HistoryModal
  visible={showHistory}
  history={history}                        // HistoryItem[]
  onClose={() => setShowHistory(false)}
  onTexture={textureHistoryItem}           // (item) => void
  onExport={exportHistoryItem}             // (item) => void — opens URL
  onPreview={(item) => { setShowHistory(false); open3DViewer(item.meshPath); }}
  onDelete={deleteHistoryItem}             // (id) => void
  getServerFileUrl={getServerFileUrl}      // (path) => string | null
  styles={S}                               // ← App.js StyleSheet
/>
```

**HistoryItem shape:**
```js
{
  id: string,               // job_id or Date.now().toString()
  label: string,            // user-provided name
  timestamp: number,        // Date.now()
  meshPath: string,         // server path to .glb file
  thumbPath: string,        // server path to thumbnail image
  isTextured: boolean,      // true if mesh_textured_glb exists
  backend: string,          // e.g. 'hunyuan_remote'
  meshSummary: object,      // reconstruction.mesh metadata
}
```

---

## Animation System

**Only `Animated` from React Native. No Reanimated, no Moti, no CSS.**

```js
// Screen fade-in — already in App.js, don't duplicate:
const fadeAnim = useRef(new Animated.Value(0)).current;
useEffect(() => {
  Animated.timing(fadeAnim, {
    toValue: 1, duration: 320, useNativeDriver: true
  }).start();
}, [screen]);

// Wrap animated elements:
<Animated.View style={{ opacity: fadeAnim }}>...</Animated.View>

// Rules:
// - useNativeDriver: true for opacity and transform (ALWAYS)
// - useNativeDriver: false only for layout props (height, width) — avoid these
// - Duration range: 200–400ms for most transitions
// - For CTA pulse: use a looping Animated.sequence with scale
```

---

## Styling Conventions

### StyleSheet naming
Every file defines its styles as a local `const S = StyleSheet.create({ ... })`.
`App.js` passes `styles={S}` to modals that need to share its style definitions.

### Typography
| Role | `fontWeight` | Color |
|---|---|---|
| Screen title / header | `'900'` | `C.textPrimary` |
| Card title, modal title | `'800'` | `C.white` or `C.textPrimary` |
| Body / label | `'600'`–`'700'` | `C.textSecondary` |
| Hint / date / muted | `'400'` | `C.textMuted` |
| CTA button text | `'800'` | `'#fff'` |

### Border radius scale
| Element | `borderRadius` |
|---|---|
| Badges, small chips | `8`–`10` |
| History action buttons | `12` |
| Primary/secondary buttons | `14`–`18` |
| Cards, modals | `20`–`24` |
| Bottom-sheet modals (top only) | `32` |

### Spacing
Padding inside panels/cards: `20`. Gap between action buttons: `10`–`12`.
Status/badge padding: `{ paddingVertical: 8, paddingHorizontal: 12 }`.

---

## Layout Rules

### Camera overlay
```js
// cropArea must fill remaining space — always flex: 1
<View style={S.cropArea} onLayout={e => setCropAreaLayout(e.nativeEvent.layout)}>

// The imageRect() helper computes scaled image bounds from cropAreaLayout + capturedPhoto
// Always use onLayout — never hardcode dimensions for anything camera-related
```

### Safe areas
```js
// Camera top bar:
paddingTop: Platform.OS === 'ios' ? 60 : 40

// Intro screen scroll:
paddingTop: Platform.OS === 'ios' ? 50 : 0

// Modal/viewer footer (home indicator):
paddingBottom: Platform.OS === 'ios' ? 34 : 14
```

---

## Loading / Error / Empty / Success States

Every network-dependent UI block must handle all four:

```js
// Loading — inline in button
{isProcessing
  ? <ActivityIndicator color="#fff" />
  : <Text style={S.priBtnText}>Reconstruct Full</Text>}

// Error
catch (e) {
  setCameraStatus(`Error: ${shortErrorMessage(e.message)}`);
  setProcessingStage('error');
}

// Empty
<View style={S.emptyBox}>
  <Text style={S.emptyIcon}>⬡</Text>
  <Text style={S.emptyHistory}>No scans found yet</Text>
</View>

// Success
setCameraStatus('3D model ready!');
setProcessingStage('done');
```

---

## API / Backend Integration — DO NOT REWRITE

These functions in App.js are battle-tested — preserve them exactly:

| Function | Purpose |
|---|---|
| `waitForReconstructionJob(payload)` | Polls `/reconstruction-jobs/:id` until `status === 'done'` |
| `reconstructManualBbox()` | Posts bbox crop → `/reconstruct-bbox` → polls |
| `captureAndReconstructFull()` | Takes photo → posts → `/reconstruct-image` |
| `paintTexture()` | Posts job_id → `/paint-texture` → polls |
| `getServerFileUrl(path)` | Prepends `API_BASE_URL` if path is relative |
| `saveToHistory(reconstruction, segment, label)` | Upserts into `AsyncStorage` |
| `clearObjectState()` | Resets all camera/result state to initial values |

Polling config lives in `CONFIG` in `src/theme.js`:
- `RECON_POLL_INTERVAL_MS`: 5000ms
- `RECON_POLL_TIMEOUT_MS`: 45 minutes
- `TEXTURE_POLL_INTERVAL_MS`: 5000ms
- `TEXTURE_POLL_TIMEOUT_MS`: 45 minutes

---

## ❌ Anti-Patterns

```js
// ❌ Hardcoded color
backgroundColor: '#1A2233'
// ✅
backgroundColor: C.bgCardAlt

// ❌ Opening viewer without URL check
setShow3DViewer(true)   // crashes if viewerModelUrl is null
// ✅
const url = getServerFileUrl(path);
if (url) { setViewerModelUrl(url); setShow3DViewer(true); }

// ❌ Fixed height for camera area
style={{ height: 400 }}
// ✅
style={{ flex: 1 }}

// ❌ useNativeDriver: false for opacity/transform
Animated.timing(anim, { toValue: 1, useNativeDriver: false })
// ✅
Animated.timing(anim, { toValue: 1, useNativeDriver: true })

// ❌ Rewriting polling logic
while (true) { /* custom poll */ }
// ✅ Reuse waitForReconstructionJob() / paintTexture()

// ❌ Adding a navigation library
import { NavigationContainer } from '@react-navigation/native'
// ✅
setScreen('camera')

// ❌ Processing stage string not in mapping table
setProcessingStage('uploading')   // Timeline shows nothing
// ✅ Use exact mapped strings
setProcessingStage('preprocess')

// ❌ Using C.accentActive (undefined — see Bug note above)
borderColor: C.accentActive
// ✅ Until fixed in theme.js:
borderColor: C.accentLight
```

---

## Definition of Done Checklist

Before any UI change is complete:

- [ ] Every color references `C.*` from `src/theme.js` — zero hardcoded hex values
- [ ] New tokens added to `src/theme.js` before use, not after
- [ ] Safe area handled with `Platform.OS === 'ios'` — status bar top, home indicator bottom
- [ ] Dynamic containers use `onLayout` (camera area, bounding box, model preview)
- [ ] All 4 states covered: loading, error, empty, success
- [ ] `ProcessingTimeline` stage strings match the mapping table exactly
- [ ] `Viewer3DModal` only opens when `modelUrl` is a non-empty string
- [ ] Animations use `useNativeDriver: true`
- [ ] No React Navigation or new routing library added
- [ ] Existing API/polling functions preserved verbatim
- [ ] Minimum touch target: `44×44` for interactive elements
- [ ] Dark navy aesthetic maintained: `C.bg` screen, `C.accent` CTA, `C.green` success

---

## Resources

- `src/theme.js` — color tokens, API URL, CONFIG polling/timeout constants
- `src/utils.js` — `delay(ms)`, `shortErrorMessage(msg)`, `addPaddingToBbox(bbox, factor)`, `clamp01(v)`
- `src/components/` — all reusable UI components (see API Reference above)
- `App.js` — all state declarations, API actions, screen rendering, StyleSheet `S`