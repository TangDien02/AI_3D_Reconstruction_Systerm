---
name: front-end-skill-design
description: Procedural guidance for UI/UX design and frontend development in the 3DRecon mobile app. Use when editing or creating new React Native / Expo UI components, ensuring consistent styling, animations, loading states, and professional dark-mode aesthetics.
---

# Front-end Design Skill

This skill provides expert guidance for maintaining and evolving the 3DRecon mobile app frontend.

Use this skill whenever creating, editing, or refactoring UI screens/components in the 3DRecon mobile app.

## Project Context

- App type: React Native / Expo mobile app.
- Product: 3DRecon, a mobile-to-3D reconstruction app.
- Main flow: capture/upload object image → process through backend → show reconstruction progress → preview/export 3D model.
- Priority: professional demo-ready UI, stable layout, minimal code disruption.

## Core Design Principles

### 1. Modern Dark Mode Aesthetic

Use the app theme from `src/theme.js`.

Preferred visual direction:

- Base Background: `#0A0E1A` deep navy/black
- Primary Accent: `#2D6BE4` electric blue
- Secondary Accent: `#10D98A` emerald green
- Border Style: subtle, semi-transparent `rgba(99,179,237,0.15)`
- Typography:
  - Heavy headers: `800-900`
  - Body text: `400-600`
  - Avoid small low-contrast text

Do not hardcode colors directly inside components. If a needed color/token does not exist, add it to `src/theme.js` first.

### 2. Interaction & Feedback

Always provide clear feedback for user actions:

- Use `ActivityIndicator` for immediate network actions.
- Use `ProcessingTimeline` for long-running reconstruction jobs.
- Use disabled states for buttons while requests are running.
- Show error states clearly with retry options.
- Show success states when model generation/export is complete.

Use subtle animations only:

- Fade in/out for modal and screen transitions.
- Pulse for primary CTA/loading emphasis.
- Avoid excessive motion or distracting effects.

### 3. Component Guidelines

#### Viewer3DModal

- Always validate the model URL before mounting the viewer.
- Handle loading, error, empty URL, and unsupported format states.
- Do not open the modal if the model URL is missing or invalid.
- Keep the viewer controls simple and mobile-friendly.

#### LogoMark

- Use `LogoMark` for consistent branding across screens.
- Do not recreate the logo manually in each screen.
- Keep logo spacing consistent with the app header style.

#### ProcessingTimeline

- Use `ProcessingTimeline` for backend processing status.
- Map backend `stage` strings to the 4-step visual flow:
  1. Upload
  2. Preprocess
  3. Generate 3D
  4. Preview / Export
- Unknown stages should fall back to a safe “Processing...” state.

## UI Modification Workflow

1. Check `src/theme.js`
   - Use existing colors, radius, spacing, shadows, and typography tokens.
   - Add missing tokens to the theme instead of hardcoding values.

2. Preserve existing logic
   - Do not rewrite working backend/API logic unless required.
   - Do not change navigation behavior unless the task asks for it.
   - Keep existing props and function names when possible.

3. Modularize components
   - If a component exceeds around 100 lines, extract reusable parts to `src/components/`.
   - Keep screen files focused on layout and state orchestration.
   - Keep pure UI components reusable and prop-driven.

4. Verify layout
   - Use `onLayout` for dynamic sizing, especially camera overlays, crop boxes, bounding boxes, and model preview containers.
   - Avoid fixed dimensions that break on small phones.
   - Respect safe areas, status bar, and bottom home indicator.

5. Platform consistency
   - Use `Platform.select` where needed for iOS/Android differences.
   - Check touch targets are large enough for mobile.
   - Avoid web-only CSS or DOM APIs.

6. Loading and error states
   - Every network-dependent UI must have loading, error, empty, and success states.
   - Do not leave the user stuck without feedback during reconstruction.

## Code Quality Rules

- Prefer simple, readable React Native styles.
- Avoid large inline anonymous style objects when reused.
- Avoid unnecessary dependencies.
- Do not introduce web-only libraries.
- Keep naming consistent with existing project files.
- Do not remove existing comments or unrelated code.
- Make the smallest safe change needed for the requested UI improvement.

## Definition of Done

A UI change is complete only when:

- It uses `src/theme.js`.
- It preserves existing app logic.
- It handles loading/error/empty states.
- It works on common mobile screen sizes.
- It keeps the dark 3DRecon visual identity.
- It does not hardcode theme colors inside components.
- It does not break existing navigation or backend integration.

## Resources

- See `src/theme.js` for color palette and global UI configs.
- See `src/utils.js` for geometry and math helpers.
- See `src/components/` for reusable UI components.