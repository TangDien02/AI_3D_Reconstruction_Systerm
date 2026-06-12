export const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export const shortErrorMessage = (message) => {
  if (!message) return 'Unknown error';
  return String(message).replace(/\s+/g, ' ').slice(0, 240);
};

export const addPaddingToBbox = (bbox, paddingFactor = 0.15) => {
  const padW = bbox.width * paddingFactor;
  const padH = bbox.height * paddingFactor;
  return {
    x: Math.max(0, bbox.x - padW / 2),
    y: Math.max(0, bbox.y - padH / 2),
    width: bbox.width + padW,
    height: bbox.height + padH,
  };
};

export const clamp01 = (value) => Math.max(0, Math.min(1, value));
