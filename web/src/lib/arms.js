// Shared arm palette + labels for the concealworld dashboard. Matches the colours used
// in ByLayerChart.jsx; extra entries cover a future steps-ahead sweep (2/4 steps).
// "N steps ahead" = how far the predict-ahead objective looks (formerly "horizon N").
export const COLORS = {
  gpt: '#f2754f', nextlat: '#4fd1c5', mtp: '#b07cff', jtp: '#7c9cff',
  nextlat_h1: '#5ee0a0', nextlat_h2: '#9be07a', nextlat_h4: '#3fb59b',
}
export const LABEL = {
  gpt: 'GPT (baseline)', nextlat: 'NextLat · 8 steps ahead', mtp: 'MTP', jtp: 'JTP',
  nextlat_h1: 'NextLat · 1 step ahead', nextlat_h2: 'NextLat · 2 steps ahead', nextlat_h4: 'NextLat · 4 steps ahead',
}
const FALLBACK = ['#e0a23f', '#d65db1', '#6fb1ff', '#8fd14f', '#ff8c6b']
export const colorFor = (arm) =>
  COLORS[arm] || FALLBACK[Math.abs([...arm].reduce((a, c) => a * 31 + c.charCodeAt(0), 7)) % FALLBACK.length]
export const labelFor = (arm) => LABEL[arm] || arm
