// Shared arm palette + labels for the concealworld dashboard. Matches the colours used
// in ByLayerChart.jsx; extra entries cover a future horizon sweep (h2/h4).
export const COLORS = {
  gpt: '#f2754f', nextlat: '#4fd1c5', mtp: '#b07cff', jtp: '#7c9cff',
  nextlat_h1: '#5ee0a0', nextlat_h2: '#9be07a', nextlat_h4: '#3fb59b',
}
export const LABEL = {
  gpt: 'GPT (baseline)', nextlat: 'NextLat (h8)', mtp: 'MTP', jtp: 'JTP',
  nextlat_h1: 'NextLat-h1', nextlat_h2: 'NextLat-h2', nextlat_h4: 'NextLat-h4',
}
const FALLBACK = ['#e0a23f', '#d65db1', '#6fb1ff', '#8fd14f', '#ff8c6b']
export const colorFor = (arm) =>
  COLORS[arm] || FALLBACK[Math.abs([...arm].reduce((a, c) => a * 31 + c.charCodeAt(0), 7)) % FALLBACK.length]
export const labelFor = (arm) => LABEL[arm] || arm
