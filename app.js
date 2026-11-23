/* Recurrent-state report: all surfaces are measured token states, never interpolated. */
(() => {
  'use strict';

  const tabs = [...document.querySelectorAll('button[data-layer]')];
  const panels = [...document.querySelectorAll('[data-layer-panel]')];
  if (tabs.length && panels.length) {
    function activateTab(tab, focus = false) {
      tabs.forEach(button => {
        const active = button === tab;
        button.setAttribute('aria-selected', String(active));
        button.tabIndex = active ? 0 : -1;
      });
      panels.forEach(panel => {
        panel.hidden = panel.dataset.layerPanel !== tab.dataset.layer;
      });
      if (focus) tab.focus();
    }
    tabs.forEach((tab, index) => {
      const panel = panels.find(item => item.dataset.layerPanel === tab.dataset.layer);
      tab.id ||= `layer-tab-${tab.dataset.layer}`;
      tab.setAttribute('role', 'tab');
      if (panel) {
        panel.id ||= `layer-panel-${tab.dataset.layer}`;
        panel.setAttribute('role', 'tabpanel');
        panel.setAttribute('aria-labelledby', tab.id);
        panel.tabIndex = 0;
        tab.setAttribute('aria-controls', panel.id);
      }
      tab.addEventListener('click', () => activateTab(tab));
      tab.addEventListener('keydown', event => {
        let next;
        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
        if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = tabs.length - 1;
        if (next !== undefined) {
          event.preventDefault();
          activateTab(tabs[next], true);
        }
      });
    });
    activateTab(tabs.find(tab => tab.getAttribute('aria-selected') === 'true') || tabs[0]);
  }

  const byId = id => document.getElementById(id);
  const plot = byId('state-plot');
  if (!plot) return;
  const ui = {
    layer: byId('surface-layer'), head: byId('surface-head'),
    mode: byId('surface-mode'), time: byId('surface-time'),
    play: byId('surface-play'), label: byId('surface-time-label'),
    status: byId('surface-status'), scale: byId('surface-scale'),
    grid: byId('surface-grid'), reset: byId('surface-reset'),
  };
  if (Object.values(ui).some(element => !element)) {
    if (ui.status) ui.status.textContent = 'Viewer controls are unavailable. The static plots below remain accessible.';
    return;
  }

  let stateData, info, pairBounds, globalBound;
  const pairCache = new Map();
  let pairRequest = 0, pairAbort, loadedPair;
  let loading = false, ready = false, busy = false, pending = false;
  let playing = false, playbackId = 0;
  const defaultCamera = () => ({
    eye: {x: 1.5, y: 1.5, z: 1.15}, center: {x: 0, y: 0, z: 0},
    up: {x: 0, y: 0, z: 1}, projection: {type: 'perspective'},
  });
  let camera = defaultCamera(), cameraRevision = 0;
  const copyCamera = value => JSON.parse(JSON.stringify(value));
  function rememberCamera(event) {
    if (event['scene.camera']) camera = copyCamera(event['scene.camera']);
  }
  const controls = [ui.layer, ui.head, ui.mode, ui.time, ui.play, ui.scale, ui.grid, ui.reset];
  controls.forEach(control => { control.disabled = true; });
  ui.status.setAttribute('role', 'status');
  ui.status.setAttribute('aria-live', 'polite');
  ui.status.textContent = 'Loading measured per-token states. Each selected layer/head is approximately 16 MiB.';
  ui.time.min = '0';
  ui.time.max = '255';
  ui.time.step = '1';
  ui.time.value = '255';
  ui.layer.value = '8';
  ui.head.value = '0';
  ui.mode.value = 'magnitude';
  ui.scale.value = 'pair';
  ui.grid.value = '1';
  ui.label.textContent = 't = 256';
  ui.play.setAttribute('aria-pressed', 'false');

  const format = value => Number(value).toLocaleString('en-US', {maximumSignificantDigits: 5});
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

  function pause() {
    playing = false;
    playbackId += 1;
    ui.play.textContent = info && Number(ui.time.value) === info.times.length - 1 ? 'Replay' : 'Play';
    ui.play.setAttribute('aria-pressed', 'false');
    ui.play.setAttribute('aria-label', 'Play measured token states in forward order; restart at token 1 when at the end');
  }

  function selected() {
    const t = Number(ui.time.value);
    const li = info.layers.indexOf(Number(ui.layer.value));
    const hi = info.heads.indexOf(Number(ui.head.value));
    if (li < 0 || hi < 0 || !Number.isInteger(t) || t < 0 || t >= info.times.length) {
      throw new Error('Invalid checkpoint selection');
    }
    return {t, li, hi};
  }

  function surface() {
    const {t, li, hi} = selected();
    const signed = ui.mode.value === 'signed';
    const stride = Number(ui.grid.value) === 2 ? 2 : 1;
    const [, L, H, K, V] = info.shape;
    const base = t * K * V;
    const x = Array.from({length: Math.ceil(V / stride)}, (_, j) => j * stride);
    const y = Array.from({length: Math.ceil(K / stride)}, (_, i) => i * stride);
    const z = y.map(k => x.map(v => {
      const value = stateData[base + k * V + v];
      return signed ? value : Math.abs(value);
    }));
    let max = 0, energy = 0;
    for (let j = 0; j < K * V; j += 1) {
      const value = stateData[base + j];
      max = Math.max(max, Math.abs(value));
      energy += value * value;
    }
    const bound = Math.max(ui.scale.value === 'global' ? globalBound : pairBounds[li * H + hi], 1e-12);
    const lower = signed ? -bound : 0;
    ui.label.textContent = `t = ${info.times[t]}`;
    ui.time.setAttribute('aria-valuetext', `After ${info.times[t]} tokens, state ${t + 1} of ${info.times.length}`);
    ui.status.textContent = `L${info.layers[li]} · H${info.heads[hi]} · ${info.times[t]} tokens · max |S| ${format(max)} · RMS ${format(Math.sqrt(energy / (K * V)))} · fixed z [${format(lower)}, ${format(bound)}]`;
    plot.setAttribute('aria-label', `Recurrent state after ${info.times[t]} tokens, layer ${info.layers[li]}, head ${info.heads[hi]}. X: value channel. Y: key channel. Height: ${signed ? 'signed state' : 'state magnitude'}.`);
    return {
      data: [{type: 'surface', x, y, z,
        colorscale: signed ? 'RdBu' : 'Viridis', reversescale: signed,
        cauto: false, cmin: lower, cmax: bound,
        colorbar: {title: {text: signed ? 'S' : '|S|'}, thickness: 13, len: 0.7},
        hovertemplate: 'Key %{y} · Value %{x}<br>Height %{z:.6g}<extra></extra>',
      }],
      layout: {
        autosize: true, paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
        margin: {l: 0, r: 12, t: 14, b: 0},
        font: {family: 'Arial, sans-serif', size: 13, color: '#404b5a'},
        uirevision: `measured-state-camera-${cameraRevision}`,
        scene: {
          // Keep the complete user view explicit on every frame, including zoom and pan.
          camera: copyCamera(camera),
          xaxis: {title: {text: 'Value channel'}, range: [0, V - 1], autorange: false, backgroundcolor: '#fafbfc', gridcolor: '#e3e8ef'},
          yaxis: {title: {text: 'Key channel'}, range: [0, K - 1], autorange: false, backgroundcolor: '#fafbfc', gridcolor: '#e3e8ef'},
          zaxis: {title: {text: signed ? 'Signed state' : '|State|'}, range: [lower, bound], autorange: false, backgroundcolor: '#fafbfc', gridcolor: '#e3e8ef'},
          aspectmode: 'manual', aspectratio: {x: 1, y: 1, z: 0.7},
          dragmode: 'orbit', uirevision: `measured-state-camera-${cameraRevision}`,
        },
      },
    };
  }

  async function render() {
    if (!ready || loadedPair !== `L${ui.layer.value}-H${ui.head.value}`) return;
    if (busy) { pending = true; return; }
    busy = true;
    try {
      do {
        pending = false;
        const content = surface();
        const config = {responsive: true, displaylogo: false,
          scrollZoom: window.matchMedia('(pointer: fine)').matches,
          modeBarButtonsToRemove: ['toImage', 'resetCameraDefault3d', 'resetCameraLastSave3d'],
          doubleClick: false,
        };
        if (!plot.data) {
          plot.replaceChildren();
          await window.Plotly.newPlot(plot, content.data, content.layout, config);
          plot.on('plotly_relayout', rememberCamera);
          plot.on('plotly_relayouting', rememberCamera);
        } else {
          await window.Plotly.react(plot, content.data, content.layout, config);
        }
      } while (pending && ready && loadedPair === `L${ui.layer.value}-H${ui.head.value}`);
    } catch (error) {
      pause();
      ui.status.textContent = 'The 3D surface could not be rendered. Use a WebGL-capable browser or explore the static plots below.';
      console.error('State surface render failed:', error);
    } finally {
      busy = false;
    }
  }

  async function loadPair() {
    if (!info) return;
    pause();
    const key = `L${ui.layer.value}-H${ui.head.value}`;
    const request = ++pairRequest;
    if (pairAbort) pairAbort.abort();
    ready = false;
    stateData = null;
    [ui.time, ui.play].forEach(control => { control.disabled = true; });
    plot.style.opacity = '0.35';
    ui.status.textContent = `Loading ${key} · 256 measured token states · approximately 16 MiB`;
    try {
      if (pairCache.has(key)) {
        const cached = pairCache.get(key);
        pairCache.delete(key);
        pairCache.set(key, cached);
      } else {
        // At most two pairs remain resident; an in-flight replacement evicts first.
        while (pairCache.size >= 2) pairCache.delete(pairCache.keys().next().value);
        pairAbort = new AbortController();
        const file = info.files[key];
        if (typeof file !== 'string' || !/^states-L\d+-H\d+\.f32$/.test(file)) throw new Error('Invalid state asset name');
        const response = await fetch(`assets/${file}`, {signal: pairAbort.signal});
        if (!response.ok) throw new Error('State asset unavailable');
        const buffer = await response.arrayBuffer();
        if (request !== pairRequest) return;
        const count = info.shape[0] * info.shape[3] * info.shape[4];
        if (buffer.byteLength !== count * 4) throw new Error('Incomplete state data');
        const littleEndian = new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] === 4;
        let values;
        if (littleEndian) values = new Float32Array(buffer);
        else {
          const reader = new DataView(buffer);
          values = new Float32Array(count);
          for (let i = 0; i < count; i += 1) values[i] = reader.getFloat32(i * 4, true);
        }
        for (const value of values) if (!Number.isFinite(value)) throw new Error('Non-finite state');
        pairCache.set(key, values);
      }
      if (request !== pairRequest) return;
      stateData = pairCache.get(key);
      loadedPair = key;
      ready = true;
      controls.forEach(control => { control.disabled = false; });
      plot.style.opacity = '1';
      await render();
    } catch (error) {
      if (request !== pairRequest || error.name === 'AbortError') return;
      ui.status.textContent = 'The selected states could not be loaded. Choose another layer/head or refresh. The static plots below remain available.';
      console.error('State pair loading failed:', error);
    }
  }

  async function load() {
    if (loading || info) return;
    loading = true;
    try {
      if (!window.Plotly) throw new Error('Plotly is unavailable');
      const metaResponse = await fetch('assets/experiment.json');
      if (!metaResponse.ok) throw new Error('Metadata unavailable');
      const source = await metaResponse.json();
      info = {shape: source.state_shape, times: source.token_indices,
              layers: source.layer_ids, heads: source.head_ids, files: source.state_files};
      if (!Array.isArray(info.shape) || info.shape.length !== 5 ||
          info.shape[3] !== 128 || info.shape[4] !== 128 ||
          !Array.isArray(info.times) || !Array.isArray(info.layers) || !Array.isArray(info.heads) ||
          info.times.length !== info.shape[0] || info.layers.length !== info.shape[1] || info.heads.length !== info.shape[2]) {
        throw new Error('Unexpected state schema');
      }
      pairBounds = source.pair_max_abs.flat();
      globalBound = source.global_max_abs;
      if (pairBounds.length !== info.layers.length * info.heads.length ||
          !pairBounds.every(value => Number.isFinite(value) && value >= 0) ||
          !Number.isFinite(globalBound) || globalBound < 0) throw new Error('Invalid fixed ranges');
      ui.time.max = String(info.times.length - 1);
      ui.time.value = String(info.times.length - 1);
      [ui.layer, ui.head, ui.mode, ui.scale, ui.grid].forEach(control => { control.disabled = false; });
      await loadPair();
    } catch (error) {
      info = null;
      ui.status.textContent = 'The measured data or 3D viewer could not be loaded. Refresh or use the static plots and data downloads below.';
      console.error('State explorer initialization failed:', error);
    } finally {
      loading = false;
    }
  }

  [ui.layer, ui.head].forEach(control => control.addEventListener('change', loadPair));
  [ui.mode, ui.scale, ui.grid].forEach(control => control.addEventListener('change', render));
  ui.reset.addEventListener('click', () => {
    camera = defaultCamera();
    cameraRevision += 1;
    render();
  });
  let sliderFrame = 0;
  ui.time.addEventListener('input', () => {
    pause();
    if (info) {
      ui.label.textContent = `t = ${info.times[Number(ui.time.value)]}`;
      ui.time.setAttribute('aria-valuetext', `After ${info.times[Number(ui.time.value)]} tokens`);
    }
    if (!sliderFrame) sliderFrame = requestAnimationFrame(() => { sliderFrame = 0; render(); });
  });
  ui.play.addEventListener('click', async () => {
    if (playing) { pause(); return; }
    if (!ready) return;
    playing = true;
    const id = ++playbackId;
    ui.play.textContent = 'Pause';
    ui.play.setAttribute('aria-label', 'Pause token-state playback');
    ui.play.setAttribute('aria-pressed', 'true');
    if (Number(ui.time.value) >= info.times.length - 1) ui.time.value = '0';
    await render();
    while (playing && id === playbackId) {
      await sleep(160);
      if (!playing || id !== playbackId) break;
      if (busy) continue;
      if (Number(ui.time.value) >= info.times.length - 1) { pause(); break; }
      ui.time.value = String(Number(ui.time.value) + 1);
      await render();
    }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) { observer.disconnect(); load(); }
    }, {rootMargin: '350px'});
    observer.observe(plot);
  } else load();
})();
