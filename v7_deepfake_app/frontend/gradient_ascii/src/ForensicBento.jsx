import React, { useRef, useState, useEffect } from 'react';
import MagicBento from './MagicBento';
import './MagicBento.css';

/* =========================================================================
   1. Forensic Visual Component: Acoustic Waveform Preprocessing
   ========================================================================= */
const WAVE_BARS = [18, 35, 52, 78, 64, 45, 82, 95, 68, 54, 76, 40, 60, 30, 48, 22];

function WaveformVisual() {
  const [scrubIndex, setScrubIndex] = useState(null);
  const containerRef = useRef(null);

  const handlePointerMove = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const idx = Math.min(WAVE_BARS.length - 1, Math.floor((x / rect.width) * WAVE_BARS.length));
    setScrubIndex(idx);
  };

  const handlePointerLeave = () => {
    setScrubIndex(null);
  };

  const activeTime = scrubIndex !== null ? ((scrubIndex / 16) * 4.0).toFixed(2) : null;
  const activeAmp = scrubIndex !== null ? ((WAVE_BARS[scrubIndex] / 100) * 0.95).toFixed(2) : null;

  return (
    <div
      ref={containerRef}
      className="forensic-visual"
      onPointerMove={handlePointerMove}
      onPointerDown={handlePointerMove}
      onPointerLeave={handlePointerLeave}
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        height: '100%',
        width: '100%',
        padding: '6px 0',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '8px' }}>
        <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#F3F5F8', letterSpacing: '-0.01em' }}>
          {activeAmp ? `${activeAmp} FS` : '16,000 Hz'}
        </span>
        <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#8B9CFF', background: 'rgba(139, 156, 255, 0.12)', padding: '2px 7px', borderRadius: '4px' }}>
          {activeTime ? `${activeTime}s / 4.0s` : '64k Samples'}
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: '46px', gap: '3px' }}>
        {WAVE_BARS.map((height, i) => {
          const isSelected = scrubIndex === i;
          return (
            <div
              key={i}
              style={{
                flex: 1,
                height: `${height}%`,
                background: isSelected
                  ? '#8B9CFF'
                  : 'linear-gradient(180deg, rgba(139, 156, 255, 0.75) 0%, rgba(101, 214, 196, 0.4) 100%)',
                borderRadius: '2px',
                transition: 'height 0.15s ease, background 0.15s ease',
                boxShadow: isSelected ? '0 0 8px rgba(139, 156, 255, 0.8)' : 'none',
              }}
            />
          );
        })}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '0.7rem', color: '#7C8798' }}>
        <span>Polyphase Mono</span>
        <span>Peak RMS Normalized</span>
      </div>
    </div>
  );
}

/* =========================================================================
   2. Forensic Visual Component: Spectro-Temporal Decomposition (FrFT & WPD)
   ========================================================================= */
const SPECTRUM_POINTS = [
  [0, 36], [15, 28], [30, 42], [45, 18], [60, 24], [75, 12], [90, 20], [105, 8], [120, 16], [135, 30], [150, 40]
];

function SpectrumVisual() {
  const [scrubX, setScrubX] = useState(null);
  const containerRef = useRef(null);

  const handlePointerMove = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    setScrubX(x);
  };

  const handlePointerLeave = () => {
    setScrubX(null);
  };

  const freq = scrubX !== null ? (((scrubX / 160) * 8.0).toFixed(1)) : 'α = 0.50';
  const subband = scrubX !== null ? `Subband ${Math.min(16, Math.max(1, Math.floor((scrubX / 160) * 16) + 1))}` : '16 Subbands';

  return (
    <div
      ref={containerRef}
      className="forensic-visual"
      onPointerMove={handlePointerMove}
      onPointerDown={handlePointerMove}
      onPointerLeave={handlePointerLeave}
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        height: '100%',
        width: '100%',
        padding: '6px 0',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '8px' }}>
        <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#F3F5F8', letterSpacing: '-0.01em' }}>
          {scrubX !== null ? `${freq} kHz` : 'FrFT α = 0.50'}
        </span>
        <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#65D6C4', background: 'rgba(101, 214, 196, 0.12)', padding: '2px 7px', borderRadius: '4px' }}>
          {subband}
        </span>
      </div>

      <div style={{ position: 'relative', height: '46px', width: '100%' }}>
        <svg viewBox="0 0 150 48" preserveAspectRatio="none" style={{ width: '100%', height: '100%', overflow: 'visible' }}>
          <defs>
            <linearGradient id="spectrumGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#65D6C4" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#65D6C4" stopOpacity="0.0" />
            </linearGradient>
          </defs>
          <path
            d="M0,36 C20,24 35,45 50,18 C65,10 80,26 100,12 C120,6 135,28 150,40 L150,48 L0,48 Z"
            fill="url(#spectrumGrad)"
          />
          <path
            d="M0,36 C20,24 35,45 50,18 C65,10 80,26 100,12 C120,6 135,28 150,40"
            fill="none"
            stroke="#65D6C4"
            strokeWidth="2.2"
            strokeLinecap="round"
          />
        </svg>

        {scrubX !== null && (
          <div
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: `${scrubX}px`,
              width: '1px',
              background: '#F3F5F8',
              boxShadow: '0 0 6px #65D6C4',
              pointerEvents: 'none',
            }}
          />
        )}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '0.7rem', color: '#7C8798' }}>
        <span>Fractional Phase Rotation</span>
        <span>Wavelet Packet (db4)</span>
      </div>
    </div>
  );
}

/* =========================================================================
   3. Forensic Visual Component: LCNN-2D Latent Representation
   ========================================================================= */
const LATENT_WEIGHTS = [0.42, 0.88, 0.65, 0.94, 0.31, 0.72, 0.58, 0.85, 0.49, 0.91, 0.63, 0.77];

function LatentVisual() {
  const [hoverDim, setHoverDim] = useState(null);
  const containerRef = useRef(null);

  const handlePointerMove = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const dim = Math.min(LATENT_WEIGHTS.length - 1, Math.floor((x / rect.width) * LATENT_WEIGHTS.length));
    setHoverDim(dim);
  };

  const handlePointerLeave = () => {
    setHoverDim(null);
  };

  return (
    <div
      ref={containerRef}
      className="forensic-visual"
      onPointerMove={handlePointerMove}
      onPointerDown={handlePointerMove}
      onPointerLeave={handlePointerLeave}
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        height: '100%',
        width: '100%',
        padding: '6px 0',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '8px' }}>
        <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#F3F5F8', letterSpacing: '-0.01em' }}>
          {hoverDim !== null ? `Act ${LATENT_WEIGHTS[hoverDim].toFixed(2)}` : '128D Latent'}
        </span>
        <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#E8B86D', background: 'rgba(232, 184, 109, 0.12)', padding: '2px 7px', borderRadius: '4px' }}>
          {hoverDim !== null ? `Unit ${hoverDim * 10}` : '770k Params'}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: '4px', height: '46px', alignItems: 'end' }}>
        {LATENT_WEIGHTS.map((weight, i) => {
          const isSelected = hoverDim === i;
          return (
            <div
              key={i}
              style={{
                height: `${weight * 100}%`,
                background: isSelected
                  ? '#E8B86D'
                  : 'linear-gradient(180deg, rgba(232, 184, 109, 0.85) 0%, rgba(139, 156, 255, 0.3) 100%)',
                borderRadius: '2px',
                transition: 'height 0.12s ease, background 0.12s ease',
                boxShadow: isSelected ? '0 0 8px rgba(232, 184, 109, 0.7)' : 'none',
              }}
            />
          );
        })}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '0.7rem', color: '#7C8798' }}>
        <span>Max-Feature-Map (MFM)</span>
        <span>Latent Artifact Manifold</span>
      </div>
    </div>
  );
}

/* =========================================================================
   4. Forensic Visual Component: Decision Threshold & Aggregation
   ========================================================================= */
function ThresholdVisual() {
  const [hoverVal, setHoverVal] = useState(null);
  const containerRef = useRef(null);

  const handlePointerMove = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const score = (x / rect.width).toFixed(3);
    setHoverVal(score);
  };

  const handlePointerLeave = () => {
    setHoverVal(null);
  };

  const thresholdPos = 48.2; // 0.4820 threshold at 48.2%
  const currentPos = hoverVal !== null ? `${(hoverVal * 100).toFixed(1)}%` : `${thresholdPos}%`;
  const isSpoof = hoverVal !== null ? parseFloat(hoverVal) >= 0.482 : false;

  return (
    <div
      ref={containerRef}
      className="forensic-visual"
      onPointerMove={handlePointerMove}
      onPointerDown={handlePointerMove}
      onPointerLeave={handlePointerLeave}
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        height: '100%',
        width: '100%',
        padding: '6px 0',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '8px' }}>
        <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#F3F5F8', letterSpacing: '-0.01em' }}>
          {hoverVal !== null ? `Score ${hoverVal}` : 'τ = 0.4820'}
        </span>
        <span
          style={{
            fontSize: '0.75rem',
            fontWeight: 600,
            color: isSpoof ? '#F07878' : '#4CC9A6',
            background: isSpoof ? 'rgba(240, 120, 120, 0.12)' : 'rgba(76, 201, 166, 0.12)',
            padding: '2px 7px',
            borderRadius: '4px',
          }}
        >
          {hoverVal !== null ? (isSpoof ? 'Synthetic Indication' : 'Genuine Concordance') : 'Candidate Boundary'}
        </span>
      </div>

      <div style={{ position: 'relative', height: '46px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        {/* Track bar */}
        <div
          style={{
            height: '12px',
            width: '100%',
            borderRadius: '6px',
            background: 'linear-gradient(90deg, #4CC9A6 0%, #4CC9A6 42%, #E8B86D 48%, #F07878 54%, #F07878 100%)',
            opacity: 0.85,
            position: 'relative',
          }}
        />

        {/* Threshold marker needle */}
        <div
          style={{
            position: 'absolute',
            left: `${thresholdPos}%`,
            top: '4px',
            bottom: '4px',
            width: '2px',
            background: '#FFFFFF',
            boxShadow: '0 0 6px rgba(255, 255, 255, 0.9)',
          }}
        />

        {/* Dynamic pointer if hovering */}
        {hoverVal !== null && (
          <div
            style={{
              position: 'absolute',
              left: currentPos,
              top: '0px',
              width: '4px',
              height: '20px',
              background: '#8B9CFF',
              borderRadius: '2px',
              transform: 'translateX(-50%)',
              boxShadow: '0 0 8px #8B9CFF',
              pointerEvents: 'none',
            }}
          />
        )}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '0.7rem', color: '#7C8798' }}>
        <span>0.00 (Genuine)</span>
        <span style={{ color: '#F3F5F8', fontWeight: 600 }}>Boundary 0.482</span>
        <span>1.00 (Spoof)</span>
      </div>
    </div>
  );
}

/* =========================================================================
   Icons for Forensic Cards (SVG)
   ========================================================================= */
const WaveformIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2 10v4" /><path d="M6 6v12" /><path d="M10 3v18" /><path d="M14 7v10" /><path d="M18 9v6" /><path d="M22 11v2" />
  </svg>
);

const SpectrumIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" /><path d="m10 15 5-3-5-3v6Z" fill="currentColor" opacity="0.3" /><path d="M12 2a10 10 0 0 1 10 10" />
  </svg>
);

const LatentIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <rect width="7" height="7" x="3" y="3" rx="1" /><rect width="7" height="7" x="14" y="3" rx="1" /><rect width="7" height="7" x="14" y="14" rx="1" /><rect width="7" height="7" x="3" y="14" rx="1" />
  </svg>
);

const DecisionIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2v20" /><path d="m17 5-5-3-5 3" /><path d="m17 19-5 3-5-3" /><circle cx="12" cy="12" r="3" />
  </svg>
);

/* =========================================================================
   Forensic Bento Items Definition
   ========================================================================= */
const FORENSIC_ITEMS = [
  {
    id: 'stage-1',
    label: '01 · Signal Processing',
    icon: <WaveformIcon />,
    title: '16 kHz Polyphase Framing',
    description: 'Waveforms are polyphase-resampled to 16 kHz mono, peak RMS normalized, and segmented into exact 4.0s frames (64,000 samples).',
    visual: <WaveformVisual />,
    colSpan: 1,
    rowSpan: 1,
  },
  {
    id: 'stage-2',
    label: '02 · Dual-Domain Features',
    icon: <SpectrumIcon />,
    title: 'FrFT (α=0.5) & 4-Level WPD',
    description: 'Fractional Fourier rotation coupled with 4-level Wavelet Packet Decomposition exposes phase shifts and synthetic artifacts.',
    visual: <SpectrumVisual />,
    colSpan: 1,
    rowSpan: 1,
  },
  {
    id: 'stage-3',
    label: '03 · Neural Architecture',
    icon: <LatentIcon />,
    title: 'LCNN-2D Max-Feature-Map',
    description: 'Light CNN utilizes 2D Max-Feature-Map competitive activation units, projecting 2D spectrograms into a 128D latent manifold.',
    visual: <LatentVisual />,
    colSpan: 1,
    rowSpan: 1,
  },
  {
    id: 'stage-4',
    label: '04 · Decision Engine',
    icon: <DecisionIcon />,
    title: 'Candidate Threshold 0.4820',
    description: 'Calibrated candidate threshold classifies segments; multi-window recordings aggregate mean and peak spoof probabilities.',
    visual: <ThresholdVisual />,
    colSpan: 1,
    rowSpan: 1,
  },
];

/* =========================================================================
   Main Forensic Bento Export
   ========================================================================= */
export default function ForensicBento() {
  const [columns, setColumns] = useState(4);

  // Dynamic responsive columns based on viewport width
  useEffect(() => {
    const updateColumns = () => {
      const width = window.innerWidth;
      if (width < 620) {
        setColumns(1);
      } else if (width < 1080) {
        setColumns(2);
      } else {
        setColumns(4);
      }
    };

    updateColumns();
    window.addEventListener('resize', updateColumns);
    return () => window.removeEventListener('resize', updateColumns);
  }, []);

  // Forward parent pointer events so spotlight and glow track mouse smoothly
  useEffect(() => {
    if (typeof window !== 'undefined' && window.parent && window.parent !== window) {
      try {
        const forwardPointer = (e) => {
          const ifr = window.frameElement;
          if (!ifr) return;
          const rect = ifr.getBoundingClientRect();
          const clientX = e.clientX - rect.left;
          const clientY = e.clientY - rect.top;

          window.dispatchEvent(
            new PointerEvent(e.type, {
              clientX,
              clientY,
              bubbles: true,
            })
          );
        };

        window.parent.addEventListener('pointermove', forwardPointer, { passive: true });
        window.parent.addEventListener('pointerdown', forwardPointer, { passive: true });

        return () => {
          try {
            window.parent.removeEventListener('pointermove', forwardPointer);
            window.parent.removeEventListener('pointerdown', forwardPointer);
          } catch (e) {}
        };
      } catch (err) {}
    }
  }, []);

  return (
    <div
      style={{
        width: '100%',
        boxSizing: 'border-box',
        padding: '8px 4px',
      }}
    >
      <MagicBento
        items={FORENSIC_ITEMS}
        columns={columns}
        rowHeight={285}
        gap={14}
        radius={14}
        theme="dark"
        glowColor="#8B9CFF"
        enableBorderGlow={true}
        enableSpotlight={true}
        spotlightRadius={260}
        enableTilt={false}
        enableStars={false}
        textAutoHide={false}
        intro={false}
      />
    </div>
  );
}
