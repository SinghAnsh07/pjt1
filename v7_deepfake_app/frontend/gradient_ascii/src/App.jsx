import React, { useEffect, useRef } from 'react';
import PatternWaves from './PatternWaves';
import './App.css';

export default function App() {
  const containerRef = useRef(null);

  useEffect(() => {
    // Automatically configure parent iframe if running inside Streamlit
    try {
      const ifr = window.frameElement;
      if (ifr) {
        ifr.style.position = 'fixed';
        ifr.style.top = '0';
        ifr.style.left = '0';
        ifr.style.width = '100vw';
        ifr.style.height = '100vh';
        ifr.style.zIndex = '-1';
        ifr.style.border = 'none';
        ifr.style.pointerEvents = 'none';
        ifr.title = 'pattern_waves_outer_bg';

        const parentElem = ifr.closest('.element-container') || ifr.parentElement;
        if (parentElem) {
          parentElem.style.position = 'fixed';
          parentElem.style.top = '0';
          parentElem.style.left = '0';
          parentElem.style.width = '0';
          parentElem.style.height = '0';
          parentElem.style.margin = '0';
          parentElem.style.padding = '0';
          parentElem.style.overflow = 'visible';
          parentElem.style.zIndex = '-1';
        }
      }
    } catch (e) {
      // Cross-origin restriction fallback
    }

    // Forward pointer events from parent Streamlit window so interactive ripples react
    if (typeof window !== 'undefined' && window.parent && window.parent !== window) {
      try {
        const forwardPointer = (e) => {
          window.dispatchEvent(
            new PointerEvent(e.type, {
              clientX: e.clientX,
              clientY: e.clientY,
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
      } catch (err) {
        // Cross-origin error fallback
      }
    }
  }, []);

  return (
    <div className="outer-pattern-waves-container" ref={containerRef}>
      {/* Official React Bits PatternWaves Component */}
      <PatternWaves
        preset="silk"
        color="#ffffff"
        backgroundColor="#000000"
        fade="edges"
        interactive={true}
        cursorSize={50}
        cursorStrength={0.6}
        style={{ width: '100%', height: '100%' }}
      />

      {/* Subtle dark vignette overlay for depth & text readability */}
      <div className="outer-dark-overlay" aria-hidden="true" />
    </div>
  );
}
