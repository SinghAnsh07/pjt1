import React, { useEffect, useState, useRef } from 'react';
import SpecularButton from './SpecularButton';
import './SpecularButton.css';

export default function AnalyzeSpecularButton() {
  const [disabled, setDisabled] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    // Check initial disabled state from parent configuration
    if (typeof window !== 'undefined') {
      if (window.__BUTTON_DISABLED__ !== undefined) {
        setDisabled(Boolean(window.__BUTTON_DISABLED__));
      }
    }

    // Forward pointer events from parent Streamlit window so specular shine follows mouse seamlessly
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
      } catch (err) {
        // Cross-origin restriction fallback
      }
    }
  }, []);

  const handleClick = () => {
    if (disabled) return;
    try {
      if (typeof window !== 'undefined' && window.parent && window.parent.document) {
        const parentDoc = window.parent.document;
        // Locate the target primary button in the Streamlit DOM
        const primaryBtns = Array.from(
          parentDoc.querySelectorAll('button[kind="primary"], [data-testid="baseButton-primary"]')
        );
        let target = primaryBtns.find(
          (b) =>
            b.innerText?.toLowerCase().includes('analyze') ||
            b.textContent?.toLowerCase().includes('analyze')
        );
        if (!target && primaryBtns.length > 0) {
          target = primaryBtns[0];
        }
        if (target) {
          target.click();
        } else {
          console.warn('Native Streamlit analyze button not found');
        }
      }
    } catch (err) {
      console.error('Failed to trigger Streamlit analysis:', err);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '100%',
        height: '100%',
        padding: '8px 12px',
        boxSizing: 'border-box',
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <SpecularButton
        size="md"
        radius={12}
        tint="#182238"
        tintOpacity={0.7}
        textColor="#F3F5F8"
        lineColor="#8B9CFF"
        baseColor="#324168"
        intensity={1.25}
        idleIntensity={0.38}
        shineSize={14}
        shineFade={32}
        thickness={1.3}
        glow={1.3}
        followMouse={true}
        proximity={280}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        aria-label="Analyze Audio Recording"
      >
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '0.98rem',
            fontWeight: 600,
            letterSpacing: '0.02em',
          }}
        >
          <svg
            width="17"
            height="17"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.4"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ color: '#8B9CFF' }}
          >
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
          <span>Analyze Audio</span>
        </span>
      </SpecularButton>
    </div>
  );
}
