import React from 'react';
import ReactDOM from 'react-dom/client';
import ForensicBento from './ForensicBento';

function AutoResizeWrapper() {
  React.useEffect(() => {
    const syncHeight = () => {
      try {
        const ifr = window.frameElement;
        if (!ifr) return;
        const root = document.getElementById('root');
        if (!root) return;
        const h = root.offsetHeight || root.scrollHeight;
        if (h > 50) {
          ifr.style.height = `${h + 12}px`;
          const parent = ifr.closest('.element-container') || ifr.parentElement;
          if (parent) {
            parent.style.height = `${h + 12}px`;
          }
        }
      } catch (err) {
        // Cross-origin fallback
      }
    };

    syncHeight();
    const ro = new ResizeObserver(syncHeight);
    const root = document.getElementById('root');
    if (root) ro.observe(root);
    window.addEventListener('resize', syncHeight);

    const timer1 = setTimeout(syncHeight, 150);
    const timer2 = setTimeout(syncHeight, 600);

    return () => {
      ro.disconnect();
      window.removeEventListener('resize', syncHeight);
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, []);

  return <ForensicBento />;
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AutoResizeWrapper />
  </React.StrictMode>
);
