/* Clean, zero-error 4-step interactive lifecycle controller for [data-ss] */
(() => {
  const root = document.querySelector('[data-ss]');
  if (!root) return;

  const tabs = Array.from(root.querySelectorAll('.ss-tab[data-step]'));
  const panes = Array.from(root.querySelectorAll('.ss-pane[data-pane]'));
  if (!tabs.length || !panes.length) return;

  let current = 0;
  let timer = null;
  let typeTimers = [];

  function clearTyping() {
    typeTimers.forEach((id) => clearTimeout(id));
    typeTimers = [];
  }

  function animatePane(pane) {
    clearTyping();
    const ty = pane.querySelector('.ty[data-text]');
    if (ty) {
      const full = ty.getAttribute('data-text') || '';
      ty.textContent = '';
      for (let i = 1; i <= full.length; i++) {
        typeTimers.push(
          setTimeout(() => {
            ty.textContent = full.slice(0, i);
          }, Math.min(i * 8, 520))
        );
      }
    }
    const lines = Array.from(pane.querySelectorAll('.tl--out'));
    lines.forEach((line, idx) => {
      line.style.opacity = '0';
      line.style.transition = 'opacity 0.22s ease';
      const delay = Number(line.getAttribute('data-t')) || (idx + 1) * 320;
      typeTimers.push(
        setTimeout(() => {
          line.style.opacity = '1';
        }, Math.min(delay, 1400))
      );
    });
  }

  function activate(stepIndex) {
    current = (stepIndex + tabs.length) % tabs.length;
    tabs.forEach((tab, idx) => {
      const active = idx === current;
      tab.classList.toggle('is-active', active);
      tab.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    panes.forEach((pane, idx) => {
      const active = idx === current;
      pane.classList.toggle('is-active', active);
      if (active) animatePane(pane);
    });
  }

  function startAuto() {
    if (timer) clearInterval(timer);
    timer = setInterval(() => {
      activate(current + 1);
    }, 6000);
  }

  tabs.forEach((tab, idx) => {
    tab.addEventListener('click', () => {
      activate(idx);
      startAuto();
    });
  });

  activate(0);
  startAuto();
})();
