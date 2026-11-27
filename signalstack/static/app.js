// Small progressive enhancements: spinner on slow (LLM) forms, live job progress polling.
document.addEventListener('submit', (e) => {
  const f = e.target;
  if (f.dataset.slow !== undefined) {
    const s = document.getElementById('spinner');
    if (s) { s.querySelector('div').textContent = f.dataset.slow || 'Working…'; s.style.display = 'flex'; }
  }
});

async function pollJobs() {
  const els = document.querySelectorAll('[data-job]');
  if (!els.length) return;
  let anyRunning = false;
  for (const el of els) {
    try {
      const r = await fetch('/api/jobs/' + el.dataset.job);
      const j = await r.json();
      el.querySelector('.p').textContent = j.progress || '';
      el.querySelector('.s').textContent = j.status;
      el.querySelector('.s').className = 's chip ' + j.status;
      if (j.status === 'running') anyRunning = true;
    } catch (_) {}
  }
  if (anyRunning) setTimeout(pollJobs, 3000);
  else if (document.body.dataset.reloadOnDone !== undefined) setTimeout(() => location.reload(), 800);
}
document.addEventListener('DOMContentLoaded', pollJobs);
