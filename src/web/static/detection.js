const button = document.getElementById('detection-toggle');
const status = document.getElementById('detection-status');
let enabled = false;
let busy = false;

function render(data) {
  enabled = data.enabled;
  button.setAttribute('aria-pressed', String(enabled));
  button.textContent = enabled ? 'Выключить детекцию людей' : 'Включить детекцию людей';
  const labels = {
    disabled: 'Детекция выключена',
    loading: 'Загрузка модели…',
    running: 'Детекция включена',
  };
  status.textContent = data.error || labels[data.state] || 'Состояние неизвестно';
  document.querySelectorAll('[data-channel]').forEach((element) => {
    const count = data.cameras[element.dataset.channel];
    element.textContent = `Людей: ${count ?? '—'}`;
  });
}

async function update(value) {
  if (busy) return;
  busy = true;
  button.disabled = true;
  try {
    const options = value === undefined ? {} : {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({enabled: value}),
    };
    const response = await fetch('/api/detection', {
      ...options, cache: 'no-store', signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) throw new Error('Request failed');
    render(await response.json());
  } catch {
    status.textContent = 'Не удалось получить состояние детекции. Повторите попытку.';
  } finally {
    busy = false;
    button.disabled = false;
  }
}

button.addEventListener('click', () => update(!enabled));
async function poll() {
  await update();
  window.setTimeout(poll, 1500);
}
poll();
