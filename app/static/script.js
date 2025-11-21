const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const colorizeBtn = document.getElementById('colorizeBtn');
const status = document.getElementById('status');
const preview = document.getElementById('preview');

let selectedFile = null;
let previewUrl = null;

const initialHint = 'Drop an image or video to colorize';

const updateHint = (file) => {
  if (file) {
    dropzone.querySelector('span').innerText = file.name;
  } else {
    dropzone.querySelector('span').innerText = initialHint;
  }
};

updateHint(null);

const setSelectedFile = (files) => {
  if (!files || !files.length) {
    selectedFile = null;
  } else {
    selectedFile = files[0];
  }
  updateHint(selectedFile);
};

dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropzone.classList.add('dropzone--active');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dropzone--active');
});

dropzone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropzone.classList.remove('dropzone--active');
  setSelectedFile(event.dataTransfer.files);
});

fileInput.addEventListener('change', (event) => setSelectedFile(event.target.files));

const resetPreview = () => {
  preview.innerHTML = `<div class="preview-placeholder">
    <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
      <path d="M32 4C18.745 4 8 14.745 8 28s10.745 24 24 24 24-10.745 24-24S45.255 4 32 4zm0 44c-11.598 0-21-9.402-21-21S20.402 6 32 6s21 9.402 21 21-9.402 21-21 21z"/>
      <path d="M32 18l-8 12h16zM18 42h28v4H18z"/>
    </svg>
    Waiting for your masterpiece...
  </div>`;
};

resetPreview();

const displayStatus = (message, tone = 'info') => {
  status.textContent = message;
  status.style.color = tone === 'error' ? '#ff8da1' : tone === 'success' ? '#3edbd1' : 'rgba(244, 246, 251, 0.75)';
};

const createPreview = (blob, mimeType) => {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(blob);
  preview.innerHTML = '';
  if (mimeType.startsWith('video')) {
    const el = document.createElement('video');
    el.controls = true;
    el.autoplay = false;
    el.src = previewUrl;
    preview.appendChild(el);
  } else {
    const el = document.createElement('img');
    el.src = previewUrl;
    el.alt = 'Colorized result';
    preview.appendChild(el);
  }
};

colorizeBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    displayStatus('Select an image or video first.', 'error');
    return;
  }

  const form = new FormData();
  form.append('file', selectedFile);

  colorizeBtn.disabled = true;
  displayStatus('Colorizing your file—this might take a few seconds...');

  try {
    const response = await fetch('/colorize', {
      method: 'POST',
      body: form,
    });

    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || 'Colorization failed');
    }

    const mimeType = response.headers.get('content-type') || 'application/octet-stream';
    const blob = await response.blob();
    createPreview(blob, mimeType);
    displayStatus('Done! See your colorized version below.', 'success');
  } catch (error) {
    console.error(error);
    displayStatus(error.message || 'Something went wrong', 'error');
  } finally {
    colorizeBtn.disabled = false;
  }
});
