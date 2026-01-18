(() => {
  const providerSelect = document.getElementById('provider-select');
  const apiKeyVisible = document.getElementById('api-key-visible');
  const providerInput = document.getElementById('provider-input');
  const apiKeyInput = document.getElementById('api-key-input');
  const appInstance = document.body.dataset.appInstance;

  const STORAGE_PROVIDER = 'prompt_alchemy_provider';
  const STORAGE_SESSION = 'prompt_alchemy_session';
  const providerLabels = {
    gemini: 'Gemini',
    openai: 'OpenAI',
    grok: 'Grok'
  };
  const keyCache = {};

  const clearLocalState = () => {
    localStorage.removeItem('gemini_api_key');
    localStorage.removeItem('openai_api_key');
    localStorage.removeItem('grok_api_key');
    localStorage.removeItem(STORAGE_PROVIDER);
  };

  const ensureFreshLocalStorage = () => {
    const storedSession = localStorage.getItem(STORAGE_SESSION);
    if (storedSession !== appInstance) {
      clearLocalState();
      localStorage.setItem(STORAGE_SESSION, appInstance);
    }
  };

  const getStorageKey = (provider) => `${provider}_api_key`;

  const maskKey = (key) => {
    const length = Math.max(6, Math.min(key.length, 32));
    return '*'.repeat(length);
  };

  const applyMaskedKey = (provider, key) => {
    if (!key) return;
    keyCache[provider] = key;
    apiKeyInput.value = key;
    apiKeyVisible.value = maskKey(key);
    apiKeyVisible.dataset.masked = 'true';
  };

  const hydrateFromEnv = (provider) => {
    fetch(`/api/keys?provider=${encodeURIComponent(provider)}`)
      .then((res) => res.json())
      .then((data) => {
        if (!data || !data.api_key) return;
        applyMaskedKey(provider, data.api_key);
        localStorage.setItem(getStorageKey(provider), data.api_key);
      })
      .catch(() => {});
  };

  const syncProvider = (provider) => {
    providerInput.value = provider;
    localStorage.setItem(STORAGE_PROVIDER, provider);
    const storedKey = localStorage.getItem(getStorageKey(provider)) || '';
    apiKeyVisible.placeholder = `Paste ${providerLabels[provider]} API Key`;
    if (storedKey) {
      applyMaskedKey(provider, storedKey);
    } else {
      apiKeyVisible.value = '';
      apiKeyInput.value = '';
      apiKeyVisible.dataset.masked = 'false';
      hydrateFromEnv(provider);
    }
  };

  let keyUpdateTimer;
  const scheduleKeySync = (provider, key) => {
    if (keyUpdateTimer) {
      clearTimeout(keyUpdateTimer);
    }
    keyUpdateTimer = setTimeout(() => {
      const formData = new FormData();
      formData.append('provider', provider);
      formData.append('api_key', key);
      fetch('/api/keys', { method: 'POST', body: formData }).catch(() => {});
    }, 400);
  };

  ensureFreshLocalStorage();

  const initialProvider = localStorage.getItem(STORAGE_PROVIDER) || 'gemini';
  providerSelect.value = initialProvider;
  syncProvider(initialProvider);

  providerSelect.addEventListener('change', (event) => {
    syncProvider(event.target.value);
  });

  apiKeyVisible.addEventListener('focus', () => {
    if (apiKeyVisible.dataset.masked === 'true') {
      apiKeyVisible.value = '';
      apiKeyVisible.dataset.masked = 'false';
    }
  });

  apiKeyVisible.addEventListener('blur', () => {
    const provider = providerInput.value;
    const key = keyCache[provider];
    if (!apiKeyVisible.value && key) {
      applyMaskedKey(provider, key);
    }
  });

  apiKeyVisible.addEventListener('input', (event) => {
    const key = event.target.value;
    const provider = providerInput.value;
    if (!key) {
      apiKeyInput.value = '';
      keyCache[provider] = '';
      localStorage.removeItem(getStorageKey(provider));
      return;
    }
    keyCache[provider] = key;
    apiKeyInput.value = key;
    localStorage.setItem(getStorageKey(provider), key);
    scheduleKeySync(provider, key);
  });

  const focusPills = document.querySelectorAll('.focus-pill');
  focusPills.forEach((pill) => {
    const checkbox = pill.querySelector('input[type="checkbox"]');
    if (!checkbox) return;
    checkbox.addEventListener('change', () => {
      pill.classList.toggle('active', checkbox.checked);
    });
  });

  const familyButtons = document.querySelectorAll('.family-button');
  const familyInput = document.getElementById('model-family-input');

  const setActiveFamily = (familyId) => {
    familyInput.value = familyId;
    familyButtons.forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.familyId === familyId);
    });
    if (window.htmx) {
      window.htmx.ajax('GET', `/partials/checkpoints?family_id=${familyId}`, '#checkpoint-container');
    }
  };

  familyButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      setActiveFamily(btn.dataset.familyId);
    });
  });

  const creativityInput = document.getElementById('creativity-input');
  const creativityValue = document.getElementById('creativity-value');
  const creativityBar = document.getElementById('creativity-bar');
  const creativityThumb = document.getElementById('creativity-thumb');

  const updateCreativity = () => {
    const value = parseFloat(creativityInput.value || '0');
    const pct = Math.round(value * 100);
    creativityValue.textContent = `${pct}%`;
    creativityBar.style.width = `${pct}%`;
    creativityThumb.style.left = `calc(${pct}% - 8px)`;
  };
  creativityInput.addEventListener('input', updateCreativity);
  updateCreativity();

  const dropzone = document.getElementById('dropzone');
  const imageInput = document.getElementById('image-input');
  const imagePreview = document.getElementById('image-preview');
  const generateButton = document.getElementById('generate-button');

  let imageItems = [];

  const rebuildInputFiles = () => {
    const dataTransfer = new DataTransfer();
    imageItems.forEach((item) => dataTransfer.items.add(item.file));
    imageInput.files = dataTransfer.files;
  };

  const updateGenerateState = () => {
    const disabled = imageItems.length === 0;
    generateButton.disabled = disabled;
    generateButton.classList.toggle('opacity-60', disabled);
    generateButton.classList.toggle('pointer-events-none', disabled);
    generateButton.setAttribute('aria-disabled', disabled ? 'true' : 'false');
  };

  const renderPreviews = () => {
    imagePreview.innerHTML = '';
    if (imageItems.length === 0) {
      imagePreview.classList.add('hidden');
    } else {
      imagePreview.classList.remove('hidden');
    }

    imageItems.forEach((item) => {
      const card = document.createElement('div');
      card.className = 'relative group aspect-[3/4] rounded-lg overflow-hidden border border-gray-700/50 shadow-md';
      card.innerHTML = `
        <img src="${item.url}" alt="Reference" class="w-full h-full object-cover" />
        <div class="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
        <button type="button" class="absolute top-2 right-2 p-1.5 bg-red-500/80 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600 backdrop-blur-md" data-remove-id="${item.id}">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-3.5 h-3.5">
            <path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12" />
          </svg>
        </button>
      `;
      imagePreview.appendChild(card);
    });
  };

  const addFiles = (files) => {
    Array.from(files).forEach((file) => {
      if (!file.type.startsWith('image/')) return;
      if (imageItems.length >= 6) return;
      const id = Math.random().toString(36).slice(2, 9);
      imageItems.push({ id, file, url: URL.createObjectURL(file) });
    });
    rebuildInputFiles();
    renderPreviews();
    updateGenerateState();
  };

  const removeFileById = (id) => {
    const target = imageItems.find((item) => item.id === id);
    if (target) {
      URL.revokeObjectURL(target.url);
    }
    imageItems = imageItems.filter((item) => item.id !== id);
    rebuildInputFiles();
    renderPreviews();
    updateGenerateState();
  };

  dropzone.addEventListener('click', () => imageInput.click());
  dropzone.addEventListener('dragover', (event) => {
    event.preventDefault();
  });
  dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    addFiles(event.dataTransfer.files);
  });
  imageInput.addEventListener('change', (event) => {
    addFiles(event.target.files);
    imageInput.value = '';
  });

  imagePreview.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-remove-id]');
    if (!button) return;
    removeFileById(button.dataset.removeId);
  });

  updateGenerateState();

  const handleCopy = (target) => {
    const text = document.getElementById(`${target}-text`);
    if (!text) return;
    navigator.clipboard.writeText(text.textContent || '').then(() => {
      const buttons = document.querySelectorAll(`.copy-button[data-copy-target="${target}"]`);
      buttons.forEach((btn) => {
        const label = btn.querySelector('.copy-label');
        if (!label) return;
        label.textContent = 'Copied!';
        setTimeout(() => {
          label.textContent = 'Copy';
        }, 2000);
      });
    });
  };

  document.addEventListener('click', (event) => {
    const button = event.target.closest('.copy-button');
    if (!button) return;
    handleCopy(button.dataset.copyTarget);
  });

  document.addEventListener('htmx:afterSwap', () => {
    updateGenerateState();
  });

  const form = document.getElementById('prompt-form');
  const buttonLabel = generateButton.querySelector('span');
  const buttonIcon = generateButton.querySelector('svg');
  let loading = false;

  const setButtonLoading = (isLoading) => {
    if (loading === isLoading) return;
    loading = isLoading;
    if (isLoading) {
      buttonIcon.classList.add('hidden');
      buttonLabel.textContent = 'Synthesizing Prompt...';
    } else {
      buttonIcon.classList.remove('hidden');
      buttonLabel.textContent = 'Generate Perfect Prompt';
    }
  };

  document.addEventListener('htmx:beforeRequest', (event) => {
    if (event.target === form) {
      setButtonLoading(true);
    }
  });

  document.addEventListener('htmx:afterRequest', (event) => {
    if (event.target === form) {
      setButtonLoading(false);
    }
  });
})();
