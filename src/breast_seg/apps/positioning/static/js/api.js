// Thin HTTP layer. Every call resolves to parsed JSON or throws an Error whose message is already user-facing —
// callers never have to work out whether a failure was network, HTTP or payload shaped.
//
// Every request carries a deadline. Without one, a connection that dies mid-flight (a dropped tunnel, a sleeping
// VPN) leaves the promise pending forever and the UI sits on a spinner with no way out. Every pending request also
// drives the thin progress line at the top of the page.

import { t } from './i18n.js';

const DEFAULT_TIMEOUT_MS = 30_000;
const UPLOAD_TIMEOUT_MS = 10 * 60_000;

let pending = 0;
function busy(delta) {
  pending = Math.max(0, pending + delta);
  const line = document.getElementById('top-progress');
  if (line) line.hidden = pending === 0;
}

async function request(url, { timeout = DEFAULT_TIMEOUT_MS, signal, quiet = false, ...options } = {}) {
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener('abort', onAbort, { once: true });
  }
  const timer = setTimeout(() => controller.abort('timeout'), timeout);

  let response;
  if (!quiet) busy(+1);
  try {
    response = await fetch(url, { ...options, signal: controller.signal });
  } catch (cause) {
    if (signal && signal.aborted) throw new Error(t('cancelled'));
    if (controller.signal.aborted) throw new Error(t('err_timeout'));
    throw new Error(t('err_network'));
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', onAbort);
    if (!quiet) busy(-1);
  }

  const isJson = (response.headers.get('content-type') || '').includes('application/json');
  const payload = isJson ? await response.json().catch(() => null) : null;
  if (!response.ok) throw new Error(errorMessage(payload, response.status));
  return payload;
}

// FastAPI sends a string `detail` for the app's own errors and a list of field errors for request-validation
// failures; both need to end up readable.
function errorMessage(payload, status) {
  const detail = payload && payload.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail) && detail.length) {
    const first = detail[0];
    if (first && typeof first.msg === 'string') return t('err_invalid', { msg: first.msg });
  }
  return t('err_http', { status });
}

/**
 * Multipart upload with progress events:
 *   onProgress({ phase: 'upload', loaded, total })  while the bytes go out
 *   onProgress({ phase: 'server' })                 once they are all sent (DICOM decoding + inference)
 * Resolves to the study record; `signal` aborts the request.
 */
function createStudy(files, { onProgress = () => {}, signal } = {}) {
  return new Promise((resolve, reject) => {
    const body = new FormData();
    // Send the path the file came in under, not just its basename: in a folder upload the view lives in the
    // directory name, and the file inside is usually a scanner serial number that says nothing.
    for (const file of files) {
      body.append('files', file, file.webkitRelativePath || file.relativePath || file.name);
    }
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/studies');
    xhr.responseType = 'json';
    xhr.timeout = UPLOAD_TIMEOUT_MS;
    let serverPhase = false;
    const toServer = () => { if (!serverPhase) { serverPhase = true; onProgress({ phase: 'server' }); } };
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress({ phase: 'upload', loaded: event.loaded, total: event.total });
        if (event.loaded >= event.total) toServer();
      }
    };
    xhr.upload.onload = toServer;
    const done = () => { busy(-1); if (signal) signal.removeEventListener('abort', abort); };
    xhr.onload = () => {
      done();
      const payload = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
      else reject(new Error(errorMessage(payload, xhr.status)));
    };
    xhr.onerror = () => { done(); reject(new Error(t('err_network'))); };
    xhr.ontimeout = () => { done(); reject(new Error(t('err_timeout'))); };
    xhr.onabort = () => { done(); reject(new Error(t('cancelled'))); };
    const abort = () => xhr.abort();
    if (signal) {
      if (signal.aborted) { reject(new Error(t('cancelled'))); return; }
      signal.addEventListener('abort', abort, { once: true });
    }
    busy(+1);
    xhr.send(body);
  });
}

const id = (value) => encodeURIComponent(value);

export const api = {
  health: () => request('/api/health', { timeout: 8000, quiet: true }),
  listStudies: () => request('/api/studies'),
  getStudy: (studyId) => request(`/api/studies/${id(studyId)}`),
  getReport: (studyId, lang) => request(`/api/studies/${id(studyId)}/report?lang=${id(lang)}`),
  getLabels: (studyId) => request(`/api/studies/${id(studyId)}/labels`),
  deleteAll: () => request('/api/studies', { method: 'DELETE' }),
  deleteStudy: (studyId) => request(`/api/studies/${id(studyId)}`, { method: 'DELETE' }),

  putLabels: (studyId, payload) => request(`/api/studies/${id(studyId)}/labels`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    timeout: 60_000,
  }),

  putRules: (studyId, rules) => request(`/api/studies/${id(studyId)}/rules`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(rules),
  }),

  createStudy,

  imageUrl: (studyId, viewKey) => `/api/studies/${id(studyId)}/image/${id(viewKey)}.png`,
};
