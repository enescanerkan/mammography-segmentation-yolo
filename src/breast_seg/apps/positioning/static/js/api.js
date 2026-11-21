// Thin HTTP layer. Every call resolves to parsed JSON or throws an Error whose message is already user-facing —
// callers never have to work out whether a failure was network, HTTP or payload shaped.
//
// Every request carries a deadline. Without one, a connection that dies mid-flight (a dropped tunnel, a sleeping
// VPN) leaves the promise pending forever and the UI sits on a spinner with no way out. Every pending request also
// drives the thin progress line at the top of the page.

import { pathOf } from './dataset.js';
import { serverError, t } from './i18n.js';

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
  if (!response.ok) throw httpError(payload, response.status);
  return payload;
}

/** A user-facing Error that also carries the HTTP status (callers tell "gone" from "failed"). */
function httpError(payload, status) {
  const error = new Error(errorMessage(payload, status));
  error.status = status;
  return error;
}

// FastAPI sends a string `detail` for the app's own errors (with a `code` the page translates by, when a reader can
// meet it) and a list of field errors for request-validation failures; both need to end up readable.
function errorMessage(payload, status) {
  const detail = payload && payload.detail;
  if (typeof detail === 'string') return serverError(payload);
  if (Array.isArray(detail) && detail.length) {
    const first = detail[0];
    if (first && typeof first.msg === 'string') return t('err_invalid', { msg: first.msg });
  }
  return t('err_http', { status });
}

/** Up to three file names, and how many more there are: what a message names. */
export function fileList(files) {
  const names = files.slice(0, 3).map(pathOf);
  return files.length > 3 ? `${names.join(', ')} ${t('more_files', { n: files.length - 3 })}` : names.join(', ');
}

/**
 * The files the browser cannot read any more. A file that was moved, deleted or changed after it was picked fails the
 * request before a byte reaches the server, and the browser reports that exactly like a connection that failed; a
 * one-byte read of each file tells the two apart.
 */
export async function unreadableFiles(files) {
  const failed = await Promise.all(files.map((file) => file.slice(0, 1).arrayBuffer().then(() => false, () => true)));
  return files.filter((_, index) => failed[index]);
}

/**
 * Multipart upload with progress events:
 *   onProgress({ phase: 'upload', loaded, total })  while the bytes go out
 *   onProgress({ phase: 'server' })                 once they are all sent (DICOM decoding + inference)
 * Resolves to the study record; `signal` aborts the request. A dataset upload also says three things about the
 * study: `studyKey` (dataset.js), stored with it so the same study is found again when its folder is opened another
 * time; `unopened`, that it is analysed ahead of the reviewer (it is not listed among the recent studies until
 * `markOpened`); and `leftOut`, how many of its files were not sent because of the per-upload limit.
 */
function createStudy(files, { onProgress = () => {}, signal, studyKey = '', unopened = false, leftOut = 0 } = {}) {
  return new Promise((resolve, reject) => {
    const body = new FormData();
    // Send the path the file came in under, not just its basename: in a folder upload the view lives in the
    // directory name, and the file inside is usually a scanner serial number that says nothing.
    for (const file of files) {
      body.append('files', file, file.webkitRelativePath || file.relativePath || file.name);
    }
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/studies');
    if (studyKey) xhr.setRequestHeader('X-Study-Key', studyKey);
    if (unopened) xhr.setRequestHeader('X-Study-Unopened', '1');
    if (leftOut > 0) xhr.setRequestHeader('X-Study-Left-Out', String(leftOut));
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
      else reject(httpError(payload, xhr.status));
    };
    xhr.onerror = () => {
      done();
      // Say so when it is a file the browser could not read, not the server it could not reach.
      unreadableFiles(files).then((bad) => reject(new Error(
        bad.length ? t('err_file_unreadable', { file: fileList(bad) }) : t('err_network'))));
    };
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
  // The studies somebody opened, newest first, with `total` (everything stored) and `unopened` (analysed ahead,
  // not opened yet, so not listed). `limit` caps the rows (0: only the counts, no record is read).
  listStudies: ({ limit } = {}) => request(limit === undefined ? '/api/studies' : `/api/studies?limit=${Number(limit)}`),
  // Which of these study keys the server holds: { found: { key: study id } }.
  lookupStudies: (keys, { signal } = {}) => request('/api/studies/lookup', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ keys }),
    signal,
  }),
  getStudy: (studyId, { signal } = {}) => request(`/api/studies/${id(studyId)}`, { signal }),
  getReport: (studyId, lang) => request(`/api/studies/${id(studyId)}/report?lang=${id(lang)}`),
  getLabels: (studyId) => request(`/api/studies/${id(studyId)}/labels`),
  deleteAll: (expected) => request(`/api/studies?expected=${Number(expected)}`, { method: 'DELETE' }),
  deleteStudy: (studyId) => request(`/api/studies/${id(studyId)}`, { method: 'DELETE' }),
  // The study is on screen for review: from now on it is listed among the recent studies.
  markOpened: (studyId) => request(`/api/studies/${id(studyId)}/opened`, { method: 'POST', quiet: true }),

  putLabels: (studyId, payload) => request(`/api/studies/${id(studyId)}/labels`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    timeout: 60_000,
  }),

  // The editor's "Smooth / clean labels": the view's contours cleaned by the server's rules (nothing is stored).
  cleanView: (studyId, viewKey, polygons) => request(`/api/studies/${id(studyId)}/clean`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ view_key: viewKey, polygons }),
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
