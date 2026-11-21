// Dataset browsing: a folder of many studies, opened one study at a time.
//
// The folder never leaves the browser as a whole. Each file's StudyInstanceUID is read from its header, here, only to
// group the files; the UID is not shown and not sent (the server reads no identifiers). A study is then uploaded like
// any other, when the reviewer gets to it. What is sent with it is a study key: HMAC-SHA-256 of the UID (with the
// number and size of the study's files) under a random salt this installation made for itself, so the same folder
// opened again finds the studies already stored (with their corrections) instead of uploading them a second time.
//
// The salt keeps the UID itself out of the server's storage and makes the keys of two installations incomparable.
// It is not a secret from the clients of this server: every page is given it, so whoever can call the API and
// already knows a UID can compute its key and ask whether that study is stored (the API has no authentication and
// lists every study as it is).
//
// This module has no imports and touches no DOM, so its parser also runs under Node (tests/test_dataset_js.py).

const FIRST_BYTES = 16 * 1024;    // the study UID normally sits within the first few kilobytes
const MAX_BYTES = 256 * 1024;     // grown to this only when the header is longer; past it, group by folder
const READ_PARALLEL = 8;
const UNDEFINED = 0xffffffff;

// A study folder usually ships more than images: label sidecars, reports, thumbnails, OS junk. These never contain
// pixel data. Shared with the single-study upload in main.js.
export const NOT_AN_IMAGE = /\.(txt|json|csv|tsv|xml|ya?ml|md|pdf|docx?|xlsx?|zip|gz|log|ini|png|jpe?g|gif|bmp|tiff?|svg|webp|mp4|avi|db|sqlite)$/i;
export const DICOM_EXT = /\.(dcm|dicom|ima)$/i;
// A picture: an image file, but not one that can be measured (no pixel spacing, no view or side in a header).
export const PICTURE = /\.(png|jpe?g|gif|bmp|tiff?|webp)$/i;
// A path that names a view: "RMLO", "L_CC", "cc/", "MLO.dcm" ... (shared with the single-study upload).
export const VIEW_IN_PATH = /(^|[^a-z])(?:[lr][\W_]*)?(?:mlo|cc)([^a-z]|$)/i;
export const pathOf = (file) => file.webkitRelativePath || file.relativePath || file.name;
const folderOf = (path) => path.split('/').slice(0, -1).join('/');

/** Files that could be images: not empty, not hidden, not an obvious sidecar. */
export function usableFiles(files) {
  return files.filter((file) => {
    const base = pathOf(file).split('/').pop();
    return file.size > 0 && !base.startsWith('.') && !NOT_AN_IMAGE.test(base);
  });
}

// ── DICOM header walker ─────────────────────────────────────────────────────
// Value representations whose explicit-VR header has two reserved bytes and a 4-byte length (PS3.5 7.1.2).
const LONG_VR = new Set(['OB', 'OD', 'OF', 'OL', 'OV', 'OW', 'SQ', 'SV', 'UC', 'UN', 'UR', 'UT', 'UV']);
const IMPLICIT_LE = '1.2.840.10008.1.2';
const EXPLICIT_BE = '1.2.840.10008.1.2.2';
const DEFLATED = '1.2.840.10008.1.2.1.99';
const MEDIA_DIRECTORY = '1.2.840.10008.1.3.10';   // Media Storage Directory Storage: a DICOMDIR, never an image
const MORE = Symbol('more');   // thrown when the bytes end before the answer
const BAD = Symbol('bad');     // thrown when the bytes stop looking like DICOM

/**
 * The top-level StudyInstanceUID (0020,000D) in the first bytes of a DICOM file, and whether the file is an image.
 *
 * Walks the elements in order (preamble + "DICM", file meta group 0002 in explicit VR little endian, then the data
 * set in the transfer syntax's encoding), jumping over every value and over whole sequences, defined length or not,
 * so a Study Instance UID nested in a sequence (e.g. Related Series, 0008,1250) is never taken for the file's own.
 * The walk goes on past the UID to the pixel data element, to tell an image from a DICOM file that only describes
 * one (a DICOMDIR, a report).
 *
 * @param {Uint8Array} bytes  the start of the file
 * @returns {{dicom: boolean, uid: string, more: boolean, sopClass: string, absent: boolean, pixels: ?boolean,
 *            end: boolean}}
 *   `more`: the study UID lies past these bytes; `sopClass`: the SOP class UID, when read; `absent`: the data set
 *   has no top-level study UID; `pixels`: a pixel data element was reached (true), cannot follow (false) or was not
 *   reached within these bytes (null); `end`: the walk stopped exactly where these bytes end.
 */
export function parseStudyUid(bytes) {
  const size = bytes.length;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const need = (end) => { if (end > size) throw MORE; };
  const u16 = (at, little) => { need(at + 2); return view.getUint16(at, little); };
  const u32 = (at, little) => { need(at + 4); return view.getUint32(at, little); };
  const text = (at, length) => {
    need(at + length);
    return String.fromCharCode(...bytes.subarray(at, at + length)).replace(/[\0 ]+$/, '');
  };
  const looksLikeVr = (at) => {
    need(at + 2);
    return bytes[at] >= 0x41 && bytes[at] <= 0x5a && bytes[at + 1] >= 0x41 && bytes[at + 1] <= 0x5a;
  };

  function header(at, explicit, little) {
    const group = u16(at, little);
    const element = u16(at + 2, little);
    // Items and delimiters (FFFE,xxxx) carry no VR in any transfer syntax.
    if (group === 0xfffe || !explicit) return { group, element, vr: '', length: u32(at + 4, little), value: at + 8 };
    if (!looksLikeVr(at + 4)) throw BAD;
    const vr = String.fromCharCode(bytes[at + 4], bytes[at + 5]);
    if (LONG_VR.has(vr)) return { group, element, vr, length: u32(at + 8, little), value: at + 12 };
    return { group, element, vr, length: u16(at + 6, little), value: at + 8 };
  }

  // Offset just past the value of element `h`. An undefined length is a sequence (or UN, which then holds implicit
  // VR little endian): walk its items to the sequence delimiter.
  function skip(h, explicit, little) {
    if (h.length !== UNDEFINED) return h.value + h.length;
    const innerExplicit = h.vr === 'UN' ? false : explicit;
    const innerLittle = h.vr === 'UN' ? true : little;
    let at = h.value;
    for (;;) {
      const item = header(at, false, innerLittle);
      if (item.group !== 0xfffe) throw BAD;
      if (item.element === 0xe0dd) return item.value;              // sequence delimitation
      if (item.element !== 0xe000) throw BAD;
      if (item.length !== UNDEFINED) { at = item.value + item.length; continue; }
      at = item.value;                                              // undefined-length item: to its delimiter
      for (;;) {
        const inner = header(at, innerExplicit, innerLittle);
        if (inner.group === 0xfffe && inner.element === 0xe00d) { at = inner.value; break; }
        at = skip(inner, innerExplicit, innerLittle);
      }
    }
  }

  let dicom = false;
  let decided = false;           // the study UID question is answered
  const found = { uid: '', sopClass: '', absent: false, pixels: null, end: false };
  const result = (more) => ({ dicom, more, ...found });
  try {
    const magic = size >= 132 && bytes[128] === 0x44 && bytes[129] === 0x49 && bytes[130] === 0x43 && bytes[131] === 0x4d;
    let at = magic ? 132 : 0;
    if (!magic) {
      // No preamble: a bare data set, accepted only if it starts like one.
      const group = u16(0, true);
      if (group !== 0x0002 && group !== 0x0008) return result(false);
    }
    dicom = true;

    let syntax = '';
    while (u16(at, true) === 0x0002) {                              // file meta: explicit VR little endian
      const h = header(at, true, true);
      if (h.length === UNDEFINED) throw BAD;
      if (h.element === 0x0010) syntax = text(h.value, h.length);
      if (h.element === 0x0002) found.sopClass = text(h.value, h.length);   // Media Storage SOP Class UID
      at = h.value + h.length;
    }
    if (syntax === DEFLATED) return result(false);                  // cannot be walked: grouped by folder
    const little = syntax !== EXPLICIT_BE;
    const explicit = syntax ? syntax !== IMPLICIT_LE : looksLikeVr(at + 4);

    for (;;) {
      if (at === size) { found.end = true; return result(!decided); }   // nothing follows in these bytes
      const h = header(at, explicit, little);
      const tag = h.group * 0x10000 + h.element;                    // elements come in tag order
      if (tag === 0x00080016 && !found.sopClass && h.length !== UNDEFINED && h.length <= 64) {
        found.sopClass = text(h.value, h.length);
      }
      if (!decided && tag === 0x0020000d) {
        if (h.length !== UNDEFINED && h.length <= 64) {
          const uid = text(h.value, h.length);
          found.uid = /^[0-9]+(\.[0-9]+)*$/.test(uid) ? uid : '';
        }
        decided = true;
      } else if (!decided && tag > 0x0020000d) {
        found.absent = true;
        decided = true;
      }
      if (h.group === 0x7fe0 && (h.element === 0x0008 || h.element === 0x0009 || h.element === 0x0010)) {
        found.pixels = true;                                        // float, double or ordinary pixel data
        return result(false);
      }
      if (tag > 0x7fe00010) { found.pixels = false; return result(false); }
      at = skip(h, explicit, little);
    }
  } catch (error) {
    if (error === MORE) return result(!decided);
    if (error === BAD) return result(false);
    throw error;
  }
}

/**
 * StudyInstanceUID of one file ('' when it has none we can read), reading 16 KB first and 256 KB at most.
 *
 * `image` is false for a DICOM file that cannot be a view of a study: a media directory (DICOMDIR), a data set with
 * no top-level study UID, or a file read to its end without pixel data (a report, a presentation state). A file
 * whose header is too long to decide stays an image. A file that is not DICOM comes back as { dicom: false } unless
 * its extension says otherwise.
 */
export async function readStudyUid(file) {
  let length = Math.min(file.size, FIRST_BYTES);
  for (;;) {
    const head = parseStudyUid(new Uint8Array(await file.slice(0, length).arrayBuffer()));
    const whole = length >= file.size;
    // Read on while the study UID is still ahead, or while a small file has not shown whether it holds pixel data.
    const open = head.more || (head.dicom && head.pixels === null && !head.end && file.size <= MAX_BYTES);
    if (!open || whole || length >= MAX_BYTES) {
      const complete = whole && head.end;                           // walked to the very end of the file
      const noStudy = head.absent || (complete && head.more);
      const noPixels = head.pixels === false || (complete && head.pixels === null);
      return {
        dicom: head.dicom || DICOM_EXT.test(file.name),
        uid: head.more ? '' : head.uid,
        image: !(head.dicom && (head.sopClass === MEDIA_DIRECTORY || noStudy || noPixels)),
      };
    }
    length = Math.min(file.size, MAX_BYTES);
  }
}

// ── study key ───────────────────────────────────────────────────────────────
// SHA-256 and HMAC written out: crypto.subtle exists only on https and localhost, and the app is also served over
// plain http inside a network.
const K256 = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]);

/** SHA-256 of a Uint8Array, as a Uint8Array of 32 bytes. */
export function sha256(bytes) {
  const length = bytes.length;
  const padded = new Uint8Array((((length + 8) >> 6) + 1) << 6);
  padded.set(bytes);
  padded[length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 8, Math.floor(length / 0x20000000));
  view.setUint32(padded.length - 4, (length << 3) >>> 0);
  const h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const w = new Uint32Array(64);
  const rotr = (x, n) => (x >>> n) | (x << (32 - n));
  for (let offset = 0; offset < padded.length; offset += 64) {
    for (let i = 0; i < 16; i += 1) w[i] = view.getUint32(offset + 4 * i);
    for (let i = 16; i < 64; i += 1) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, k] = h;
    for (let i = 0; i < 64; i += 1) {
      const t1 = (k + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i]) >>> 0;
      const t2 = ((rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      k = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
  }
  const out = new Uint8Array(32);
  const outView = new DataView(out.buffer);
  h.forEach((word, i) => outView.setUint32(4 * i, word));
  return out;
}

/** HMAC-SHA-256 (RFC 2104) of `message` under `key`, both Uint8Array. */
export function hmacSha256(key, message) {
  const block = new Uint8Array(64);
  block.set(key.length > 64 ? sha256(key) : key);
  const inner = new Uint8Array(64 + message.length);
  const outer = new Uint8Array(64 + 32);
  for (let i = 0; i < 64; i += 1) { inner[i] = block[i] ^ 0x36; outer[i] = block[i] ^ 0x5c; }
  inner.set(message, 64);
  outer.set(sha256(inner), 64);
  return sha256(outer);
}

/**
 * The key a study is stored under, in hex: HMAC-SHA-256, under the installation's salt (64 hex characters, as the
 * server hands it out), of the StudyInstanceUID with the number and the total size of the study's files. '' without
 * a salt or a UID. The same files give the same key; a study whose files changed has another key and is analysed
 * afresh instead of being shown from a record that no longer matches it.
 */
export function studyKey(saltHex, uid, files = []) {
  if (!/^[0-9a-f]{64}$/.test(saltHex || '') || !uid) return '';
  const salt = new Uint8Array(32);
  for (let i = 0; i < 32; i += 1) salt[i] = parseInt(saltHex.slice(2 * i, 2 * i + 2), 16);
  const bytes = files.reduce((sum, file) => sum + (file.size || 0), 0);
  const digest = hmacSha256(salt, new TextEncoder().encode(`${uid}\n${files.length}\n${bytes}`));
  return [...digest].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

/**
 * The files of one study the server will take: all of them, or, past its limit per study, those whose path names a
 * view first and then the rest, each in path order. `left` is how many stay behind.
 */
export function studyUpload(files, limit) {
  if (files.length <= limit) return { files, left: 0 };
  const named = files.filter((file) => VIEW_IN_PATH.test(pathOf(file)));
  const others = files.filter((file) => !VIEW_IN_PATH.test(pathOf(file)));
  return { files: [...named, ...others].slice(0, limit), left: files.length - limit };
}

// ── grouping ────────────────────────────────────────────────────────────────
async function mapLimited(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const index = next;
      next += 1;
      out[index] = await fn(items[index]);
    }
  }));
  return out;
}

const natural = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

/**
 * Split the files of a folder into studies, in folder order.
 *
 * Files are grouped by StudyInstanceUID; a file whose UID could not be read (a header too long, a deflated data
 * set) is grouped with its own folder. A DICOM file that is not an image of a study (a DICOMDIR, a file with no
 * study UID or no pixel data) is skipped, like a file that is not DICOM. Each study is named after the folder that
 * holds its files (the deepest folder they share), never after the UID; studies that share a folder also carry the
 * name of their first file, so no two entries read the same.
 *
 * @returns {{ name: string, studies: {name: string, files: File[], uid: string}[], skipped: number }}
 *   `uid` ('' for a study grouped by folder) is for making the study key only: it is never shown and never sent.
 */
export async function buildDataset(files, onProgress = () => {}) {
  const candidates = usableFiles(files);
  let done = 0;
  const heads = await mapLimited(candidates, READ_PARALLEL, async (file) => {
    let head = { dicom: false, uid: '', image: false };
    try { head = await readStudyUid(file); } catch (error) { /* unreadable: skipped below */ }
    done += 1;
    onProgress(done, candidates.length);
    return head;
  });

  const groups = new Map();
  let skipped = files.length - candidates.length;
  candidates.forEach((file, index) => {
    const { dicom, uid, image } = heads[index];
    if (!dicom || !image) { skipped += 1; return; }
    const key = uid ? `uid:${uid}` : `dir:${folderOf(pathOf(file))}`;
    if (!groups.has(key)) groups.set(key, { uid, files: [] });
    groups.get(key).files.push(file);
  });

  const first = pathOf(files[0] || { name: '' });
  const root = first.includes('/') ? first.split('/')[0] : ''; // a folder pick; loose files have no root
  const studies = [...groups.values()].map(({ uid, files: group }) => {
    group.sort((a, b) => natural.compare(pathOf(a), pathOf(b)));
    const folders = group.map((file) => folderOf(pathOf(file)).split('/'));
    const shared = [];
    for (let depth = 0; folders.every((parts) => depth < parts.length && parts[depth] === folders[0][depth]); depth += 1) {
      shared.push(folders[0][depth]);
    }
    const folder = shared.length > 1 ? shared.slice(1).join('/') : '';
    const first = pathOf(group[0]);
    return {
      name: folder || first.split('/').pop(),
      inFolder: first.split('/').slice(shared.length).join('/'),    // the first file, as seen from that folder
      files: group,
      uid,
      sortKey: first,
    };
  });
  studies.sort((a, b) => natural.compare(a.sortKey, b.sortKey));
  // Two studies in one folder would read the same in the list: tell them apart by their first file.
  const uses = new Map();
  for (const study of studies) uses.set(study.name, (uses.get(study.name) || 0) + 1);
  for (const study of studies) {
    if (uses.get(study.name) > 1 && study.inFolder && study.inFolder !== study.name) {
      study.name = `${study.name} · ${study.inFolder}`;
    }
  }
  return { name: root, studies: studies.map(({ name, files: list, uid }) => ({ name, files: list, uid })), skipped };
}
