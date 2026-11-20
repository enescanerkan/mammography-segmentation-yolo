// Dataset browsing: a folder of many studies, opened one study at a time.
//
// The folder never leaves the browser as a whole. Each file's StudyInstanceUID is read from its header, here, only to
// group the files; the UID is not shown and not sent (the server reads no identifiers). A study is then uploaded like
// any other, when the reviewer gets to it.
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
const MORE = Symbol('more');   // thrown when the bytes end before the answer
const BAD = Symbol('bad');     // thrown when the bytes stop looking like DICOM

/**
 * The top-level StudyInstanceUID (0020,000D) in the first bytes of a DICOM file.
 *
 * Walks the elements in order (preamble + "DICM", file meta group 0002 in explicit VR little endian, then the data
 * set in the transfer syntax's encoding), jumping over every value and over whole sequences, defined length or not,
 * so a Study Instance UID nested in a sequence (e.g. Related Series, 0008,1250) is never taken for the file's own.
 *
 * @param {Uint8Array} bytes  the start of the file
 * @returns {{dicom: boolean, uid: string, more: boolean}}  `more`: the answer lies past these bytes
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
  try {
    const magic = size >= 132 && bytes[128] === 0x44 && bytes[129] === 0x49 && bytes[130] === 0x43 && bytes[131] === 0x4d;
    let at = magic ? 132 : 0;
    if (!magic) {
      // No preamble: a bare data set, accepted only if it starts like one.
      const group = u16(0, true);
      if (group !== 0x0002 && group !== 0x0008) return { dicom: false, uid: '', more: false };
    }
    dicom = true;

    let syntax = '';
    while (u16(at, true) === 0x0002) {                              // file meta: explicit VR little endian
      const h = header(at, true, true);
      if (h.length === UNDEFINED) throw BAD;
      if (h.element === 0x0010) syntax = text(h.value, h.length);
      at = h.value + h.length;
    }
    if (syntax === DEFLATED) return { dicom: true, uid: '', more: false };
    const little = syntax !== EXPLICIT_BE;
    const explicit = syntax ? syntax !== IMPLICIT_LE : looksLikeVr(at + 4);

    for (;;) {
      const h = header(at, explicit, little);
      const tag = h.group * 0x10000 + h.element;
      if (tag === 0x0020000d) {
        if (h.length === UNDEFINED || h.length > 64) return { dicom: true, uid: '', more: false };
        const uid = text(h.value, h.length);
        return { dicom: true, uid: /^[0-9]+(\.[0-9]+)*$/.test(uid) ? uid : '', more: false };
      }
      if (tag > 0x0020000d) return { dicom: true, uid: '', more: false };   // elements come in tag order
      at = skip(h, explicit, little);
    }
  } catch (error) {
    if (error === MORE) return { dicom, uid: '', more: true };
    if (error === BAD) return { dicom, uid: '', more: false };
    throw error;
  }
}

/**
 * StudyInstanceUID of one file ('' when it has none we can read), reading 16 KB first and 256 KB at most.
 * A file that is not DICOM comes back as { dicom: false } unless its extension says otherwise.
 */
export async function readStudyUid(file) {
  let length = Math.min(file.size, FIRST_BYTES);
  for (;;) {
    const head = parseStudyUid(new Uint8Array(await file.slice(0, length).arrayBuffer()));
    if (!head.more || length >= file.size || length >= MAX_BYTES) {
      return { dicom: head.dicom || DICOM_EXT.test(file.name), uid: head.more ? '' : head.uid };
    }
    length = Math.min(file.size, MAX_BYTES);
  }
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
 * Files are grouped by StudyInstanceUID; a file without one is grouped with its own folder. Each study is named
 * after the folder that holds its files (the deepest folder they share), never after the UID.
 *
 * @returns {{ name: string, studies: {name: string, files: File[]}[], skipped: number }}
 */
export async function buildDataset(files, onProgress = () => {}) {
  const candidates = usableFiles(files);
  let done = 0;
  const heads = await mapLimited(candidates, READ_PARALLEL, async (file) => {
    let head = { dicom: false, uid: '' };
    try { head = await readStudyUid(file); } catch (error) { /* unreadable: skipped below */ }
    done += 1;
    onProgress(done, candidates.length);
    return head;
  });

  const groups = new Map();
  let skipped = files.length - candidates.length;
  candidates.forEach((file, index) => {
    const { dicom, uid } = heads[index];
    if (!dicom) { skipped += 1; return; }
    const key = uid ? `uid:${uid}` : `dir:${folderOf(pathOf(file))}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(file);
  });

  const first = pathOf(files[0] || { name: '' });
  const root = first.includes('/') ? first.split('/')[0] : ''; // a folder pick; loose files have no root
  const studies = [...groups.values()].map((group) => {
    group.sort((a, b) => natural.compare(pathOf(a), pathOf(b)));
    const folders = group.map((file) => folderOf(pathOf(file)).split('/'));
    const shared = [];
    for (let depth = 0; folders.every((parts) => depth < parts.length && parts[depth] === folders[0][depth]); depth += 1) {
      shared.push(folders[0][depth]);
    }
    const folder = shared.length > 1 ? shared.slice(1).join('/') : '';
    return { name: folder || pathOf(group[0]).split('/').pop(), files: group, sortKey: pathOf(group[0]) };
  });
  studies.sort((a, b) => natural.compare(a.sortKey, b.sortKey));
  return { name: root, studies: studies.map(({ name, files: list }) => ({ name, files: list })), skipped };
}
