// data.js
// In-browser parsing for MovieLens 100K: u.item (metadata), u.data (ratings)
// Exports global data structures and loadData()

/** @typedef {{rawId:number,index:number,title:string,year?:number,genres:string[],genreVec:Float32Array}} Movie */

// ---------- Exported globals ----------
/** @type {Movie[]} */
let movies = [];
/** @type {Map<number, number>} rawItemId -> dense index i */
let movieIndexByRawId = new Map();
/** @type {Map<number, number>} rawUserIndex (rawId -> dense index u) */
let userIndexByRawId = new Map();
/** @type {{u:number,i:number,r:number}[]} */
let ratingsTriples = [];
/** @type {Map<number, Set<number>>} dense user u -> set of dense item i */
let userRatedItems = new Map();
/** @type {{nUsers:number,nItems:number,nRatings:number,mean:number}} */
const STATS = { nUsers: 0, nItems: 0, nRatings: 0, mean: 0 };

/** MovieLens 100K genre flags order (19) */
const GENRES = [
  "unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
  "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
  "Mystery","Romance","Sci-Fi","Thriller","War","Western"
];

/**
 * Load and parse u.item and u.data from the current directory.
 * Populates movies, mappings, ratingsTriples, userRatedItems, STATS.
 */
async function loadData() {
  // Fetch files (ensure they are served via http(s) — file:// won't work)
  const [itemResp, dataResp] = await Promise.all([
    fetch('./u.item'),
    fetch('./u.data')
  ]);
  if (!itemResp.ok) throw new Error(`Failed to fetch u.item: ${itemResp.status}`);
  if (!dataResp.ok) throw new Error(`Failed to fetch u.data: ${dataResp.status}`);

  const [itemText, dataText] = await Promise.all([itemResp.text(), dataResp.text()]);

  // Parse u.item
  parseItems(itemText);

  // Parse u.data
  parseRatings(dataText);

  // Compute stats
  STATS.nItems = movies.length;
  STATS.nUsers = userIndexByRawId.size;
  STATS.nRatings = ratingsTriples.length;
  STATS.mean = ratingsTriples.length
    ? ratingsTriples.reduce((s, t) => s + t.r, 0) / ratingsTriples.length
    : 0;
}

// ------------------ Internal parsers ------------------

/**
 * Parse u.item (pipe-separated), fill movies[] and movieIndexByRawId
 * @param {string} text
 */
function parseItems(text) {
  movies = [];
  movieIndexByRawId.clear();

  const lines = text.split('\n');
  let denseIndex = 0;

  for (const lineRaw of lines) {
    const line = lineRaw.trim();
    if (!line) continue;

    // Expected: movieId|title|releaseDate|videoReleaseDate|imdbURL|g0|...|g18
    const parts = line.split('|');
    if (parts.length < 5 + 19) continue;

    const rawId = Number(parts[0]);
    const title = parts[1] || `Movie ${rawId}`;
    const year = extractYearFromTitle(title);

    const flags = parts.slice(5, 5 + 19).map(x => Number(x) || 0);
    const genres = [];
    for (let k = 0; k < 19; k++) if (flags[k] === 1) genres.push(GENRES[k]);

    const genreVec = new Float32Array(19);
    for (let k = 0; k < 19; k++) genreVec[k] = flags[k] ? 1 : 0;

    movieIndexByRawId.set(rawId, denseIndex);
    movies.push({ rawId, index: denseIndex, title, year, genres, genreVec });
    denseIndex++;
  }
}

/**
 * Parse u.data (tab-separated), fill ratingsTriples, userIndexByRawId, userRatedItems
 * @param {string} text
 */
function parseRatings(text) {
  ratingsTriples = [];
  userIndexByRawId.clear();
  userRatedItems.clear();

  const lines = text.split('\n');
  let userDenseCounter = 0;

  for (const lineRaw of lines) {
    const line = lineRaw.trim();
    if (!line) continue;

    // userId\titemId\trating\ttimestamp
    const parts = line.split('\t');
    if (parts.length < 3) continue;

    const rawU = Number(parts[0]);
    const rawI = Number(parts[1]);
    const r = Number(parts[2]);
    if (!Number.isFinite(rawU) || !Number.isFinite(rawI) || !Number.isFinite(r)) continue;

    // Map raw IDs to dense indices
    if (!userIndexByRawId.has(rawU)) {
      userIndexByRawId.set(rawU, userDenseCounter++);
    }
    const u = userIndexByRawId.get(rawU);

    const i = movieIndexByRawId.get(rawI);
    if (i === undefined) continue; // item not present in u.item

    ratingsTriples.push({ u, i, r });

    if (!userRatedItems.has(u)) userRatedItems.set(u, new Set());
    userRatedItems.get(u).add(i);
  }
}

/**
 * Attempt to extract year from a title like "Toy Story (1995)".
 * @param {string} t
 * @returns {number|undefined}
 */
function extractYearFromTitle(t) {
  const m = t.match(/\((\d{4})\)\s*$/);
  if (m) {
    const y = Number(m[1]);
    if (y >= 1900 && y <= 2100) return y;
  }
  return undefined;
}
