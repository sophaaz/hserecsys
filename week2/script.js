// script.js — Content-based movie recommender (Cosine ranking + Jaccard display)

/* ============================
   App bootstrap
   ============================ */

// Initialize the application when the window loads
window.onload = async function() {
    try {
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';

        // Load data (defined in data.js)
        await loadData();

        populateMoviesDropdown();
        resultElement.textContent = "Data loaded. Please select a movie — cosine similarity will find recs.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js (if loadData throws)
    }
};

/* ============================
   UI helpers
   ============================ */

function populateMoviesDropdown() {
    const selectElement = document.getElementById('movie-select');

    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));

    // Add movies to dropdown
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

/* ============================
   Similarity utilities
   ============================ */

// Cache for genre sets to avoid recreating Set objects repeatedly
const genreSetCache = new Map();

function getGenreSet(movie) {
    let set = genreSetCache.get(movie.id);
    if (!set) {
        set = new Set(movie.genres || []);
        genreSetCache.set(movie.id, set);
    }
    return set;
}

// Cosine similarity for binary genre vectors:
// cos = |A ∩ B| / (sqrt(|A|) * sqrt(|B|))
function cosineSimilarityGenres(setA, setB) {
    const sizeA = setA.size;
    const sizeB = setB.size;
    if (sizeA === 0 || sizeB === 0) return 0;

    let intersectionCount = 0;
    const [small, large] = sizeA < sizeB ? [setA, setB] : [setB, setA];
    for (const g of small) {
        if (large.has(g)) intersectionCount++;
    }

    const denom = Math.sqrt(sizeA) * Math.sqrt(sizeB);
    return denom > 0 ? intersectionCount / denom : 0;
}

// Jaccard similarity for sets:
// j = |A ∩ B| / |A ∪ B|
function jaccardSimilarityGenres(setA, setB) {
    const sizeA = setA.size;
    const sizeB = setB.size;
    if (sizeA === 0 && sizeB === 0) return 0;

    let intersectionCount = 0;
    const [small, large] = sizeA < sizeB ? [setA, setB] : [setB, setA];
    for (const g of small) {
        if (large.has(g)) intersectionCount++;
    }
    const unionSize = sizeA + sizeB - intersectionCount;
    return unionSize > 0 ? intersectionCount / unionSize : 0;
}

// Safe HTML escape for titles/genres
function escapeHTML(str) {
    return String(str).replace(/[&<>"']/g, s => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[s]);
}

function fmtScore(x) {
    return Number.isFinite(x) ? x.toFixed(3) : '0.000';
}

/* ============================
   Core: recommendations flow
   ============================ */

function getRecommendations() {
    const resultElement = document.getElementById('result');

    try {
        // Step 1: Get user input
        const selectElement = document.getElementById('movie-select');
        const selectedMovieId = parseInt(selectElement.value);

        if (isNaN(selectedMovieId)) {
            resultElement.textContent = "Please select a movie first.";
            resultElement.className = 'error';
            return;
        }

        // Step 2: Find the liked movie
        const likedMovie = movies.find(movie => movie.id === selectedMovieId);
        if (!likedMovie) {
            resultElement.textContent = "Error: Selected movie not found in database.";
            resultElement.className = 'error';
            return;
        }

        // Show loading message while processing
        resultElement.textContent = "Calculating recommendations (cosine + jaccard)...";
        resultElement.className = 'loading';

        // Allow UI to update before heavy computation
        setTimeout(() => {
            try {
                // Step 3: Prepare for similarity calculation
                const likedGenresSet = getGenreSet(likedMovie);
                const likedGenresList = [...likedGenresSet];
                const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);

                // Step 4: Score candidates (cosine + jaccard)
                const scoredMovies = candidateMovies.map(candidate => {
                    const candSet = getGenreSet(candidate);
                    const cosine = cosineSimilarityGenres(likedGenresSet, candSet);
                    const jaccard = jaccardSimilarityGenres(likedGenresSet, candSet);
                    return { ...candidate, cosine, jaccard };
                });

                // Step 5: Sort by cosine (desc), tie-breaker by title
                scoredMovies.sort((a, b) => {
                    if (b.cosine !== a.cosine) return b.cosine - a.cosine;
                    return a.title.localeCompare(b.title);
                });

                // Step 6: Top-K
                const TOP_K = 2;
                const topRecommendations = scoredMovies.slice(0, TOP_K);

                // Step 7: Display results
                if (topRecommendations.length > 0 && topRecommendations[0].cosine > 0) {
                    const likedGenresText = likedGenresList.length ? likedGenresList.join(', ') : '—';

                    const lines = topRecommendations.map((r, i) => {
                        const rGenres = [...getGenreSet(r)];
                        const rGenresText = rGenres.length ? rGenres.join(', ') : '—';
                        return `${i + 1}) “${escapeHTML(r.title)}” (genres: ${escapeHTML(rGenresText)}) — cosine: ${fmtScore(r.cosine)}, jaccard: ${fmtScore(r.jaccard)}`;
                    }).join('<br>');

                    resultElement.innerHTML =
                        `Because you liked “${escapeHTML(likedMovie.title)}” ` +
                        `<span style="color: var(--muted)">(genres: ${escapeHTML(likedGenresText)})</span>, ` +
                        `we recommend:<br>${lines}`;

                    resultElement.className = 'success';
                } else {
                    resultElement.textContent = `No strong matches found for “${likedMovie.title}”.`;
                    resultElement.className = 'error';
                }
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 50);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}

