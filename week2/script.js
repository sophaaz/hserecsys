// script.js — Content-based movie recommender (Cosine similarity, vanilla JS)

// ============================
// App bootstrap
// ============================

// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';
        
        // Load data (defined in data.js)
        await loadData();
        
        // Populate dropdown and update status
        populateMoviesDropdown();
        resultElement.textContent = "Data loaded. Please select a movie - cosine similairy will find recs.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js (if loadData throws)
    }
};

// ============================
// UI helpers
// ============================

// Populate the movies dropdown with sorted movie titles
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

// ============================
// Similarity utilities (Cosine)
// ============================

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
    // Iterate through the smaller set for efficiency
    const [small, large] = sizeA < sizeB ? [setA, setB] : [setB, setA];
    for (const g of small) {
        if (large.has(g)) intersectionCount++;
    }

    const denom = Math.sqrt(sizeA) * Math.sqrt(sizeB);
    return denom > 0 ? intersectionCount / denom : 0;
}

// ============================
// Core: recommendations flow
// ============================

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
        resultElement.textContent = "Calculating recommendations (cosine similarity)...";
        resultElement.className = 'loading';
        
        // Use setTimeout to allow the UI to update before heavy computation
        setTimeout(() => {
            try {
                // Step 3: Prepare for similarity calculation
                const likedGenres = getGenreSet(likedMovie);
                const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);
                
                // Step 4: Calculate Cosine similarity scores (instead of Jaccard)
                const scoredMovies = candidateMovies.map(candidate => {
                    const candidateGenres = getGenreSet(candidate);
                    const score = cosineSimilarityGenres(likedGenres, candidateGenres);
                    return { ...candidate, score };
                });
                
                // Step 5: Sort by score in descending order; tie-breaker by title for stability
                scoredMovies.sort((a, b) => {
                    if (b.score !== a.score) return b.score - a.score;
                    return a.title.localeCompare(b.title);
                });
                
                // Step 6: Select top recommendations
                const TOP_K = 2;
                const topRecommendations = scoredMovies.slice(0, TOP_K);
                
                // Step 7: Display results
                if (topRecommendations.length > 0 && topRecommendations[0].score > 0) {
                    const recommendationTitles = topRecommendations.map(movie => movie.title);
                    resultElement.textContent = `Because you liked "${likedMovie.title}", we recommend: ${recommendationTitles.join(', ')}`;
                    resultElement.className = 'success';
                } else {
                    resultElement.textContent = `No strong cosine-based matches found for "${likedMovie.title}".`;
                    resultElement.className = 'error';
                }
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}
