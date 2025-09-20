You are an expert full-stack web developer who creates robust, well-commented, and modular web applications using only vanilla HTML, CSS, and JavaScript.

Your task is to generate the complete code for a "Content-Based Movie Recommender" web application based on the detailed specifications below. The application logic will be split into two separate JavaScript files: `data.js` for data loading and parsing, and `script.js` for UI and recommendation logic. Please provide the code for each of the four files—`index.html`, `style.css`, `data.js`, and `script.js`—separately and clearly labeled.

---

### **Project Specification: Content-Based Movie Recommender (Modular)**

#### **1. Overall Goal**

Build a single-page web application that recommends movies. The application will use `data.js` to load and parse movie and rating data from local files (`u.item`, `u.data`). The `script.js` file will then use this parsed data to populate the UI and calculate content-based recommendations using the Jaccard similarity index when a user makes a selection.

#### **2. File `index.html` - The Application Structure**

-   **DOCTYPE and Language:** The document should start with `<!DOCTYPE html>` and the `<html>` tag should specify `lang="en"`.
-   **Title:** The page title should be "Content-Based Movie Recommender".
-   **Main Heading:** Include an `<h1>` with the text "Content-Based Movie Recommender".
-   **Instructions:** Add a `<p>` tag with instructions like, "Select a movie you like, and we'll find similar ones for you!"
-   **Dropdown Menu:** Include a `<select>` element with the ID `movie-select`. This will be populated dynamically by JavaScript.
-   **Button:** Include a `<button>` with the text "Get Recommendations". When clicked, it must call the `getRecommendations()` JavaScript function.
-   **Result Display Area:** Include a `<div>` with the ID `result-box`. Inside this div, add a `<p>` tag with the ID `result`. This will be used to show loading messages and the final recommendations.
-   **File Linking:** This is a critical step. At the end of the `<body>`, link to **both** JavaScript files. `data.js` must be loaded **before** `script.js` because `script.js` depends on the functions and variables defined in `data.js`.
    ```
    <script src="data.js"></script>
    <script src="script.js"></script>
    ```

#### **3. File `style.css` - The Application Design**

-   **Layout:** Create a professional, modern, and user-friendly layout. All content should be centered on the page within a main container.
-   **Background:** The `<body>` should have a light, neutral background color (e.g., `#f4f7f6`).
-   **Container:** The main container holding all elements should have a white background, rounded corners (`border-radius`), and a subtle box shadow to make it pop.
-   **Typography:** Use a clean, sans-serif font like 'Helvetica' or 'Arial'.
-   **Controls:** The `<select>` dropdown and `<button>` should have consistent styling, with adequate padding and a clear visual hierarchy.
-   **Button:** The button should be inviting, with a distinct background color (e.g., a shade of blue), white text, and a hover effect (e.g., slightly darker background) to indicate interactivity.
-   **Result Area:** The `#result-box` should have some padding and a light background to separate it from the controls. The recommendation text inside `#result` should be bold and easy to read.

#### **4. File `data.js` - The Data Handling Module**

This file is responsible only for fetching and parsing the data from local files.

1.  **Global Variables:**
    -   Declare two global `let` variables, `movies` and `ratings`, initialized as empty arrays.

2.  **Primary Function: `loadData()`**
    -   This must be an `async` function.
    -   It will use the `fetch()` API to read `u.item` and `u.data`. Assume these files are in the same directory as `index.html`.
    -   Implement `try...catch` error handling to manage potential file loading failures. If a file fails to load, display an error message in the `#result` paragraph.
    -   Inside the `try` block, first `await` the fetch call for `u.item`, convert the response to text, and pass it to the `parseItemData` function.
    -   Then, `await` the fetch call for `u.data`, convert it to text, and pass it to the `parseRatingData` function.
    -   The function should implicitly return a `Promise` that resolves when the asynchronous operations are complete.

3.  **Parsing Function: `parseItemData(text)`**
    -   This function takes the raw text from `u.item` as input.
    -   It should define an array of the 18 genre names (from "Action" to "Western").
    -   It will split the input text into individual lines. For each line, it will:
        -   Split the line by the `|` delimiter.
        -   Extract the movie `id` (field 0) and `title` (field 1).
        -   Iterate through the last 19 fields to build an array of `genres` for the movie where the value is '1'.
        -   Create a movie object `{ id, title, genres }` and push it to the global `movies` array.

4.  **Parsing Function: `parseRatingData(text)`**
    -   This function takes the raw text from `u.data` as input.
    -   It will split the text into lines. For each line, it will:
        -   Split the line by the `\t` (tab) delimiter.
        -   Create a rating object `{ userId, itemId, rating, timestamp }` and push it to the global `ratings` array.

#### **5. File `script.js` - The UI and Logic Module**

This file handles the user interface and the recommendation logic. It will depend on the data loaded by `data.js`.

1.  **Initialization Logic:**
    -   Use `window.onload` to create an `async` function that initializes the application.
    -   Inside this function, `await` the `loadData()` function from `data.js`.
    -   After the data is successfully loaded, call `populateMoviesDropdown()` and set an initial status message in the result box (e.g., "Data loaded. Please select a movie.").

2.  **UI Function: `populateMoviesDropdown()`**
    -   This function gets the `<select>` element by its ID.
    -   It should sort the `movies` array alphabetically by title to improve user experience.
    -   It will then loop through the sorted `movies` array and create an `<option>` for each movie, setting its `value` to the movie `id` and its `innerText` to the movie `title`.

3.  **Core Logic: `getRecommendations()`**
    -   This is the main function for content-based filtering, triggered by the button click. It must perform the following steps in order:
        -   **Step 1 (Get User Input):** Get the `value` of the currently selected option from the `#movie-select` dropdown. This value is the movie ID as a string. Convert it to an integer.
        -   **Step 2 (Find Liked Movie):** Search the global `movies` array to find the movie object whose `id` matches the selected movie ID. Store this in a `likedMovie` variable. If no movie is found, display an error and exit.
        -   **Step 3 (Prepare for Similarity):** Create a JavaScript `Set` from the `genres` array of the `likedMovie`. Create a `candidateMovies` array by filtering the global `movies` array to exclude the `likedMovie`.
        -   **Step 4 (Calculate Scores):** Create a `scoredMovies` array by mapping over the `candidateMovies`. For each `candidateMovie`, calculate the **Jaccard Similarity Index** between its genre set and the `likedMovie`'s genre set. The formula is `(Size of Intersection) / (Size of Union)`. The resulting objects in the new array should be in the format `{...candidate, score: jaccardScore}`.
        -   **Step 5 (Sort by Score):** Sort the `scoredMovies` array in descending order based on the `score`.
        -   **Step 6 (Select Top Recommendations):** Take the first two movies from the sorted array using `.slice(0, 2)`.
        -   **Step 7 (Display Result):** Construct a user-friendly output string (e.g., "Because you liked '[Liked Movie Title]', we recommend: [Movie 1 Title], [Movie 2 Title]") and set it as the `innerText` of the `#result` paragraph.

---
Please now generate the complete code for the `index.html`, `style.css`, `data.js`, and `script.js` files based on these final, detailed specifications.
