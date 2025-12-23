// static/predictor/js/theme.js

(function () {
    const STORAGE_KEY = "site_theme";

    function applyTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
    }

    function getInitialTheme() {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved === "light" || saved === "dark") return saved;
        return "light";
    }

    function toggleTheme() {
        const current = document.documentElement.getAttribute("data-theme") || "light";
        const next = current === "dark" ? "light" : "dark";
        applyTheme(next);
        localStorage.setItem(STORAGE_KEY, next);
    }

    // Apply on load
    applyTheme(getInitialTheme());

    // Bind button when it exists
    window.addEventListener("DOMContentLoaded", () => {
        const btn = document.getElementById("themeToggle");
        if (btn) btn.addEventListener("click", toggleTheme);
    });
})();
