function getStoredTheme() {
    return localStorage.getItem('theme');
}
function setStoredTheme(theme) {
    localStorage.setItem('theme', theme);
}
function getPreferredTheme() {
    var storedTheme = getStoredTheme();
    if (storedTheme) {
        return storedTheme;
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
function setTheme(theme) {
    if (theme === 'auto') {
        document.documentElement.setAttribute('data-bs-theme', (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
    } else {
        document.documentElement.setAttribute('data-bs-theme', theme);
    }
}
setTheme(getPreferredTheme());
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    const storedTheme = getStoredTheme()
    if (storedTheme !== 'light' && storedTheme !== 'dark') {
      setTheme(getPreferredTheme())
    }
});
window.addEventListener('DOMContentLoaded', () => {
    document.getElementById("theme-toggle").addEventListener("click", () => {
        const theme = document.documentElement.getAttribute('data-bs-theme');
        if (theme === "dark") {
            setStoredTheme('light');
            setTheme('light');
        } else {
            setStoredTheme('dark');
            setTheme('dark');
        }
    });
});

