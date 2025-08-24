// lê ?api=... da URL ou cai no localhost para dev local
window.API_BASE_URL = new URLSearchParams(location.search).get('api') || 'http://localhost:8000';
