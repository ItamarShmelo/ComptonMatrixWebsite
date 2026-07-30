(() => {
  'use strict';

  const root = document.documentElement;
  root.classList.add('reveal-animations');
  const themeToggle = document.querySelector('#theme-toggle');
  const navToggle = document.querySelector('#nav-toggle');
  const navLinks = document.querySelector('#primary-links');

  const preferredTheme = localStorage.getItem('mace-theme');
  if (preferredTheme === 'light' || preferredTheme === 'dark') {
    root.dataset.theme = preferredTheme;
  }

  function updateThemeButton() {
    if (!themeToggle) return;
    const light = root.dataset.theme === 'light';
    themeToggle.setAttribute('aria-label', light ? 'Switch to dark theme' : 'Switch to light theme');
  }

  updateThemeButton();

  themeToggle?.addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
    localStorage.setItem('mace-theme', root.dataset.theme);
    updateThemeButton();
  });

  navToggle?.addEventListener('click', () => {
    const open = navToggle.getAttribute('aria-expanded') === 'true';
    navToggle.setAttribute('aria-expanded', String(!open));
    navLinks?.classList.toggle('open', !open);
  });

  navLinks?.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      navToggle?.setAttribute('aria-expanded', 'false');
      navLinks.classList.remove('open');
    });
  });

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });

  document.querySelectorAll('.reveal').forEach((element) => observer.observe(element));
})();
