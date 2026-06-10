import '@testing-library/jest-dom/vitest'

// jsdom does not implement scrollIntoView — stub it globally
window.HTMLElement.prototype.scrollIntoView = function () {}
