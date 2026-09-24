/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#18201d',
        canvas: '#f3f1eb',
        accent: '#167d69',
      },
      boxShadow: {
        panel: '0 1px 2px rgba(24, 32, 29, 0.06), 0 12px 36px rgba(24, 32, 29, 0.045)',
      },
    },
  },
  plugins: [],
}
