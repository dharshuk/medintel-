/** Tailwind config for MedIntel - Perplexity + ChatGPT + Apple Vision hybrid theme */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#28F7CE',
        accent2: '#00B8FF',
        danger: '#FF4D6D',
        success: '#23c68b',
        warning: '#FFB800',
        'bg-dark': '#050618',
        'bg-navy': '#061225',
        'glass-border': 'rgba(255, 255, 255, 0.04)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      backdropBlur: {
        8: '8px',
        12: '12px',
      },
      boxShadow: {
        glow: '0 0 25px rgba(40, 247, 206, 0.35)',
        'glow-intense': '0 0 40px rgba(40, 247, 206, 0.6)',
        glass: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
      },
      animation: {
        pulseGlow: 'pulseGlow 3s ease-in-out infinite',
        heartbeat: 'heartbeat 2.5s ease-in-out infinite',
        slideIn: 'slideIn 0.3s ease-out',
        fadeIn: 'fadeIn 0.4s ease-in',
        wave: 'wave 1.2s ease-in-out infinite',
      },
      keyframes: {
        pulseGlow: {
          '0%, 100%': { opacity: 0.7, transform: 'scale(1)', boxShadow: '0 0 20px rgba(40, 247, 206, 0.3)' },
          '50%': { opacity: 1, transform: 'scale(1.05)', boxShadow: '0 0 40px rgba(40, 247, 206, 0.6)' },
        },
        heartbeat: {
          '0%, 100%': { transform: 'scaleX(1)' },
          '25%': { transform: 'scaleX(1.1)' },
          '50%': { transform: 'scaleX(0.92)' },
          '75%': { transform: 'scaleX(1.05)' },
        },
        slideIn: {
          '0%': { transform: 'translateX(100%)', opacity: 0 },
          '100%': { transform: 'translateX(0)', opacity: 1 },
        },
        fadeIn: {
          '0%': { opacity: 0 },
          '100%': { opacity: 1 },
        },
        wave: {
          '0%, 100%': { transform: 'scaleY(0.5)' },
          '50%': { transform: 'scaleY(1)' },
        },
      },
    },
  },
  plugins: [],
};
