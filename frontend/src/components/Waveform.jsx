// Canvas-based waveform visualization reacting to voice activity

import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

function Waveform({ isActive }) {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const dataRef = useRef(Array(32).fill(0.2));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    // Set canvas size
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    };
    resize();
    window.addEventListener('resize', resize);

    // Animation loop
    const animate = () => {
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;

      ctx.clearRect(0, 0, width, height);

      const barCount = dataRef.current.length;
      const barWidth = width / barCount;
      const gap = 2;

      dataRef.current = dataRef.current.map((value) => {
        if (isActive) {
          // Animated values when active
          return Math.max(0.1, Math.min(1, value + (Math.random() - 0.5) * 0.3));
        } else {
          // Decay to baseline when inactive
          return Math.max(0.2, value * 0.95);
        }
      });

      dataRef.current.forEach((value, i) => {
        const barHeight = value * height * 0.8;
        const x = i * barWidth;
        const y = (height - barHeight) / 2;

        // Gradient
        const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
        gradient.addColorStop(0, isActive ? '#28F7CE' : '#28F7CE80');
        gradient.addColorStop(1, isActive ? '#00B8FF' : '#00B8FF40');

        ctx.fillStyle = gradient;
        ctx.fillRect(x + gap / 2, y, barWidth - gap, barHeight);
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive]);

  return (
    <motion.div
      className="glass-card overflow-hidden p-3"
      animate={isActive ? { borderColor: '#28F7CE40' } : { borderColor: 'rgba(255,255,255,0.04)' }}
      transition={{ duration: 0.3 }}
    >
      <canvas
        ref={canvasRef}
        className="h-16 w-full"
        style={{ display: 'block' }}
      />
    </motion.div>
  );
}

export default Waveform;
