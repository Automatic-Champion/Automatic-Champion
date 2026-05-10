export default function PitchSVG() {
  // Stripe rows for realistic grass effect — more vibrant greens
  const stripeCount = 14;
  const stripeHeight = 1050 / stripeCount;

  // Generate crowd dots (random-looking but deterministic)
  const crowdDotsTop: { cx: number; cy: number; r: number }[] = [];
  const crowdDotsBottom: { cx: number; cy: number; r: number }[] = [];
  for (let i = 0; i < 60; i++) {
    const seed = i * 17 + 7;
    crowdDotsTop.push({
      cx: 40 + (seed * 13) % 600,
      cy: 5 + (seed * 7) % 30,
      r: 2.0 + (seed % 3) * 0.8,
    });
    crowdDotsBottom.push({
      cx: 40 + ((seed + 100) * 11) % 600,
      cy: 1020 + ((seed + 50) * 7) % 25,
      r: 2.0 + ((seed + 3) % 3) * 0.8,
    });
  }

  return (
    <svg
      viewBox="0 0 680 1050"
      className="w-full h-auto"
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Grass stripes — vibrant */}
      {Array.from({ length: stripeCount }, (_, i) => (
        <rect
          key={i}
          x={0}
          y={i * stripeHeight}
          width={680}
          height={stripeHeight}
          fill={i % 2 === 0 ? "#1f8a42" : "#25a34f"}
        />
      ))}

      {/* Crowd dots — top */}
      {crowdDotsTop.map((dot, i) => (
        <circle
          key={`ct-${i}`}
          cx={dot.cx}
          cy={dot.cy}
          r={dot.r}
          fill="white"
          opacity={0.15}
        />
      ))}

      {/* Crowd dots — bottom */}
      {crowdDotsBottom.map((dot, i) => (
        <circle
          key={`cb-${i}`}
          cx={dot.cx}
          cy={dot.cy}
          r={dot.r}
          fill="white"
          opacity={0.15}
        />
      ))}

      {/* Vignette overlay — dark edges */}
      <defs>
        <radialGradient id="vignette" cx="50%" cy="50%" r="60%">
          <stop offset="0%" stopColor="transparent" stopOpacity="0" />
          <stop offset="100%" stopColor="#000000" stopOpacity="0.1" />
        </radialGradient>
      </defs>
      <rect x={0} y={0} width={680} height={1050} fill="url(#vignette)" />

      {/* Outer boundary */}
      <rect
        x={40}
        y={40}
        width={600}
        height={970}
        fill="none"
        stroke="white"
        strokeWidth={2}
      />

      {/* Center line */}
      <line
        x1={40}
        y1={525}
        x2={640}
        y2={525}
        stroke="white"
        strokeWidth={2}
      />

      {/* Center circle */}
      <circle
        cx={340}
        cy={525}
        r={91.5}
        fill="none"
        stroke="white"
        strokeWidth={2}
      />

      {/* Center spot */}
      <circle cx={340} cy={525} r={4} fill="white" />

      {/* Top penalty area (attackers' end) */}
      <rect
        x={148}
        y={40}
        width={384}
        height={165}
        fill="none"
        stroke="white"
        strokeWidth={2}
      />
      {/* Top goal area */}
      <rect
        x={228}
        y={40}
        width={224}
        height={55}
        fill="none"
        stroke="white"
        strokeWidth={2}
      />
      {/* Top penalty spot */}
      <circle cx={340} cy={152} r={4} fill="white" />
      {/* Top penalty arc */}
      <path
        d="M 280 205 A 91.5 91.5 0 0 0 400 205"
        fill="none"
        stroke="white"
        strokeWidth={2}
      />

      {/* Bottom penalty area (goalkeeper's end) */}
      <rect
        x={148}
        y={845}
        width={384}
        height={165}
        fill="none"
        stroke="white"
        strokeWidth={2}
      />
      {/* Bottom goal area */}
      <rect
        x={228}
        y={955}
        width={224}
        height={55}
        fill="none"
        stroke="white"
        strokeWidth={2}
      />
      {/* Bottom penalty spot */}
      <circle cx={340} cy={898} r={4} fill="white" />
      {/* Bottom penalty arc */}
      <path
        d="M 280 845 A 91.5 91.5 0 0 1 400 845"
        fill="none"
        stroke="white"
        strokeWidth={2}
      />

      {/* Corner arcs */}
      <path d="M 40 52 A 12 12 0 0 1 52 40" fill="none" stroke="white" strokeWidth={2} />
      <path d="M 628 40 A 12 12 0 0 1 640 52" fill="none" stroke="white" strokeWidth={2} />
      <path d="M 40 998 A 12 12 0 0 0 52 1010" fill="none" stroke="white" strokeWidth={2} />
      <path d="M 628 1010 A 12 12 0 0 0 640 998" fill="none" stroke="white" strokeWidth={2} />
    </svg>
  );
}
