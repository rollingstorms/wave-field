import { useMemo } from "react";
import { sampleBoardDisplay } from "../field/boardDisplay";
import { boardDisplayColor } from "../field/continuousColor";
import { BOARD_SIZE } from "../game/constants";

interface ContinuousPatternProps {
  values: number[][];
  samplesPerSquare?: number;
  normalize?: boolean;
}

export function ContinuousPattern({ values, samplesPerSquare = 6, normalize = false }: ContinuousPatternProps) {
  const samples = useMemo(() => sampleBoardDisplay(values, samplesPerSquare), [values, samplesPerSquare]);
  const resolution = BOARD_SIZE * samplesPerSquare;
  const maximum = Math.max(...values.flat().map(Math.abs), 0.001);
  const colorScale = normalize ? 2.4 / maximum : 1;

  return (
    <svg className="continuous-pattern" viewBox={`0 0 ${resolution} ${resolution}`} preserveAspectRatio="none" aria-hidden="true">
      {samples.flatMap((row, y) => row.map((sample, x) => (
        <rect
          key={`${x}-${y}`}
          x={x}
          y={y}
          width="1"
          height="1"
          fill={boardDisplayColor(sample.value * colorScale, sample.activity * colorScale)}
        />
      )))}
    </svg>
  );
}
