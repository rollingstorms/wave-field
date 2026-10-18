import { useLayoutEffect, useMemo, useRef } from "react";
import { sampleBoardDisplay } from "../field/boardDisplay";
import { boardDisplayColor } from "../field/continuousColor";
import { BOARD_SIZE } from "../game/constants";

interface ContinuousPatternProps {
  values: number[][];
  samplesPerSquare?: number;
  normalize?: boolean;
}

export function ContinuousPattern({ values, samplesPerSquare = 6, normalize = false }: ContinuousPatternProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const samples = useMemo(() => sampleBoardDisplay(values, samplesPerSquare), [values, samplesPerSquare]);
  const resolution = BOARD_SIZE * samplesPerSquare;
  const maximum = Math.max(...values.flat().map(Math.abs), 0.001);
  const colorScale = normalize ? 2.4 / maximum : 1;
  useLayoutEffect(() => {
    const context = canvasRef.current?.getContext("2d");
    if (!context) return;
    samples.forEach((row, y) => row.forEach((sample, x) => {
      context.fillStyle = boardDisplayColor(sample.value * colorScale, sample.activity * colorScale);
      context.fillRect(x, y, 1, 1);
    }));
  }, [colorScale, samples]);

  return (
    <canvas ref={canvasRef} className="continuous-pattern" width={resolution} height={resolution} aria-hidden="true" />
  );
}
