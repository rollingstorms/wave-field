import { build } from "esbuild";

const result = await build({
  stdin: {
    contents: `
      import { profileMobilityDiagnostics, positionProfileMobilityDiagnostics } from "./src/game/mechanicsAnalysis.ts";
      const isolated = profileMobilityDiagnostics();
      const opening = positionProfileMobilityDiagnostics();
      const rows = isolated.map((metric) => {
        const positions = opening.filter((row) => row.pieceType === metric.pieceType
          && row.profile.join(",") === metric.profile.join(","));
        return {
          piece: metric.pieceType,
          phase: metric.profile.map((value) => value > 0 ? "+" : "−").join(""),
          isolatedMoves: Number(metric.averageMoves.toFixed(2)),
          deadOrigins: metric.deadOrigins,
          openingReachable: positions.reduce((sum, row) => sum + row.reachableMoves, 0),
          openingPlayable: positions.reduce((sum, row) => sum + row.playableMoves, 0),
          openingSafe: positions.reduce((sum, row) => sum + row.safeMoves, 0),
          piecesWithoutSafeMoves: positions.reduce((sum, row) => sum + row.piecesWithoutSafeMoves, 0),
        };
      });
      console.table(rows);
    `,
    resolveDir: process.cwd(),
    sourcefile: "pattern-mobility-audit.ts",
  },
  bundle: true,
  platform: "node",
  format: "esm",
  write: false,
});

await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString("base64")}`);
