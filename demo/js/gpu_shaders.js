// Step24:
// packed direct draw の正式 shader 契約をここに固定する。
//
// 正式属性契約:
// - aCenterPx      : packed.centerPx.xy
// - aRadiusPx      : packed.radiusPx
// - aColorAlpha    : packed.colorAlpha.rgba
// - aConic         : packed.conic.xyz
//
// 重要:
// - 正式 alpha は aColorAlpha.a のみ
// - separate opacity attribute / uniform は持たない
// - CPU pack / upload descriptor / shader の契約を一本化する

export const GPU_STEP_VERTEX_SHADER = `#version 300 es
precision highp float;

in vec2 aCenterPx;
in float aRadiusPx;
in vec4 aColorAlpha;
in vec3 aConic;

uniform vec2 uViewportPx;

out vec4 vColorAlpha;
out float vRadiusPx;
out vec3 vConic;
out vec2 vCenterPx;

void main() {
  // Packed centers are consumed here in the same screen-space convention used by
  // the legacy/tile-debug overlays: pixel coordinates from a top-left origin.
  // CUDA-style point centers live on integer pixel indices, so the GL point
  // must be placed at the corresponding pixel center (+0.5 in OpenGL space).
  float x = ((aCenterPx.x + 0.5) / uViewportPx.x) * 2.0 - 1.0;
  float y = 1.0 - ((aCenterPx.y + 0.5) / uViewportPx.y) * 2.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
  gl_PointSize = max(1.0, aRadiusPx * 2.0);

  // Step24:
  // packed.colorAlpha.rgba をそのまま後段へ渡す。
  // 正式 alpha は vColorAlpha.a。
  vColorAlpha = aColorAlpha;
  vRadiusPx = aRadiusPx;
  vConic = aConic;
  vCenterPx = aCenterPx;
}
`;

export const GPU_STEP_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec4 vColorAlpha;
in float vRadiusPx;
in vec3 vConic;
in vec2 vCenterPx;

uniform vec2 uViewportPx;

out vec4 outColor;

void main() {
  vec2 pixelIndexPx = vec2(
    gl_FragCoord.x - 0.5,
    uViewportPx.y - gl_FragCoord.y - 0.5
  );
  vec2 d = pixelIndexPx - vCenterPx;
  float dx = d.x;
  float dy = d.y;

  float power =
    -0.5 * (vConic.x * dx * dx + vConic.z * dy * dy)
    - vConic.y * dx * dy;
  if (power > 0.0) discard;
  float packedAlpha = vColorAlpha.a;
  float gaussianAlpha = packedAlpha * exp(power);
  float finalAlpha = clamp(gaussianAlpha, 0.0, 0.99);
  if (finalAlpha < (1.0 / 255.0)) discard;

  outColor = vec4(vColorAlpha.rgb, finalAlpha);
}
`;
