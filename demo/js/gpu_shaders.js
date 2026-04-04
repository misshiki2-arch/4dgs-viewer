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

void main() {
  vec2 ndc = vec2(
    (aCenterPx.x / uViewportPx.x) * 2.0 - 1.0,
    1.0 - (aCenterPx.y / uViewportPx.y) * 2.0
  );

  gl_Position = vec4(ndc, 0.0, 1.0);
  gl_PointSize = max(1.0, aRadiusPx * 2.0);

  // Step24:
  // packed.colorAlpha.rgba をそのまま後段へ渡す。
  // 正式 alpha は vColorAlpha.a。
  vColorAlpha = aColorAlpha;
  vRadiusPx = aRadiusPx;
  vConic = aConic;
}
`;

export const GPU_STEP_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec4 vColorAlpha;
in float vRadiusPx;
in vec3 vConic;

out vec4 outColor;

void main() {
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  vec2 d = uv * vRadiusPx;
  float dx = d.x;
  float dy = d.y;

  // conic = [xx, xy, yy]
  float power =
    -0.5 * (vConic.x * dx * dx + vConic.z * dy * dy)
    - vConic.y * dx * dy;

  if (power > 0.0) {
    discard;
  }

  // Step24:
  // 正式 alpha は vColorAlpha.a のみ。
  // 旧 opacity のような別概念は shader 側に持ち込まない。
  float packedAlpha = vColorAlpha.a;
  float gaussianAlpha = packedAlpha * exp(power);
  float finalAlpha = min(0.99, gaussianAlpha);

  if (finalAlpha < (1.0 / 255.0)) {
    discard;
  }

  outColor = vec4(vColorAlpha.rgb, finalAlpha);
}
`;
