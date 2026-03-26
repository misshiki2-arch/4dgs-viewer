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

  float power = -0.5 * (vConic.x * dx * dx + vConic.z * dy * dy) - vConic.y * dx * dy;
  if (power > 0.0) discard;

  float alpha = min(0.99, vColorAlpha.a * exp(power));
  if (alpha < (1.0 / 255.0)) discard;

  outColor = vec4(vColorAlpha.rgb, alpha);
}
`;
