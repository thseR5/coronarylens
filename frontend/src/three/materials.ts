import * as THREE from 'three'

/** Shared uniforms: one clock for all striped materials. */
export const stripeUniforms = { uTime: { value: 0 } }

/**
 * Vessel material: MeshStandardMaterial + moving diagonal stripes for conformal-"uncertain" vessels.
 * Stripes alternate the base colour with near-black/near-white (picked by luminance), so the pattern
 * is readable without colour vision. `userData.stripes.value` (0..1) fades the pattern in/out.
 */
export type LitMaterial = THREE.MeshStandardMaterial | THREE.MeshLambertMaterial

/** `lite` = MeshLambertMaterial (cheaper per fragment) for software rasterisers. */
export function makeVesselMaterial(lite = false): LitMaterial {
  const m: LitMaterial = lite ? new THREE.MeshLambertMaterial() : new THREE.MeshStandardMaterial({ roughness: 0.42, metalness: 0.05 })
  const stripes = { value: 0 }
  m.userData.stripes = stripes
  m.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = stripeUniforms.uTime
    shader.uniforms.uStripes = stripes
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vStripePos;')
      .replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\nvStripePos = (modelMatrix * vec4(transformed, 1.0)).xyz;',
      )
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vStripePos;\nuniform float uTime;\nuniform float uStripes;')
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
        if (uStripes > 0.001) {
          float band = fract(dot(vStripePos, vec3(0.62, 0.55, 0.56)) * 16.0 - uTime * 0.9);
          float s = smoothstep(0.42, 0.5, band) * (1.0 - smoothstep(0.92, 1.0, band));
          float lum = dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722));
          vec3 alt = lum > 0.18 ? vec3(0.02) : vec3(0.92);
          diffuseColor.rgb = mix(diffuseColor.rgb, alt, s * 0.85 * uStripes);
        }`,
      )
  }
  m.customProgramCacheKey = () => 'cl-vessel-stripes'
  return m
}

/** Fresnel "x-ray" material: rim-bright, centre transparent, so the vessels read through the heart. */
export function makeXrayMaterial(color: string): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(color) }, uBase: { value: 0.04 } },
    vertexShader: /* glsl */ `
      varying vec3 vN; varying vec3 vV;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vN = normalize(normalMatrix * normal); vV = -mv.xyz;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor; uniform float uBase;
      varying vec3 vN; varying vec3 vV;
      void main() {
        float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.2);
        gl_FragColor = vec4(uColor, clamp(uBase + 0.85 * f, 0.0, 1.0));
        #include <colorspace_fragment>
      }`,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  })
}

export function makeContextMaterial(color: string, opacity = 1, lite = false): LitMaterial {
  const translucent = opacity < 1
  const Ctor = lite ? THREE.MeshLambertMaterial : THREE.MeshStandardMaterial
  return new Ctor({
    color,
    ...(lite ? {} : { roughness: 0.7, metalness: 0 }),
    transparent: translucent,
    opacity,
    depthWrite: !translucent,
    side: translucent ? THREE.FrontSide : THREE.DoubleSide,
  })
}
