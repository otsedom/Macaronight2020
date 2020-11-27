precision mediump float;

// grab texcoords from the vertex shader
varying vec2 vTexCoord;

// our texture coming from p5
uniform sampler2D tex0;


void main() {

  vec2 uv = vTexCoord;
  // flip the y uvs
  uv.y = 1.0 - uv.y;

  // get the webcam as a vec4 using texture2D
  vec4 tex = texture2D(tex0, uv);
  
  // lets invert the colors just for kicks
  tex.rgb = 1.0 - tex.rgb;

  // output the grayscale value in all three rgb color channels
  gl_FragColor = tex;
}

