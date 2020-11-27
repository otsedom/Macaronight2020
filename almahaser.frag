// Inspired by Alma Haser https://scene360.com/art/112440/alma-hasers/

precision mediump float;

// grab texcoords from the vertex shader
varying vec2 vTexCoord;

// our texture coming from p5
uniform sampler2D tex0;
uniform float mouseX;
uniform vec2 u_resolution;
uniform float mouseY;
//varying vec4 vertColor;
//varying vec4 vertTexCoord;
//uniform vec2 texOffset; // sing processing
const vec2 texOffset = vec2(1.0, 1.0); // From https://www.nuomiphp.com/eplan/en/167810.html



float scale = 1.0;
float step = 0.05;

void main(void) {  
  vec2 stc = gl_FragCoord.xy/u_resolution;
  stc.y = 1.0 -stc.y;
  
  vec2 uv = vTexCoord;
  // flip the y uvs
  uv.y = 1.0 - uv.y;
  
  // get the webcam as a vec4 using texture2D
  //vec4 tex = vec4(0.0,0.0,0.0,1.0);
  vec4 tex = texture2D(tex0, uv);
  //if (uv.x>mouseX && uv.y > mouseY && uv.y<mouseY+4.0*step){
  if (stc.x>mouseX && stc.y > mouseY && stc.y<mouseY+12.0*step){
    
    // Alternating rows
	  if ( mod(stc.y*scale,2.0*step) <= step){
      // Positive displacement using step
      if ( mod(stc.x*scale,2.0*step) <= step)
        uv.x += (step)*texOffset.s;
      else // Negative displacement
        uv.x -= (step)*texOffset.s;
    }
    tex = texture2D(tex0, uv);
  }
  
  // output the grayscale value in all three rgb color channels
  gl_FragColor = tex;
  
}